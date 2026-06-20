"""
Unified RuntimeLogger service for browser-use observability.

This module provides the RuntimeLogger adapter layer. All entry points
(Agent steps, browser commands, MCP tools, CLI tasks, sandbox/cloud
execution, skill_cli daemon) emit events ONLY through RuntimeLogger.

The entry layer (Agent / Browser / CLI / MCP / etc.) configures WHICH
sinks receive events:
  - console   : standard Python logging / stdout
  - event_bus : bubus EventBus (in-process pub/sub)
  - telemetry : PostHog anonymous product telemetry
  - cloud     : Cloud sync events (if BROWSER_USE_CLOUD_SYNC enabled)

The entry layer NEVER hand-builds payloads for individual sinks.
"""

from __future__ import annotations

import contextvars
import logging
import os
import socket
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from pydantic import BaseModel

from browser_use.config import CONFIG, is_running_in_docker
from browser_use.utils import get_browser_use_version, singleton

from .views import (
	ErrorInfo,
	EventSeverity,
	EventSource,
	EventType,
	OutputFile,
	RuntimeEvent,
	TokenUsage,
)

logger = logging.getLogger(__name__)


# ── RuntimeContext: thread/task-local linking state ────────────────────────


class _RuntimeContext:
	"""
	Thread / async-task-local context that auto-populates
	task_id / session_id / step / action_index for every event.

	Uses contextvars so nested async tasks and thread pools each
	get an independent copy that inherits from the parent.
	"""

	def __init__(self) -> None:
		self._cv_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('runtime_ctx_task_id', default=None)
		self._cv_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('runtime_ctx_session_id', default=None)
		self._cv_step: contextvars.ContextVar[int | None] = contextvars.ContextVar('runtime_ctx_step', default=None)
		self._cv_action_index: contextvars.ContextVar[int | None] = contextvars.ContextVar(
			'runtime_ctx_action_index', default=None
		)
		self._cv_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar('runtime_ctx_run_id', default=None)
		self._cv_parent_event_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
			'runtime_ctx_parent_event_id', default=None
		)

	# ── Getters ────────────────────────────────────────────────────────

	def snapshot(self) -> dict[str, Any]:
		return {
			'task_id': self._cv_task_id.get(),
			'session_id': self._cv_session_id.get(),
			'step': self._cv_step.get(),
			'action_index': self._cv_action_index.get(),
			'run_id': self._cv_run_id.get(),
			'parent_event_id': self._cv_parent_event_id.get(),
		}

	# ── Sync context managers ──────────────────────────────────────────

	def set_context_raw(
		self,
		*,
		task_id: str | None = None,
		session_id: str | None = None,
		step: int | None = None,
		action_index: int | None = None,
		run_id: str | None = None,
		parent_event_id: str | None = None,
	) -> dict[str, Any]:
		"""Non-contextmanager variant: returns tokens dict for later reset."""
		tokens: dict[str, Any] = {}
		if task_id is not None:
			tokens['task_id'] = self._cv_task_id.set(task_id)
		if session_id is not None:
			tokens['session_id'] = self._cv_session_id.set(session_id)
		if step is not None:
			tokens['step'] = self._cv_step.set(step)
		if action_index is not None:
			tokens['action_index'] = self._cv_action_index.set(action_index)
		if run_id is not None:
			tokens['run_id'] = self._cv_run_id.set(run_id)
		if parent_event_id is not None:
			tokens['parent_event_id'] = self._cv_parent_event_id.set(parent_event_id)
		return tokens

	def reset_context(self, tokens: dict[str, Any]) -> None:
		"""Reset contextvars using tokens returned by set_context_raw."""
		for key, token in reversed(tokens.items()):
			cv = getattr(self, f'_cv_{key}')
			cv.reset(token)

	@contextmanager
	def set_context(
		self,
		*,
		task_id: str | None = None,
		session_id: str | None = None,
		step: int | None = None,
		action_index: int | None = None,
		run_id: str | None = None,
		parent_event_id: str | None = None,
	) -> Iterator[None]:
		tokens = self.set_context_raw(
			task_id=task_id,
			session_id=session_id,
			step=step,
			action_index=action_index,
			run_id=run_id,
			parent_event_id=parent_event_id,
		)
		try:
			yield
		finally:
			self.reset_context(tokens)

	# ── Async context managers ─────────────────────────────────────────

	@asynccontextmanager
	async def async_set_context(
		self,
		*,
		task_id: str | None = None,
		session_id: str | None = None,
		step: int | None = None,
		action_index: int | None = None,
		run_id: str | None = None,
		parent_event_id: str | None = None,
	) -> AsyncIterator[None]:
		with self.set_context(
			task_id=task_id,
			session_id=session_id,
			step=step,
			action_index=action_index,
			run_id=run_id,
			parent_event_id=parent_event_id,
		):
			yield


# ── Sink configuration ─────────────────────────────────────────────────────


class SinkConfig(BaseModel):
	"""Which sinks a RuntimeLogger instance will deliver to."""

	console: bool = True
	event_bus: bool = False
	telemetry: bool = True
	cloud: bool = False
	min_console_severity: EventSeverity = EventSeverity.INFO
	min_telemetry_severity: EventSeverity = EventSeverity.WARNING


# ── RuntimeLogger: single entry point for ALL events ───────────────────────


@singleton
class RuntimeLogger:
	"""
	Unified adapter for ALL browser-use observability output.

	Typical usage by an entry layer::

	    from browser_use.observability_runtime import RuntimeLogger, EventSource

	    rl = RuntimeLogger(source=EventSource.AGENT)
	    rl.set_sinks(console=True, event_bus=True, telemetry=True, cloud=False)

	    with rl.context(task_id=agent.task_id, session_id=agent.session_id):
	        rl.step_start(step=0, message='Beginning step 0')
	        ...
	        rl.info('Searching page for elements', data={'url': url})
	        ...
	        rl.step_end(step=0, duration_ms=elapsed_ms)
	"""

	def __init__(self, source: EventSource = EventSource.AGENT) -> None:
		self._default_source = source
		self._ctx = _RuntimeContext()
		self._sink_config = SinkConfig()
		self._sinks_configured = False
		self._event_bus = None  # lazy bubus EventBus
		self._telemetry_service = None  # lazy ProductTelemetry
		self._cloud_sync_client = None  # lazy cloud sync (set by entry layer)

		# Cached environmental fields (set once at construction)
		try:
			self._hostname: str | None = socket.gethostname()
		except Exception:
			self._hostname = None
		self._pid: int = os.getpid()
		try:
			self._version: str | None = get_browser_use_version()
		except Exception:
			self._version = None
		self._is_docker: bool = is_running_in_docker()

	# ── Configuration (entry layers use these) ─────────────────────────

	def set_source(self, source: EventSource) -> None:
		"""Change the default source component for this logger instance."""
		self._default_source = source

	def set_sinks(
		self,
		*,
		console: bool | None = None,
		event_bus: bool | None = None,
		telemetry: bool | None = None,
		cloud: bool | None = None,
		min_console_severity: EventSeverity | None = None,
		min_telemetry_severity: EventSeverity | None = None,
	) -> None:
		"""
		Choose which sinks this logger emits to.

		Called ONLY by the entry layer (Agent / CLI / MCP / ...).
		Business logic code should NEVER call this.
		"""
		if console is not None:
			self._sink_config.console = console
		if event_bus is not None:
			self._sink_config.event_bus = event_bus
		if telemetry is not None:
			self._sink_config.telemetry = telemetry
		if cloud is not None:
			self._sink_config.cloud = cloud
		if min_console_severity is not None:
			self._sink_config.min_console_severity = min_console_severity
		if min_telemetry_severity is not None:
			self._sink_config.min_telemetry_severity = min_telemetry_severity
		self._sinks_configured = True

	def set_cloud_sync_client(self, client: Any) -> None:
		"""Attach a cloud sync client (agent.cloud_sync). Only set by the Agent."""
		self._cloud_sync_client = client

	# ── Context management (delegate to _RuntimeContext) ───────────────

	def set_context(self, **kwargs: Any) -> dict[str, Any]:
		"""Sync: set linking fields and return a reset token dict.

		Usage::

		    token = rl.set_context(task_id='...', session_id='...')
		    try:
		        ...
		    finally:
		        rl.reset_context(token)
		"""
		return self._ctx.set_context_raw(**kwargs)

	def reset_context(self, token: Any) -> None:
		"""Reset context to a previous state (returned by set_context)."""
		self._ctx.reset_context(token)

	def context(self, **kwargs: Any) -> Any:
		"""Sync context manager for linking state."""
		return self._ctx.set_context(**kwargs)

	def async_context(self, **kwargs: Any) -> Any:
		"""Async context manager for linking state."""
		return self._ctx.async_set_context(**kwargs)

	# ── Duck-typed telemetry bridge (accessed by entry layers) ────────

	@property
	def _telemetry(self) -> _TelemetryBridge:
		"""Backwards-compat alias: entry layers set _telemetry.telemetry_client.

		We expose a tiny bridge object with a ``telemetry_client`` setter
		that writes through to ``self._telemetry_service``.
		"""
		return _TelemetryBridge(self)

	# ── Direct event emission ──────────────────────────────────────────

	def emit(self, event: RuntimeEvent) -> RuntimeEvent:
		"""
		Emit a fully-built RuntimeEvent to all configured sinks.

		This is the single dispatch point. Every other method on this
		class ends up calling emit() exactly once.
		"""
		# Auto-populate missing linking fields from context
		snap = self._ctx.snapshot()
		changed = False
		for field_name in ('task_id', 'session_id', 'step', 'action_index', 'run_id', 'parent_event_id'):
			if getattr(event, field_name) is None and snap.get(field_name) is not None:
				setattr(event, field_name, snap[field_name])
				changed = True
		# Auto-populate environmental fields if unset
		if event.hostname is None:
			event.hostname = self._hostname
		if event.pid is None:
			event.pid = self._pid
		if event.version is None:
			event.version = self._version
		if event.is_docker is None:
			event.is_docker = self._is_docker

		# Dispatch to each sink
		try:
			self._dispatch_to_sinks(event)
		except Exception as exc:  # pragma: no cover - defensive
			# Never let observability break business logic
			logger.debug('RuntimeLogger dispatch failed: %s', exc)

		return event

	# ── Convenience API (preferred over building RuntimeEvent by hand) ──

	def log(
		self,
		message: str,
		*,
		severity: EventSeverity = EventSeverity.INFO,
		event_type: EventType = EventType.INFO,
		source: EventSource | None = None,
		data: dict[str, Any] | None = None,
		duration_ms: float | None = None,
		error: BaseException | ErrorInfo | None = None,
		output_files: list[OutputFile] | None = None,
		tokens: TokenUsage | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		"""
		Primary convenience method — log a message at a given severity.

		Examples::

		    rt.info('Browser started', data={'cdp_url': url})
		    rt.warning('Navigation slow', duration_ms=1200.0)
		"""
		if isinstance(error, BaseException):
			err_info = ErrorInfo.from_exception(error)
		else:
			err_info = error

		evt = RuntimeEvent(
			source=source or self._default_source,
			event_type=event_type,
			severity=severity,
			message=message,
			data=data or {},
			duration_ms=duration_ms,
			error=err_info,
			output_files=output_files or [],
			tokens=tokens,
			**link_fields,
		)
		return self.emit(evt)

	def debug(self, message: str, **kwargs: Any) -> RuntimeEvent:
		return self.log(message, severity=EventSeverity.DEBUG, event_type=EventType.DEBUG, **kwargs)

	def info(self, message: str, **kwargs: Any) -> RuntimeEvent:
		return self.log(message, severity=EventSeverity.INFO, event_type=EventType.INFO, **kwargs)

	def warning(self, message: str, **kwargs: Any) -> RuntimeEvent:
		return self.log(message, severity=EventSeverity.WARNING, event_type=EventType.WARNING, **kwargs)

	def warn(self, message: str, **kwargs: Any) -> RuntimeEvent:
		return self.warning(message, **kwargs)

	def error(self, message: str, **kwargs: Any) -> RuntimeEvent:
		return self.log(message, severity=EventSeverity.ERROR, event_type=EventType.ERROR, **kwargs)

	def critical(self, message: str, **kwargs: Any) -> RuntimeEvent:
		return self.log(message, severity=EventSeverity.CRITICAL, event_type=EventType.ERROR, **kwargs)

	def exception(
		self,
		exc: BaseException,
		message: str | None = None,
		*,
		severity: EventSeverity = EventSeverity.ERROR,
		event_type: EventType = EventType.EXCEPTION,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Log an exception (with automatic traceback capture)."""
		return self.log(
			message=message or str(exc),
			severity=severity,
			event_type=event_type,
			error=exc,
			**kwargs,
		)

	# ── Agent-specific helpers ─────────────────────────────────────────

	def task_start(self, *, task: str, model: str | None = None, **link_fields: Any) -> RuntimeEvent:
		return self.log(
			f'Starting task: {task[:120]}{"…" if len(task) > 120 else ""}',
			event_type=EventType.TASK_START,
			data={'task': task, 'model': model} if model else {'task': task},
			**link_fields,
		)

	def task_end(
		self,
		*,
		success: bool | None = None,
		duration_ms: float | None = None,
		final_result: str | None = None,
		tokens: TokenUsage | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		data: dict[str, Any] = {'success': success}
		if final_result is not None:
			data['final_result'] = final_result[:2000] if len(final_result) > 2000 else final_result
		return self.log(
			'Task ended' if success is None else f'Task {"succeeded" if success else "failed"}',
			event_type=EventType.TASK_END,
			duration_ms=duration_ms,
			tokens=tokens,
			data=data,
			**link_fields,
		)

	def step_start(self, *, step: int, message: str | None = None, **link_fields: Any) -> RuntimeEvent:
		return self.log(
			message or f'Step {step} starting',
			event_type=EventType.STEP_START,
			step=step,
			**link_fields,
		)

	def step_end(
		self,
		*,
		step: int,
		duration_ms: float | None = None,
		n_actions: int | None = None,
		success: bool | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		data: dict[str, Any] = {}
		if n_actions is not None:
			data['n_actions'] = n_actions
		if success is not None:
			data['success'] = success
		return self.log(
			f'Step {step} completed',
			event_type=EventType.STEP_END,
			step=step,
			duration_ms=duration_ms,
			data=data,
			**link_fields,
		)

	def agent_action(
		self,
		*,
		action_name: str,
		action_params: dict[str, Any] | None = None,
		action_index: int | None = None,
		step: int | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		return self.log(
			f'Action: {action_name}',
			event_type=EventType.AGENT_ACTION,
			severity=EventSeverity.DEBUG,
			action_index=action_index,
			step=step,
			data={'action_name': action_name, 'action_params': action_params or {}},
			**link_fields,
		)

	def agent_action_result(
		self,
		*,
		action_name: str,
		action_index: int | None = None,
		step: int | None = None,
		duration_ms: float | None = None,
		error: BaseException | ErrorInfo | None = None,
		extracted_content: str | None = None,
		output_files: list[OutputFile] | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		data: dict[str, Any] = {'action_name': action_name}
		if extracted_content is not None:
			# Truncate long content for event payload
			data['extracted_content'] = extracted_content[:1000] + '…' if len(extracted_content) > 1000 else extracted_content
		return self.log(
			f'Action result: {action_name}',
			event_type=EventType.AGENT_ACTION_RESULT,
			severity=EventSeverity.DEBUG if error is None else EventSeverity.ERROR,
			action_index=action_index,
			step=step,
			duration_ms=duration_ms,
			error=error,
			data=data,
			output_files=output_files,
			**link_fields,
		)

	# ── Browser-specific helpers ───────────────────────────────────────

	def browser_event(
		self,
		*,
		event_type: EventType,
		message: str,
		severity: EventSeverity = EventSeverity.INFO,
		url: str | None = None,
		duration_ms: float | None = None,
		data: dict[str, Any] | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		merged_data = dict(data or {})
		if url is not None:
			merged_data['url'] = url
		return self.log(
			message,
			source=EventSource.BROWSER,
			event_type=event_type,
			severity=severity,
			duration_ms=duration_ms,
			data=merged_data,
			**link_fields,
		)

	# ── MCP-specific helpers ───────────────────────────────────────────

	def mcp_tool_call(
		self,
		*,
		tool_name: str,
		arguments: dict[str, Any] | None = None,
		server_name: str | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		data: dict[str, Any] = {'tool_name': tool_name}
		if arguments is not None:
			# Sanitize: don't log full argument contents by default
			data['arg_keys'] = list(arguments.keys())
		if server_name is not None:
			data['server_name'] = server_name
		return self.log(
			f'MCP tool call: {tool_name}',
			event_type=EventType.MCP_TOOL_CALL,
			data=data,
			**link_fields,
		)

	def mcp_tool_result(
		self,
		*,
		tool_name: str,
		duration_ms: float | None = None,
		error: BaseException | ErrorInfo | None = None,
		server_name: str | None = None,
		**link_fields: Any,
	) -> RuntimeEvent:
		data: dict[str, Any] = {'tool_name': tool_name}
		if server_name is not None:
			data['server_name'] = server_name
		return self.log(
			f'MCP tool result: {tool_name}',
			event_type=EventType.MCP_TOOL_RESULT,
			severity=EventSeverity.DEBUG if error is None else EventSeverity.ERROR,
			duration_ms=duration_ms,
			error=error,
			data=data,
			**link_fields,
		)

	# ── Timing helpers ─────────────────────────────────────────────────

	@contextmanager
	def timed(
		self,
		message: str,
		*,
		severity: EventSeverity = EventSeverity.DEBUG,
		event_type: EventType = EventType.METRIC,
		**kwargs: Any,
	) -> Iterator[RuntimeEvent]:
		"""
		Context manager that measures wall time and emits a RuntimeEvent
		on exit with ``duration_ms`` populated.

		Usage::

		    with rt.timed('DOM serialization', source=EventSource.DOM) as evt:
		        result = dom_service.serialize()
		    # evt.duration_ms is now set
		"""
		start = time.perf_counter()
		emitted: RuntimeEvent | None = None
		try:
			# Yield a placeholder event so caller can attach data
			placeholder = RuntimeEvent(
				source=kwargs.get('source') or self._default_source,
				event_type=event_type,
				severity=severity,
				message=message,
				data=kwargs.get('data') or {},
			)
			yield placeholder
			# If caller mutated placeholder.data, merge it back into kwargs
			if placeholder.data:
				kwargs.setdefault('data', {})
				kwargs['data'].update(placeholder.data)
		except BaseException as exc:
			duration_ms = (time.perf_counter() - start) * 1000.0
			emitted = self.log(
				message,
				severity=EventSeverity.ERROR,
				event_type=event_type,
				duration_ms=duration_ms,
				error=exc,
				**kwargs,
			)
			raise
		else:
			duration_ms = (time.perf_counter() - start) * 1000.0
			emitted = self.log(
				message,
				severity=severity,
				event_type=event_type,
				duration_ms=duration_ms,
				**kwargs,
			)
		finally:
			pass

	@asynccontextmanager
	async def async_timed(
		self,
		message: str,
		*,
		severity: EventSeverity = EventSeverity.DEBUG,
		event_type: EventType = EventType.METRIC,
		**kwargs: Any,
	) -> AsyncIterator[RuntimeEvent]:
		"""Async version of :meth:`timed`."""
		start = time.perf_counter()
		placeholder = RuntimeEvent(
			source=kwargs.get('source') or self._default_source,
			event_type=event_type,
			severity=severity,
			message=message,
			data=kwargs.get('data') or {},
		)
		try:
			yield placeholder
			if placeholder.data:
				kwargs.setdefault('data', {})
				kwargs['data'].update(placeholder.data)
		except BaseException as exc:
			duration_ms = (time.perf_counter() - start) * 1000.0
			self.log(
				message,
				severity=EventSeverity.ERROR,
				event_type=event_type,
				duration_ms=duration_ms,
				error=exc,
				**kwargs,
			)
			raise
		else:
			duration_ms = (time.perf_counter() - start) * 1000.0
			self.log(
				message,
				severity=severity,
				event_type=event_type,
				duration_ms=duration_ms,
				**kwargs,
			)

	# ── Internal: sink dispatch ────────────────────────────────────────

	def _dispatch_to_sinks(self, event: RuntimeEvent) -> None:
		"""Route an event to each enabled sink, respecting min-severity filters."""
		if self._sink_config.console and self._meets_severity(event.severity, self._sink_config.min_console_severity):
			self._sink_console(event)

		if self._sink_config.event_bus:
			self._sink_event_bus(event)

		if self._sink_config.telemetry and self._meets_severity(event.severity, self._sink_config.min_telemetry_severity):
			self._sink_telemetry(event)

		if self._sink_config.cloud:
			self._sink_cloud(event)

	@staticmethod
	def _meets_severity(actual: EventSeverity, minimum: EventSeverity) -> bool:
		ordering = {
			EventSeverity.DEBUG: 0,
			EventSeverity.INFO: 1,
			EventSeverity.WARNING: 2,
			EventSeverity.ERROR: 3,
			EventSeverity.CRITICAL: 4,
		}
		return ordering[actual] >= ordering[minimum]

	def _sink_console(self, event: RuntimeEvent) -> None:
		"""Deliver event via standard Python logging."""
		# Use the event's source as the logger name so users can filter
		logger_name = f'browser_use.rt.{event.source.value}'
		py_logger = logging.getLogger(logger_name)
		# Build extra dict for structured loggers
		extra: dict[str, Any] = {
			'runtime_event_id': event.event_id,
			'runtime_event_type': event.event_type.value,
			'runtime_source': event.source.value,
		}
		if event.task_id:
			extra['task_id'] = event.task_id
		if event.session_id:
			extra['session_id'] = event.session_id
		if event.step is not None:
			extra['step'] = event.step
		line = event.to_console_line()
		py_logger.log(event.severity.to_logging_level(), line, extra=extra)
		# Also log traceback separately if present
		if event.error and event.error.error_stack:
			py_logger.log(
				event.severity.to_logging_level(),
				'Traceback:\n%s',
				event.error.error_stack,
				extra=extra,
			)

	def _sink_event_bus(self, event: RuntimeEvent) -> None:
		"""Deliver event via bubus EventBus (in-process pub/sub)."""
		try:
			if self._event_bus is None:
				from bubus import EventBus

				self._event_bus = EventBus()
			# Use RuntimeEventBusBridge to wrap RuntimeEvent in a
			# bubus.BaseEvent-compatible object for dispatch.
			from .views import RuntimeEventBusBridge

			bridge = RuntimeEventBusBridge.wrap(event)
			self._event_bus.dispatch(bridge)  # type: ignore[arg-type]
		except Exception as exc:  # pragma: no cover - defensive
			logger.debug('bubus EventBus dispatch failed: %s', exc)

	def _sink_telemetry(self, event: RuntimeEvent) -> None:
		"""Deliver event to PostHog via ProductTelemetry."""
		if not CONFIG.ANONYMIZED_TELEMETRY:
			return
		try:
			if self._telemetry_service is None:
				from browser_use.telemetry.service import ProductTelemetry

				self._telemetry_service = ProductTelemetry()  # type: ignore[call-arg]

			# Bridge: create a BaseTelemetryEvent-compatible wrapper
			# using the RuntimeEvent's to_telemetry_properties() output
			props = event.to_telemetry_properties()
			wrapper = _RuntimeTelemetryWrapper(name=self._telemetry_event_name(event), properties=props)
			self._telemetry_service.capture(wrapper)  # type: ignore[arg-type]
		except Exception as exc:  # pragma: no cover - defensive
			logger.debug('Telemetry dispatch failed: %s', exc)

	@staticmethod
	def _telemetry_event_name(event: RuntimeEvent) -> str:
		"""Map a RuntimeEvent to a stable PostHog event name."""
		# A smaller set of categories than the full EventType enum
		# keeps the PostHog schema manageable.
		source_prefix = event.source.value
		if event.event_type in {EventType.TASK_START, EventType.TASK_END}:
			return f'{source_prefix}_task'
		if event.event_type in {EventType.STEP_START, EventType.STEP_END}:
			return f'{source_prefix}_step'
		if event.event_type in {EventType.AGENT_ACTION, EventType.AGENT_ACTION_RESULT}:
			return f'{source_prefix}_action'
		if event.event_type in {EventType.EXCEPTION, EventType.ERROR}:
			return f'{source_prefix}_error'
		if event.event_type in {EventType.LLM_CALL, EventType.LLM_RESULT}:
			return f'{source_prefix}_llm'
		if event.event_type in {EventType.MCP_TOOL_CALL, EventType.MCP_TOOL_RESULT}:
			return f'{source_prefix}_mcp_tool'
		if event.event_type in {EventType.SANDBOX_START, EventType.SANDBOX_END}:
			return f'{source_prefix}_sandbox'
		return f'{source_prefix}_event'

	def _sink_cloud(self, event: RuntimeEvent) -> None:
		"""Deliver event to browser-use cloud sync (if client attached)."""
		if self._cloud_sync_client is None:
			return
		try:
			# Only sync high-signal events to cloud (not every debug line)
			if event.severity in {EventSeverity.DEBUG, EventSeverity.INFO} and event.event_type not in {
				EventType.STEP_START,
				EventType.STEP_END,
				EventType.TASK_START,
				EventType.TASK_END,
				EventType.SESSION_START,
				EventType.SESSION_END,
				EventType.AGENT_ACTION,
				EventType.AGENT_ACTION_RESULT,
			}:
				return

			sync_client = self._cloud_sync_client
			# The cloud sync interface uses a per-event-type dispatch.
			# We delegate to a small helper method on the sync client
			# if it supports it, otherwise no-op.
			handler = getattr(sync_client, 'handle_runtime_event', None)
			if callable(handler):
				handler(event)
		except Exception as exc:  # pragma: no cover - defensive
			logger.debug('Cloud sync dispatch failed: %s', exc)


# ── Bridge class for telemetry ─────────────────────────────────────────────


class _RuntimeTelemetryWrapper:
	"""
	Duck-typed wrapper that satisfies ``BaseTelemetryEvent`` protocol
	for ProductTelemetry.capture().

	ProductTelemetry only reads ``event.name`` and ``event.properties``,
	so we don't need to actually inherit from BaseTelemetryEvent.
	"""

	def __init__(self, name: str, properties: dict[str, Any]) -> None:
		self.name = name
		self.properties = properties


class _TelemetryBridge:
	"""
	Tiny backwards-compat bridge: entry layers write
	``runtime_logger._telemetry.telemetry_client = product_telemetry``,
	we translate that into ``runtime_logger._telemetry_service``.
	"""

	__slots__ = ('_logger',)

	def __init__(self, logger: Any) -> None:
		self._logger = logger

	@property
	def telemetry_client(self) -> Any:
		return self._logger._telemetry_service

	@telemetry_client.setter
	def telemetry_client(self, value: Any) -> None:
		self._logger._telemetry_service = value
