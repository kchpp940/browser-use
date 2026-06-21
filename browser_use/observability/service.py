"""RuntimeLogger adapter layer for unified event routing.

This module provides the RuntimeLogger which accepts RuntimeEvent objects and routes
them to configured output sinks. Each sink adapts the unified event format to the
specific requirements of the output target (console, event bus, telemetry, cloud).

Entry points (Agent, CLI, MCP, etc.) only need to emit RuntimeEvent objects - they
no longer need to hand-write different log formats or event payloads for different
output targets.
"""

from __future__ import annotations

import json
import logging
import traceback
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from browser_use.config import CONFIG
from browser_use.utils import singleton

from .views import (
	EventSeverity,
	EventSource,
	EventType,
	RuntimeContext,
	RuntimeEvent,
)

logger = logging.getLogger(__name__)


S = TypeVar('S', bound='BaseSink')


class BaseSink(ABC):
	"""Abstract base class for event sinks."""

	def __init__(self, name: str, enabled: bool = True):
		self.name = name
		self.enabled = enabled
		self.filter_event_types: set[str] | None = None
		self.filter_severity: EventSeverity = EventSeverity.DEBUG
		self.filter_sources: set[str] | None = None

	def configure(
		self,
		filter_event_types: Sequence[str] | None = None,
		filter_severity: EventSeverity = EventSeverity.DEBUG,
		filter_sources: Sequence[str] | None = None,
		enabled: bool = True,
	) -> None:
		"""Configure filter rules for this sink."""
		self.enabled = enabled
		self.filter_event_types = set(filter_event_types) if filter_event_types else None
		self.filter_severity = filter_severity
		self.filter_sources = set(filter_sources) if filter_sources else None

	def should_accept(self, event: RuntimeEvent) -> bool:
		"""Check if this sink should accept the event based on filters."""
		if not self.enabled:
			return False

		severity_order = [
			EventSeverity.DEBUG,
			EventSeverity.INFO,
			EventSeverity.WARNING,
			EventSeverity.ERROR,
			EventSeverity.RESULT,
		]
		if severity_order.index(event.severity) < severity_order.index(self.filter_severity):
			return False

		if self.filter_event_types and event.event_type not in self.filter_event_types:
			return False

		if self.filter_sources and event.source not in self.filter_sources:
			return False

		return True

	@abstractmethod
	def emit(self, event: RuntimeEvent) -> None:
		"""Emit the event to this sink. Must be implemented by subclasses."""
		pass

	async def emit_async(self, event: RuntimeEvent) -> None:
		"""Async version of emit for sinks that need async I/O."""
		self.emit(event)

	def flush(self) -> None:
		"""Flush any buffered events."""
		pass

	def cleanup(self) -> None:
		"""Clean up resources."""
		pass


class ConsoleSink(BaseSink):
	"""Sink that outputs formatted events to the console using standard logging.

	Adapts RuntimeEvent to the existing logging infrastructure, preserving
	the current log format and behavior.
	"""

	_SEVERITY_TO_LOG_LEVEL = {
		EventSeverity.DEBUG: logging.DEBUG,
		EventSeverity.INFO: logging.INFO,
		EventSeverity.WARNING: logging.WARNING,
		EventSeverity.ERROR: logging.ERROR,
		EventSeverity.RESULT: 35,  # Custom RESULT level
	}

	def __init__(self, name: str = 'console', enabled: bool = True):
		super().__init__(name, enabled)
		self._logger = logging.getLogger('browser_use')

	def _get_component_name(self, event: RuntimeEvent) -> str:
		"""Get a clean component name for the log message."""
		source = event.source
		if isinstance(source, EventSource):
			source = source.value

		if source == 'agent':
			task_suffix = f'🅰 {event.task_id[-4:]}' if event.task_id else ''
			browser_suffix = f'⇢ 🅑 {event.browser_session_id[-4:]}' if event.browser_session_id else ''
			if task_suffix or browser_suffix:
				return f'Agent{task_suffix} {browser_suffix}'.strip()
			return 'Agent'
		elif source == 'browser':
			return 'BrowserSession'
		elif source == 'tools':
			return 'tools'
		elif source == 'dom':
			return 'dom'
		elif source == 'cli':
			return 'CLI'
		elif source == 'mcp':
			return 'MCP'
		elif source == 'sandbox':
			return 'Sandbox'
		elif source == 'skill_cli':
			return 'SkillCLI'
		elif source == 'llm':
			return 'LLM'

		return source.capitalize() if source else 'Unknown'

	def _format_message(self, event: RuntimeEvent) -> str:
		"""Format the event message for console output."""
		parts = []

		if event.step is not None:
			parts.append(f'[Step {event.step}]')

		if event.message:
			parts.append(event.message)

		if event.data and CONFIG.BROWSER_USE_LOGGING_LEVEL == 'debug':
			# In debug mode, include structured data
			try:
				data_str = json.dumps(event.data, default=str, indent=2)
				if len(data_str) > 500:
					data_str = data_str[:500] + '... (truncated)'
				parts.append(f'\n{data_str}')
			except Exception:
				pass

		return ' '.join(parts) if parts else event.event_type

	def emit(self, event: RuntimeEvent) -> None:
		"""Emit the event to the console logger."""
		if not self.should_accept(event):
			return

		log_level = self._SEVERITY_TO_LOG_LEVEL.get(event.severity, logging.INFO)
		component_name = self._get_component_name(event)
		message = self._format_message(event)

		# Get or create a logger with the component name
		logger = logging.getLogger(f'browser_use.{component_name}')
		logger.log(log_level, message)


class EventBusSink(BaseSink):
	"""Sink that dispatches events to a bubus EventBus.

	Adapts RuntimeEvent to the existing cloud_events format for backward
	compatibility. Existing EventBus listeners continue to work unchanged.
	"""

	def __init__(
		self,
		name: str = 'eventbus',
		eventbus: Any = None,
		enabled: bool = True,
	):
		super().__init__(name, enabled)
		self._eventbus = eventbus
		self._event_type_mapping: dict[str, type] = {}

	def set_eventbus(self, eventbus: Any) -> None:
		"""Set the EventBus instance to dispatch to."""
		self._eventbus = eventbus

	def register_event_type(self, runtime_event_type: str, event_class: type) -> None:
		"""Register a mapping from RuntimeEvent type to cloud_event class."""
		self._event_type_mapping[runtime_event_type] = event_class

	def _convert_to_cloud_event(self, event: RuntimeEvent) -> Any:
		"""Convert RuntimeEvent to the appropriate cloud_event type for backward compatibility."""
		# Lazy import to avoid circular dependencies
		from browser_use.agent.cloud_events import (
			CreateAgentOutputFileEvent,
			CreateAgentStepEvent,
			CreateAgentTaskEvent,
			UpdateAgentTaskEvent,
		)

		event_type = event.event_type
		event_data = event.data or {}

		if event_type == EventType.AGENT_TASK_START and event.task_id:
			return CreateAgentTaskEvent(
				id=event.task_id,
				user_id='',
				device_id=None,
				agent_session_id=event.session_id or '',
				llm_model=event_data.get('model', 'unknown'),
				task=event_data.get('task', ''),
				started_at=event.timestamp,
			)
		elif event_type == EventType.AGENT_STEP_END and event.task_id:
			return CreateAgentStepEvent(
				agent_task_id=event.task_id,
				step=event.step or 0,
				evaluation_previous_goal=event_data.get('evaluation_previous_goal', ''),
				memory=event_data.get('memory', ''),
				next_goal=event_data.get('next_goal', ''),
				actions=event_data.get('actions', []),
				url=event_data.get('url', ''),
				screenshot_url=event_data.get('screenshot_url'),
			)
		elif event_type == EventType.FILE_OUTPUT and event.task_id:
			return CreateAgentOutputFileEvent(
				task_id=event.task_id,
				file_name=Path(event.output_path or 'unknown').name,
				file_content=event_data.get('file_content'),
				content_type=event_data.get('content_type'),
			)
		elif event_type == EventType.AGENT_TASK_END and event.task_id:
			done_output = event_data.get('final_result')
			if done_output and len(done_output) > 500000:
				done_output = done_output[:500000]
			return UpdateAgentTaskEvent(
				id=event.task_id,
				user_id='',
				device_id=None,
				stopped=event_data.get('stopped', False),
				paused=event_data.get('paused', False),
				done_output=done_output,
				finished_at=event.timestamp,
				agent_state=event_data.get('agent_state', {}),
				user_feedback_type=None,
				user_comment=None,
				gif_url=event_data.get('gif_url'),
			)

		# For other event types, return None - they don't have cloud_event equivalents
		return None

	def emit(self, event: RuntimeEvent) -> None:
		"""Dispatch the event to the EventBus if there's a mapped cloud_event type."""
		if not self.should_accept(event) or self._eventbus is None:
			return

		cloud_event = self._convert_to_cloud_event(event)
		if cloud_event is not None:
			try:
				self._eventbus.dispatch(cloud_event)
			except Exception as e:
				logger.debug(f'Failed to dispatch to EventBus: {e}')

	async def emit_async(self, event: RuntimeEvent) -> None:
		"""Async dispatch to EventBus."""
		if not self.should_accept(event) or self._eventbus is None:
			return

		cloud_event = self._convert_to_cloud_event(event)
		if cloud_event is not None:
			try:
				await self._eventbus.dispatch_async(cloud_event)
			except Exception as e:
				logger.debug(f'Failed to dispatch async to EventBus: {e}')


class TelemetrySink(BaseSink):
	"""Sink that sends events to PostHog via ProductTelemetry.

	Adapts RuntimeEvent to the existing BaseTelemetryEvent format for
	backward compatibility with the telemetry system.
	"""

	def __init__(self, name: str = 'telemetry', enabled: bool | None = None):
		super().__init__(name, enabled if enabled is not None else CONFIG.ANONYMIZED_TELEMETRY)
		self._telemetry: Any = None

	def _get_telemetry(self) -> Any:
		"""Lazy load ProductTelemetry to avoid circular imports."""
		if self._telemetry is None:
			from browser_use.telemetry.service import ProductTelemetry

			self._telemetry = ProductTelemetry()
		return self._telemetry

	def _convert_to_telemetry_event(self, event: RuntimeEvent) -> Any:
		"""Convert RuntimeEvent to the appropriate telemetry event type."""
		from browser_use.telemetry.views import (
			AgentTelemetryEvent,
			CLITelemetryEvent,
			MCPClientTelemetryEvent,
			MCPServerTelemetryEvent,
		)

		event_type = event.event_type
		event_data = event.data or {}

		if event_type == EventType.AGENT_TASK_END:
			return AgentTelemetryEvent(
				task=event_data.get('task', ''),
				model=event_data.get('model', 'unknown'),
				model_provider=event_data.get('model_provider', 'unknown'),
				max_steps=event_data.get('max_steps', 0),
				max_actions_per_step=event_data.get('max_actions_per_step', 0),
				use_vision=event_data.get('use_vision', False),
				version=event.source_version,
				source=event_data.get('source', 'unknown'),
				cdp_url=event_data.get('cdp_url'),
				agent_type=event_data.get('agent_type'),
				action_errors=event_data.get('action_errors', []),
				action_history=event_data.get('action_history', []),
				urls_visited=event_data.get('urls_visited', []),
				steps=event.step or event_data.get('steps', 0),
				total_input_tokens=event.tokens_input or 0,
				total_output_tokens=event.tokens_output or 0,
				prompt_cached_tokens=event_data.get('prompt_cached_tokens', 0),
				total_tokens=event.tokens_total or 0,
				total_duration_seconds=(event.duration_ms or 0) / 1000,
				success=event_data.get('success'),
				final_result_response=event_data.get('final_result'),
				error_message=event.error_message,
				judge_verdict=event_data.get('judge_verdict'),
				judge_reasoning=event_data.get('judge_reasoning'),
				judge_failure_reason=event_data.get('judge_failure_reason'),
				judge_reached_captcha=event_data.get('judge_reached_captcha'),
				judge_impossible_task=event_data.get('judge_impossible_task'),
			)
		elif event_type in (EventType.MCP_CLIENT_CONNECT, EventType.MCP_CLIENT_DISCONNECT):
			action = (
				'connect'
				if event_type == EventType.MCP_CLIENT_CONNECT
				else 'disconnect'
			)
			return MCPClientTelemetryEvent(
				server_name=event_data.get('server_name', 'unknown'),
				command=event_data.get('command', ''),
				tools_discovered=event_data.get('tools_discovered', 0),
				version=event.source_version,
				action=action,
				tool_name=event_data.get('tool_name'),
				duration_seconds=(event.duration_ms or 0) / 1000 if event.duration_ms else None,
				error_message=event.error_message,
			)
		elif event_type in (EventType.MCP_SERVER_START, EventType.MCP_SERVER_STOP, EventType.MCP_TOOL_CALL_END):
			action = (
				'start'
				if event_type == EventType.MCP_SERVER_START
				else 'stop'
				if event_type == EventType.MCP_SERVER_STOP
				else 'tool_call'
			)
			return MCPServerTelemetryEvent(
				version=event.source_version,
				action=action,
				tool_name=event_data.get('tool_name'),
				duration_seconds=(event.duration_ms or 0) / 1000 if event.duration_ms else None,
				error_message=event.error_message,
				parent_process_cmdline=event_data.get('parent_process_cmdline'),
			)
		elif event_type in (EventType.CLI_TASK_START, EventType.CLI_TASK_END, EventType.CLI_ERROR):
			action = (
				'start'
				if event_type == EventType.CLI_TASK_START
				else 'task_completed'
				if event_type == EventType.CLI_TASK_END
				else 'error'
			)
			return CLITelemetryEvent(
				version=event.source_version,
				action=action,
				mode=event_data.get('mode', 'interactive'),
				model=event_data.get('model'),
				model_provider=event_data.get('model_provider'),
				duration_seconds=(event.duration_ms or 0) / 1000 if event.duration_ms else None,
				error_message=event.error_message,
			)

		return None

	def emit(self, event: RuntimeEvent) -> None:
		"""Send the event to telemetry if there's a mapped telemetry event type."""
		if not self.should_accept(event):
			return

		try:
			telemetry_event = self._convert_to_telemetry_event(event)
			if telemetry_event is not None:
				telemetry = self._get_telemetry()
				telemetry.capture(telemetry_event)
		except Exception as e:
			logger.debug(f'Failed to send telemetry event: {e}')

	def flush(self) -> None:
		"""Flush telemetry queue."""
		if self._telemetry is not None:
			try:
				self._telemetry.flush()
			except Exception as e:
				logger.debug(f'Failed to flush telemetry: {e}')


class SandboxSink(BaseSink):
	"""Sink that adapts RuntimeEvent to Sandbox SSE format.

	Converts RuntimeEvent to SSEEvent format for streaming sandbox
	execution output to clients.
	"""

	def __init__(self, name: str = 'sandbox', enabled: bool = True):
		super().__init__(name, enabled)
		self._sse_callback: Any = None

	def set_sse_callback(self, callback: Any) -> None:
		"""Set the callback function to send SSE events."""
		self._sse_callback = callback

	def _convert_to_sse_event(self, event: RuntimeEvent) -> Any:
		"""Convert RuntimeEvent to SSEEvent format."""
		from browser_use.sandbox.views import (
			BrowserCreatedData,
			ErrorData,
			ExecutionResponse,
			LogData,
			ResultData,
			SSEEvent,
			SSEEventType,
		)

		event_type = event.event_type
		event_data = event.data or {}

		if event_type == EventType.SANDBOX_INSTANCE_CREATE:
			return SSEEvent(
				type=SSEEventType.BROWSER_CREATED,
				data=BrowserCreatedData(
					session_id=event.browser_session_id or '',
					live_url=event_data.get('live_url', ''),
					status=event_data.get('status', 'created'),
				),
				timestamp=event.timestamp.isoformat(),
			)
		elif event_type == EventType.SANDBOX_LOG:
			return SSEEvent(
				type=SSEEventType.LOG,
				data=LogData(
					message=event.message or '',
					level=event_data.get('level', 'info'),
				),
				timestamp=event.timestamp.isoformat(),
			)
		elif event_type == EventType.SANDBOX_RESULT:
			return SSEEvent(
				type=SSEEventType.RESULT,
				data=ResultData(
					execution_response=ExecutionResponse(
						success=event_data.get('success', True),
						result=event_data.get('result'),
						error=event.error_message,
						traceback=event.error_traceback,
					),
				),
				timestamp=event.timestamp.isoformat(),
			)
		elif event_type == EventType.SANDBOX_ERROR:
			return SSEEvent(
				type=SSEEventType.ERROR,
				data=ErrorData(
					error=event.error_message or 'Unknown error',
					traceback=event.error_traceback,
					status_code=event_data.get('status_code', 500),
				),
				timestamp=event.timestamp.isoformat(),
			)
		elif event_type == EventType.SANDBOX_INSTANCE_READY:
			return SSEEvent(
				type=SSEEventType.INSTANCE_READY,
				data={},
				timestamp=event.timestamp.isoformat(),
			)

		return None

	def emit(self, event: RuntimeEvent) -> None:
		"""Send the event as an SSE event if there's a callback."""
		if not self.should_accept(event) or self._sse_callback is None:
			return

		sse_event = self._convert_to_sse_event(event)
		if sse_event is not None:
			try:
				self._sse_callback(sse_event)
			except Exception as e:
				logger.debug(f'Failed to send SSE event: {e}')


class FileSink(BaseSink):
	"""Sink that writes structured JSON events to a file.

	Writes one JSON object per line (JSON Lines format) for easy
	post-processing and analysis.
	"""

	def __init__(
		self,
		name: str = 'file',
		file_path: str | Path | None = None,
		enabled: bool = True,
	):
		super().__init__(name, enabled)
		self._file_path = Path(file_path) if file_path else None
		self._file_handle: Any = None
		self._buffer: list[str] = []
		self._buffer_size = 100

	def set_file_path(self, file_path: str | Path) -> None:
		"""Set the file path and open the file for appending."""
		self.close()
		self._file_path = Path(file_path)
		self._file_path.parent.mkdir(parents=True, exist_ok=True)
		self._file_handle = open(self._file_path, 'a', encoding='utf-8')

	def _flush_buffer(self) -> None:
		"""Flush the buffer to disk."""
		if self._file_handle and self._buffer:
			try:
				self._file_handle.write('\n'.join(self._buffer) + '\n')
				self._file_handle.flush()
				self._buffer.clear()
			except Exception as e:
				logger.debug(f'Failed to flush file sink buffer: {e}')

	def emit(self, event: RuntimeEvent) -> None:
		"""Write the event to the file as JSON."""
		if not self.should_accept(event):
			return

		if self._file_handle is None and self._file_path is not None:
			self.set_file_path(self._file_path)

		if self._file_handle is None:
			return

		try:
			event_json = event.to_json()
			self._buffer.append(event_json)

			if len(self._buffer) >= self._buffer_size:
				self._flush_buffer()
		except Exception as e:
			logger.debug(f'Failed to write event to file: {e}')

	def flush(self) -> None:
		"""Flush any buffered events to disk."""
		self._flush_buffer()

	def close(self) -> None:
		"""Close the file handle."""
		self.flush()
		if self._file_handle:
			try:
				self._file_handle.close()
			except Exception:
				pass
			self._file_handle = None

	def cleanup(self) -> None:
		"""Clean up resources."""
		self.close()


@singleton
class RuntimeLogger:
	"""Unified logger that routes RuntimeEvent objects to configured sinks.

	This is the main entry point for all components to emit events. Components
	create RuntimeEvent objects (or use convenience methods) and the RuntimeLogger
	handles routing to all configured sinks.

	Usage:
	    from browser_use.observability import runtime_logger, EventType, EventSource

	    # Simple event emission
	    runtime_logger.emit(
	        event_type=EventType.AGENT_STEP_END,
	        source=EventSource.AGENT,
	        message='Step completed',
	        step=5,
	        task_id='task-123',
	        data={'actions': [...]},
	    )

	    # Convenience methods
	    runtime_logger.info('Agent started', source=EventSource.AGENT)
	    runtime_logger.error('Something failed', error=exception, source=EventSource.AGENT)

	    # Context propagation
	    with runtime_logger.context(task_id='task-123', step=5):
	        runtime_logger.info('This event gets task_id and step automatically')
	"""

	def __init__(self) -> None:
		self._sinks: dict[str, BaseSink] = {}
		self._context_stack: list[RuntimeContext] = []
		self._default_context: RuntimeContext = RuntimeContext()

		# Initialize default sinks
		self._initialize_default_sinks()

	def _initialize_default_sinks(self) -> None:
		"""Initialize the default set of sinks."""
		# Console sink is always enabled by default
		console_sink = ConsoleSink()
		self.add_sink(console_sink)

		# Telemetry sink - follows ANONYMIZED_TELEMETRY config
		telemetry_sink = TelemetrySink()
		self.add_sink(telemetry_sink)

		# EventBus sink - enabled, but only dispatches when eventbus is set
		eventbus_sink = EventBusSink()
		self.add_sink(eventbus_sink)

		# Sandbox sink - enabled, but only dispatches when callback is set
		sandbox_sink = SandboxSink()
		self.add_sink(sandbox_sink)

		# File sink - disabled by default, enable via configure
		file_sink = FileSink(enabled=False)
		self.add_sink(file_sink)

	def add_sink(self, sink: BaseSink) -> None:
		"""Add a sink to the logger."""
		self._sinks[sink.name] = sink

	def remove_sink(self, name: str) -> None:
		"""Remove a sink by name."""
		if name in self._sinks:
			self._sinks[name].cleanup()
			del self._sinks[name]

	def get_sink(self, name: str) -> BaseSink | None:
		"""Get a sink by name."""
		return self._sinks.get(name)

	def configure_sink(
		self,
		name: str,
		filter_event_types: Sequence[str] | None = None,
		filter_severity: EventSeverity = EventSeverity.DEBUG,
		filter_sources: Sequence[str] | None = None,
		enabled: bool | None = None,
		**kwargs: Any,
	) -> None:
		"""Configure a specific sink."""
		sink = self._sinks.get(name)
		if sink is None:
			return

		if enabled is not None:
			sink.enabled = enabled

		sink.configure(
			filter_event_types=filter_event_types,
			filter_severity=filter_severity,
			filter_sources=filter_sources,
			enabled=sink.enabled,
		)

		# Handle sink-specific configuration
		if isinstance(sink, FileSink) and 'file_path' in kwargs:
			sink.set_file_path(kwargs['file_path'])
		elif isinstance(sink, EventBusSink) and 'eventbus' in kwargs:
			sink.set_eventbus(kwargs['eventbus'])
		elif isinstance(sink, SandboxSink) and 'sse_callback' in kwargs:
			sink.set_sse_callback(kwargs['sse_callback'])

	def set_default_context(self, **kwargs: Any) -> None:
		"""Set the default context that applies to all events."""
		context = RuntimeContext(**{k: v for k, v in kwargs.items() if k in RuntimeContext.model_fields})
		self._default_context = context

	def context(self, **kwargs: Any) -> 'RuntimeContextManager':
		"""Create a context manager that temporarily sets identity fields.

		Usage:
		    with runtime_logger.context(task_id='task-123', step=5):
		        runtime_logger.info('Event with context')
		"""
		return RuntimeContextManager(self, **kwargs)

	def _get_current_context(self) -> RuntimeContext:
		"""Get the merged context from stack and default."""
		context = self._default_context
		for ctx in self._context_stack:
			context = context.merge(ctx)
		return context

	def _push_context(self, context: RuntimeContext) -> None:
		"""Push a context onto the stack."""
		self._context_stack.append(context)

	def _pop_context(self) -> None:
		"""Pop the last context from the stack."""
		if self._context_stack:
			self._context_stack.pop()

	def _apply_context(self, event: RuntimeEvent) -> RuntimeEvent:
		"""Apply the current context to an event."""
		context = self._get_current_context()
		return context.apply_to_event(event)

	def create_event(
		self,
		event_type: str | EventType,
		source: str | EventSource,
		message: str | None = None,
		data: dict[str, Any] | None = None,
		severity: EventSeverity = EventSeverity.INFO,
		error: Exception | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Create a RuntimeEvent with the current context applied."""
		event_kwargs = {
			'event_type': event_type.value if isinstance(event_type, EventType) else event_type,
			'source': source.value if isinstance(source, EventSource) else source,
			'message': message,
			'data': data,
			'severity': severity,
		}

		# Add identity fields from kwargs
		for field in ['task_id', 'session_id', 'browser_session_id', 'run_id', 'step']:
			if field in kwargs and kwargs[field] is not None:
				event_kwargs[field] = kwargs[field]

		# Add performance fields from kwargs
		for field in ['duration_ms', 'tokens_input', 'tokens_output', 'tokens_total']:
			if field in kwargs and kwargs[field] is not None:
				event_kwargs[field] = kwargs[field]

		# Add output fields from kwargs
		for field in ['output_path', 'output_url']:
			if field in kwargs and kwargs[field] is not None:
				event_kwargs[field] = kwargs[field]

		# Handle error from exception
		if error is not None:
			event_kwargs['error_type'] = type(error).__name__
			event_kwargs['error_message'] = str(error)
			event_kwargs['error_traceback'] = traceback.format_exc()

		# Handle explicit error fields (for pre-captured errors)
		if 'error_type' in kwargs and kwargs['error_type'] is not None:
			event_kwargs['error_type'] = kwargs['error_type']
		if 'error_message' in kwargs and kwargs['error_message'] is not None:
			event_kwargs['error_message'] = kwargs['error_message']
		if 'error_traceback' in kwargs and kwargs['error_traceback'] is not None:
			event_kwargs['error_traceback'] = kwargs['error_traceback']

		event = RuntimeEvent(**event_kwargs)
		return self._apply_context(event)

	def emit(
		self,
		event_type: str | EventType,
		source: str | EventSource,
		message: str | None = None,
		data: dict[str, Any] | None = None,
		severity: EventSeverity = EventSeverity.INFO,
		error: Exception | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Create and emit an event to all sinks."""
		event = self.create_event(
			event_type=event_type,
			source=source,
			message=message,
			data=data,
			severity=severity,
			error=error,
			**kwargs,
		)

		for sink in self._sinks.values():
			try:
				sink.emit(event)
			except Exception as e:
				logger.debug(f'Sink {sink.name} failed to emit event: {e}')

		return event

	async def emit_async(
		self,
		event_type: str | EventType,
		source: str | EventSource,
		message: str | None = None,
		data: dict[str, Any] | None = None,
		severity: EventSeverity = EventSeverity.INFO,
		error: Exception | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Async version of emit for sinks that need async I/O."""
		event = self.create_event(
			event_type=event_type,
			source=source,
			message=message,
			data=data,
			severity=severity,
			error=error,
			**kwargs,
		)

		for sink in self._sinks.values():
			try:
				await sink.emit_async(event)
			except Exception as e:
				logger.debug(f'Sink {sink.name} failed to emit async event: {e}')

		return event

	def emit_event(self, event: RuntimeEvent) -> RuntimeEvent:
		"""Emit an already-created RuntimeEvent."""
		event = self._apply_context(event)

		for sink in self._sinks.values():
			try:
				sink.emit(event)
			except Exception as e:
				logger.debug(f'Sink {sink.name} failed to emit event: {e}')

		return event

	# Convenience methods for common log levels
	def debug(
		self,
		message: str,
		source: str | EventSource = EventSource.AGENT,
		data: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Emit a debug-level event."""
		return self.emit(
			event_type=EventType.LOG_DEBUG,
			source=source,
			message=message,
			data=data,
			severity=EventSeverity.DEBUG,
			**kwargs,
		)

	def info(
		self,
		message: str,
		source: str | EventSource = EventSource.AGENT,
		data: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Emit an info-level event."""
		return self.emit(
			event_type=EventType.LOG_INFO,
			source=source,
			message=message,
			data=data,
			severity=EventSeverity.INFO,
			**kwargs,
		)

	def warning(
		self,
		message: str,
		source: str | EventSource = EventSource.AGENT,
		data: dict[str, Any] | None = None,
		error: Exception | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Emit a warning-level event."""
		return self.emit(
			event_type=EventType.LOG_WARNING,
			source=source,
			message=message,
			data=data,
			severity=EventSeverity.WARNING,
			error=error,
			**kwargs,
		)

	def error(
		self,
		message: str,
		source: str | EventSource = EventSource.AGENT,
		data: dict[str, Any] | None = None,
		error: Exception | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Emit an error-level event."""
		return self.emit(
			event_type=EventType.LOG_ERROR,
			source=source,
			message=message,
			data=data,
			severity=EventSeverity.ERROR,
			error=error,
			**kwargs,
		)

	def result(
		self,
		message: str,
		source: str | EventSource = EventSource.AGENT,
		data: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Emit a result-level event."""
		return self.emit(
			event_type=EventType.LOG_RESULT,
			source=source,
			message=message,
			data=data,
			severity=EventSeverity.RESULT,
			**kwargs,
		)

	def flush(self) -> None:
		"""Flush all sinks."""
		for sink in self._sinks.values():
			try:
				sink.flush()
			except Exception as e:
				logger.debug(f'Sink {sink.name} failed to flush: {e}')

	def cleanup(self) -> None:
		"""Clean up all sinks."""
		for sink in self._sinks.values():
			try:
				sink.cleanup()
			except Exception as e:
				logger.debug(f'Sink {sink.name} failed to cleanup: {e}')
		self._sinks.clear()


class RuntimeContextManager:
	"""Context manager for temporarily setting RuntimeLogger context."""

	def __init__(self, logger: RuntimeLogger, **kwargs: Any):
		self._logger = logger
		self._context = RuntimeContext(**{k: v for k, v in kwargs.items() if k in RuntimeContext.model_fields})

	def __enter__(self) -> 'RuntimeContextManager':
		self._logger._push_context(self._context)
		return self

	def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
		self._logger._pop_context()


# Global singleton instance
runtime_logger = RuntimeLogger()
