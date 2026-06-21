"""
Unified RuntimeEvent data model for browser-use observability.

This module defines a single structured event schema that is used across all
entry points (Agent steps, browser commands, MCP tools, CLI tasks, sandbox/cloud
execution, skill_cli daemon).

Every event emitted by the RuntimeLogger will conform to this schema, ensuring
consistent task_id/session_id/step numbering, error typing, and output file
tracking regardless of where the event originates.
"""

from __future__ import annotations

import logging
import traceback
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid_extensions import uuid7str


class EventSource(str, Enum):
	"""Source component that emitted the event."""

	AGENT = 'agent'
	BROWSER = 'browser'
	MCP_SERVER = 'mcp_server'
	MCP_CLIENT = 'mcp_client'
	CLI = 'cli'
	SANDBOX = 'sandbox'
	CLOUD = 'cloud'
	SKILL_CLI = 'skill_cli'
	TELEMETRY = 'telemetry'
	LLM = 'llm'
	TOOLS = 'tools'
	DOM = 'dom'


class EventType(str, Enum):
	"""Semantic type of the runtime event."""

	# Lifecycle events
	SESSION_START = 'session_start'
	SESSION_END = 'session_end'
	TASK_START = 'task_start'
	TASK_END = 'task_end'
	STEP_START = 'step_start'
	STEP_END = 'step_end'

	# Agent events
	AGENT_THINKING = 'agent_thinking'
	AGENT_ACTION = 'agent_action'
	AGENT_ACTION_RESULT = 'agent_action_result'
	AGENT_EVALUATION = 'agent_evaluation'
	AGENT_MEMORY_UPDATE = 'agent_memory_update'
	AGENT_PLAN_UPDATE = 'agent_plan_update'
	AGENT_LOOP_DETECTED = 'agent_loop_detected'

	# Browser events
	BROWSER_START = 'browser_start'
	BROWSER_STOP = 'browser_stop'
	BROWSER_NAVIGATE = 'browser_navigate'
	BROWSER_CLICK = 'browser_click'
	BROWSER_TYPE = 'browser_type'
	BROWSER_SCROLL = 'browser_scroll'
	BROWSER_TAB_SWITCH = 'browser_tab_switch'
	BROWSER_TAB_CLOSE = 'browser_tab_close'
	BROWSER_DOWNLOAD = 'browser_download'
	BROWSER_SCREENSHOT = 'browser_screenshot'
	BROWSER_ERROR = 'browser_error'
	BROWSER_RECONNECT = 'browser_reconnect'
	BROWSER_CAPTCHA = 'browser_captcha'

	# MCP events
	MCP_TOOL_CALL = 'mcp_tool_call'
	MCP_TOOL_RESULT = 'mcp_tool_result'
	MCP_SERVER_START = 'mcp_server_start'
	MCP_SERVER_STOP = 'mcp_server_stop'
	MCP_CLIENT_CONNECT = 'mcp_client_connect'
	MCP_CLIENT_DISCONNECT = 'mcp_client_disconnect'

	# CLI events
	CLI_START = 'cli_start'
	CLI_MESSAGE = 'cli_message'
	CLI_COMMAND = 'cli_command'
	CLI_TASK_COMPLETE = 'cli_task_complete'

	# Sandbox / Cloud events
	SANDBOX_START = 'sandbox_start'
	SANDBOX_END = 'sandbox_end'
	CLOUD_BROWSER_PROVISION = 'cloud_browser_provision'
	CLOUD_BROWSER_TEARDOWN = 'cloud_browser_teardown'
	CLOUD_EXECUTION_START = 'cloud_execution_start'
	CLOUD_EXECUTION_END = 'cloud_execution_end'
	CLOUD_SYNC = 'cloud_sync'

	# Skill CLI daemon events
	SKILL_DAEMON_START = 'skill_daemon_start'
	SKILL_DAEMON_STOP = 'skill_daemon_stop'
	SKILL_SESSION_CREATE = 'skill_session_create'
	SKILL_COMMAND = 'skill_command'
	SKILL_IDLE_TIMEOUT = 'skill_idle_timeout'

	# LLM / Tools events
	LLM_CALL = 'llm_call'
	LLM_RESULT = 'llm_result'
	TOOL_EXECUTE = 'tool_execute'
	TOOL_RESULT = 'tool_result'

	# Error events
	ERROR = 'error'
	EXCEPTION = 'exception'
	WARNING = 'warning'

	# Generic info / debug
	INFO = 'info'
	DEBUG = 'debug'
	METRIC = 'metric'


class EventSeverity(str, Enum):
	"""Severity level of the event (mirrors standard logging levels)."""

	DEBUG = 'debug'
	INFO = 'info'
	WARNING = 'warning'
	ERROR = 'error'
	CRITICAL = 'critical'

	@classmethod
	def from_logging_level(cls, level: int) -> EventSeverity:
		if level >= logging.CRITICAL:
			return cls.CRITICAL
		if level >= logging.ERROR:
			return cls.ERROR
		if level >= logging.WARNING:
			return cls.WARNING
		if level >= logging.INFO:
			return cls.INFO
		return cls.DEBUG

	def to_logging_level(self) -> int:
		return {
			'debug': logging.DEBUG,
			'info': logging.INFO,
			'warning': logging.WARNING,
			'error': logging.ERROR,
			'critical': logging.CRITICAL,
		}[self.value]


class OutputFile(BaseModel):
	"""A file produced as output of an event (screenshot, GIF, CSV, PDF, etc.)."""

	model_config = ConfigDict(extra='forbid')

	path: str
	file_name: str | None = None
	content_type: str | None = None
	size_bytes: int | None = None
	description: str | None = None


class ErrorInfo(BaseModel):
	"""Structured error information for an event."""

	model_config = ConfigDict(extra='forbid')

	error_type: str = Field(..., description='Fully qualified exception class name, e.g. "ValueError"')
	error_message: str = Field(..., description='Exception message (str(e))')
	error_stack: str | None = Field(default=None, description='Formatted traceback string')
	error_module: str | None = Field(default=None, description='Module where the exception was defined')

	@classmethod
	def from_exception(cls, exc: BaseException, include_stack: bool = True) -> ErrorInfo:
		tb_str = None
		if include_stack and exc.__traceback__:
			tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
		return cls(
			error_type=f'{type(exc).__module__}.{type(exc).__name__}',
			error_message=str(exc),
			error_stack=tb_str,
			error_module=type(exc).__module__,
		)


class TokenUsage(BaseModel):
	"""Token usage for LLM-related events."""

	model_config = ConfigDict(extra='forbid')

	input_tokens: int | None = None
	output_tokens: int | None = None
	prompt_cached_tokens: int | None = None
	total_tokens: int | None = None
	cost_usd: float | None = None
	model_name: str | None = None
	model_provider: str | None = None


class RuntimeEvent(BaseModel):
	"""
	Unified runtime event schema for all browser-use observability.

	Every event logged through RuntimeLogger conforms to this schema,
	guaranteeing consistent fields across Agent steps, browser commands,
	MCP tool calls, CLI tasks, sandbox execution, and the skill_cli daemon.

	Key invariants:
	- ``event_id`` is always a UUID7 (time-sortable)
	- ``timestamp`` is always UTC ISO 8601
	- ``task_id``, ``session_id``, ``step`` are populated from the
	  active RuntimeContext when available; callers may override
	- ``data`` is an arbitrary JSON-able dict for event-specific payloads
	- ``error`` and ``output_files`` are consistently typed across sources
	"""

	model_config = ConfigDict(extra='allow', populate_by_name=True)

	# ── Identity ───────────────────────────────────────────────────────────
	event_id: str = Field(default_factory=uuid7str, description='UUID7, time-sortable event identifier')
	timestamp: datetime = Field(
		default_factory=lambda: datetime.now(timezone.utc),
		description='UTC timestamp of the event',
	)

	# ── Source / Type ───────────────────────────────────────────────────────
	source: EventSource = Field(..., description='Component that emitted the event')
	event_type: EventType = Field(..., description='Semantic event category')
	severity: EventSeverity = Field(default=EventSeverity.INFO, description='Log-style severity level')

	# ── Linking / Traceability ─────────────────────────────────────────────
	task_id: str | None = Field(default=None, description='Agent task identifier (uuid7)')
	session_id: str | None = Field(default=None, description='Browser session identifier (uuid7)')
	step: int | None = Field(default=None, ge=0, description='Agent step number (0-indexed)')
	action_index: int | None = Field(default=None, ge=0, description='Action index within the step')
	parent_event_id: str | None = Field(default=None, description='Parent event for causal linking')
	run_id: str | None = Field(default=None, description='Higher-level orchestration run ID')

	# ── Human-readable content ─────────────────────────────────────────────
	message: str = Field(default='', description='Human-readable event description')

	# ── Structured payload ─────────────────────────────────────────────────
	data: dict[str, Any] = Field(
		default_factory=dict,
		description='Event-type-specific structured payload',
	)

	# ── Errors ─────────────────────────────────────────────────────────────
	error: ErrorInfo | None = Field(default=None, description='Structured error info if event represents a failure')

	# ── Output artifacts ───────────────────────────────────────────────────
	output_files: list[OutputFile] = Field(
		default_factory=list,
		description='Files produced by this event (screenshots, CSVs, etc.)',
	)

	# ── Performance / Cost ─────────────────────────────────────────────────
	duration_ms: float | None = Field(default=None, ge=0, description='Duration of the operation in milliseconds')
	tokens: TokenUsage | None = Field(default=None, description='Token usage / cost for LLM events')

	# ── Execution environment ──────────────────────────────────────────────
	hostname: str | None = Field(default=None, description='Hostname of the machine (populated automatically when available)')
	pid: int | None = Field(default=None, description='OS process ID')
	version: str | None = Field(default=None, description='browser-use version')
	is_docker: bool | None = Field(default=None, description='Whether running inside a Docker container')

	# ── Convenience validators ─────────────────────────────────────────────

	@field_validator('timestamp')
	@classmethod
	def _ensure_utc(cls, v: datetime) -> datetime:
		if v.tzinfo is None:
			return v.replace(tzinfo=timezone.utc)
		return v.astimezone(timezone.utc)

	@field_validator('task_id', 'session_id', 'parent_event_id', 'run_id')
	@classmethod
	def _coerce_uuid_to_str(cls, v: Any) -> str | None:
		if isinstance(v, UUID):
			return str(v)
		return v

	# ── Convenience factory methods ─────────────────────────────────────────

	@classmethod
	def from_exception(
		cls,
		exc: BaseException,
		*,
		source: EventSource,
		event_type: EventType = EventType.EXCEPTION,
		severity: EventSeverity = EventSeverity.ERROR,
		include_stack: bool = True,
		**kwargs: Any,
	) -> RuntimeEvent:
		"""Create a RuntimeEvent wrapping an exception."""
		return cls(
			source=source,
			event_type=event_type,
			severity=severity,
			error=ErrorInfo.from_exception(exc, include_stack=include_stack),
			message=str(exc),
			**kwargs,
		)

	def to_console_line(self) -> str:
		"""Render a short, human-readable line for console output."""
		parts: list[str] = []
		ts = self.timestamp.strftime('%H:%M:%S')
		parts.append(f'[{ts}]')
		parts.append(f'{self.severity.value.upper():<8}')
		parts.append(f'{self.source.value:<12}')

		tag_bits: list[str] = []
		if self.task_id:
			tag_bits.append(f'task={self.task_id[-6:]}')
		if self.session_id:
			tag_bits.append(f'sess={self.session_id[-6:]}')
		if self.step is not None:
			tag_bits.append(f'step={self.step}')
		if tag_bits:
			parts.append('(' + ' '.join(tag_bits) + ')')

		parts.append(self.message)

		if self.duration_ms is not None:
			parts.append(f'({self.duration_ms:.1f}ms)')

		if self.error:
			parts.append(f'[ERR: {self.error.error_type}]')

		return ' '.join(p for p in parts if p)

	def to_console_extra_dict(self) -> dict[str, Any]:
		"""Build the ``extra=`` dict for Python ``logging.Logger.log()``.

		All six canonical linking / identity fields come through a single
		adapter so every console output uses the same key names.
		"""
		extra: dict[str, Any] = {
			'runtime_event_id': self.event_id,
			'runtime_event_type': self.event_type.value,
			'runtime_source': self.source.value,
		}
		if self.task_id:
			extra['task_id'] = self.task_id
		if self.session_id:
			extra['session_id'] = self.session_id
		if self.step is not None:
			extra['step'] = self.step
		return extra

	def to_telemetry_event_name(self) -> str:
		"""Map this event to a stable, coarse-grained PostHog event name.

		The PostHog schema intentionally uses a small number of categories
		(agent_task / agent_step / agent_action / error / llm / mcp_tool /
		sandbox) rather than mirroring the full ``EventType`` enum.
		"""
		source_prefix = self.source.value
		if self.event_type in {EventType.TASK_START, EventType.TASK_END}:
			return f'{source_prefix}_task'
		if self.event_type in {EventType.STEP_START, EventType.STEP_END}:
			return f'{source_prefix}_step'
		if self.event_type in {EventType.AGENT_ACTION, EventType.AGENT_ACTION_RESULT}:
			return f'{source_prefix}_action'
		if self.event_type in {EventType.EXCEPTION, EventType.ERROR}:
			return f'{source_prefix}_error'
		if self.event_type in {EventType.LLM_CALL, EventType.LLM_RESULT}:
			return f'{source_prefix}_llm'
		if self.event_type in {EventType.MCP_TOOL_CALL, EventType.MCP_TOOL_RESULT}:
			return f'{source_prefix}_mcp_tool'
		if self.event_type in {EventType.SANDBOX_START, EventType.SANDBOX_END}:
			return f'{source_prefix}_sandbox'
		return f'{source_prefix}_event'

	def to_telemetry_properties(self) -> dict[str, Any]:
		"""Convert to a flat dict suitable for PostHog / telemetry ingestion."""
		props: dict[str, Any] = {
			'event_id': self.event_id,
			'timestamp': self.timestamp.isoformat(),
			'source': self.source.value,
			'event_type': self.event_type.value,
			'severity': self.severity.value,
			'message': self.message,
		}
		if self.task_id:
			props['task_id'] = self.task_id
		if self.session_id:
			props['session_id'] = self.session_id
		if self.step is not None:
			props['step'] = self.step
		if self.action_index is not None:
			props['action_index'] = self.action_index
		if self.duration_ms is not None:
			props['duration_ms'] = self.duration_ms
		if self.error:
			props['error_type'] = self.error.error_type
			props['error_message'] = self.error.error_message
		if self.tokens:
			for k, v in self.tokens.model_dump(exclude_none=True).items():
				props[f'tokens_{k}'] = v
		if self.output_files:
			props['output_file_count'] = len(self.output_files)
		if self.version:
			props['version'] = self.version
		if self.is_docker is not None:
			props['is_docker'] = self.is_docker
		for k, v in self.data.items():
			if isinstance(v, (str, int, float, bool, type(None))):
				props[f'data_{k}'] = v
		return props

	def to_cloud_event_dict(self) -> dict[str, Any]:
		"""Serialize this RuntimeEvent to a cloud-sync compatible payload.

		The cloud sync endpoint accepts the same canonical linking fields
		(task_id / session_id / step / error / output_files / source) as
		every other sink; callers should never hand-build payloads.
		"""
		payload: dict[str, Any] = {
			'event_id': self.event_id,
			'timestamp': self.timestamp.isoformat(),
			'source': self.source.value,
			'event_type': self.event_type.value,
			'severity': self.severity.value,
			'message': self.message,
			'data': self.data,
		}
		if self.task_id:
			payload['task_id'] = self.task_id
		if self.session_id:
			payload['session_id'] = self.session_id
		if self.step is not None:
			payload['step'] = self.step
		if self.action_index is not None:
			payload['action_index'] = self.action_index
		if self.duration_ms is not None:
			payload['duration_ms'] = self.duration_ms
		if self.error:
			payload['error'] = self.error.model_dump(mode='json', exclude_none=True)
		if self.output_files:
			payload['output_files'] = [f.model_dump(mode='json', exclude_none=True) for f in self.output_files]
		if self.tokens:
			payload['tokens'] = self.tokens.model_dump(mode='json', exclude_none=True)
		if self.version:
			payload['version'] = self.version
		if self.hostname:
			payload['hostname'] = self.hostname
		if self.pid is not None:
			payload['pid'] = self.pid
		if self.is_docker is not None:
			payload['is_docker'] = self.is_docker
		return payload


# ── Bubus EventBus bridge ──────────────────────────────────────────────────

try:
	from bubus import BaseEvent  # type: ignore
except Exception:  # pragma: no cover
	BaseEvent = None  # type: ignore


if BaseEvent is not None:

	class RuntimeEventBusBridge(BaseEvent):  # type: ignore[misc,valid-type]
		"""
		Bubus BaseEvent subclass that wraps a RuntimeEvent for dispatch
		through an existing bubus EventBus.

		This exists because bubus.EventBus.dispatch() requires events
		that inherit from bubus.BaseEvent (with event_id, event_created_at,
		event_type, event_schema fields). RuntimeEventBusBridge inherits
		from bubus.BaseEvent and carries a RuntimeEvent as its payload.

		Subscribers filter by ``event_type == 'RuntimeEventBusBridge'``
		and access the wrapped event via ``event.runtime_event``.
		"""

		runtime_event: RuntimeEvent

		@classmethod
		def wrap(cls, event: RuntimeEvent) -> RuntimeEventBusBridge:
			"""Create a bridge event wrapping a RuntimeEvent."""
			return cls(runtime_event=event)  # type: ignore[return-value]

else:  # pragma: no cover - bubus is a core dependency

	class RuntimeEventBusBridge:  # type: ignore[no-redef]
		"""Fallback stub when bubus is not available."""

		@classmethod
		def wrap(cls, event: RuntimeEvent) -> RuntimeEventBusBridge:
			return cls()
