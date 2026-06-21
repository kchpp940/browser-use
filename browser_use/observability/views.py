"""Unified RuntimeEvent schema for consistent observability across all browser-use components.

This module defines the standardized event schema that all components (Agent, CLI, MCP,
browser session, sandbox, skill_cli) use to emit events. The RuntimeLogger adapter
layer then routes these events to appropriate sinks (console, event bus, telemetry, cloud).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field, field_serializer
from uuid_extensions import uuid7str

from browser_use.utils import get_browser_use_version

T = TypeVar('T', bound=BaseModel)


class EventSource(str, Enum):
	"""Source component that emitted the event."""

	AGENT = 'agent'
	BROWSER = 'browser'
	CLI = 'cli'
	MCP = 'mcp'
	SANDBOX = 'sandbox'
	SKILL_CLI = 'skill_cli'
	LLM = 'llm'
	TOOLS = 'tools'
	DOM = 'dom'


class EventType(str, Enum):
	"""Standardized event types across all components."""

	# Agent events
	AGENT_TASK_START = 'agent.task.start'
	AGENT_TASK_END = 'agent.task.end'
	AGENT_STEP_START = 'agent.step.start'
	AGENT_STEP_END = 'agent.step.end'
	AGENT_ACTION_START = 'agent.action.start'
	AGENT_ACTION_END = 'agent.action.end'
	AGENT_ERROR = 'agent.error'

	# Browser events
	BROWSER_SESSION_START = 'browser.session.start'
	BROWSER_SESSION_END = 'browser.session.end'
	BROWSER_COMMAND_START = 'browser.command.start'
	BROWSER_COMMAND_END = 'browser.command.end'
	BROWSER_NAVIGATION = 'browser.navigation'
	BROWSER_DOWNLOAD = 'browser.download'
	BROWSER_ERROR = 'browser.error'

	# CLI events
	CLI_TASK_START = 'cli.task.start'
	CLI_TASK_END = 'cli.task.end'
	CLI_MESSAGE = 'cli.message'
	CLI_ERROR = 'cli.error'

	# MCP events
	MCP_SERVER_START = 'mcp.server.start'
	MCP_SERVER_STOP = 'mcp.server.stop'
	MCP_TOOL_CALL_START = 'mcp.tool_call.start'
	MCP_TOOL_CALL_END = 'mcp.tool_call.end'
	MCP_CLIENT_CONNECT = 'mcp.client.connect'
	MCP_CLIENT_DISCONNECT = 'mcp.client.disconnect'
	MCP_ERROR = 'mcp.error'

	# Sandbox/Cloud events
	SANDBOX_INSTANCE_CREATE = 'sandbox.instance.create'
	SANDBOX_INSTANCE_READY = 'sandbox.instance.ready'
	SANDBOX_EXECUTION_START = 'sandbox.execution.start'
	SANDBOX_EXECUTION_END = 'sandbox.execution.end'
	SANDBOX_LOG = 'sandbox.log'
	SANDBOX_RESULT = 'sandbox.result'
	SANDBOX_ERROR = 'sandbox.error'

	# Skill CLI events
	SKILL_DAEMON_START = 'skill.daemon.start'
	SKILL_DAEMON_STOP = 'skill.daemon.stop'
	SKILL_COMMAND = 'skill.command'
	SKILL_ERROR = 'skill.error'

	# Generic events
	LOG_DEBUG = 'log.debug'
	LOG_INFO = 'log.info'
	LOG_WARNING = 'log.warning'
	LOG_ERROR = 'log.error'
	LOG_RESULT = 'log.result'
	LLM_CALL = 'llm.call'
	FILE_OUTPUT = 'file.output'


class EventSeverity(str, Enum):
	"""Event severity level."""

	DEBUG = 'debug'
	INFO = 'info'
	WARNING = 'warning'
	ERROR = 'error'
	RESULT = 'result'


class RuntimeEvent(BaseModel):
	"""Unified runtime event schema for all browser-use components.

	All components emit events with this schema. The RuntimeLogger routes
	these events to appropriate sinks based on configuration.

	Fields are designed to be as consistent as possible across all entry points:
	- task_id: Always the agent task ID when an agent is involved
	- session_id: The agent session ID (not browser session, use browser_session_id for that)
	- browser_session_id: The browser session ID specifically
	- run_id: For CLI/MCP/sandbox runs that don't have a task_id
	- step: 1-indexed step number (consistent across all components)
	"""

	# Core identification - always present
	event_id: str = Field(default_factory=uuid7str)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	event_type: str
	event_version: str = '1.0'

	# Identity fields - optional depending on context
	task_id: str | None = None
	session_id: str | None = None
	browser_session_id: str | None = None
	run_id: str | None = None
	step: int | None = None

	# Source fields - always populated by RuntimeLogger
	source: EventSource | str
	source_version: str = Field(default_factory=get_browser_use_version)

	# Content fields
	message: str | None = None
	data: dict[str, Any] | None = None

	# Error fields
	error_type: str | None = None
	error_message: str | None = None
	error_traceback: str | None = None

	# Performance fields
	duration_ms: float | None = None
	tokens_input: int | None = None
	tokens_output: int | None = None
	tokens_total: int | None = None

	# Output fields
	output_path: str | None = None
	output_url: str | None = None

	# Severity - used for console output filtering
	severity: EventSeverity = EventSeverity.INFO

	@field_serializer('timestamp')
	def serialize_timestamp(self, v: datetime) -> str:
		"""Serialize timestamp to ISO 8601 format."""
		return v.isoformat()

	def to_json(self) -> str:
		"""Serialize event to JSON string."""
		return self.model_dump_json()

	def to_dict(self) -> dict[str, Any]:
		"""Serialize event to dictionary."""
		return self.model_dump()

	def get_identity_fields(self) -> dict[str, str | int | None]:
		"""Get all identity fields for routing/filtering."""
		return {
			'task_id': self.task_id,
			'session_id': self.session_id,
			'browser_session_id': self.browser_session_id,
			'run_id': self.run_id,
			'step': self.step,
		}


class RuntimeContext(BaseModel):
	"""Context that propagates identity fields through call chains.

	Use this to pass task_id, session_id, etc. through nested calls
	so that events emitted by nested components automatically get
	the correct identity fields.
	"""

	task_id: str | None = None
	session_id: str | None = None
	browser_session_id: str | None = None
	run_id: str | None = None
	step: int | None = None
	source: EventSource | str | None = None

	def merge(self, other: 'RuntimeContext | dict[str, Any] | None') -> 'RuntimeContext':
		"""Merge another context into this one, with other taking precedence."""
		if other is None:
			return self.model_copy()
		if isinstance(other, dict):
			other = RuntimeContext(**{k: v for k, v in other.items() if k in self.model_fields})
		merged = self.model_dump()
		for field, value in other.model_dump().items():
			if value is not None:
				merged[field] = value
		return RuntimeContext(**merged)

	def apply_to_event(self, event: RuntimeEvent) -> RuntimeEvent:
		"""Apply this context's identity fields to an event (only if not already set)."""
		update = {}
		for field in ['task_id', 'session_id', 'browser_session_id', 'run_id', 'step']:
			if getattr(event, field) is None and getattr(self, field) is not None:
				update[field] = getattr(self, field)
		if event.source == EventSource.AGENT and self.source is not None:
			update['source'] = self.source
		if update:
			return event.model_copy(update=update)
		return event
