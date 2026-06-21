"""Unified observability infrastructure for browser-use.

This module provides a unified RuntimeEvent / RuntimeLogger adapter layer that
standardizes event emission across all browser-use components (Agent, CLI, MCP,
browser session, sandbox, skill_cli daemon).

Key components:
- RuntimeEvent: Unified event schema with consistent fields for task_id,
  session_id, step, event_type, etc.
- RuntimeLogger: Adapter layer that routes RuntimeEvent objects to configured
  output sinks (console, event bus, telemetry, cloud, file).
- EventSink implementations: ConsoleSink, EventBusSink, TelemetrySink,
  SandboxSink, FileSink - each adapts RuntimeEvent to the target format.

Entry points only need to emit RuntimeEvent objects - the RuntimeLogger handles
routing to all configured outputs. No more hand-writing different log formats
or event payloads for different output targets.
"""

# LMNR integration (optional tracing)
from ._lmnr import (
	get_observability_status,
	is_debug_mode,
	is_lmnr_available,
	observe,
	observe_debug,
)
from .service import (
	BaseSink,
	ConsoleSink,
	EventBusSink,
	FileSink,
	RuntimeContextManager,
	RuntimeLogger,
	SandboxSink,
	TelemetrySink,
	runtime_logger,
)

# New unified observability infrastructure
from .views import (
	EventSeverity,
	EventSource,
	EventType,
	RuntimeContext,
	RuntimeEvent,
)

__all__ = [
	# LMNR integration (optional tracing)
	'observe',
	'observe_debug',
	'is_lmnr_available',
	'is_debug_mode',
	'get_observability_status',
	# New unified event schema
	'EventSeverity',
	'EventSource',
	'EventType',
	'RuntimeContext',
	'RuntimeEvent',
	# New logger and sinks
	'BaseSink',
	'ConsoleSink',
	'EventBusSink',
	'FileSink',
	'RuntimeContextManager',
	'RuntimeLogger',
	'SandboxSink',
	'TelemetrySink',
	'runtime_logger',
]
