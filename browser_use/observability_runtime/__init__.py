"""
Unified observability runtime for browser-use.

This package provides the RuntimeEvent / RuntimeLogger adapter layer
that standardizes event emission across:

- Agent steps & actions
- Browser commands (navigate, click, type, ...)
- MCP tools (server & client)
- CLI tasks & commands
- Sandbox / cloud execution
- skill_cli daemon

Entry layers should ONLY configure sinks via :class:`RuntimeLogger.set_sinks`
and then call the convenience methods (info, warning, step_start, ...).
NEVER hand-build payloads for individual sinks.
"""

from __future__ import annotations

# ── Direct imports for all types ──────────────────────────────────────────
# All types are imported directly (not lazily) so that pyright can resolve
# constructor signatures correctly.  These are pure type definitions (enums,
# Pydantic models, or @singleton services) with negligible import cost.
from browser_use.observability_runtime.service import RuntimeLogger, SinkConfig, create_sink_config, validate_event_consistency
from browser_use.observability_runtime.views import (
	ErrorInfo,
	EventSeverity,
	EventSource,
	EventType,
	OutputFile,
	RuntimeEvent,
	TokenUsage,
)

__all__ = (
	'RuntimeEvent',
	'RuntimeLogger',
	'SinkConfig',
	'create_sink_config',
	'validate_event_consistency',
	'EventSource',
	'EventType',
	'EventSeverity',
	'ErrorInfo',
	'OutputFile',
	'TokenUsage',
)
