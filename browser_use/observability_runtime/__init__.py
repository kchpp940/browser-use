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

from typing import TYPE_CHECKING

from browser_use.observability_runtime.service import SinkConfig

# ── Direct imports for lightweight types (enums, Pydantic models) ─────────
# These are pure type definitions with zero initialization cost. Importing
# them directly gives pyright full visibility and avoids "expected 0
# positional arguments" errors on SinkConfig() etc.
from browser_use.observability_runtime.views import (
	ErrorInfo,
	EventSeverity,
	EventSource,
	EventType,
	OutputFile,
	RuntimeEvent,
	TokenUsage,
)

if TYPE_CHECKING:
	from browser_use.observability_runtime.service import RuntimeLogger

# ── Lazy import only for RuntimeLogger (has initialization cost) ───────────

_LAZY_IMPORTS = {
	'RuntimeLogger': ('browser_use.observability_runtime.service', 'RuntimeLogger'),
}


def __getattr__(name: str):
	"""Lazy import mechanism — only import heavy modules when accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			attr = getattr(module, attr_name)
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = (
	'RuntimeEvent',
	'RuntimeLogger',
	'SinkConfig',
	'EventSource',
	'EventType',
	'EventSeverity',
	'ErrorInfo',
	'OutputFile',
	'TokenUsage',
)
