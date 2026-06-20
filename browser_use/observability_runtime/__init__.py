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

# ── Lazy imports (follow the pattern used in browser_use/__init__.py) ──────

if TYPE_CHECKING:
	from browser_use.observability_runtime.service import RuntimeLogger, SinkConfig
	from browser_use.observability_runtime.views import (
		ErrorInfo,
		EventSeverity,
		EventSource,
		EventType,
		OutputFile,
		RuntimeEvent,
		TokenUsage,
	)

_LAZY_IMPORTS = {
	# Core data models
	'RuntimeEvent': ('browser_use.observability_runtime.views', 'RuntimeEvent'),
	'EventSource': ('browser_use.observability_runtime.views', 'EventSource'),
	'EventType': ('browser_use.observability_runtime.views', 'EventType'),
	'EventSeverity': ('browser_use.observability_runtime.views', 'EventSeverity'),
	'ErrorInfo': ('browser_use.observability_runtime.views', 'ErrorInfo'),
	'OutputFile': ('browser_use.observability_runtime.views', 'OutputFile'),
	'TokenUsage': ('browser_use.observability_runtime.views', 'TokenUsage'),
	# Logger service
	'RuntimeLogger': ('browser_use.observability_runtime.service', 'RuntimeLogger'),
	'SinkConfig': ('browser_use.observability_runtime.service', 'SinkConfig'),
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


__all__ = list(_LAZY_IMPORTS.keys())
