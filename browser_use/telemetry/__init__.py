"""
Telemetry for Browser Use.

Also re-exports the unified RuntimeEvent / RuntimeLogger observability API
from :mod:`browser_use.observability_runtime` for convenience, since
telemetry is one of the primary sinks.
"""

from typing import TYPE_CHECKING

# Type stubs for lazy imports
if TYPE_CHECKING:
	from browser_use.observability_runtime import (
		ErrorInfo,
		EventSeverity,
		EventSource,
		EventType,
		OutputFile,
		RuntimeEvent,
		RuntimeLogger,
		SinkConfig,
		TokenUsage,
	)
	from browser_use.telemetry.service import ProductTelemetry
	from browser_use.telemetry.views import (
		BaseTelemetryEvent,
		CLITelemetryEvent,
		MCPClientTelemetryEvent,
		MCPServerTelemetryEvent,
	)

# Lazy imports mapping
_LAZY_IMPORTS = {
	# Legacy telemetry classes
	'ProductTelemetry': ('browser_use.telemetry.service', 'ProductTelemetry'),
	'BaseTelemetryEvent': ('browser_use.telemetry.views', 'BaseTelemetryEvent'),
	'CLITelemetryEvent': ('browser_use.telemetry.views', 'CLITelemetryEvent'),
	'MCPClientTelemetryEvent': ('browser_use.telemetry.views', 'MCPClientTelemetryEvent'),
	'MCPServerTelemetryEvent': ('browser_use.telemetry.views', 'MCPServerTelemetryEvent'),
	# Unified observability runtime (re-exported here for ergonomics)
	'RuntimeEvent': ('browser_use.observability_runtime', 'RuntimeEvent'),
	'RuntimeLogger': ('browser_use.observability_runtime', 'RuntimeLogger'),
	'SinkConfig': ('browser_use.observability_runtime', 'SinkConfig'),
	'EventSource': ('browser_use.observability_runtime', 'EventSource'),
	'EventType': ('browser_use.observability_runtime', 'EventType'),
	'EventSeverity': ('browser_use.observability_runtime', 'EventSeverity'),
	'ErrorInfo': ('browser_use.observability_runtime', 'ErrorInfo'),
	'OutputFile': ('browser_use.observability_runtime', 'OutputFile'),
	'TokenUsage': ('browser_use.observability_runtime', 'TokenUsage'),
}


def __getattr__(name: str):
	"""Lazy import mechanism for telemetry components."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	# Legacy telemetry
	'BaseTelemetryEvent',
	'ProductTelemetry',
	'CLITelemetryEvent',
	'MCPClientTelemetryEvent',
	'MCPServerTelemetryEvent',
	# Unified observability runtime
	'RuntimeEvent',
	'RuntimeLogger',
	'SinkConfig',
	'EventSource',
	'EventType',
	'EventSeverity',
	'ErrorInfo',
	'OutputFile',
	'TokenUsage',
]
