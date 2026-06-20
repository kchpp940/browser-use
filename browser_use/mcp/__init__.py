"""MCP (Model Context Protocol) support for browser-use.

This module provides integration with MCP servers and clients for browser automation.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from browser_use.mcp.client import MCPClient
	from browser_use.mcp.controller import MCPToolWrapper
	from browser_use.mcp.server import BrowserUseServer

_LAZY_IMPORTS = {
	'MCPClient': ('browser_use.mcp.client', 'MCPClient'),
	'MCPToolWrapper': ('browser_use.mcp.controller', 'MCPToolWrapper'),
	'BrowserUseServer': ('browser_use.mcp.server', 'BrowserUseServer'),
}


def __getattr__(name: str):
	"""Lazy import — defer loading MCP SDK dependent modules until accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		from importlib import import_module

		module = import_module(module_path)
		attr = getattr(module, attr_name)
		globals()[name] = attr
		return attr
	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ['MCPClient', 'MCPToolWrapper', 'BrowserUseServer']
