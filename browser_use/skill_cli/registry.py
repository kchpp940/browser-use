"""Unified skill_cli command registry.

This module centralizes ALL browser skill_cli command metadata definitions.

Before this refactor, skill_cli maintained THREE independent tables in
commands/browser.py:
    COMMANDS                 — set of valid command names
    COMMAND_TO_TOOL_MAPPING  — command name -> core tool name
    CLI_SPECIFIC_COMMANDS    — commands that don't map to core tools

These tables have been replaced with a single declarative list of ToolCapability
objects, from which COMMANDS and COMMAND_TO_TOOL_MAPPING are *derived* automatically
via the helper functions in this module.

This is the single source of truth for skill_cli command metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from browser_use.tools.registry.views import ToolCapability

# ---------------------------------------------------------------------------
# Command-specific parameter models (optional — only for commands with
# well-defined param shapes that we want to document/validate)
# ---------------------------------------------------------------------------


class OpenParams(BaseModel):
	url: str = Field(description='URL to open. Auto-prefixes with https:// if missing.')


class ClickParams(BaseModel):
	args: list = Field(
		default_factory=list,
		description='Either [index] to click by element index, or [x, y] to click by coordinates.',
	)


class TypeParams(BaseModel):
	text: str = Field(description='Text to type into the currently focused input.')


class InputParams(BaseModel):
	index: int = Field(description='Element index.')
	text: str = Field(description='Text to enter into the element.')


class ScrollParams(BaseModel):
	direction: str = Field(default='down', description='"up" or "down"')
	amount: int = Field(default=500, description='Pixel amount to scroll.')


class ScreenshotParams(BaseModel):
	full: bool = Field(default=False, description='Whether to capture the full page.')
	path: str | None = Field(default=None, description='Optional file path to save PNG to.')


class KeysParams(BaseModel):
	keys: str = Field(description='Keys to send (e.g. "Enter", "Tab", "ArrowDown").')


class SelectParams(BaseModel):
	index: int = Field(description='Dropdown element index.')
	value: str = Field(description='Option value to select.')


class UploadParams(BaseModel):
	index: int = Field(description='File input element index.')
	path: str = Field(description='Local file path to upload.')


class EvalParams(BaseModel):
	js: str = Field(description='JavaScript expression to evaluate.')


class ExtractParams(BaseModel):
	query: str = Field(description='Description of content to extract from the page.')


class HoverParams(BaseModel):
	index: int = Field(description='Element index to hover over.')


class WaitParams(BaseModel):
	wait_command: str = Field(description='"selector" or "text"')


class GetParams(BaseModel):
	get_command: str = Field(description='"title", "html", "text", "value", "attributes", or "bbox"')


class CookiesParams(BaseModel):
	cookies_command: str = Field(description='"get", "set", "clear", "export", or "import"')


class RecordParams(BaseModel):
	record_command: str = Field(description='"start", "stop", or "status"')


class TabParams(BaseModel):
	tab_command: str = Field(description='"list", "new", "switch", or "close"')


# ---------------------------------------------------------------------------
# Declarative command list — THE SINGLE SOURCE OF TRUTH
#
# Each entry is a ToolCapability with:
#   name           — CLI command name (derives COMMANDS set)
#   description    — human-readable documentation
#   category       — same categories as core tools (navigation/interaction/...)
#   param_schema   — optional Pydantic model for structured params
#   core_tool_name — maps to a core tool for unified availability filtering
#                    (None = CLI-specific, always allowed unless explicitly blocked)
#   requires_browser — whether the command needs an active browser session
# ---------------------------------------------------------------------------


CLI_COMMAND_CAPABILITIES: list[ToolCapability] = [
	ToolCapability(
		name='open',
		description='Navigate to a URL. Auto-prefixes https:// if no scheme given.',
		category='navigation',
		param_schema=OpenParams,
		core_tool_name='navigate',
		requires_browser=True,
	),
	ToolCapability(
		name='click',
		description='Click by element index or viewport coordinates.',
		category='interaction',
		param_schema=ClickParams,
		core_tool_name='click',
		requires_browser=True,
	),
	ToolCapability(
		name='type',
		description='Type text into the currently focused input field.',
		category='interaction',
		param_schema=TypeParams,
		core_tool_name='input',
		requires_browser=True,
	),
	ToolCapability(
		name='input',
		description='Click an element by index, then type text into it.',
		category='interaction',
		param_schema=InputParams,
		core_tool_name='input',
		requires_browser=True,
	),
	ToolCapability(
		name='scroll',
		description='Scroll the page up or down by a pixel amount.',
		category='interaction',
		param_schema=ScrollParams,
		core_tool_name='scroll',
		requires_browser=True,
	),
	ToolCapability(
		name='back',
		description='Go back to the previous page in history.',
		category='navigation',
		core_tool_name='go_back',
		requires_browser=True,
	),
	ToolCapability(
		name='screenshot',
		description='Take a screenshot of the current page (viewport or full).',
		category='extraction',
		param_schema=ScreenshotParams,
		core_tool_name='screenshot',
		requires_browser=True,
	),
	ToolCapability(
		name='state',
		description='Get the current DOM state as structured text for the agent.',
		category='extraction',
		core_tool_name='extract',
		requires_browser=True,
	),
	ToolCapability(
		name='tab',
		description='List, create, switch to, or close browser tabs.',
		category='tab_management',
		param_schema=TabParams,
		core_tool_name='switch',
		requires_browser=True,
	),
	ToolCapability(
		name='keys',
		description='Send special keys (Enter, Escape, Tab, ArrowDown, etc.).',
		category='interaction',
		param_schema=KeysParams,
		core_tool_name='send_keys',
		requires_browser=True,
	),
	ToolCapability(
		name='select',
		description='Select an option from a dropdown by value.',
		category='interaction',
		param_schema=SelectParams,
		core_tool_name='select_dropdown',
		requires_browser=True,
	),
	ToolCapability(
		name='upload',
		description='Upload a local file into a file input element by index.',
		category='interaction',
		param_schema=UploadParams,
		core_tool_name='upload_file',
		requires_browser=True,
	),
	ToolCapability(
		name='eval',
		description='Evaluate arbitrary JavaScript on the page via CDP.',
		category='interaction',
		param_schema=EvalParams,
		core_tool_name='evaluate',
		requires_browser=True,
	),
	ToolCapability(
		name='extract',
		description='Extract structured content from the page using an LLM query.',
		category='extraction',
		param_schema=ExtractParams,
		core_tool_name='extract',
		requires_browser=True,
	),
	ToolCapability(
		name='cookies',
		description='Get, set, clear, export, or import browser cookies.',
		category='system',
		param_schema=CookiesParams,
		core_tool_name='evaluate',
		requires_browser=True,
	),
	ToolCapability(
		name='wait',
		description='Wait for a CSS selector or specific text to appear on the page.',
		category='system',
		param_schema=WaitParams,
		core_tool_name='wait',
		requires_browser=True,
	),
	ToolCapability(
		name='hover',
		description='Move the mouse cursor over an element by index.',
		category='interaction',
		param_schema=HoverParams,
		core_tool_name='click',
		requires_browser=True,
	),
	ToolCapability(
		name='dblclick',
		description='Double-click an element by index.',
		category='interaction',
		param_schema=HoverParams,
		core_tool_name='click',
		requires_browser=True,
	),
	ToolCapability(
		name='rightclick',
		description='Right-click an element by index.',
		category='interaction',
		param_schema=HoverParams,
		core_tool_name='click',
		requires_browser=True,
	),
	ToolCapability(
		name='get',
		description='Retrieve page info: title, html, element text, value, attributes, or bbox.',
		category='extraction',
		param_schema=GetParams,
		core_tool_name='extract',
		requires_browser=True,
	),
	ToolCapability(
		name='record',
		description='Start, stop, or check the status of a screen recording.',
		category='system',
		param_schema=RecordParams,
		core_tool_name='evaluate',
		requires_browser=True,
	),
]


# ---------------------------------------------------------------------------
# Derived collections — built automatically from CLI_COMMAND_CAPABILITIES.
# Import these from commands/browser.py instead of maintaining duplicate tables.
# ---------------------------------------------------------------------------


def get_commands() -> set[str]:
	"""Derive the COMMANDS set from the capability list."""
	return {cap.name for cap in CLI_COMMAND_CAPABILITIES}


def get_command_to_tool_mapping() -> dict[str, str]:
	"""Derive the COMMAND_TO_TOOL_MAPPING dict from the capability list.

	Only includes commands that actually map to a core tool (core_tool_name is not None).
	"""
	return {cap.name: cap.core_tool_name for cap in CLI_COMMAND_CAPABILITIES if cap.core_tool_name is not None}


def get_cli_specific_commands() -> set[str]:
	"""Derive the CLI_SPECIFIC_COMMANDS set from the capability list.

	These are commands with no corresponding core tool (core_tool_name is None).
	"""
	return {cap.name for cap in CLI_COMMAND_CAPABILITIES if cap.core_tool_name is None}


def get_capability(command_name: str) -> ToolCapability | None:
	"""Look up a command's ToolCapability by name."""
	for cap in CLI_COMMAND_CAPABILITIES:
		if cap.name == command_name:
			return cap
	return None


# Pre-computed module-level constants for convenient import
# (recomputed once at import time — the capability list is static).
COMMANDS: set[str] = get_commands()
COMMAND_TO_TOOL_MAPPING: dict[str, str] = get_command_to_tool_mapping()
CLI_SPECIFIC_COMMANDS: set[str] = get_cli_specific_commands()
