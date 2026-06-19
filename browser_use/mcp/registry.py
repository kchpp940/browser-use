"""Unified MCP tool registry.

This module centralizes ALL tool metadata definitions for the MCP server,
including:
- Core tool name mappings (browser_* prefix -> canonical core tool name)
- Custom descriptions for core tools when exposed via MCP
- MCP-specific tool capabilities (browser_get_state, browser_get_html, etc.)
- Executor factories that bind tool execution to a BrowserUseServer instance

Before this refactor, all tool capabilities and executor closures were defined
inline inside mcp/server.py's _init_tool_registry() method. This module moves
them to a single source of truth so server.py only handles runtime binding.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from browser_use.mcp.views import (
	CloseAllSessionsParams,
	CloseSessionParams,
	GetHtmlParams,
	GetStateParams,
	ListSessionsParams,
	ListTabsParams,
	RetryWithAgentParams,
	TypeTextParams,
)
from browser_use.tools.registry.views import ToolCapability

# ---------------------------------------------------------------------------
# Protocol for the server methods that executors need access to.
# This keeps the registry decoupled from the concrete BrowserUseServer class.
# ---------------------------------------------------------------------------


class MCPServerProtocol(Protocol):
	"""Subset of BrowserUseServer API required by MCP tool executors."""

	async def _get_browser_state(self, include_screenshot: bool) -> tuple[str, str | None]: ...

	async def _get_html(self, selector: str | None) -> str: ...

	async def _list_tabs(self) -> str: ...

	async def _retry_with_browser_use_agent(
		self,
		task: str,
		max_steps: int = 100,
		model: str | None = None,
		allowed_domains: list[str] | None = None,
		use_vision: bool = True,
	) -> str: ...

	async def _list_sessions(self) -> str: ...

	async def _close_session(self, session_id: str) -> str: ...

	async def _close_all_sessions(self) -> str: ...


# ---------------------------------------------------------------------------
# Core tool mappings: MCP name -> canonical core tool name
# ---------------------------------------------------------------------------

CORE_TOOL_MAPPINGS: dict[str, str] = {
	'browser_navigate': 'navigate',
	'browser_click': 'click',
	'browser_type': 'input',
	'browser_scroll': 'scroll',
	'browser_go_back': 'go_back',
	'browser_switch_tab': 'switch',
	'browser_close_tab': 'close',
	'browser_screenshot': 'screenshot',
	'browser_extract_content': 'extract',
}

# ---------------------------------------------------------------------------
# Custom descriptions for core tools when exposed through the MCP interface.
# These are tuned for LLM consumers of the MCP API, distinct from the core
# tool descriptions which target the browser-use Agent prompt.
# ---------------------------------------------------------------------------

CORE_TOOL_MCP_DESCRIPTIONS: dict[str, str] = {
	'browser_navigate': 'Navigate to a URL in the browser',
	'browser_scroll': 'Scroll the page',
	'browser_go_back': 'Go back to the previous page',
	'browser_switch_tab': 'Switch to a different tab',
	'browser_close_tab': 'Close a tab',
	'browser_screenshot': (
		'Take a screenshot of the current page. Returns viewport metadata as text and the screenshot as an image.'
	),
	'browser_extract_content': 'Extract structured content from the current page based on a query',
}


# ---------------------------------------------------------------------------
# MCP-SPECIFIC TOOL CAPABILITIES
# These tools exist only in the MCP server and are NOT part of the core registry.
# ---------------------------------------------------------------------------


def build_get_state_capability() -> ToolCapability:
	"""Capability for browser_get_state."""
	return ToolCapability(
		name='browser_get_state',
		description='Get the current state of the page including all interactive elements',
		category='extraction',
		param_schema=GetStateParams,
		requires_browser=True,
		result_is_structured=True,
	)


def build_get_html_capability() -> ToolCapability:
	"""Capability for browser_get_html."""
	return ToolCapability(
		name='browser_get_html',
		description='Get the raw HTML of the current page or a specific element by CSS selector',
		category='extraction',
		param_schema=GetHtmlParams,
		requires_browser=True,
	)


def build_list_tabs_capability() -> ToolCapability:
	"""Capability for browser_list_tabs."""
	return ToolCapability(
		name='browser_list_tabs',
		description='List all open tabs',
		category='tab_management',
		param_schema=ListTabsParams,
		requires_browser=True,
	)


def build_retry_agent_capability() -> ToolCapability:
	"""Capability for retry_with_browser_use_agent."""
	return ToolCapability(
		name='retry_with_browser_use_agent',
		description=(
			'Retry a task using the browser-use agent. Only use this as a last resort '
			'if you fail to interact with a page multiple times.'
		),
		category='system',
		param_schema=RetryWithAgentParams,
		requires_browser=True,
		requires_llm=True,
	)


def build_list_sessions_capability() -> ToolCapability:
	"""Capability for browser_list_sessions."""
	return ToolCapability(
		name='browser_list_sessions',
		description='List all active browser sessions with their details and last activity time',
		category='system',
		param_schema=ListSessionsParams,
		requires_browser=False,
	)


def build_close_session_capability() -> ToolCapability:
	"""Capability for browser_close_session."""
	return ToolCapability(
		name='browser_close_session',
		description='Close a specific browser session by its ID',
		category='system',
		param_schema=CloseSessionParams,
		requires_browser=False,
	)


def build_close_all_sessions_capability() -> ToolCapability:
	"""Capability for browser_close_all."""
	return ToolCapability(
		name='browser_close_all',
		description='Close all active browser sessions and clean up resources',
		category='system',
		param_schema=CloseAllSessionsParams,
		requires_browser=False,
	)


# ---------------------------------------------------------------------------
# Custom MCP overrides for core tools (different param schema, descriptions)
# ---------------------------------------------------------------------------


def build_type_text_capability(base_cap: ToolCapability) -> ToolCapability:
	"""Override for browser_type: uses cleaner MCP-specific param schema."""
	return ToolCapability(
		name='browser_type',
		description=('Type text into an input field. Clears existing text by default; pass text="" to clear only.'),
		category=base_cap.category,
		param_schema=TypeTextParams,
		domains=base_cap.domains,
		terminates_sequence=base_cap.terminates_sequence,
		requires_browser=base_cap.requires_browser,
		requires_llm=base_cap.requires_llm,
		result_is_structured=base_cap.result_is_structured,
	)


def build_click_capability(base_cap: ToolCapability) -> ToolCapability:
	"""Override for browser_click: improved MCP-focused description."""
	return ToolCapability(
		name='browser_click',
		description=(
			'Click an element by index or at specific viewport coordinates. '
			'Use index for elements from browser_get_state, or coordinate_x/coordinate_y '
			'for pixel-precise clicking.'
		),
		category=base_cap.category,
		param_schema=base_cap.param_schema,
		domains=base_cap.domains,
		terminates_sequence=base_cap.terminates_sequence,
		requires_browser=base_cap.requires_browser,
		requires_llm=base_cap.requires_llm,
		result_is_structured=base_cap.result_is_structured,
	)


def build_core_override_capability(mcp_name: str, base_cap: ToolCapability, custom_description: str) -> ToolCapability:
	"""Build an override capability for a core tool with a custom MCP description."""
	return ToolCapability(
		name=mcp_name,
		description=custom_description,
		category=base_cap.category,
		param_schema=base_cap.param_schema,
		domains=base_cap.domains,
		terminates_sequence=base_cap.terminates_sequence,
		requires_browser=base_cap.requires_browser,
		requires_llm=base_cap.requires_llm,
		result_is_structured=base_cap.result_is_structured,
	)


# ---------------------------------------------------------------------------
# Executor factories — create executor callables bound to a server instance.
#
# These take a server (implementing MCPServerProtocol) and return an async
# callable suitable for passing to ToolRegistryAdapter.register_external_capability().
#
# Using factories instead of inline closures in server.py keeps tool definitions
# declarative and testable.
# ---------------------------------------------------------------------------


def make_get_state_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for browser_get_state bound to the given server."""

	async def _executor(
		include_screenshot: bool = False,
		browser_session: Any | None = None,
	) -> Any:
		state_json, screenshot_b64 = await server._get_browser_state(include_screenshot)
		result: dict[str, Any] = {'_state_json': state_json}
		if screenshot_b64:
			result['_screenshot_b64'] = screenshot_b64
		return result

	return _executor


def make_get_html_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for browser_get_html bound to the given server."""

	async def _executor(
		selector: str | None = None,
		browser_session: Any | None = None,
	) -> Any:
		return await server._get_html(selector)

	return _executor


def make_list_tabs_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for browser_list_tabs bound to the given server."""

	async def _executor(
		browser_session: Any | None = None,
	) -> Any:
		return await server._list_tabs()

	return _executor


def make_retry_agent_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for retry_with_browser_use_agent bound to the given server."""

	async def _executor(
		task: str,
		max_steps: int = 100,
		model: str | None = None,
		allowed_domains: list[str] | None = None,
		use_vision: bool = True,
		browser_session: Any | None = None,
	) -> Any:
		return await server._retry_with_browser_use_agent(
			task=task,
			max_steps=max_steps,
			model=model,
			allowed_domains=allowed_domains,
			use_vision=use_vision,
		)

	return _executor


def make_list_sessions_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for browser_list_sessions bound to the given server."""

	async def _executor() -> Any:
		return await server._list_sessions()

	return _executor


def make_close_session_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for browser_close_session bound to the given server."""

	async def _executor(session_id: str) -> Any:
		return await server._close_session(session_id)

	return _executor


def make_close_all_executor(
	server: MCPServerProtocol,
) -> Callable[..., Any]:
	"""Create executor for browser_close_all bound to the given server."""

	async def _executor() -> Any:
		return await server._close_all_sessions()

	return _executor


def make_type_text_executor(
	adapter: Any,
) -> Callable[..., Any]:
	"""Create executor for browser_type that delegates to the core 'input' tool.

	This maps MCP params (index, text) to the core tool's params transparently.
	"""

	async def _executor(
		index: int,
		text: str,
		browser_session: Any | None = None,
	) -> Any:
		return await adapter.execute_tool(
			tool_name='input',
			arguments={'index': index, 'text': text},
			browser_session=browser_session,
		)

	return _executor


# ---------------------------------------------------------------------------
# Top-level registration helper — wires everything into a ToolRegistryAdapter.
# ---------------------------------------------------------------------------


def register_all_mcp_tools(
	adapter: Any,
	server: MCPServerProtocol,
) -> None:
	"""Register ALL MCP tools (core mappings + MCP-specific) on the given adapter.

	This is the single entry point called from BrowserUseServer._init_tool_registry().
	It replaces ~300 lines of inline closure/register calls in server.py with a
	declarative, centrally managed registration process.
	"""

	# --- Step 1: Register name mappings for core tools ---
	for mcp_name, core_name in CORE_TOOL_MAPPINGS.items():
		adapter.register_name_mapping(mcp_name, core_name)

	# --- Step 2: Register custom overrides for core tools with MCP-specific UX ---
	for mcp_name, core_name in CORE_TOOL_MAPPINGS.items():
		base_cap = adapter.get_tool_capability(core_name)
		if base_cap is None:
			continue

		if mcp_name == 'browser_type':
			mcp_cap = build_type_text_capability(base_cap)
			executor = make_type_text_executor(adapter)
			adapter.register_external_capability(
				capability=mcp_cap,
				executor=executor,
				canonical_alias=mcp_name,
			)

		elif mcp_name == 'browser_click':
			mcp_cap = build_click_capability(base_cap)
			adapter.register_external_capability(
				capability=mcp_cap,
				executor=None,  # Falls through to name-mapped execute_tool
				canonical_alias=mcp_name,
			)

		elif mcp_name in CORE_TOOL_MCP_DESCRIPTIONS:
			mcp_cap = build_core_override_capability(
				mcp_name=mcp_name,
				base_cap=base_cap,
				custom_description=CORE_TOOL_MCP_DESCRIPTIONS[mcp_name],
			)
			adapter.register_external_capability(
				capability=mcp_cap,
				executor=None,
				canonical_alias=mcp_name,
			)

	# --- Step 3: Register MCP-specific external tools with their executors ---
	mcp_specific_tools: list[tuple[Callable[[], ToolCapability], Callable[[MCPServerProtocol], Callable[..., Any]]]] = [
		(build_get_state_capability, make_get_state_executor),
		(build_get_html_capability, make_get_html_executor),
		(build_list_tabs_capability, make_list_tabs_executor),
		(build_retry_agent_capability, make_retry_agent_executor),
		(build_list_sessions_capability, make_list_sessions_executor),
		(build_close_session_capability, make_close_session_executor),
		(build_close_all_sessions_capability, make_close_all_executor),
	]

	for cap_builder, exec_factory in mcp_specific_tools:
		capability = cap_builder()
		executor = exec_factory(server)
		adapter.register_external_capability(capability=capability, executor=executor)
