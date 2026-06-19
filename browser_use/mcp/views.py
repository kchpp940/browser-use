"""Pydantic models for MCP-specific tool parameters.

These models define parameter schemas for tools that are specific to the MCP server
and are not part of the core browser-use tools registry.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GetStateParams(BaseModel):
	"""Parameters for browser_get_state tool."""

	include_screenshot: bool = Field(
		default=False,
		description='Whether to include a screenshot of the current page',
	)


class GetHtmlParams(BaseModel):
	"""Parameters for browser_get_html tool."""

	selector: str | None = Field(
		default=None,
		description='Optional CSS selector to get HTML of a specific element. If omitted, returns full page HTML.',
	)


class ListTabsParams(BaseModel):
	"""Parameters for browser_list_tabs tool."""

	pass


class RetryWithAgentParams(BaseModel):
	"""Parameters for retry_with_browser_use_agent tool."""

	task: str = Field(
		description='The high-level goal and detailed step-by-step description of the task the AI browser agent needs to attempt, along with any relevant data needed to complete the task and info about previous attempts.',
	)
	max_steps: int = Field(
		default=100,
		description='Maximum number of steps an agent can take.',
	)
	model: str | None = Field(
		default=None,
		description='LLM model to use (e.g., gpt-4o, claude-3-opus-20240229). Defaults to the configured model.',
	)
	allowed_domains: list[str] | None = Field(
		default=None,
		description='List of domains the agent is allowed to visit (security feature). Omit to use the server-configured profile defaults. An empty list is treated the same as omitting the argument and will NOT disable server-configured restrictions.',
	)
	use_vision: bool = Field(
		default=True,
		description='Whether to use vision capabilities (screenshots) for the agent',
	)


class ListSessionsParams(BaseModel):
	"""Parameters for browser_list_sessions tool."""

	pass


class CloseSessionParams(BaseModel):
	"""Parameters for browser_close_session tool."""

	session_id: str = Field(
		description='The browser session ID to close (get from browser_list_sessions)',
	)


class CloseAllSessionsParams(BaseModel):
	"""Parameters for browser_close_all tool."""

	pass


class TypeTextParams(BaseModel):
	"""Parameters for browser_type tool.

	Replicates the core InputTextAction with MCP naming.
	"""

	index: int = Field(
		description='The index of the input element (from browser_get_state)',
	)
	text: str = Field(
		description='The text to type. Pass an empty string ("") to clear the field without typing.',
	)
