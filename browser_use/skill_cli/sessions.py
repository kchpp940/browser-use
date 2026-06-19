"""Session data — SessionInfo dataclass and browser session factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from browser_use.skill_cli.browser import CLIBrowserSession
from browser_use.skill_cli.python_session import PythonSession

if TYPE_CHECKING:
	from browser_use.browser.session import BrowserSession
	from browser_use.skill_cli.actions import ActionHandler

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
	"""Information about a browser session."""

	name: str
	headed: bool
	profile: str | None
	cdp_url: str | None
	browser_session: BrowserSession
	actions: ActionHandler | None = None
	python_session: PythonSession = field(default_factory=PythonSession)
	use_cloud: bool = False

	# Unified tool registry adapter for consistent tool metadata and
	# command filtering across all entry points (Agent, MCP, skill_cli).
	# This replaces per-module COMMANDS whitelists with a single source of truth.
	tool_registry_adapter: Any | None = None
	allowed_commands: set[str] | None = None

	def init_tool_registry(
		self,
		allowed_tool_names: list[str] | None = None,
		allowed_categories: list[str] | None = None,
	) -> None:
		"""Initialize the unified ToolRegistryAdapter and filter allowed commands.

		This is the single entry point for skill_cli command whitelisting.
		It uses the same ToolRegistryAdapter.filter_allowed_tools() mechanism
		as task templates and MCP servers, ensuring consistent tool availability
		checks across all entry points.

		Args:
			allowed_tool_names: Optional list of allowed core tool names.
				If provided, only commands mapping to these tools are allowed.
			allowed_categories: Optional list of allowed categories.
				If provided, only commands in these categories are allowed.
		"""
		from browser_use.skill_cli.commands.browser import (
			CLI_SPECIFIC_COMMANDS,
			COMMAND_TO_TOOL_MAPPING,
		)
		from browser_use.tools.service import Tools

		# Create a unified Tools registry and get its adapter
		tools = Tools()
		adapter = tools.get_tool_registry_adapter()

		# Apply unified filtering if any filter was specified.
		# This uses the same filter_allowed_tools() mechanism as task templates
		# and MCP servers, ensuring consistent tool availability checks.
		if allowed_tool_names is not None or allowed_categories is not None:
			# Always include 'done' action (required for task completion semantics)
			effective_names = None
			if allowed_tool_names is not None:
				effective_names = list(allowed_tool_names) + ['done']
			adapter = adapter.filter_allowed_tools(
				allowed_tool_names=effective_names,
				allowed_categories=allowed_categories,
				resolve_aliases=True,
			)

		# Build set of allowed core tool names from the filtered adapter
		allowed_core_tools: set[str] = {cap.name for cap in adapter.list_tool_capabilities()}

		# Build set of allowed CLI commands by checking the mapping
		allowed_commands: set[str] = set()

		# Add CLI-specific commands that don't map to core tools
		allowed_commands.update(CLI_SPECIFIC_COMMANDS)

		# Add commands whose mapped core tools are in the allowed set
		for cmd, core_tool in COMMAND_TO_TOOL_MAPPING.items():
			if core_tool in allowed_core_tools:
				allowed_commands.add(cmd)

		# If no filters were specified, allow everything in COMMANDS
		if allowed_tool_names is None and allowed_categories is None:
			from browser_use.skill_cli.commands.browser import COMMANDS

			allowed_commands = set(COMMANDS)

		self.tool_registry_adapter = adapter
		self.allowed_commands = allowed_commands

		logger.debug(
			f'Session tool registry initialized: '
			f'{len(allowed_commands)} commands allowed, '
			f'{len(allowed_core_tools)} core tools available'
		)

	def is_command_allowed(self, command: str) -> bool:
		"""Check if a command is allowed in this session.

		This uses the unified filter_allowed_tools() mechanism for consistent
		command availability checking across all entry points.
		"""
		if self.allowed_commands is None:
			# If registry was never initialized, allow all commands by default
			# (backward compatibility with existing code paths)
			return True
		return command in self.allowed_commands


async def create_browser_session(
	headed: bool,
	profile: str | None,
	cdp_url: str | None = None,
	use_cloud: bool = False,
	cloud_profile_id: str | None = None,
	cloud_proxy_country_code: str | None = None,
	cloud_timeout: int | None = None,
) -> CLIBrowserSession:
	"""Create BrowserSession based on connection mode.

	- CDP URL: Connect to existing browser (cdp_url takes precedence)
	- Cloud: Provision a cloud browser via BrowserSession(use_cloud=True)
	- With profile: User's real Chrome with the specified profile
	- No profile: Playwright-managed Chromium (default)
	"""
	if cdp_url is not None:
		return CLIBrowserSession(cdp_url=cdp_url)  # type: ignore[call-arg]

	if use_cloud:
		kwargs: dict = {'use_cloud': True}
		if cloud_profile_id is not None:
			kwargs['cloud_profile_id'] = cloud_profile_id
		if cloud_proxy_country_code is not None:
			kwargs['cloud_proxy_country_code'] = cloud_proxy_country_code
		if cloud_timeout is not None:
			kwargs['cloud_timeout'] = cloud_timeout
		return CLIBrowserSession(**kwargs)  # type: ignore[call-arg]

	if profile is None:
		return CLIBrowserSession(headless=not headed)  # type: ignore[call-arg]

	from browser_use.skill_cli.utils import find_chrome_executable, get_chrome_profile_path, list_chrome_profiles

	chrome_path = find_chrome_executable()
	if not chrome_path:
		raise RuntimeError('Could not find Chrome executable. Please install Chrome or omit --profile to use Chromium.')

	# Always get the Chrome user data directory (not the profile subdirectory)
	user_data_dir = get_chrome_profile_path(None)

	# Resolve profile: accept directory names ("Default", "Profile 1") and
	# display names ("Person 1", "Work"). Directory names take precedence.
	# If profile metadata can't be read, fall back to using the value as-is.
	known_profiles = list_chrome_profiles()
	directory_names = {p['directory'] for p in known_profiles}

	if not known_profiles or profile in directory_names:
		profile_directory = profile
	else:
		# Try case-insensitive display name match
		profile_directory = None
		profile_lower = profile.lower()
		for p in known_profiles:
			if p['name'].lower() == profile_lower:
				profile_directory = p['directory']
				break
		# Also try case-insensitive directory name match
		if profile_directory is None:
			for d in directory_names:
				if d.lower() == profile_lower:
					profile_directory = d
					break

		if profile_directory is None:
			lines = [f'Unknown profile {profile!r}. Available profiles:']
			for p in known_profiles:
				lines.append(f'  "{p["name"]}" ({p["directory"]})')
			raise RuntimeError('\n'.join(lines))

	return CLIBrowserSession(
		executable_path=chrome_path,  # type: ignore[call-arg]
		user_data_dir=user_data_dir,  # type: ignore[call-arg]
		profile_directory=profile_directory,  # type: ignore[call-arg]
		headless=not headed,  # type: ignore[call-arg]
	)
