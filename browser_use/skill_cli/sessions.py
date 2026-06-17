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


async def create_browser_session(
	headed: bool,
	profile: str | None,
	cdp_url: str | None = None,
	use_cloud: bool = False,
	cloud_profile_id: str | None = None,
	cloud_proxy_country_code: str | None = None,
	cloud_timeout: int | None = None,
) -> CLIBrowserSession:
	"""Create BrowserSession using unified config merge path.

	All configuration sources (CLI args, env vars, config.json) are merged
	with consistent priority via BrowserSession.from_config_sources().
	"""
	from browser_use.browser.session import BrowserSession

	# Build CLI args dict with all explicit parameters
	cli_args: dict[str, Any] = {
		'headed': headed,
		'cdp_url': cdp_url,
		'use_cloud': use_cloud,
		'cloud_profile_id': cloud_profile_id,
		'cloud_proxy_country_code': cloud_proxy_country_code,
		'cloud_timeout': cloud_timeout,
	}

	# Handle Chrome profile resolution (profile name -> directory)
	direct_kwargs: dict[str, Any] = {}
	if not cdp_url and not use_cloud and profile is not None:
		from browser_use.skill_cli.utils import find_chrome_executable, get_chrome_profile_path, list_chrome_profiles

		chrome_path = find_chrome_executable()
		if not chrome_path:
			raise RuntimeError('Could not find Chrome executable. Please install Chrome or omit --profile to use Chromium.')

		user_data_dir = get_chrome_profile_path(None)
		known_profiles = list_chrome_profiles()
		directory_names = {p['directory'] for p in known_profiles}

		if not known_profiles or profile in directory_names:
			profile_directory = profile
		else:
			profile_directory = None
			profile_lower = profile.lower()
			for p in known_profiles:
				if p['name'].lower() == profile_lower:
					profile_directory = p['directory']
					break
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

		direct_kwargs.update(
			{
				'executable_path': chrome_path,
				'user_data_dir': user_data_dir,
				'profile_directory': profile_directory,
			}
		)

	# Use the UNIFIED factory method — same path as Python API, TUI, beta agent
	session = BrowserSession.from_config_sources(
		direct_kwargs=direct_kwargs,
		cli_args=cli_args,
		load_from_env=True,
		load_from_config_file=True,
		skip_watchdogs=True,
	)

	return session  # type: ignore[return-value]
