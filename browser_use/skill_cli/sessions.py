"""Session data — SessionInfo dataclass and browser session factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from browser_use.runtime.adapter import _browser_config_to_profile_kwargs
from browser_use.runtime.config import BrowserSessionConfig
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
	"""Create BrowserSession based on connection mode.

	Uses CommandRuntimeAdapter's BrowserSessionConfig-to-profile conversion
	for standard cases, with skill_cli-specific Chrome profile discovery.

	- CDP URL: Connect to existing browser (cdp_url takes precedence)
	- Cloud: Provision a cloud browser via BrowserSession(use_cloud=True)
	- With profile: User's real Chrome with the specified profile
	- No profile: Playwright-managed Chromium (default)
	"""
	if cdp_url is not None:
		bs_config = BrowserSessionConfig(cdp_url=cdp_url, headed=headed)
		profile_kwargs = _browser_config_to_profile_kwargs(bs_config)
		return CLIBrowserSession(**profile_kwargs)

	if use_cloud:
		bs_config = BrowserSessionConfig(
			use_cloud=True,
			headed=headed,
			cloud_profile_id=cloud_profile_id,
			cloud_proxy_country_code=cloud_proxy_country_code,
			cloud_timeout=cloud_timeout,
		)
		profile_kwargs = _browser_config_to_profile_kwargs(bs_config)
		return CLIBrowserSession(**profile_kwargs)

	if profile is None:
		bs_config = BrowserSessionConfig(headless=not headed, headed=headed)
		profile_kwargs = _browser_config_to_profile_kwargs(bs_config)
		return CLIBrowserSession(**profile_kwargs)

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

	bs_config = BrowserSessionConfig(
		executable_path=chrome_path,
		user_data_dir=user_data_dir,
		profile_directory=profile_directory,
		headless=not headed,
		headed=headed,
	)
	profile_kwargs = _browser_config_to_profile_kwargs(bs_config)
	return CLIBrowserSession(**profile_kwargs)
