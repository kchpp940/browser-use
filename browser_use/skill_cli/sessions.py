"""Session data — SessionInfo dataclass and browser session factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
	profile_directory: str | None
	cdp_url: str | None
	browser_session: BrowserSession
	actions: ActionHandler | None = None
	python_session: PythonSession = field(default_factory=PythonSession)
	use_cloud: bool = False
	profile: str | None = None


async def create_browser_session(
	headed: bool,
	profile_directory: str | None,
	cdp_url: str | None = None,
	use_cloud: bool = False,
	cloud_profile_id: str | None = None,
	cloud_proxy_country_code: str | None = None,
	cloud_timeout: int | None = None,
	profile: str | None = None,
) -> CLIBrowserSession:
	"""Create BrowserSession based on connection mode.

	- CDP URL: Connect to existing browser (cdp_url takes precedence)
	- Cloud: Provision a cloud browser via BrowserSession(use_cloud=True)
	- With profile_directory: User's real Chrome with the specified profile directory
	- No profile_directory: Playwright-managed Chromium (default)
	- Profile: Load configuration from a profile preset (overrides other settings)
	"""
	if cdp_url is not None:
		return CLIBrowserSession(cdp_url=cdp_url, profile=profile)  # type: ignore[call-arg]

	if use_cloud:
		kwargs: dict = {'use_cloud': True, 'profile': profile}
		if cloud_profile_id is not None:
			kwargs['cloud_profile_id'] = cloud_profile_id
		if cloud_proxy_country_code is not None:
			kwargs['cloud_proxy_country_code'] = cloud_proxy_country_code
		if cloud_timeout is not None:
			kwargs['cloud_timeout'] = cloud_timeout
		return CLIBrowserSession(**kwargs)  # type: ignore[call-arg]

	if profile_directory is None:
		return CLIBrowserSession(headless=not headed, profile=profile)  # type: ignore[call-arg]

	from browser_use.skill_cli.utils import find_chrome_executable, get_chrome_profile_path, list_chrome_profiles

	chrome_path = find_chrome_executable()
	if not chrome_path:
		raise RuntimeError('Could not find Chrome executable. Please install Chrome or omit --profile-directory to use Chromium.')

	# Always get the Chrome user data directory (not the profile subdirectory)
	user_data_dir = get_chrome_profile_path(None)

	# Resolve profile_directory: accept directory names ("Default", "Profile 1") and
	# display names ("Person 1", "Work"). Directory names take precedence.
	# If profile_directory metadata can't be read, fall back to using the value as-is.
	known_profiles = list_chrome_profiles()
	directory_names = {p['directory'] for p in known_profiles}

	if not known_profiles or profile_directory in directory_names:
		resolved_profile_directory = profile_directory
	else:
		# Try case-insensitive display name match
		resolved_profile_directory = None
		profile_lower = profile_directory.lower()
		for p in known_profiles:
			if p['name'].lower() == profile_lower:
				resolved_profile_directory = p['directory']
				break
		# Also try case-insensitive directory name match
		if resolved_profile_directory is None:
			for d in directory_names:
				if d.lower() == profile_lower:
					resolved_profile_directory = d
					break

		if resolved_profile_directory is None:
			lines = [f'Unknown profile directory {profile_directory!r}. Available profiles:']
			for p in known_profiles:
				lines.append(f'  "{p["name"]}" ({p["directory"]})')
			raise RuntimeError('\n'.join(lines))

	browser_kwargs = {
		'executable_path': chrome_path,
		'user_data_dir': user_data_dir,
		'profile_directory': resolved_profile_directory,
		'headless': not headed,
		'profile': profile,
	}

	return CLIBrowserSession(**browser_kwargs)  # type: ignore[call-arg]
