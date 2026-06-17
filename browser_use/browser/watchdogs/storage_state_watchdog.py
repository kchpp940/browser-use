"""Storage state watchdog for managing browser cookies and storage persistence."""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from bubus import BaseEvent
from cdp_use.cdp.network import Cookie
from pydantic import Field, PrivateAttr

from browser_use.browser.events import (
	BrowserConnectedEvent,
	BrowserStopEvent,
	LoadStorageStateEvent,
	SaveStorageStateEvent,
	StorageStateLoadedEvent,
	StorageStateSavedEvent,
)
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.utils import create_task_with_error_handling

_SESSION_COOKIE_EXPIRES_VALUES = frozenset({0, 0.0, -1, -1.0})


def _normalize_cookie(cookie: dict[str, Any]) -> dict[str, Any]:
	"""Normalize a single cookie dict for consistent storage.

	- Session cookies (expires=0 or -1) have the expires key removed so that
	  CDP does not treat them as already-expired when re-loaded.
	- ``sameSite`` is defaulted to ``'Lax'`` when absent or ``'None'`` (which
	  is not a valid CDP value).
	"""
	c = dict(cookie)
	expires = c.get('expires')
	if expires in _SESSION_COOKIE_EXPIRES_VALUES:
		c.pop('expires', None)
	same_site = c.get('sameSite')
	if not same_site or same_site == 'None':
		c['sameSite'] = 'Lax'
	return c


def _normalize_storage_state(state: dict[str, Any]) -> dict[str, Any]:
	"""Return a copy of *state* with every cookie normalized."""
	normalized = dict(state)
	if 'cookies' in normalized:
		normalized['cookies'] = [_normalize_cookie(c) if isinstance(c, dict) else c for c in normalized['cookies']]
	return normalized


class StorageStateWatchdog(BaseWatchdog):
	"""Monitors and persists browser storage state including cookies and localStorage."""

	LISTENS_TO: ClassVar[list[type[BaseEvent]]] = [
		BrowserConnectedEvent,
		BrowserStopEvent,
		SaveStorageStateEvent,
		LoadStorageStateEvent,
	]
	EMITS: ClassVar[list[type[BaseEvent]]] = [
		StorageStateSavedEvent,
		StorageStateLoadedEvent,
	]

	auto_save_interval: float = Field(default=30.0)
	save_on_change: bool = Field(default=True)

	_monitoring_task: asyncio.Task | None = PrivateAttr(default=None)
	_last_cookie_state: list[dict] = PrivateAttr(default_factory=list)
	_save_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

	async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
		"""Start monitoring when browser starts."""
		self.logger.debug('[StorageStateWatchdog] 🍪 Initializing auth/cookies sync <-> with storage_state.json file')

		await self._start_monitoring()

		await self.event_bus.dispatch(LoadStorageStateEvent())

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		"""Stop monitoring when browser stops.

		Performs a final save before cancelling the monitoring loop so that
		any state accumulated since the last auto-save is persisted.
		"""
		self.logger.debug('[StorageStateWatchdog] Stopping storage_state monitoring')
		await self._save_storage_state()
		await self._stop_monitoring()

	async def on_SaveStorageStateEvent(self, event: SaveStorageStateEvent) -> None:
		"""Handle storage state save request."""
		path = event.path
		if path is None:
			if self.browser_session.browser_profile.storage_state:
				path = str(self.browser_session.browser_profile.storage_state)
			else:
				path = None
		await self._save_storage_state(path)

	async def on_LoadStorageStateEvent(self, event: LoadStorageStateEvent) -> None:
		"""Handle storage state load request."""
		path = event.path
		if path is None:
			if self.browser_session.browser_profile.storage_state:
				path = str(self.browser_session.browser_profile.storage_state)
			else:
				path = None
		await self._load_storage_state(path)

	async def _start_monitoring(self) -> None:
		"""Start the monitoring task."""
		if self._monitoring_task and not self._monitoring_task.done():
			return

		assert self.browser_session.cdp_client is not None

		self._monitoring_task = create_task_with_error_handling(
			self._monitor_storage_changes(), name='monitor_storage_changes', logger_instance=self.logger, suppress_exceptions=True
		)

	async def _stop_monitoring(self) -> None:
		"""Stop the monitoring task."""
		if self._monitoring_task and not self._monitoring_task.done():
			self._monitoring_task.cancel()
			try:
				await self._monitoring_task
			except asyncio.CancelledError:
				pass

	async def _check_for_cookie_changes_cdp(self, event: dict) -> None:
		"""Check if a CDP network event indicates cookie changes."""
		try:
			headers = event.get('headers', {})
			if 'set-cookie' in headers or 'Set-Cookie' in headers:
				self.logger.debug('[StorageStateWatchdog] Cookie change detected via CDP')
				if self.save_on_change:
					await self._save_storage_state()
		except Exception as e:
			self.logger.warning(f'[StorageStateWatchdog] Error checking for cookie changes: {e}')

	async def _monitor_storage_changes(self) -> None:
		"""Periodically check for storage changes and auto-save."""
		while True:
			try:
				await asyncio.sleep(self.auto_save_interval)
				if await self._have_cookies_changed():
					self.logger.debug('[StorageStateWatchdog] Detected changes to sync with storage_state.json')
					await self._save_storage_state()
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f'[StorageStateWatchdog] Error in monitoring loop: {e}')

	async def _have_cookies_changed(self) -> bool:
		"""Check if cookies have changed since last save."""
		if not self.browser_session.cdp_client:
			return False

		try:
			current_cookies = await self.browser_session._cdp_get_cookies()

			current_cookie_set = {
				(c.get('name', ''), c.get('domain', ''), c.get('path', '')): c.get('value', '') for c in current_cookies
			}

			last_cookie_set = {
				(c.get('name', ''), c.get('domain', ''), c.get('path', '')): c.get('value', '') for c in self._last_cookie_state
			}

			return current_cookie_set != last_cookie_set
		except Exception as e:
			self.logger.debug(f'[StorageStateWatchdog] Error comparing cookies: {e}')
			return False

	async def _save_storage_state(self, path: str | None = None) -> None:
		"""Save browser storage state to file.

		The write is performed atomically via a temp file + ``os.replace`` so
		that a crash at any point never leaves the destination file in a
		partial or missing state.
		"""
		async with self._save_lock:
			save_path = path or self.browser_session.browser_profile.storage_state
			if not save_path:
				return

			if isinstance(save_path, dict):
				self.logger.debug('[StorageStateWatchdog] Storage state is already a dict, skipping file save')
				return

			try:
				cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=None)
				if not cdp_session:
					self.logger.warning('[StorageStateWatchdog] No CDP session available, skipping save')
					return

				storage_state = await self.browser_session._cdp_get_storage_state()
				normalized_new = _normalize_storage_state(dict(storage_state))

				new_cookies = normalized_new.get('cookies', [])
				new_origins = normalized_new.get('origins', [])

				self._last_cookie_state = new_cookies.copy()

				json_path = Path(save_path).expanduser().resolve()
				json_path.parent.mkdir(parents=True, exist_ok=True)

				if json_path.exists():
					try:
						existing_state = _normalize_storage_state(json.loads(json_path.read_text(encoding='utf-8')))
						existing_cookies = existing_state.get('cookies', [])
						existing_origins = existing_state.get('origins', [])

						if not new_cookies and not new_origins and (existing_cookies or existing_origins):
							self.logger.warning(
								'[StorageStateWatchdog] CDP returned empty state but file has data; '
								'preserving existing file to avoid overwriting login state'
							)
							return

						merged_state = self._merge_storage_states(existing_state, normalized_new)
					except Exception as e:
						self.logger.error(f'[StorageStateWatchdog] Failed to merge with existing state, preserving file: {e}')
						return
				else:
					merged_state = normalized_new

				merged_cookies = merged_state.get('cookies', [])
				if not merged_cookies and not merged_state.get('origins', []):
					self.logger.debug('[StorageStateWatchdog] Merged state is empty, skipping save')
					return

				fd, tmp_path = tempfile.mkstemp(
					dir=str(json_path.parent),
					prefix=json_path.stem + '_',
					suffix='.json.tmp',
				)
				try:
					with os.fdopen(fd, 'w', encoding='utf-8') as f:
						f.write(json.dumps(merged_state, indent=4, ensure_ascii=False))
					os.replace(tmp_path, str(json_path))
					tmp_path = None
				except BaseException:
					if tmp_path and os.path.exists(tmp_path):
						os.unlink(tmp_path)
					raise

				self.event_bus.dispatch(
					StorageStateSavedEvent(
						path=str(json_path),
						cookies_count=len(merged_cookies),
						origins_count=len(merged_state.get('origins', [])),
					)
				)

				self.logger.debug(
					f'[StorageStateWatchdog] Saved storage state to {json_path} '
					f'({len(merged_cookies)} cookies, '
					f'{len(merged_state.get("origins", []))} origins)'
				)

			except Exception as e:
				self.logger.error(f'[StorageStateWatchdog] Failed to save storage state: {e}')

	async def _load_storage_state(self, path: str | None = None) -> None:
		"""Load browser storage state from file."""
		if not self.browser_session.cdp_client:
			self.logger.warning('[StorageStateWatchdog] No CDP client available for loading')
			return

		load_path = path or self.browser_session.browser_profile.storage_state
		if not load_path or not os.path.exists(str(load_path)):
			return

		try:
			import anyio

			content = await anyio.Path(str(load_path)).read_text()
			storage = _normalize_storage_state(json.loads(content))

			if 'cookies' in storage and storage['cookies']:
				normalized_cookies: list[Cookie] = []
				for cookie in storage['cookies']:
					if isinstance(cookie, dict):
						normalized_cookies.append(Cookie(**cookie))
					else:
						normalized_cookies.append(cookie)  # type: ignore[arg-type]

				await self.browser_session._cdp_set_cookies(normalized_cookies)
				self._last_cookie_state = storage['cookies'].copy()
				self.logger.debug(f'[StorageStateWatchdog] Added {len(storage["cookies"])} cookies from storage state')

			if 'origins' in storage and storage['origins']:
				for origin in storage['origins']:
					origin_value = origin.get('origin')
					if not origin_value:
						continue

					if origin.get('localStorage'):
						lines = []
						for item in origin['localStorage']:
							lines.append(f'window.localStorage.setItem({json.dumps(item["name"])}, {json.dumps(item["value"])});')
						script = (
							'(function(){\n'
							f'  if (window.location && window.location.origin !== {json.dumps(origin_value)}) return;\n'
							'  try {\n'
							f'    {" ".join(lines)}\n'
							'  } catch (e) {}\n'
							'})();'
						)
						await self.browser_session._cdp_add_init_script(script)

					if origin.get('sessionStorage'):
						lines = []
						for item in origin['sessionStorage']:
							lines.append(
								f'window.sessionStorage.setItem({json.dumps(item["name"])}, {json.dumps(item["value"])});'
							)
						script = (
							'(function(){\n'
							f'  if (window.location && window.location.origin !== {json.dumps(origin_value)}) return;\n'
							'  try {\n'
							f'    {" ".join(lines)}\n'
							'  } catch (e) {}\n'
							'})();'
						)
						await self.browser_session._cdp_add_init_script(script)
				self.logger.debug(
					f'[StorageStateWatchdog] Applied localStorage/sessionStorage from {len(storage["origins"])} origins'
				)

			self.event_bus.dispatch(
				StorageStateLoadedEvent(
					path=str(load_path),
					cookies_count=len(storage.get('cookies', [])),
					origins_count=len(storage.get('origins', [])),
				)
			)

			self.logger.debug(f'[StorageStateWatchdog] Loaded storage state from: {load_path}')

		except Exception as e:
			self.logger.error(f'[StorageStateWatchdog] Failed to load storage state: {e}')

	@staticmethod
	def _merge_storage_states(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
		"""Merge two storage states with new values taking precedence.

		Cookies are keyed by ``(name, domain, path)``; new overwrites existing
		on collision.  Origins are keyed by their ``origin`` string and
		localStorage/sessionStorage entries are **deep-merged** so that items
		present in *existing* but absent in *new* are not lost.
		"""
		merged: dict[str, Any] = {}

		# --- cookies ---
		existing_cookies: dict[tuple[str, str, str], dict[str, Any]] = {
			(c['name'], c['domain'], c['path']): c for c in existing.get('cookies', []) if isinstance(c, dict)
		}
		for cookie in new.get('cookies', []):
			if not isinstance(cookie, dict):
				continue
			key = (cookie['name'], cookie['domain'], cookie['path'])
			existing_cookies[key] = cookie
		merged['cookies'] = list(existing_cookies.values())

		# --- origins (deep-merge localStorage / sessionStorage items) ---
		def _merge_storage_items(
			existing_items: list[dict[str, str]] | None,
			new_items: list[dict[str, str]] | None,
		) -> list[dict[str, str]] | None:
			if not existing_items and not new_items:
				return None
			items_by_name: dict[str, dict[str, str]] = {}
			for item in existing_items or []:
				items_by_name[item['name']] = item
			for item in new_items or []:
				items_by_name[item['name']] = item
			return list(items_by_name.values())

		existing_origins: dict[str, dict[str, Any]] = {o['origin']: o for o in existing.get('origins', [])}
		for origin in new.get('origins', []):
			origin_name = origin.get('origin')
			if not origin_name:
				continue
			if origin_name not in existing_origins:
				existing_origins[origin_name] = origin
			else:
				merged_origin: dict[str, Any] = {'origin': origin_name}
				merged_origin['localStorage'] = _merge_storage_items(
					existing_origins[origin_name].get('localStorage'),
					origin.get('localStorage'),
				)
				merged_origin['sessionStorage'] = _merge_storage_items(
					existing_origins[origin_name].get('sessionStorage'),
					origin.get('sessionStorage'),
				)
				if merged_origin['localStorage'] is None:
					merged_origin.pop('localStorage')
				if merged_origin['sessionStorage'] is None:
					merged_origin.pop('sessionStorage')
				existing_origins[origin_name] = merged_origin

		merged['origins'] = list(existing_origins.values())

		return merged

	async def get_current_cookies(self) -> list[dict[str, Any]]:
		"""Get current cookies using CDP."""
		if not self.browser_session.cdp_client:
			return []

		try:
			cookies = await self.browser_session._cdp_get_cookies()
			return [dict(cookie) for cookie in cookies]
		except Exception as e:
			self.logger.error(f'[StorageStateWatchdog] Failed to get cookies: {e}')
			return []

	async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
		"""Add cookies using CDP."""
		if not self.browser_session.cdp_client:
			self.logger.warning('[StorageStateWatchdog] No CDP client available for adding cookies')
			return

		try:
			cookie_objects = [Cookie(**cookie_dict) if isinstance(cookie_dict, dict) else cookie_dict for cookie_dict in cookies]
			await self.browser_session._cdp_set_cookies(cookie_objects)
			self.logger.debug(f'[StorageStateWatchdog] Added {len(cookies)} cookies')
		except Exception as e:
			self.logger.error(f'[StorageStateWatchdog] Failed to add cookies: {e}')
