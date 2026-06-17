"""Storage state watchdog for managing browser cookies and storage persistence."""

from __future__ import annotations

import asyncio
import json
import os
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
from browser_use.browser.storage_state import (
	StorageStateSource,
	merge_storage_states,
	normalize_storage_state,
	resolve_storage_state_path,
	write_storage_state_atomically,
)
from browser_use.browser.watchdog_base import BaseWatchdog
from browser_use.utils import create_task_with_error_handling


class StorageStateWatchdog(BaseWatchdog):
	"""Monitors and persists browser storage state.

	Reads the initial storage state from ``BrowserProfile.storage_state``,
	which may be:

	* **A file path** (``str | Path``) → load from the file and keep writing
	  updated cookies / storage back to it (auto-save + stop-save).
	* **A dict** → load into the browser once; never write back (read-only
	  seed data).
	* ``None`` → nothing to load or save.
	"""

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
	_last_cookie_state: list[dict[str, Any]] = PrivateAttr(default_factory=list)
	_save_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

	@property
	def _profile_storage_state(self) -> str | Path | dict[str, Any] | None:
		"""Raw storage_state value from the browser profile."""
		return self.browser_session.browser_profile.storage_state

	@property
	def _save_target_path(self) -> Path | None:
		"""Resolved file path to save state to, or ``None`` if not persistent.

		Only file-path-based storage states are writable; dict-based ones
		are read-only seed data.
		"""
		return resolve_storage_state_path(self._profile_storage_state)

	# ------------------------------------------------------------------
	# Event handlers
	# ------------------------------------------------------------------

	async def on_BrowserConnectedEvent(self, event: BrowserConnectedEvent) -> None:
		"""Start monitoring and load initial state on browser connect."""
		self.logger.debug("[StorageStateWatchdog] 🍪 Initializing auth/cookies sync")
		await self._start_monitoring()
		await self.event_bus.dispatch(LoadStorageStateEvent())

	async def on_BrowserStopEvent(self, event: BrowserStopEvent) -> None:
		"""Final save then stop monitoring on browser stop.

		Save happens *before* cancelling the monitoring task so that any
		state accumulated since the last auto-save is persisted.
		"""
		self.logger.debug("[StorageStateWatchdog] Stopping storage_state monitoring")
		await self._save_storage_state()
		await self._stop_monitoring()

	async def on_SaveStorageStateEvent(self, event: SaveStorageStateEvent) -> None:
		"""Handle explicit storage state save request."""
		await self._save_storage_state(path=event.path)

	async def on_LoadStorageStateEvent(self, event: LoadStorageStateEvent) -> None:
		"""Handle explicit storage state load request."""
		await self._load_storage_state(path=event.path)

	# ------------------------------------------------------------------
	# Monitoring loop
	# ------------------------------------------------------------------

	async def _start_monitoring(self) -> None:
		"""Start the periodic monitoring task."""
		if self._monitoring_task and not self._monitoring_task.done():
			return

		assert self.browser_session.cdp_client is not None

		self._monitoring_task = create_task_with_error_handling(
			self._monitor_storage_changes(),
			name="monitor_storage_changes",
			logger_instance=self.logger,
			suppress_exceptions=True,
		)

	async def _stop_monitoring(self) -> None:
		"""Stop the periodic monitoring task."""
		if self._monitoring_task and not self._monitoring_task.done():
			self._monitoring_task.cancel()
			try:
				await self._monitoring_task
			except asyncio.CancelledError:
				pass

	async def _check_for_cookie_changes_cdp(self, event: dict) -> None:
		"""Check if a CDP network event indicates cookie changes."""
		try:
			headers = event.get("headers", {})
			if "set-cookie" in headers or "Set-Cookie" in headers:
				self.logger.debug("[StorageStateWatchdog] Cookie change detected via CDP")
				if self.save_on_change:
					await self._save_storage_state()
		except Exception as e:
			self.logger.warning(f"[StorageStateWatchdog] Error checking for cookie changes: {e}")

	async def _monitor_storage_changes(self) -> None:
		"""Periodically check for storage changes and auto-save."""
		while True:
			try:
				await asyncio.sleep(self.auto_save_interval)
				if await self._have_cookies_changed():
					self.logger.debug(
						"[StorageStateWatchdog] Detected changes to sync with storage_state"
					)
					await self._save_storage_state()
			except asyncio.CancelledError:
				break
			except Exception as e:
				self.logger.error(f"[StorageStateWatchdog] Error in monitoring loop: {e}")

	async def _have_cookies_changed(self) -> bool:
		"""Check if cookies have changed since the last saved state."""
		if not self.browser_session.cdp_client:
			return False

		try:
			current_cookies = await self.browser_session._cdp_get_cookies()
			current_cookie_set = {
				(c.get("name", ""), c.get("domain", ""), c.get("path", "")): c.get("value", "")
				for c in current_cookies
			}
			last_cookie_set = {
				(c.get("name", ""), c.get("domain", ""), c.get("path", "")): c.get("value", "")
				for c in self._last_cookie_state
			}
			return current_cookie_set != last_cookie_set
		except Exception as e:
			self.logger.debug(f"[StorageStateWatchdog] Error comparing cookies: {e}")
			return False

	# ------------------------------------------------------------------
	# Save
	# ------------------------------------------------------------------

	async def _save_storage_state(self, path: str | None = None) -> None:
		"""Save browser storage state.

		If *path* is given, always write there.  Otherwise use the profile's
		``storage_state`` value — but only if it resolves to a file path
		(dict-based storage states are read-only and never written back).
		"""
		async with self._save_lock:
			if path is not None:
				json_path = Path(path).expanduser().resolve()
			else:
				json_path = self._save_target_path

			if json_path is None:
				return

			try:
				cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=None)
				if not cdp_session:
					self.logger.warning(
						"[StorageStateWatchdog] No CDP session available, skipping save"
					)
					return

				storage_state = await self.browser_session._cdp_get_storage_state()
				normalized_new = normalize_storage_state(dict(storage_state))

				new_cookies = normalized_new.get("cookies", [])
				new_origins = normalized_new.get("origins", [])

				self._last_cookie_state = [dict(c) for c in new_cookies]

				if json_path.exists():
					try:
						existing_state = normalize_storage_state(
							json.loads(json_path.read_text(encoding="utf-8"))
						)
						existing_cookies = existing_state.get("cookies", [])
						existing_origins = existing_state.get("origins", [])

						if (
							not new_cookies
							and not new_origins
							and (existing_cookies or existing_origins)
						):
							self.logger.warning(
								"[StorageStateWatchdog] CDP returned empty state but "
								"file has data; preserving existing file to avoid "
								"overwriting login state"
							)
							return

						merged_state = merge_storage_states(existing_state, normalized_new)
					except Exception as e:
						self.logger.error(
							"[StorageStateWatchdog] Failed to merge with existing state, "
							f"preserving file: {e}"
						)
						return
				else:
					merged_state = normalized_new

				merged_cookies = merged_state.get("cookies", [])
				if not merged_cookies and not merged_state.get("origins", []):
					self.logger.debug(
						"[StorageStateWatchdog] Merged state is empty, skipping save"
					)
					return

				write_storage_state_atomically(json_path, merged_state)

				self.event_bus.dispatch(
					StorageStateSavedEvent(
						path=str(json_path),
						cookies_count=len(merged_cookies),
						origins_count=len(merged_state.get("origins", [])),
					)
				)

				self.logger.debug(
					f"[StorageStateWatchdog] Saved storage state to {json_path} "
					f"({len(merged_cookies)} cookies, "
					f'{len(merged_state.get("origins", []))} origins)'
				)

			except Exception as e:
				self.logger.error(f"[StorageStateWatchdog] Failed to save storage state: {e}")

	# ------------------------------------------------------------------
	# Load
	# ------------------------------------------------------------------

	async def _load_storage_state(self, path: str | None = None) -> None:
		"""Load storage state into the browser session.

		If *path* is given, load from that file.  Otherwise resolve the
		profile's ``storage_state`` — which may be a file path or an
		in-memory dict.
		"""
		if not self.browser_session.cdp_client:
			self.logger.warning("[StorageStateWatchdog] No CDP client available for loading")
			return

		# Resolve source
		if path is not None:
			if not os.path.exists(path):
				return
			try:
				source = StorageStateSource.from_any(path)
			except Exception:
				return
		else:
			source = StorageStateSource.from_any(self._profile_storage_state)

		if source is None:
			return

		storage = normalize_storage_state(source.value)
		source_label = str(source.path) if source.path else "<in-memory dict>"

		try:
			if "cookies" in storage and storage["cookies"]:
				normalized_cookies: list[Cookie] = []
				for cookie in storage["cookies"]:
					if isinstance(cookie, dict):
						normalized_cookies.append(Cookie(**cookie))
					else:
						normalized_cookies.append(cookie)  # type: ignore[arg-type]

				await self.browser_session._cdp_set_cookies(normalized_cookies)
				self._last_cookie_state = [dict(c) for c in storage["cookies"]]
				self.logger.debug(
					f'[StorageStateWatchdog] Added {len(storage["cookies"])} cookies from storage state'
				)

			if "origins" in storage and storage["origins"]:
				for origin in storage["origins"]:
					origin_value = origin.get("origin")
					if not origin_value:
						continue

					if origin.get("localStorage"):
						lines = []
						for item in origin["localStorage"]:
							lines.append(
								f"window.localStorage.setItem("
								f'{json.dumps(item["name"])}, {json.dumps(item["value"])});'
							)
						script = (
							"(function(){\n"
							f"  if (window.location && window.location.origin !== "
							f"{json.dumps(origin_value)}) return;\n"
							"  try {\n"
							f'    {" ".join(lines)}\n'
							"  } catch (e) {}\n"
							"})();"
						)
						await self.browser_session._cdp_add_init_script(script)

					if origin.get("sessionStorage"):
						lines = []
						for item in origin["sessionStorage"]:
							lines.append(
								f"window.sessionStorage.setItem("
								f'{json.dumps(item["name"])}, {json.dumps(item["value"])});'
							)
						script = (
							"(function(){\n"
							f"  if (window.location && window.location.origin !== "
							f"{json.dumps(origin_value)}) return;\n"
							"  try {\n"
							f'    {" ".join(lines)}\n'
							"  } catch (e) {}\n"
							"})();"
						)
						await self.browser_session._cdp_add_init_script(script)

				self.logger.debug(
					"[StorageStateWatchdog] Applied localStorage/sessionStorage from "
					f'{len(storage["origins"])} origins'
				)

			self.event_bus.dispatch(
				StorageStateLoadedEvent(
					path=source_label,
					cookies_count=len(storage.get("cookies", [])),
					origins_count=len(storage.get("origins", [])),
				)
			)

			self.logger.debug(f"[StorageStateWatchdog] Loaded storage state from: {source_label}")

		except Exception as e:
			self.logger.error(f"[StorageStateWatchdog] Failed to load storage state: {e}")

	# ------------------------------------------------------------------
	# Public helpers
	# ------------------------------------------------------------------

	async def get_current_cookies(self) -> list[dict[str, Any]]:
		"""Get current cookies using CDP."""
		if not self.browser_session.cdp_client:
			return []

		try:
			cookies = await self.browser_session._cdp_get_cookies()
			return [dict(cookie) for cookie in cookies]
		except Exception as e:
			self.logger.error(f"[StorageStateWatchdog] Failed to get cookies: {e}")
			return []

	async def add_cookies(self, cookies: list[dict[str, Any]]) -> None:
		"""Add cookies using CDP."""
		if not self.browser_session.cdp_client:
			self.logger.warning("[StorageStateWatchdog] No CDP client available for adding cookies")
			return

		try:
			cookie_objects = [
				Cookie(**cookie_dict) if isinstance(cookie_dict, dict) else cookie_dict
				for cookie_dict in cookies
			]
			await self.browser_session._cdp_set_cookies(cookie_objects)
			self.logger.debug(f"[StorageStateWatchdog] Added {len(cookies)} cookies")
		except Exception as e:
			self.logger.error(f"[StorageStateWatchdog] Failed to add cookies: {e}")
