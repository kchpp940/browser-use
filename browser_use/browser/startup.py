from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from browser_use.browser.cloud.cloud import CloudBrowserAuthError, CloudBrowserError
from browser_use.browser.cloud.views import CreateBrowserRequest
from browser_use.browser.events import (
	BrowserConnectedEvent,
	BrowserLaunchEvent,
	BrowserLaunchResult,
)
from browser_use.browser.profile import BrowserProfile
from browser_use.observability import observe_debug

if TYPE_CHECKING:
	from browser_use.browser.session import BrowserSession


class BrowserConnectionMode(str, Enum):
	LOCAL = 'local'
	CDP = 'cdp'
	REMOTE = 'remote'
	CLOUD = 'cloud'


class BrowserConnectionContext(BaseModel):
	connection_mode: BrowserConnectionMode
	is_local: bool
	use_cloud: bool
	has_cdp_url: bool
	cdp_url: str | None = None
	cloud_params: CreateBrowserRequest | None = None
	headers: dict[str, str] | None = None
	needs_download_dir_init: bool = False
	needs_process_teardown: bool = False
	needs_cloud_cleanup: bool = False

	model_config = {'arbitrary_types_allowed': True}


class BrowserConnectionStrategy(ABC):
	@abstractmethod
	async def pre_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None: ...

	@abstractmethod
	async def acquire_cdp_url(self, session: BrowserSession, ctx: BrowserConnectionContext) -> str: ...

	@abstractmethod
	async def post_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None: ...

	@abstractmethod
	async def cleanup_on_failure(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None: ...

	@abstractmethod
	async def cleanup_on_stop(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None: ...


class LocalBrowserStrategy(BrowserConnectionStrategy):
	async def pre_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		ctx.needs_download_dir_init = True
		ctx.needs_process_teardown = True

	async def acquire_cdp_url(self, session: BrowserSession, ctx: BrowserConnectionContext) -> str:
		launch_event = session.event_bus.dispatch(BrowserLaunchEvent())
		await launch_event
		launch_result: BrowserLaunchResult = await launch_event.event_result(
			raise_if_none=True, raise_if_any=True
		)
		session.browser_profile.cdp_url = launch_result.cdp_url
		return launch_result.cdp_url

	async def post_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		pass

	async def cleanup_on_failure(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		from browser_use.browser.events import BrowserKillEvent

		session.logger.warning('[LocalBrowserStrategy] Cleaning up local browser process after failure')
		session.event_bus.dispatch(BrowserKillEvent())

	async def cleanup_on_stop(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		session.browser_profile.cdp_url = None


class CDPConnectionStrategy(BrowserConnectionStrategy):
	async def pre_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		ctx.needs_download_dir_init = True

	async def acquire_cdp_url(self, session: BrowserSession, ctx: BrowserConnectionContext) -> str:
		assert ctx.cdp_url is not None
		return ctx.cdp_url

	async def post_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		pass

	async def cleanup_on_failure(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		pass

	async def cleanup_on_stop(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		session.browser_profile.cdp_url = None


class RemoteBrowserStrategy(BrowserConnectionStrategy):
	async def pre_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		ctx.needs_download_dir_init = True
		ctx.headers = dict(getattr(session.browser_profile, 'headers', None) or {})
		from browser_use.utils import get_browser_use_version

		ctx.headers['User-Agent'] = f'browser-use/{get_browser_use_version()}'

	async def acquire_cdp_url(self, session: BrowserSession, ctx: BrowserConnectionContext) -> str:
		assert ctx.cdp_url is not None
		return ctx.cdp_url

	async def post_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		session.logger.debug('🌐 Connected to remote browser - downloads will be handled via CDP events only')

	async def cleanup_on_failure(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		pass

	async def cleanup_on_stop(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		pass


class CloudBrowserStrategy(BrowserConnectionStrategy):
	async def pre_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		ctx.needs_cloud_cleanup = True
		ctx.needs_download_dir_init = True
		assert ctx.cloud_params is not None

	async def acquire_cdp_url(self, session: BrowserSession, ctx: BrowserConnectionContext) -> str:
		assert ctx.cloud_params is not None
		try:
			cloud_browser_response = await session._cloud_browser_client.create_browser(ctx.cloud_params)
			session.browser_profile.cdp_url = cloud_browser_response.cdpUrl
			session.browser_profile.is_local = False
			session.logger.info('🌤️ Successfully connected to cloud browser service')
			return cloud_browser_response.cdpUrl
		except CloudBrowserAuthError:
			raise
		except CloudBrowserError as e:
			raise CloudBrowserError(f'Failed to create cloud browser: {e}')

	async def post_connect(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		pass

	async def cleanup_on_failure(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		session.logger.warning('[CloudBrowserStrategy] Cleaning up cloud browser session after failure')
		await self._cleanup_cloud_session(session)

	async def cleanup_on_stop(self, session: BrowserSession, ctx: BrowserConnectionContext) -> None:
		await self._cleanup_cloud_session(session)

	async def _cleanup_cloud_session(self, session: BrowserSession) -> None:
		try:
			if session._cloud_browser_client and session.browser_profile.cdp_url:
				await session._cloud_browser_client.stop_browser(session.browser_profile.cdp_url)
				session.logger.info('🌤️ Cloud browser session stopped')
		except Exception as e:
			session.logger.debug(f'Error stopping cloud browser session: {e}')


_MODE_STRATEGIES: dict[BrowserConnectionMode, type[BrowserConnectionStrategy]] = {
	BrowserConnectionMode.LOCAL: LocalBrowserStrategy,
	BrowserConnectionMode.CDP: CDPConnectionStrategy,
	BrowserConnectionMode.REMOTE: RemoteBrowserStrategy,
	BrowserConnectionMode.CLOUD: CloudBrowserStrategy,
}


class BrowserSessionStartupPlan:
	def __init__(self, session: BrowserSession) -> None:
		self.session = session
		self._connection_context: BrowserConnectionContext | None = None
		self._strategy: BrowserConnectionStrategy | None = None

	@observe_debug(ignore_input=True, ignore_output=True, name='startup_plan_execute')
	async def execute(self) -> dict[str, str]:
		await self._phase_attach_watchdogs()

		self._connection_context = self._phase_resolve_connection()
		self._strategy = _MODE_STRATEGIES[self._connection_context.connection_mode]()

		try:
			await self._phase_pre_connect()
			cdp_url = await self._phase_acquire_cdp_url()
			await self._phase_connect_cdp(cdp_url)
			await self._phase_post_connect()
		except Exception:
			await self._phase_cleanup_failure()
			raise

		return {'cdp_url': self.session.cdp_url}

	def _phase_resolve_connection(self) -> BrowserConnectionContext:
		profile = self.session.browser_profile

		use_cloud = profile.use_cloud or profile.cloud_browser_params is not None
		has_cdp_url = bool(profile.cdp_url)
		is_local = profile.is_local

		if use_cloud:
			mode = BrowserConnectionMode.CLOUD
			cloud_params = profile.cloud_browser_params or CreateBrowserRequest()
		elif has_cdp_url and is_local:
			mode = BrowserConnectionMode.CDP
			cloud_params = None
		elif has_cdp_url and not is_local:
			mode = BrowserConnectionMode.REMOTE
			cloud_params = None
		elif is_local:
			mode = BrowserConnectionMode.LOCAL
			cloud_params = None
		else:
			raise ValueError('Got BrowserSession(is_local=False) but no cdp_url was provided to connect to!')

		return BrowserConnectionContext(
			connection_mode=mode,
			is_local=is_local,
			use_cloud=use_cloud,
			has_cdp_url=has_cdp_url,
			cdp_url=profile.cdp_url,
			cloud_params=cloud_params,
		)

	async def _phase_pre_connect(self) -> None:
		assert self._strategy is not None
		assert self._connection_context is not None
		await self._strategy.pre_connect(self.session, self._connection_context)

	async def _phase_acquire_cdp_url(self) -> str:
		assert self._strategy is not None
		assert self._connection_context is not None

		if self._connection_context.has_cdp_url:
			assert self._connection_context.cdp_url is not None
			assert '://' in self._connection_context.cdp_url
			return self._connection_context.cdp_url

		return await self._strategy.acquire_cdp_url(self.session, self._connection_context)

	async def _phase_connect_cdp(self, cdp_url: str) -> None:
		assert self._connection_context is not None
		async with self.session._connection_lock:
			if self.session._cdp_client_root is not None:
				self.session.logger.debug('Already connected to CDP, skipping reconnection')
				return

			if self._connection_context.headers is not None:
				original_headers = getattr(self.session.browser_profile, 'headers', None)
				self.session.browser_profile.headers = self._connection_context.headers
			else:
				original_headers = None

			try:
				await asyncio.wait_for(self.session.connect(cdp_url=cdp_url), timeout=15.0)
			except TimeoutError:
				from typing import cast

				from cdp_use import CDPClient

				cdp_client = cast(CDPClient | None, self.session._cdp_client_root)
				if cdp_client is not None:
					try:
						await cdp_client.stop()
					except Exception:
						pass
					self.session._cdp_client_root = None
				manager = self.session.session_manager
				if manager is not None:
					try:
						await manager.clear()
					except Exception:
						pass
					self.session.session_manager = None
				self.session.agent_focus_target_id = None
				raise RuntimeError(
					f'connect() timed out after 15s — CDP connection to {cdp_url} is too slow or unresponsive'
				)
			finally:
				if original_headers is not None:
					self.session.browser_profile.headers = original_headers

			await self.session.event_bus.dispatch(BrowserConnectedEvent(cdp_url=self.session.cdp_url))

	async def _phase_post_connect(self) -> None:
		assert self._strategy is not None
		assert self._connection_context is not None

		await self._strategy.post_connect(self.session, self._connection_context)

		if self.session.browser_profile.demo_mode:
			try:
				demo = self.session.demo_mode
				if demo:
					await demo.ensure_ready()
			except Exception as exc:
				self.session.logger.warning(f'[DemoMode] Failed to inject demo overlay: {exc}')

	async def _phase_cleanup_failure(self) -> None:
		assert self._strategy is not None
		assert self._connection_context is not None
		await self._strategy.cleanup_on_failure(self.session, self._connection_context)

	async def cleanup_on_stop(self) -> None:
		assert self._strategy is not None
		assert self._connection_context is not None
		await self._strategy.cleanup_on_stop(self.session, self._connection_context)

	async def _phase_attach_watchdogs(self) -> None:
		if self.session._watchdogs_attached:
			self.session.logger.debug('Watchdogs already attached, skipping duplicate attachment')
			return

		from browser_use.browser.watchdogs.aboutblank_watchdog import AboutBlankWatchdog
		from browser_use.browser.watchdogs.captcha_watchdog import CaptchaWatchdog
		from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog
		from browser_use.browser.watchdogs.dom_watchdog import DOMWatchdog
		from browser_use.browser.watchdogs.downloads_watchdog import DownloadsWatchdog
		from browser_use.browser.watchdogs.har_recording_watchdog import HarRecordingWatchdog
		from browser_use.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
		from browser_use.browser.watchdogs.permissions_watchdog import PermissionsWatchdog
		from browser_use.browser.watchdogs.popups_watchdog import PopupsWatchdog
		from browser_use.browser.watchdogs.recording_watchdog import RecordingWatchdog
		from browser_use.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog
		from browser_use.browser.watchdogs.security_watchdog import SecurityWatchdog
		from browser_use.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog

		DownloadsWatchdog.model_rebuild()
		self.session._downloads_watchdog = DownloadsWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._downloads_watchdog.attach_to_session()
		if self.session.browser_profile.auto_download_pdfs:
			self.session.logger.debug('📄 PDF auto-download enabled for this session')

		should_enable_storage_state = (
			self.session.browser_profile.storage_state is not None
			or self.session.browser_profile.user_data_dir is not None
		)
		if should_enable_storage_state:
			StorageStateWatchdog.model_rebuild()
			self.session._storage_state_watchdog = StorageStateWatchdog(
				event_bus=self.session.event_bus,
				browser_session=self.session,
				auto_save_interval=60.0,
				save_on_change=False,
			)
			self.session._storage_state_watchdog.attach_to_session()
			self.session.logger.debug(
				f'🍪 StorageStateWatchdog enabled (storage_state: {bool(self.session.browser_profile.storage_state)}, user_data_dir: {bool(self.session.browser_profile.user_data_dir)})'
			)
		else:
			self.session.logger.debug(
				'🍪 StorageStateWatchdog disabled (no storage_state or user_data_dir configured)'
			)

		LocalBrowserWatchdog.model_rebuild()
		self.session._local_browser_watchdog = LocalBrowserWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._local_browser_watchdog.attach_to_session()

		SecurityWatchdog.model_rebuild()
		self.session._security_watchdog = SecurityWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._security_watchdog.attach_to_session()

		AboutBlankWatchdog.model_rebuild()
		self.session._aboutblank_watchdog = AboutBlankWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._aboutblank_watchdog.attach_to_session()

		PopupsWatchdog.model_rebuild()
		self.session._popups_watchdog = PopupsWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._popups_watchdog.attach_to_session()

		PermissionsWatchdog.model_rebuild()
		self.session._permissions_watchdog = PermissionsWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._permissions_watchdog.attach_to_session()

		DefaultActionWatchdog.model_rebuild()
		self.session._default_action_watchdog = DefaultActionWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._default_action_watchdog.attach_to_session()

		ScreenshotWatchdog.model_rebuild()
		self.session._screenshot_watchdog = ScreenshotWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._screenshot_watchdog.attach_to_session()

		DOMWatchdog.model_rebuild()
		self.session._dom_watchdog = DOMWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._dom_watchdog.attach_to_session()

		RecordingWatchdog.model_rebuild()
		self.session._recording_watchdog = RecordingWatchdog(
			event_bus=self.session.event_bus, browser_session=self.session
		)
		self.session._recording_watchdog.attach_to_session()

		if self.session.browser_profile.record_har_path:
			HarRecordingWatchdog.model_rebuild()
			self.session._har_recording_watchdog = HarRecordingWatchdog(
				event_bus=self.session.event_bus, browser_session=self.session
			)
			self.session._har_recording_watchdog.attach_to_session()

		if self.session.browser_profile.captcha_solver:
			CaptchaWatchdog.model_rebuild()
			self.session._captcha_watchdog = CaptchaWatchdog(
				event_bus=self.session.event_bus, browser_session=self.session
			)
			self.session._captcha_watchdog.attach_to_session()

		self.session._watchdogs_attached = True

	@staticmethod
	def detach_watchdogs(session: BrowserSession) -> None:
		session._crash_watchdog = None
		session._downloads_watchdog = None
		session._aboutblank_watchdog = None
		session._security_watchdog = None
		session._storage_state_watchdog = None
		session._local_browser_watchdog = None
		session._default_action_watchdog = None
		session._dom_watchdog = None
		session._screenshot_watchdog = None
		session._permissions_watchdog = None
		session._recording_watchdog = None
		session._captcha_watchdog = None
		session._har_recording_watchdog = None
		session._popups_watchdog = None
		session._watchdogs_attached = False
		if session._demo_mode:
			session._demo_mode.reset()
			session._demo_mode = None

	@staticmethod
	def resolve_browser_profile(
		browser_profile: BrowserProfile | None,
		is_local: bool,
		cdp_url: str | None,
		cloud_profile_id,
		cloud_proxy_country_code,
		cloud_timeout,
		profile_id,
		proxy_country_code,
		timeout,
		cloud_browser_params: Any | None,
		profile_kwargs: dict[str, Any],
		_unset_sentinel: Any,
	) -> BrowserProfile:
		final_profile_id = cloud_profile_id if cloud_profile_id is not None else profile_id
		final_proxy_country_code = (
			cloud_proxy_country_code
			if cloud_proxy_country_code is not _unset_sentinel
			else proxy_country_code
			if proxy_country_code is not _unset_sentinel
			else _unset_sentinel
		)
		final_timeout = cloud_timeout if cloud_timeout is not None else timeout

		if (
			final_profile_id is not None
			or final_proxy_country_code is not _unset_sentinel
			or final_timeout is not None
		):
			cloud_kwargs: dict[str, Any] = {}
			if final_profile_id is not None:
				cloud_kwargs['cloud_profile_id'] = final_profile_id
			if final_proxy_country_code is not _unset_sentinel:
				cloud_kwargs['cloud_proxy_country_code'] = final_proxy_country_code
			if final_timeout is not None:
				cloud_kwargs['cloud_timeout'] = final_timeout
			cloud_params = CreateBrowserRequest(**cloud_kwargs)
			profile_kwargs['cloud_browser_params'] = cloud_params
			profile_kwargs['use_cloud'] = True

		if 'cloud_browser' in profile_kwargs:
			profile_kwargs['use_cloud'] = profile_kwargs.pop('cloud_browser')

		if cloud_browser_params is not None:
			profile_kwargs['use_cloud'] = True

		if is_local is False and profile_kwargs.get('executable_path') is not None:
			profile_kwargs['is_local'] = True
		use_cloud = profile_kwargs.get('use_cloud') or profile_kwargs.get('cloud_browser')
		if not cdp_url and not use_cloud:
			profile_kwargs['is_local'] = True

		if browser_profile is not None:
			merged_kwargs = {**browser_profile.model_dump(exclude_unset=True), **profile_kwargs}
			return BrowserProfile(**merged_kwargs)
		else:
			return BrowserProfile(**profile_kwargs)
