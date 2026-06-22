from __future__ import annotations

import logging
import os
import time
from typing import Any

from browser_use.runtime.config import BrowserSessionConfig, LLMConfig, RunResult, TaskConfig

logger = logging.getLogger(__name__)


class CommandRuntimeAdapter:
	"""Unified runtime adapter that encapsulates task configuration parsing,
	Agent creation, BrowserSession creation, result serialization, and exit
	code handling.

	All entry points (legacy CLI, skill_cli daemon, MCP tool handler, examples)
	should delegate to this adapter instead of duplicating Agent init and result
	handling logic.
	"""

	def __init__(self, task_config: TaskConfig) -> None:
		self.task_config = task_config
		self._llm: Any | None = None
		self._browser_session: Any | None = None
		self._agent: Any | None = None

	def resolve_llm(self) -> Any:
		"""Resolve and cache an LLM instance from the task config.

		Supports: openai, anthropic, google, browser_use, aws_bedrock, and
		auto-detection based on available API keys.
		"""
		if self._llm is not None:
			return self._llm

		cfg = self.task_config.llm
		self._llm = _create_llm(cfg)
		return self._llm

	async def create_browser_session(self) -> Any:
		"""Create and start a BrowserSession from the task config.

		Returns the started BrowserSession. Caches for reuse.
		"""
		if self._browser_session is not None:
			return self._browser_session

		from browser_use.browser import BrowserProfile, BrowserSession

		profile_kwargs = _browser_config_to_profile_kwargs(self.task_config.browser)
		profile = BrowserProfile(**profile_kwargs)
		bs = BrowserSession(browser_profile=profile)
		await bs.start()
		self._browser_session = bs
		return bs

	def create_browser_session_sync(self) -> Any:
		"""Create a BrowserSession *without* starting it (for lazy start)."""
		if self._browser_session is not None:
			return self._browser_session

		from browser_use.browser import BrowserProfile, BrowserSession

		profile_kwargs = _browser_config_to_profile_kwargs(self.task_config.browser)
		profile = BrowserProfile(**profile_kwargs)
		bs = BrowserSession(browser_profile=profile)
		self._browser_session = bs
		return bs

	def create_agent(self, browser_session: Any | None = None) -> Any:
		"""Create an Agent from the task config, optional browser session, and
		resolved LLM.

		Does NOT run the agent — callers must call ``run_agent`` or
		``agent.run()`` themselves.
		"""
		from browser_use.agent.service import Agent
		from browser_use.agent.views import AgentSettings

		llm = self.resolve_llm()
		bs = browser_session or self._browser_session

		agent_settings = AgentSettings.model_validate(self.task_config.agent_settings)

		agent_kwargs: dict[str, Any] = {
			'task': self.task_config.task,
			'llm': llm,
			'use_vision': self.task_config.use_vision,
			**agent_settings.model_dump(),
		}

		if bs is not None:
			agent_kwargs['browser_session'] = bs
		else:
			profile_kwargs = _browser_config_to_profile_kwargs(self.task_config.browser)
			from browser_use.browser import BrowserProfile

			agent_kwargs['browser_profile'] = BrowserProfile(**profile_kwargs)

		if self.task_config.source:
			agent_kwargs['source'] = self.task_config.source

		self._agent = Agent(**agent_kwargs)
		return self._agent

	async def run_agent(self, max_steps: int | None = None) -> RunResult:
		"""Create a browser session, build an agent, run it, and return a
		``RunResult`` with serialized output and exit code.

		This is the single-call convenience method that covers the full
		lifecycle.  Callers that need finer control (e.g. the TUI) should use
		``create_browser_session`` + ``create_agent`` + ``agent.run()``
		instead.
		"""
		start = time.time()
		try:
			bs = await self.create_browser_session()
			agent = self.create_agent(browser_session=bs)
			history = await agent.run(max_steps=max_steps or self.task_config.max_steps)
			duration = time.time() - start
			result = RunResult.from_agent_history(history, duration=duration)
			return result
		except Exception as e:
			duration = time.time() - start
			logger.exception('Agent run failed: %s', e)
			return RunResult.from_error(e, duration=duration)
		finally:
			await self.cleanup()

	async def cleanup(self) -> None:
		"""Stop the browser session if we own it."""
		if self._browser_session is not None:
			try:
				bs = self._browser_session
				cfg = self.task_config.browser
				if cfg.cdp_url or cfg.use_cloud:
					await bs.stop()
				else:
					await bs.kill()
			except Exception:
				logger.debug('Browser cleanup error (ignored)', exc_info=True)
			finally:
				self._browser_session = None

	async def run_cli_task(
		self,
		max_steps: int | None = None,
		on_start: Any | None = None,
		on_complete: Any | None = None,
		on_error: Any | None = None,
	) -> RunResult:
		"""Run agent for CLI oneshot mode with optional lifecycle hooks.

		Parameters
		----------
		max_steps
			Override the task_config.max_steps if provided.
		on_start
			Optional sync/async callable called with ``(adapter, llm)``
			*after* LLM resolve but *before* agent start (for telemetry).
		on_complete
			Optional sync/async callable called with ``(adapter, result)``
			after successful completion (for telemetry).
		on_error
			Optional sync/async callable called with ``(adapter, error, duration)``
			if an exception escapes run_agent (for telemetry).

		Returns
		-------
		RunResult
			Always a valid RunResult (errors are converted, never raised).
		"""
		import asyncio as _asyncio

		start = time.time()
		try:
			llm = self.resolve_llm()
			if on_start is not None:
				try:
					if _asyncio.iscoroutinefunction(on_start):
						await on_start(self, llm)
					else:
						on_start(self, llm)
				except Exception:
					logger.debug('run_cli_task on_start hook failed (ignored)', exc_info=True)

			result = await self.run_agent(max_steps=max_steps)

			if on_complete is not None:
				try:
					if _asyncio.iscoroutinefunction(on_complete):
						await on_complete(self, result)
					else:
						on_complete(self, result)
				except Exception:
					logger.debug('run_cli_task on_complete hook failed (ignored)', exc_info=True)

			return result

		except Exception as e:
			duration = time.time() - start
			logger.exception('run_cli_task outer exception: %s', e)
			if on_error is not None:
				try:
					if _asyncio.iscoroutinefunction(on_error):
						await on_error(self, e, duration)
					else:
						on_error(self, e, duration)
				except Exception:
					logger.debug('run_cli_task on_error hook failed (ignored)', exc_info=True)
			return RunResult.from_error(e, duration=duration)
		finally:
			await self.cleanup()

	async def run_mcp_tool(
		self,
		max_steps: int | None = None,
		on_start: Any | None = None,
		on_complete: Any | None = None,
		on_error: Any | None = None,
	) -> str:
		"""Run agent as an MCP tool and return formatted text response.

		Equivalent to ``(await run_cli_task(...)).format_mcp()`` with the same
		lifecycle hooks, but always returns a string suitable for MCP
		``TextContent`` — never raises.
		"""
		result = await self.run_cli_task(
			max_steps=max_steps,
			on_start=on_start,
			on_complete=on_complete,
			on_error=on_error,
		)
		return result.format_mcp()

	async def run_skill_command(
		self,
		request_id: str = '',
		max_steps: int | None = None,
	) -> dict[str, Any]:
		"""Run agent as a skill_cli daemon command and return JSON response envelope.

		Returns a dict in the standard skill_cli ``{id, success, data, error}``
		envelope format, suitable for direct JSON serialization over the
		daemon socket.
		"""
		result = await self.run_agent(max_steps=max_steps)
		return result.to_skill_response(request_id=request_id)

	@classmethod
	def for_mcp(
		cls,
		task: str,
		*,
		profile_config: dict[str, Any] | None = None,
		llm_config_dict: dict[str, Any] | None = None,
		model_override: str | None = None,
		allowed_domains: list[str] | None = None,
		use_vision: bool = True,
		max_steps: int = 100,
	) -> CommandRuntimeAdapter:
		"""Build an adapter directly from MCP tool-call parameters.

		Parameters
		----------
		task
			The agent task description.
		profile_config
			Dict from ``get_default_profile()`` (or equivalent).
		llm_config_dict
			Dict from ``get_default_llm()`` (or equivalent).
		model_override
			Model name passed by the MCP client; overrides llm_config_dict.
		allowed_domains
			Client-supplied domain allowlist (non-empty overrides profile).
		use_vision
			Whether to enable vision for the agent.
		max_steps
			Maximum agent steps.

		Returns
		-------
		CommandRuntimeAdapter
			Ready to call ``run_mcp_tool()`` or ``run_agent()``.
		"""
		import os as _os

		llm_config_dict = llm_config_dict or {}
		model_provider = llm_config_dict.get('model_provider') or _os.getenv('MODEL_PROVIDER')

		llm = LLMConfig(
			provider='aws_bedrock' if model_provider and str(model_provider).lower() == 'bedrock' else 'auto',
			model=model_override or llm_config_dict.get('model'),
			temperature=float(llm_config_dict.get('temperature', 0.7)),
			api_key=llm_config_dict.get('api_key') or _os.getenv('OPENAI_API_KEY'),
			base_url=llm_config_dict.get('base_url'),
			aws_region=llm_config_dict.get('region') or _os.getenv('REGION', 'us-east-1'),
			aws_sso_auth=bool(llm_config_dict.get('aws_sso_auth', False)),
		)

		profile_config = profile_config or {}
		bs = BrowserSessionConfig(
			**{k: v for k, v in profile_config.items() if v is not None and k in BrowserSessionConfig.model_fields}
		)
		if allowed_domains:
			bs.allowed_domains = allowed_domains

		task_cfg = TaskConfig(
			task=task,
			llm=llm,
			browser=bs,
			max_steps=max_steps,
			use_vision=use_vision,
		)
		return cls(task_cfg)

	@classmethod
	def for_cli(
		cls,
		task: str,
		*,
		user_config: dict[str, Any],
		click_ctx: Any | None = None,
		source: str = 'cli',
		user_data_dir: str | None = None,
	) -> CommandRuntimeAdapter:
		"""Build an adapter from legacy CLI user-config + click context.

		Convenience wrapper that combines ``create_task_config_from_cli_dict``
		with the post-processing steps commonly done in run_prompt_mode
		(set task, source, user_data_dir).
		"""
		task_cfg = create_task_config_from_cli_dict(user_config)
		task_cfg.task = task
		task_cfg.source = source
		if user_data_dir and task_cfg.browser.user_data_dir is None:
			task_cfg.browser.user_data_dir = str(user_data_dir)
		return cls(task_cfg)

	@classmethod
	def for_skill(
		cls,
		task: str,
		*,
		llm_provider: str = 'auto',
		llm_model: str | None = None,
		llm_temperature: float = 0.0,
		llm_api_key: str | None = None,
		headed: bool = False,
		headless: bool | None = None,
		profile: str | None = None,
		cdp_url: str | None = None,
		use_cloud: bool = False,
		cloud_profile_id: str | None = None,
		cloud_proxy_country_code: str | None = None,
		cloud_timeout: int | None = None,
		max_steps: int = 100,
		use_vision: bool = True,
	) -> CommandRuntimeAdapter:
		"""Build an adapter from flat skill_cli-style keyword arguments.

		Designed for daemon dispatch where each command param comes as a
		separate keyword — no dict/config objects needed.
		"""
		llm = LLMConfig(
			provider=llm_provider,  # type: ignore[arg-type]
			model=llm_model,
			temperature=llm_temperature,
			api_key=llm_api_key,
		)

		bs = BrowserSessionConfig(
			headless=headless,
			headed=headed,
			profile_directory=profile if not cdp_url and not use_cloud else None,
			cdp_url=cdp_url,
			use_cloud=use_cloud,
			cloud_profile_id=cloud_profile_id,
			cloud_proxy_country_code=cloud_proxy_country_code,
			cloud_timeout=cloud_timeout,
		)

		# skill_cli real-Chrome profile path resolution
		if profile and not cdp_url and not use_cloud:
			try:
				from browser_use.skill_cli.utils import find_chrome_executable, get_chrome_profile_path

				chrome_path = find_chrome_executable()
				if chrome_path:
					bs.executable_path = chrome_path
				u_dir = get_chrome_profile_path(None)
				if u_dir:
					bs.user_data_dir = u_dir
			except Exception:
				logger.debug('for_skill: Chrome profile path resolution skipped', exc_info=True)

		task_cfg = TaskConfig(
			task=task,
			llm=llm,
			browser=bs,
			max_steps=max_steps,
			use_vision=use_vision,
		)
		return cls(task_cfg)


def _create_llm(cfg: LLMConfig) -> Any:
	"""Instantiate an LLM object from an LLMConfig."""
	from browser_use.config import CONFIG

	if cfg.provider == 'openai' or (cfg.provider == 'auto' and (cfg.api_key or CONFIG.OPENAI_API_KEY)):
		from browser_use.llm.openai.chat import ChatOpenAI

		api_key = cfg.api_key or CONFIG.OPENAI_API_KEY
		if not api_key:
			raise ValueError('OPENAI_API_KEY not set')
		model = cfg.model or 'gpt-5-mini'
		kwargs: dict[str, Any] = {'model': model, 'temperature': cfg.temperature, 'api_key': api_key}
		if cfg.base_url:
			kwargs['base_url'] = cfg.base_url
		return ChatOpenAI(**kwargs)

	elif cfg.provider == 'anthropic' or (cfg.provider == 'auto' and CONFIG.ANTHROPIC_API_KEY):
		from browser_use.llm.anthropic.chat import ChatAnthropic

		if not CONFIG.ANTHROPIC_API_KEY:
			raise ValueError('ANTHROPIC_API_KEY not set')
		model = cfg.model or 'claude-4-sonnet'
		return ChatAnthropic(model=model, temperature=cfg.temperature)

	elif cfg.provider == 'google' or (cfg.provider == 'auto' and CONFIG.GOOGLE_API_KEY):
		from browser_use.llm.google.chat import ChatGoogle

		if not CONFIG.GOOGLE_API_KEY:
			raise ValueError('GOOGLE_API_KEY not set')
		model = cfg.model or 'gemini-2.5-pro'
		return ChatGoogle(model=model, temperature=cfg.temperature)

	elif cfg.provider == 'browser_use':
		from browser_use.llm.browser_use.chat import ChatBrowserUse

		api_key = cfg.api_key or os.getenv('BROWSER_USE_API_KEY')
		if not api_key:
			raise ValueError('BROWSER_USE_API_KEY not set')
		model = cfg.model or 'bu-2-0'
		return ChatBrowserUse(model=model, temperature=cfg.temperature, api_key=api_key)

	elif cfg.provider == 'aws_bedrock':
		from browser_use.llm.aws import ChatAWSBedrock

		region = cfg.aws_region or os.getenv('REGION', 'us-east-1')
		model = cfg.model or 'us.anthropic.claude-sonnet-4-20250514-v1:0'
		return ChatAWSBedrock(model=model, aws_region=region, aws_sso_auth=cfg.aws_sso_auth)

	raise ValueError(
		'No API keys found for any LLM provider. Set one of: '
		'OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, BROWSER_USE_API_KEY'
	)


def _browser_config_to_profile_kwargs(cfg: BrowserSessionConfig) -> dict[str, Any]:
	"""Convert a BrowserSessionConfig into keyword arguments suitable for
	BrowserProfile(**kwargs).

	Only non-None values are included so that BrowserProfile defaults are
	respected.
	"""
	kwargs: dict[str, Any] = {}

	if cfg.headless is not None:
		kwargs['headless'] = cfg.headless
	elif cfg.headed:
		kwargs['headless'] = False

	if cfg.keep_alive is not None:
		kwargs['keep_alive'] = cfg.keep_alive
	if cfg.user_data_dir is not None:
		kwargs['user_data_dir'] = cfg.user_data_dir
	if cfg.profile_directory is not None:
		kwargs['profile_directory'] = cfg.profile_directory
	if cfg.executable_path is not None:
		kwargs['executable_path'] = cfg.executable_path
	if cfg.cdp_url is not None:
		kwargs['cdp_url'] = cfg.cdp_url
	if cfg.window_width is not None or cfg.window_height is not None:
		w = cfg.window_width or 1280
		h = cfg.window_height or 720
		kwargs['viewport'] = {'width': w, 'height': h}
	if cfg.allowed_domains is not None:
		kwargs['allowed_domains'] = cfg.allowed_domains
	if cfg.prohibited_domains is not None:
		kwargs['prohibited_domains'] = cfg.prohibited_domains
	if cfg.wait_between_actions is not None:
		kwargs['wait_between_actions'] = cfg.wait_between_actions
	if cfg.device_scale_factor is not None:
		kwargs['device_scale_factor'] = cfg.device_scale_factor
	if cfg.disable_security:
		kwargs['disable_security'] = True
	if cfg.proxy_server is not None:
		from browser_use.browser import ProxySettings

		proxy_kwargs: dict[str, Any] = {'server': cfg.proxy_server}
		if cfg.proxy_username:
			proxy_kwargs['username'] = cfg.proxy_username
		if cfg.proxy_password:
			proxy_kwargs['password'] = cfg.proxy_password
		if cfg.proxy_bypass:
			proxy_kwargs['bypass'] = cfg.proxy_bypass
		kwargs['proxy'] = ProxySettings(**proxy_kwargs)
	if cfg.use_cloud:
		kwargs['use_cloud'] = True
	if cfg.cloud_profile_id is not None:
		kwargs['cloud_profile_id'] = cfg.cloud_profile_id
	if cfg.cloud_proxy_country_code is not None:
		kwargs['cloud_proxy_country_code'] = cfg.cloud_proxy_country_code
	if cfg.cloud_timeout is not None:
		kwargs['cloud_timeout'] = cfg.cloud_timeout
	if cfg.downloads_path is not None:
		kwargs['downloads_path'] = cfg.downloads_path
	if cfg.ignore_https_errors:
		kwargs['ignore_https_errors'] = True
	if cfg.is_mobile is not None:
		kwargs['is_mobile'] = cfg.is_mobile

	return kwargs


def create_task_config_from_cli_dict(config: dict[str, Any]) -> TaskConfig:
	"""Convert the legacy CLI dict-based config into a TaskConfig.

	This bridges the existing ``load_user_config()`` / ``update_config_with_click_args()``
	pipeline to the new TaskConfig without changing the CLI argument parsing layer.
	"""
	model_cfg = config.get('model', {})
	browser_cfg = config.get('browser', {})
	agent_cfg = config.get('agent', {})

	llm_config = LLMConfig(
		provider='auto',
		model=model_cfg.get('name'),
		temperature=model_cfg.get('temperature', 0.0),
		api_key=(model_cfg.get('api_keys') or {}).get('OPENAI_API_KEY'),
	)

	bs_config = BrowserSessionConfig(
		headless=browser_cfg.get('headless'),
		keep_alive=browser_cfg.get('keep_alive', True),
		user_data_dir=browser_cfg.get('user_data_dir'),
		profile_directory=browser_cfg.get('profile_directory'),
		executable_path=browser_cfg.get('executable_path'),
		cdp_url=browser_cfg.get('cdp_url'),
		window_width=browser_cfg.get('window_width'),
		window_height=browser_cfg.get('window_height'),
		allowed_domains=browser_cfg.get('allowed_domains'),
		wait_between_actions=browser_cfg.get('wait_between_actions'),
		device_scale_factor=browser_cfg.get('device_scale_factor'),
		disable_security=browser_cfg.get('disable_security', False),
		ignore_https_errors=browser_cfg.get('ignore_https_errors', False),
		is_mobile=browser_cfg.get('is_mobile'),
	)

	proxy = browser_cfg.get('proxy')
	if proxy and isinstance(proxy, dict):
		bs_config.proxy_server = proxy.get('server')
		bs_config.proxy_username = proxy.get('username')
		bs_config.proxy_password = proxy.get('password')
		bs_config.proxy_bypass = proxy.get('bypass')

	max_steps = agent_cfg.get('max_steps')
	remain_agent_settings = {k: v for k, v in agent_cfg.items() if k != 'max_steps'}
	return TaskConfig(
		task='',
		llm=llm_config,
		browser=bs_config,
		agent_settings=remain_agent_settings,
		**({'max_steps': max_steps} if max_steps is not None else {}),
	)
