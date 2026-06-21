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

	return TaskConfig(
		task='',
		llm=llm_config,
		browser=bs_config,
		agent_settings=agent_cfg,
	)
