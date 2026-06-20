"""Configuration system for browser-use with automatic migration support."""

import json
import logging
import os
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from browser_use.runtime_config import ConfigResolver, RuntimeConfig

logger = logging.getLogger(__name__)


@cache
def is_running_in_docker() -> bool:
	"""Detect if we are running in a docker container, for the purpose of optimizing chrome launch flags (dev shm usage, gpu settings, etc.)"""
	try:
		if Path('/.dockerenv').exists() or 'docker' in Path('/proc/1/cgroup').read_text().lower():
			return True
	except Exception:
		pass

	try:
		# if init proc (PID 1) looks like uvicorn/python/uv/etc. then we're in Docker
		# if init proc (PID 1) looks like bash/systemd/init/etc. then we're probably NOT in Docker
		init_cmd = ' '.join(psutil.Process(1).cmdline())
		if ('py' in init_cmd) or ('uv' in init_cmd) or ('app' in init_cmd):
			return True
	except Exception:
		pass

	try:
		# if less than 10 total running procs, then we're almost certainly in a container
		if len(psutil.pids()) < 10:
			return True
	except Exception:
		pass

	return False


class OldConfig:
	"""Original lazy-loading configuration class for environment variables."""

	_dirs_created = False

	def __init__(self):
		self._runtime_config: RuntimeConfig = ConfigResolver(include_env=True, include_file=False).resolve()

	@property
	def BROWSER_USE_LOGGING_LEVEL(self) -> str:
		return self._runtime_config.logging.level

	@property
	def ANONYMIZED_TELEMETRY(self) -> bool:
		return self._runtime_config.telemetry.anonymized_telemetry

	@property
	def BROWSER_USE_CLOUD_SYNC(self) -> bool:
		return self._runtime_config.telemetry.cloud_sync_enabled

	@property
	def BROWSER_USE_CLOUD_API_URL(self) -> str:
		url = self._runtime_config.cloud.api_url
		assert '://' in url, 'BROWSER_USE_CLOUD_API_URL must be a valid URL'
		return url

	@property
	def BROWSER_USE_CLOUD_UI_URL(self) -> str:
		url = self._runtime_config.cloud.ui_url
		if url and '://' not in url:
			raise AssertionError('BROWSER_USE_CLOUD_UI_URL must be a valid URL if set')
		return url

	@property
	def BROWSER_USE_MODEL_PRICING_URL(self) -> str:
		url = self._runtime_config.cloud.model_pricing_url
		if url and '://' not in url:
			raise AssertionError('BROWSER_USE_MODEL_PRICING_URL must be a valid URL if set')
		return url

	@property
	def XDG_CACHE_HOME(self) -> Path:
		return Path(os.getenv('XDG_CACHE_HOME', '~/.cache')).expanduser().resolve()

	@property
	def XDG_CONFIG_HOME(self) -> Path:
		return Path(os.getenv('XDG_CONFIG_HOME', '~/.config')).expanduser().resolve()

	@property
	def BROWSER_USE_CONFIG_DIR(self) -> Path:
		path = Path(os.getenv('BROWSER_USE_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'browseruse'))).expanduser().resolve()
		self._ensure_dirs()
		return path

	@property
	def BROWSER_USE_CONFIG_FILE(self) -> Path:
		return self.BROWSER_USE_CONFIG_DIR / 'config.json'

	@property
	def BROWSER_USE_PROFILES_DIR(self) -> Path:
		path = self.BROWSER_USE_CONFIG_DIR / 'profiles'
		self._ensure_dirs()
		return path

	@property
	def BROWSER_USE_DEFAULT_USER_DATA_DIR(self) -> Path:
		return self.BROWSER_USE_PROFILES_DIR / 'default'

	@property
	def BROWSER_USE_EXTENSIONS_DIR(self) -> Path:
		path = self.BROWSER_USE_CONFIG_DIR / 'extensions'
		self._ensure_dirs()
		return path

	def _ensure_dirs(self) -> None:
		if not self._dirs_created:
			config_dir = (
				Path(os.getenv('BROWSER_USE_CONFIG_DIR', str(self.XDG_CONFIG_HOME / 'browseruse'))).expanduser().resolve()
			)
			config_dir.mkdir(parents=True, exist_ok=True)
			(config_dir / 'profiles').mkdir(parents=True, exist_ok=True)
			(config_dir / 'extensions').mkdir(parents=True, exist_ok=True)
			self._dirs_created = True

	@property
	def OPENAI_API_KEY(self) -> str:
		return self._runtime_config.llm.openai_api_key or ''

	@property
	def ANTHROPIC_API_KEY(self) -> str:
		return self._runtime_config.llm.anthropic_api_key or ''

	@property
	def GOOGLE_API_KEY(self) -> str:
		return self._runtime_config.llm.google_api_key or ''

	@property
	def DEEPSEEK_API_KEY(self) -> str:
		return self._runtime_config.llm.deepseek_api_key or ''

	@property
	def GROK_API_KEY(self) -> str:
		return self._runtime_config.llm.grok_api_key or ''

	@property
	def NOVITA_API_KEY(self) -> str:
		return self._runtime_config.llm.novita_api_key or ''

	@property
	def AZURE_OPENAI_ENDPOINT(self) -> str:
		return self._runtime_config.llm.azure_endpoint or ''

	@property
	def AZURE_OPENAI_KEY(self) -> str:
		return self._runtime_config.llm.azure_api_key or ''

	@property
	def SKIP_LLM_API_KEY_VERIFICATION(self) -> bool:
		return self._runtime_config.llm.skip_api_key_verification

	@property
	def DEFAULT_LLM(self) -> str:
		return self._runtime_config.llm.default_llm

	@property
	def IN_DOCKER(self) -> bool:
		return os.getenv('IN_DOCKER', 'false').lower()[:1] in 'ty1' or is_running_in_docker()

	@property
	def IS_IN_EVALS(self) -> bool:
		return os.getenv('IS_IN_EVALS', 'false').lower()[:1] in 'ty1'

	@property
	def BROWSER_USE_VERSION_CHECK(self) -> bool:
		return self._runtime_config.telemetry.version_check

	@property
	def WIN_FONT_DIR(self) -> str:
		return os.getenv('WIN_FONT_DIR', 'C:\\Windows\\Fonts')


class FlatEnvConfig(BaseSettings):
	"""All environment variables in a flat namespace."""

	model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=True, extra='allow')

	# Logging and telemetry
	BROWSER_USE_LOGGING_LEVEL: str = Field(default='info')
	CDP_LOGGING_LEVEL: str = Field(default='WARNING')
	BROWSER_USE_DEBUG_LOG_FILE: str | None = Field(default=None)
	BROWSER_USE_INFO_LOG_FILE: str | None = Field(default=None)
	ANONYMIZED_TELEMETRY: bool = Field(default=True)
	BROWSER_USE_CLOUD_SYNC: bool | None = Field(default=None)
	BROWSER_USE_CLOUD_API_URL: str = Field(default='https://api.browser-use.com')
	BROWSER_USE_CLOUD_UI_URL: str = Field(default='')
	BROWSER_USE_MODEL_PRICING_URL: str = Field(default='')

	# Path configuration
	XDG_CACHE_HOME: str = Field(default='~/.cache')
	XDG_CONFIG_HOME: str = Field(default='~/.config')
	BROWSER_USE_CONFIG_DIR: str | None = Field(default=None)

	# LLM API keys
	OPENAI_API_KEY: str = Field(default='')
	ANTHROPIC_API_KEY: str = Field(default='')
	GOOGLE_API_KEY: str = Field(default='')
	DEEPSEEK_API_KEY: str = Field(default='')
	GROK_API_KEY: str = Field(default='')
	NOVITA_API_KEY: str = Field(default='')
	AZURE_OPENAI_ENDPOINT: str = Field(default='')
	AZURE_OPENAI_KEY: str = Field(default='')
	SKIP_LLM_API_KEY_VERIFICATION: bool = Field(default=False)
	DEFAULT_LLM: str = Field(default='')

	# Runtime hints
	IN_DOCKER: bool | None = Field(default=None)
	IS_IN_EVALS: bool = Field(default=False)
	WIN_FONT_DIR: str = Field(default='C:\\Windows\\Fonts')
	BROWSER_USE_VERSION_CHECK: bool = Field(default=True)

	# MCP-specific env vars
	BROWSER_USE_CONFIG_PATH: str | None = Field(default=None)
	BROWSER_USE_HEADLESS: bool | None = Field(default=None)
	BROWSER_USE_ALLOWED_DOMAINS: str | None = Field(default=None)
	BROWSER_USE_LLM_MODEL: str | None = Field(default=None)

	# Proxy env vars
	BROWSER_USE_PROXY_URL: str | None = Field(default=None)
	BROWSER_USE_NO_PROXY: str | None = Field(default=None)
	BROWSER_USE_PROXY_USERNAME: str | None = Field(default=None)
	BROWSER_USE_PROXY_PASSWORD: str | None = Field(default=None)

	# Extension env vars
	BROWSER_USE_DISABLE_EXTENSIONS: bool | None = Field(default=None)


class DBStyleEntry(BaseModel):
	"""Database-style entry with UUID and metadata."""

	id: str = Field(default_factory=lambda: str(uuid4()))
	default: bool = Field(default=False)
	created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class BrowserProfileEntry(DBStyleEntry):
	"""Browser profile configuration entry - accepts any BrowserProfile fields."""

	model_config = ConfigDict(extra='allow')

	# Common browser profile fields for reference
	headless: bool | None = None
	user_data_dir: str | None = None
	allowed_domains: list[str] | None = None
	downloads_path: str | None = None


class LLMEntry(DBStyleEntry):
	"""LLM configuration entry."""

	api_key: str | None = None
	model: str | None = None
	temperature: float | None = None
	max_tokens: int | None = None


class AgentEntry(DBStyleEntry):
	"""Agent configuration entry."""

	max_steps: int | None = None
	use_vision: bool | None = None
	system_prompt: str | None = None


class DBStyleConfigJSON(BaseModel):
	"""New database-style configuration format."""

	browser_profile: dict[str, BrowserProfileEntry] = Field(default_factory=dict)
	llm: dict[str, LLMEntry] = Field(default_factory=dict)
	agent: dict[str, AgentEntry] = Field(default_factory=dict)


def create_default_config() -> DBStyleConfigJSON:
	"""Create a fresh default configuration."""
	logger.debug('Creating fresh default config.json')

	new_config = DBStyleConfigJSON()

	# Generate default IDs
	profile_id = str(uuid4())
	llm_id = str(uuid4())
	agent_id = str(uuid4())

	# Create default browser profile entry
	new_config.browser_profile[profile_id] = BrowserProfileEntry(id=profile_id, default=True, headless=False, user_data_dir=None)

	# Create default LLM entry
	new_config.llm[llm_id] = LLMEntry(id=llm_id, default=True, model='gpt-4.1-mini', api_key='your-openai-api-key-here')

	# Create default agent entry
	new_config.agent[agent_id] = AgentEntry(id=agent_id, default=True)

	return new_config


def load_and_migrate_config(config_path: Path) -> DBStyleConfigJSON:
	"""Load config.json or create fresh one if old format detected."""
	if not config_path.exists():
		# Create fresh config with defaults
		config_path.parent.mkdir(parents=True, exist_ok=True)
		new_config = create_default_config()
		with open(config_path, 'w') as f:
			json.dump(new_config.model_dump(), f, indent=2)
		return new_config

	try:
		with open(config_path) as f:
			data = json.load(f)

		# Check if it's already in DB-style format
		if all(key in data for key in ['browser_profile', 'llm', 'agent']) and all(
			isinstance(data.get(key, {}), dict) for key in ['browser_profile', 'llm', 'agent']
		):
			# Check if the values are DB-style entries (have UUIDs as keys)
			if data.get('browser_profile') and all(isinstance(v, dict) and 'id' in v for v in data['browser_profile'].values()):
				# Already in new format
				return DBStyleConfigJSON(**data)

		# Old format detected - delete it and create fresh config
		logger.debug(f'Old config format detected at {config_path}, creating fresh config')
		new_config = create_default_config()

		# Overwrite with new config
		with open(config_path, 'w') as f:
			json.dump(new_config.model_dump(), f, indent=2)

		logger.debug(f'Created fresh config.json at {config_path}')
		return new_config

	except Exception as e:
		logger.error(f'Failed to load config from {config_path}: {e}, creating fresh config')
		# On any error, create fresh config
		new_config = create_default_config()
		try:
			with open(config_path, 'w') as f:
				json.dump(new_config.model_dump(), f, indent=2)
		except Exception as write_error:
			logger.error(f'Failed to write fresh config: {write_error}')
		return new_config


class Config:
	"""Backward-compatible configuration class that merges all config sources.

	Re-reads environment variables on every access to maintain compatibility.
	"""

	def __init__(self):
		self._dirs_created = False

	def __getattr__(self, name: str) -> Any:
		if name.startswith('_'):
			raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

		old_config = OldConfig()

		if hasattr(old_config, name):
			return getattr(old_config, name)

		if name == 'get_default_profile':
			return lambda: self._get_default_profile()
		elif name == 'get_default_llm':
			return lambda: self._get_default_llm()
		elif name == 'get_default_agent':
			return lambda: self._get_default_agent()
		elif name == 'load_config':
			return lambda: self._load_config()
		elif name == '_ensure_dirs':
			return lambda: old_config._ensure_dirs()

		raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

	def _get_config_path(self) -> Path:
		env_config = FlatEnvConfig()
		if env_config.BROWSER_USE_CONFIG_PATH:
			return Path(env_config.BROWSER_USE_CONFIG_PATH).expanduser()
		elif env_config.BROWSER_USE_CONFIG_DIR:
			return Path(env_config.BROWSER_USE_CONFIG_DIR).expanduser() / 'config.json'
		else:
			xdg_config = Path(env_config.XDG_CONFIG_HOME).expanduser()
			return xdg_config / 'browseruse' / 'config.json'

	def _get_db_config(self) -> DBStyleConfigJSON:
		config_path = self._get_config_path()
		return load_and_migrate_config(config_path)

	def _get_default_profile(self) -> dict[str, Any]:
		db_config = self._get_db_config()
		for profile in db_config.browser_profile.values():
			if profile.default:
				return profile.model_dump(exclude_none=True)

		if db_config.browser_profile:
			return next(iter(db_config.browser_profile.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_llm(self) -> dict[str, Any]:
		db_config = self._get_db_config()
		for llm in db_config.llm.values():
			if llm.default:
				return llm.model_dump(exclude_none=True)

		if db_config.llm:
			return next(iter(db_config.llm.values())).model_dump(exclude_none=True)

		return {}

	def _get_default_agent(self) -> dict[str, Any]:
		db_config = self._get_db_config()
		for agent in db_config.agent.values():
			if agent.default:
				return agent.model_dump(exclude_none=True)

		if db_config.agent:
			return next(iter(db_config.agent.values())).model_dump(exclude_none=True)

		return {}

	def _load_config(self) -> dict[str, Any]:
		runtime_config = ConfigResolver(include_env=True, include_file=True).resolve()
		return {
			'browser_profile': browser_config_to_profile_dict(runtime_config.browser),
			'llm': runtime_config.llm.model_dump(exclude_none=True),
			'agent': runtime_config.agent.model_dump(exclude_none=True),
		}


# Create singleton instance
CONFIG = Config()


# Helper functions for MCP components
def load_browser_use_config() -> dict[str, Any]:
	runtime_config = ConfigResolver(include_env=True, include_file=True).resolve()
	return {
		'browser_profile': browser_config_to_profile_dict(runtime_config.browser),
		'llm': runtime_config.llm.model_dump(exclude_none=True),
		'agent': runtime_config.agent.model_dump(exclude_none=True),
	}


def get_default_profile(config: RuntimeConfig | dict[str, Any]) -> dict[str, Any]:
	if isinstance(config, RuntimeConfig):
		return browser_config_to_profile_dict(config.browser)
	return config.get('browser_profile', {})


def get_default_llm(config: RuntimeConfig | dict[str, Any]) -> dict[str, Any]:
	if isinstance(config, RuntimeConfig):
		return config.llm.model_dump(exclude_none=True)
	return config.get('llm', {})


# ============================================================================
# New unified runtime configuration system
# ============================================================================
#
# The following functions provide access to the new RuntimeConfig / ConfigResolver
# system while maintaining backward compatibility with existing code.
#
# Priority order (highest to lowest):
#   1. Explicit parameters
#   2. CLI arguments
#   3. Environment variables
#   4. Configuration file (config.json)
#   5. Model defaults
#
# ============================================================================


def get_runtime_config() -> Any:
	"""Get the unified runtime configuration.

	Returns:
	    RuntimeConfig instance with all configuration merged from all sources.

	Example:
	    >>> from browser_use.config import get_runtime_config
	    >>> config = get_runtime_config()
	    >>> print(config.logging.level)
	    >>> print(config.browser.headless)
	"""
	from browser_use.runtime_config import get_runtime_config as _get_runtime_config

	return _get_runtime_config()


def get_config_resolver() -> Any:
	"""Get the default ConfigResolver instance.

	Use this to add custom configuration sources before resolving.

	Returns:
	    ConfigResolver instance with env and file sources already added.

	Example:
	    >>> from browser_use.config import get_config_resolver
	    >>> resolver = get_config_resolver()
	    >>> resolver.add_explicit({'browser': {'headless': True}})
	    >>> config = resolver.resolve()
	"""
	from browser_use.runtime_config import get_default_resolver

	return get_default_resolver()


def create_runtime_config(**kwargs) -> Any:
	"""Create a RuntimeConfig with explicit overrides.

	Convenience function that creates a resolver, adds explicit overrides,
	and returns the resolved config.

	Args:
	    **kwargs: Configuration overrides (can be nested or flat format)

	Returns:
	    Resolved RuntimeConfig instance

	Example:
	    >>> config = create_runtime_config(browser_headless=True, logging_level='debug')
	"""
	from browser_use.runtime_config import ConfigResolver

	resolver = ConfigResolver()
	if kwargs:
		# Check if kwargs are nested or flat
		known_groups = {'logging', 'telemetry', 'cloud', 'llm', 'browser', 'agent', 'security', 'filesystem'}
		is_nested = any(k in known_groups for k in kwargs)
		resolver.add_explicit(kwargs, flat=not is_nested)
	return resolver.resolve()
