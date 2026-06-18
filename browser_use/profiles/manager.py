"""Profile manager - loads, resolves, and applies profile presets.

This is the single source of truth for profile configuration across
across all entry points (Python API, CLI, skill_cli, sandbox).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from browser_use.profiles.models import (
	EffectiveProfileConfig,
	ProfileLLMConfig,
	ProfileDefinition,
	ProfilesFile,
	ResolvedProfile,
)

logger = logging.getLogger(__name__)

# Default profile file name
DEFAULT_PROFILE_NAME = 'default'

# Environment variable for profile name
PROFILE_ENV_VAR = 'BROWSER_USE_PROFILE'

# Environment variable for profile file path
PROFILE_FILE_ENV_VAR = 'BROWSER_USE_PROFILES_FILE'

# Sentinel value for "not explicitly set" - used to distinguish between
# a parameter that was explicitly set to its default value vs not set at all
_UNSET = object()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
	"""Deep merge two dictionaries.

	Override values take precedence over base values.
	Nested dictionaries are merged recursively.
	"""
	result = base.copy()
	for key, value in override.items():
		if key in result and isinstance(result[key], dict) and isinstance(value, dict):
			result[key] = _deep_merge(result[key], value)
		else:
			result[key] = value
	return result


def _get_profiles_file_path() -> Path:
	"""Get the path to the profiles config file.

	Order of precedence:
	1. BROWSER_USE_PROFILES_FILE environment variable
	2. ~/.config/browseruse/profiles.json (XDG config home)
	"""
	env_path = os.environ.get(PROFILE_FILE_ENV_VAR)
	if env_path:
		return Path(env_path).expanduser().resolve()

	from browser_use.config import CONFIG

	return Path(CONFIG.BROWSER_USE_CONFIG_DIR) / 'profiles.json'


class ProfileManager:
	"""Manages profile presets for browser-use.

	Loads profiles from a JSON config file, resolves inheritance chains,
	and provides methods to create BrowserProfile, LLM, and Agent configs from named profiles.

	This is the single entry point for profile loading across all
	entry points to ensure consistent configuration.
	"""

	def __init__(self, profiles_file: Path | str | None = None):
		"""Initialize the profile manager.

		Args:
		    profiles_file: Path to the profiles JSON file. If None, uses default path.
		"""
		self._profiles_file = Path(profiles_file).expanduser().resolve() if profiles_file else _get_profiles_file_path()
		self._profiles_data: ProfilesFile | None = None
		self._resolved_profiles: dict[str, ResolvedProfile] = {}

	@property
	def profiles_file(self) -> Path:
		"""Path to the profiles config file."""
		return self._profiles_file

	def _load_profiles_file(self) -> ProfilesFile:
		"""Load and parse the profiles config file.

		Returns an empty ProfilesFile if the file doesn't exist.
		"""
		if self._profiles_data is not None:
			return self._profiles_data

		if not self._profiles_file.exists():
			logger.debug(f'Profiles file not found at {self._profiles_file}, using empty config')
			self._profiles_data = ProfilesFile()
			return self._profiles_data

		try:
			with open(self._profiles_file, encoding='utf-8') as f:
				data = json.load(f)
			self._profiles_data = ProfilesFile(**data)
			logger.debug(f'Loaded profiles from {self._profiles_file}: {len(self._profiles_data.profiles)} profiles')
		except (json.JSONDecodeError, ValidationError) as e:
			logger.warning(f'Failed to load profiles from {self._profiles_file}: {e}, using empty config')
			self._profiles_data = ProfilesFile()

		return self._profiles_data

	def reload(self) -> None:
		"""Force reload profiles from disk."""
		self._profiles_data = None
		self._resolved_profiles = {}
		self._load_profiles_file()

	def list_profiles(self) -> list[str]:
		"""List all available profile names."""
		profiles_data = self._load_profiles_file()
		return sorted(profiles_data.profiles.keys())

	def get_default_profile_name(self) -> str | None:
		"""Get the name of the default profile.

		Order of precedence:
		1. BROWSER_USE_PROFILE environment variable
		2. default_profile field in profiles.json
		3. None (no default)
		"""
		env_profile = os.environ.get(PROFILE_ENV_VAR)
		if env_profile:
			return env_profile

		profiles_data = self._load_profiles_file()
		return profiles_data.default_profile

	def has_profile(self, name: str) -> bool:
		"""Check if a profile with the given name exists."""
		profiles_data = self._load_profiles_file()
		return name in profiles_data.profiles

	def _resolve_profile_chain(self, name: str, visited: set[str] | None = None) -> list[ProfileDefinition]:
		"""Resolve the inheritance chain for a profile.

		Returns a list of ProfileDefinitions from base to derived (last = most specific).
		Raises ValueError if circular inheritance is detected.
		"""
		if visited is None:
			visited = set()

		if name in visited:
			raise ValueError(f'Circular profile inheritance detected: {name} is in chain {visited}')

		profiles_data = self._load_profiles_file()

		if name not in profiles_data.profiles:
			raise ValueError(f'Profile not found: {name}. Available profiles: {", ".join(self.list_profiles())}')

		visited.add(name)
		profile = profiles_data.profiles[name]

		if profile.extends:
			chain = self._resolve_profile_chain(profile.extends, visited.copy())
		else:
			chain = []

		chain.append(profile)
		return chain

	def resolve_profile(self, name: str) -> ResolvedProfile:
		"""Resolve a profile by name, applying all inheritance.

		Args:
		    name: Name of the profile to resolve.

		Returns:
		    ResolvedProfile with all inheritance applied.

		Raises:
		    ValueError: If the profile doesn't exist or has circular inheritance.
		"""
		if name in self._resolved_profiles:
			return self._resolved_profiles[name]

		chain = self._resolve_profile_chain(name)

		# Merge all profiles in the chain (each subsequent one overrides the previous)
		merged_browser: dict[str, Any] = {}
		merged_llm_dict: dict[str, Any] = {}
		merged_agent: dict[str, Any] = {}
		description = None

		for profile in chain:
			merged_browser = _deep_merge(merged_browser, profile.browser)
			merged_llm_dict = _deep_merge(merged_llm_dict, profile.llm.model_dump(exclude_none=True))
			merged_agent = _deep_merge(merged_agent, profile.agent)
			if profile.description:
				description = profile.description

		resolved = ResolvedProfile(
			name=name,
			description=description,
			browser=merged_browser,
			llm=ProfileLLMConfig(**merged_llm_dict),
			agent=merged_agent,
		)

		self._resolved_profiles[name] = resolved
		return resolved

	def get_profile(self, name: str | None = None) -> ResolvedProfile:
		"""Get a resolved profile by name.

		If name is None, uses the default profile.
		If no default profile is configured, returns an empty profile.

		Args:
		    name: Profile name, or None for default.

		Returns:
		    ResolvedProfile
		"""
		if name is None:
			name = self.get_default_profile_name()

		if name is None:
			# No profile specified and no default - return empty profile
			return ResolvedProfile(name='empty', description='Empty profile (no config)')

		return self.resolve_profile(name)

	def apply_overrides(
		self,
		profile: ResolvedProfile,
		*,
		browser_overrides: dict[str, Any] | None = None,
		llm_overrides: dict[str, Any] | None = None,
		agent_overrides: dict[str, Any] | None = None,
	) -> ResolvedProfile:
		"""Apply parameter overrides to a resolved profile.

		Override priority (highest to lowest):
		- API/CLI parameters
		- Profile config (already resolved)
		- Default values (in code)

		Args:
		    profile: The resolved profile to override.
		    browser_overrides: Browser settings to override.
		    llm_overrides: LLM settings to override.
		    agent_overrides: Agent settings to override.

		Returns:
		    New ResolvedProfile with overrides applied.
		"""
		merged_browser = _deep_merge(profile.browser, browser_overrides or {})
		merged_agent = _deep_merge(profile.agent, agent_overrides or {})

		# For LLM, merge the dict and create a new ProfileLLMConfig
		llm_dict = profile.llm.model_dump(exclude_none=True)
		merged_llm_dict = _deep_merge(llm_dict, llm_overrides or {})

		return ResolvedProfile(
			name=profile.name,
			description=profile.description,
			browser=merged_browser,
			llm=ProfileLLMConfig(**merged_llm_dict),
			agent=merged_agent,
		)

	def _get_env_overrides(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
		"""Extract configuration overrides from environment variables.

		Returns:
		    Tuple of (browser_overrides, llm_overrides, agent_overrides)
		"""
		browser_overrides: dict[str, Any] = {}
		llm_overrides: dict[str, Any] = {}
		agent_overrides: dict[str, Any] = {}

		# Browser-related env vars
		if os.environ.get('BROWSER_USE_HEADLESS') is not None:
			browser_overrides['headless'] = os.environ['BROWSER_USE_HEADLESS'].lower() in ('1', 'true', 'yes')
		if os.environ.get('BROWSER_USE_CDP_URL'):
			browser_overrides['cdp_url'] = os.environ['BROWSER_USE_CDP_URL']
		if os.environ.get('BROWSER_USE_DOWNLOADS_PATH'):
			browser_overrides['downloads_path'] = os.environ['BROWSER_USE_DOWNLOADS_PATH']
		if os.environ.get('BROWSER_USE_USER_DATA_DIR'):
			browser_overrides['user_data_dir'] = os.environ['BROWSER_USE_USER_DATA_DIR']
		if os.environ.get('BROWSER_USE_CLOUD_PROFILE_ID'):
			browser_overrides['cloud_profile_id'] = os.environ['BROWSER_USE_CLOUD_PROFILE_ID']
		if os.environ.get('BROWSER_USE_CLOUD_PROXY_COUNTRY_CODE'):
			browser_overrides['cloud_proxy_country_code'] = os.environ['BROWSER_USE_CLOUD_PROXY_COUNTRY_CODE']
		if os.environ.get('BROWSER_USE_CLOUD_TIMEOUT'):
			try:
				browser_overrides['cloud_timeout'] = int(os.environ['BROWSER_USE_CLOUD_TIMEOUT'])
			except ValueError:
				pass

		# LLM-related env vars
		if os.environ.get('BROWSER_USE_LLM_PROVIDER'):
			llm_overrides['provider'] = os.environ['BROWSER_USE_LLM_PROVIDER']
		if os.environ.get('BROWSER_USE_LLM_MODEL'):
			llm_overrides['model'] = os.environ['BROWSER_USE_LLM_MODEL']
		if os.environ.get('BROWSER_USE_LLM_TEMPERATURE') is not None:
			try:
				llm_overrides['temperature'] = float(os.environ['BROWSER_USE_LLM_TEMPERATURE'])
			except ValueError:
				pass
		if os.environ.get('BROWSER_USE_LLM_API_KEY'):
			llm_overrides['api_key'] = os.environ['BROWSER_USE_LLM_API_KEY']
		if os.environ.get('BROWSER_USE_LLM_API_BASE'):
			llm_overrides['api_base'] = os.environ['BROWSER_USE_LLM_API_BASE']

		# Agent-related env vars
		if os.environ.get('BROWSER_USE_MAX_STEPS'):
			try:
				agent_overrides['max_steps'] = int(os.environ['BROWSER_USE_MAX_STEPS'])
			except ValueError:
				pass
		if os.environ.get('BROWSER_USE_FLASH_MODE') is not None:
			agent_overrides['flash_mode'] = os.environ['BROWSER_USE_FLASH_MODE'].lower() in ('1', 'true', 'yes')
		if os.environ.get('BROWSER_USE_USE_VISION') is not None:
			val = os.environ['BROWSER_USE_USE_VISION'].lower()
			if val == 'auto':
				agent_overrides['use_vision'] = 'auto'
			else:
				agent_overrides['use_vision'] = val in ('1', 'true', 'yes')
		if os.environ.get('BROWSER_USE_MAX_FAILURES'):
			try:
				agent_overrides['max_failures'] = int(os.environ['BROWSER_USE_MAX_FAILURES'])
			except ValueError:
				pass

		return browser_overrides, llm_overrides, agent_overrides

	def build_effective_config(
		self,
		profile_name: str | None = None,
		*,
		browser_overrides: dict[str, Any] | None = None,
		llm_overrides: dict[str, Any] | None = None,
		agent_overrides: dict[str, Any] | None = None,
		source: str = 'api',
	) -> EffectiveProfileConfig:
		"""Build the final effective profile configuration.

		Applies overrides in order of priority (lowest to highest):
		1. Profile preset (from profiles.json)
		2. Environment variables
		3. Explicit CLI/API parameters

		Args:
		    profile_name: Name of the profile to load, or None for default.
		    browser_overrides: Browser settings from explicit parameters.
		    llm_overrides: LLM settings from explicit parameters.
		    agent_overrides: Agent settings from explicit parameters.
		    source: Source identifier for the config (e.g., "api", "cli", "sandbox").

		Returns:
		    EffectiveProfileConfig with all overrides applied.
		"""
		# Step 1: Load profile preset (base layer)
		resolved_profile = self.get_profile(profile_name)
		has_profile = resolved_profile.name != 'empty'

		# Step 2: Apply environment variable overrides
		env_browser, env_llm, env_agent = self._get_env_overrides()
		has_env_overrides = bool(env_browser or env_llm or env_agent)

		# Step 3: Apply explicit parameter overrides
		has_explicit_overrides = bool(browser_overrides or llm_overrides or agent_overrides)

		# Merge all layers
		# Profile -> env -> explicit
		merged_browser = resolved_profile.browser.copy()
		if env_browser:
			merged_browser = _deep_merge(merged_browser, env_browser)
		if browser_overrides:
			merged_browser = _deep_merge(merged_browser, browser_overrides)

		merged_agent = resolved_profile.agent.copy()
		if env_agent:
			merged_agent = _deep_merge(merged_agent, env_agent)
		if agent_overrides:
			merged_agent = _deep_merge(merged_agent, agent_overrides)

		# LLM merging
		llm_dict = resolved_profile.llm.model_dump(exclude_none=True)
		if env_llm:
			llm_dict = _deep_merge(llm_dict, env_llm)
		if llm_overrides:
			llm_dict = _deep_merge(llm_dict, llm_overrides)

		# Determine source
		if has_explicit_overrides:
			final_source = source
		elif has_env_overrides:
			final_source = 'env'
		elif has_profile:
			final_source = 'profile'
		else:
			final_source = 'defaults'

		return EffectiveProfileConfig(
			profile_name=resolved_profile.name if has_profile else None,
			source=final_source,
			browser=merged_browser,
			llm=ProfileLLMConfig(**llm_dict),
			agent=merged_agent,
		)

	def create_browser_profile_from_effective(self, effective: EffectiveProfileConfig):
		"""Create a BrowserProfile instance from an effective config.

		Args:
		    effective: The effective profile config.

		Returns:
		    BrowserProfile instance
		"""
		from browser_use.browser.profile import BrowserProfile

		return BrowserProfile(**effective.browser)

	def create_llm_from_effective(self, effective: EffectiveProfileConfig):
		"""Create an LLM chat model from an effective config.

		Args:
		    effective: The effective profile config.

		Returns:
		    Chat model instance, or None if no LLM config is provided.
		"""
		# Reuse the existing create_llm logic by wrapping in a ResolvedProfile-like object
		resolved = ResolvedProfile(
			name=effective.profile_name or 'effective',
			browser=effective.browser,
			llm=effective.llm,
			agent=effective.agent,
		)
		return self.create_llm(resolved)

	def log_effective_profile_info(self, effective: EffectiveProfileConfig) -> None:
		"""Log effective profile information for debugging/verification.

		Args:
		    effective: The effective profile config to log.
		"""
		if effective.profile_name:
			logger.info(f'📋 Profile loaded: {effective.profile_name}')
		else:
			logger.info('📋 No profile preset loaded')
		logger.info(f'   Source: {effective.source}')
		logger.info(f'   Signature: {effective.signature()}')

		# Log browser settings summary
		browser_summary = []
		if effective.browser:
			for key in [
				'headless', 'use_cloud', 'cdp_url', 'user_data_dir',
				'profile_directory', 'downloads_path', 'window_size',
				'cloud_profile_id', 'cloud_proxy_country_code', 'cloud_timeout',
			]:
				if key in effective.browser and effective.browser[key] is not None:
					val = effective.browser[key]
					if key == 'window_size' and isinstance(val, dict):
						browser_summary.append(f'{key}={val.get("width")}x{val.get("height")}')
					else:
						browser_summary.append(f'{key}={val}')
		if browser_summary:
			logger.info(f'   Browser: {", ".join(browser_summary)}')

		# Log LLM summary
		llm_summary = []
		if effective.llm.provider:
			llm_summary.append(f'provider={effective.llm.provider}')
		if effective.llm.model:
			llm_summary.append(f'model={effective.llm.model}')
		if effective.llm.temperature is not None:
			llm_summary.append(f'temperature={effective.llm.temperature}')
		if llm_summary:
			logger.info(f'   LLM: {", ".join(llm_summary)}')

		# Log agent settings summary
		agent_summary = []
		if effective.agent:
			for key in ['flash_mode', 'use_vision', 'max_failures', 'max_actions_per_step', 'use_thinking']:
				if key in effective.agent and effective.agent[key] is not None:
					agent_summary.append(f'{key}={effective.agent[key]}')
		if agent_summary:
			logger.info(f'   Agent: {", ".join(agent_summary)}')

	def create_browser_profile(self, resolved: ResolvedProfile):
		"""Create a BrowserProfile instance from a resolved profile.

		Args:
		    resolved: The resolved profile.

		Returns:
		    BrowserProfile instance
		"""
		from browser_use.browser.profile import BrowserProfile

		return BrowserProfile(**resolved.browser)

	def create_llm(self, resolved: ResolvedProfile):
		"""Create an LLM chat model from a resolved profile.

		Args:
		    resolved: The resolved profile.

		Returns:
		    Chat model instance, or None if no LLM config is provided.
		"""
		llm_config = resolved.llm
		provider = llm_config.provider

		if not provider and not llm_config.model:
			# Try to auto-detect provider from model name
			provider = self._detect_provider_from_model(llm_config.model)

		if not provider:
			# Auto-detect based on available API keys
			return self._auto_detect_llm(llm_config)

		# Build kwargs from llm config (excluding provider)
		llm_kwargs = llm_config.model_dump(exclude={'provider'}, exclude_none=True)

		provider_map = {
			'openai': 'ChatOpenAI',
			'anthropic': 'ChatAnthropic',
			'google': 'ChatGoogle',
			'browser_use': 'ChatBrowserUse',
			'groq': 'ChatGroq',
			'mistral': 'ChatMistral',
			'azure': 'ChatAzureOpenAI',
			'ollama': 'ChatOllama',
			'deepseek': 'ChatDeepSeek',
			'openrouter': 'ChatOpenRouter',
			'vercel': 'ChatVercel',
			'cerebras': 'ChatCerebras',
			'litellm': 'ChatLiteLLM',
		}

		if provider not in provider_map:
			raise ValueError(
				f'Unknown LLM provider: {provider}. Available: {", ".join(provider_map.keys())}'
			)

		# Lazy import the chat model class
		from importlib import import_module

		module_path, class_name = self._get_provider_import(provider)
		module = import_module(module_path)
		chat_class = getattr(module, class_name)

		return chat_class(**llm_kwargs)

	def _detect_provider_from_model(self, model_name: str) -> str | None:
		"""Try to detect the LLM provider from the model name.

		Args:
		    model_name: The model name.

		Returns:
		    Provider name string, or None if can't detect.
		"""
		model_lower = model_name.lower()

		if model_lower.startswith(('gpt-', 'o1', 'o3', 'o4', 'gpt5')):
			return 'openai'
		if model_lower.startswith('claude-'):
			return 'anthropic'
		if model_lower.startswith('gemini-'):
			return 'google'
		if model_lower.startswith('llama') or model_lower.startswith('mixtral'):
			return 'openrouter'
		if 'groq' in model_lower:
			return 'groq'

		return None

	def _auto_detect_llm(self, llm_config: ProfileLLMConfig):
		"""Auto-detect LLM provider based on available API keys.

		Args:
		    llm_config: LLM configuration.

		Returns:
		    Chat model instance.
		"""
		from browser_use.config import CONFIG

		llm_kwargs = llm_config.model_dump(exclude={'provider'}, exclude_none=True)

		# Check for browser_use first (recommended)
		if os.environ.get('BROWSER_USE_API_KEY'):
			from browser_use.llm.browser_use.chat import ChatBrowserUse

			return ChatBrowserUse(**llm_kwargs)

		# Check for OpenAI
		if CONFIG.OPENAI_API_KEY or llm_config.api_key:
			from browser_use.llm.openai.chat import ChatOpenAI

			model = llm_config.model or 'gpt-4.1-mini'
			return ChatOpenAI(model=model, **{k: v for k, v in llm_kwargs.items() if k != 'model'})

		# Check for Anthropic
		if CONFIG.ANTHROPIC_API_KEY:
			from browser_use.llm.anthropic.chat import ChatAnthropic

			model = llm_config.model or 'claude-4-sonnet-20250514'
			return ChatAnthropic(model=model, **{k: v for k, v in llm_kwargs.items() if k != 'model'})

		# Check for Google
		if CONFIG.GOOGLE_API_KEY:
			from browser_use.llm.google.chat import ChatGoogle

			model = llm_config.model or 'gemini-2.5-pro'
			return ChatGoogle(model=model, **{k: v for k, v in llm_kwargs.items() if k != 'model'})

		raise ValueError(
			'No LLM provider specified and no API keys detected. '
			'Set BROWSER_USE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY '
			'environment variable, or specify a provider in your profile.'
		)

	@staticmethod
	def _get_provider_import(provider: str) -> tuple[str, str]:
		"""Get the module path and class name for a provider.

		Args:
		    provider: Provider name.

		Returns:
		    Tuple of (module_path, class_name)
		"""
		provider_imports = {
			'openai': ('browser_use.llm.openai.chat', 'ChatOpenAI'),
			'anthropic': ('browser_use.llm.anthropic.chat', 'ChatAnthropic'),
			'google': ('browser_use.llm.google.chat', 'ChatGoogle'),
			'browser_use': ('browser_use.llm.browser_use.chat', 'ChatBrowserUse'),
			'groq': ('browser_use.llm.groq.chat', 'ChatGroq'),
			'mistral': ('browser_use.llm.mistral.chat', 'ChatMistral'),
			'azure': ('browser_use.llm.azure.chat', 'ChatAzureOpenAI'),
			'ollama': ('browser_use.llm.ollama.chat', 'ChatOllama'),
			'deepseek': ('browser_use.llm.deepseek.chat', 'ChatDeepSeek'),
			'openrouter': ('browser_use.llm.openrouter.chat', 'ChatOpenRouter'),
			'vercel': ('browser_use.llm.vercel.chat', 'ChatVercel'),
			'cerebras': ('browser_use.llm.cerebras.chat', 'ChatCerebras'),
			'litellm': ('browser_use.llm.litellm.chat', 'ChatLiteLLM'),
		}

		if provider not in provider_imports:
			raise ValueError(f'Unknown LLM provider: {provider}')

		return provider_imports[provider]

	def log_profile_info(self, resolved: ResolvedProfile) -> None:
		"""Log profile information for debugging/verification.

		Args:
		    resolved: The resolved profile to log.
		"""
		logger.info(f'📋 Profile loaded: {resolved.name}')
		if resolved.description:
			logger.info(f'   Description: {resolved.description}')
		logger.info(f'   Signature: {resolved.signature()}')

		# Log browser settings summary
		browser_summary = []
		if resolved.browser:
			for key in ['headless', 'use_cloud', 'cdp_url', 'user_data_dir', 'downloads_path', 'window_size']:
				if key in resolved.browser:
					val = resolved.browser[key]
					if key == 'window_size' and isinstance(val, dict):
						browser_summary.append(f'{key}={val.get("width")}x{val.get("height")}')
					else:
						browser_summary.append(f'{key}={val}')
		if browser_summary:
			logger.info(f'   Browser: {", ".join(browser_summary)}')

		# Log LLM summary
		llm_summary = []
		if resolved.llm.provider:
			llm_summary.append(f'provider={resolved.llm.provider}')
		if resolved.llm.model:
			llm_summary.append(f'model={resolved.llm.model}')
		if llm_summary:
			logger.info(f'   LLM: {", ".join(llm_summary)}')

		# Log agent settings summary
		agent_summary = []
		if resolved.agent:
			for key in ['max_steps', 'use_vision', 'flash_mode']:
				if key in resolved.agent:
					agent_summary.append(f'{key}={resolved.agent[key]}')
		if agent_summary:
			logger.info(f'   Agent: {", ".join(agent_summary)}')


# Singleton instance
_manager: ProfileManager | None = None


def get_profile_manager(profiles_file: Path | str | None = None) -> ProfileManager:
	"""Get the global ProfileManager instance.

	Args:
	    profiles_file: Optional path to profiles file (only used on first call).

	Returns:
	    Global ProfileManager instance.
	"""
	global _manager

	if _manager is None:
		_manager = ProfileManager(profiles_file=profiles_file)

	return _manager


def load_profile(
	name: str | None = None,
	*,
	browser_overrides: dict[str, Any] | None = None,
	llm_overrides: dict[str, Any] | None = None,
	agent_overrides: dict[str, Any] | None = None,
	profiles_file: Path | str | None = None,
) -> ResolvedProfile:
	"""Convenience function to load and resolve a profile with optional overrides.

	This is the main entry point for loading profiles across the profiles system.

	Args:
	    name: Profile name (None = default).
	    browser_overrides: Browser settings to override.
	    llm_overrides: LLM settings to override.
	    agent_overrides: Agent settings to override.
	    profiles_file: Optional path to profiles file.

	Returns:
	    ResolvedProfile with overrides applied.
	"""
	manager = get_profile_manager(profiles_file=profiles_file)
	profile = manager.get_profile(name)

	if browser_overrides or llm_overrides or agent_overrides:
		profile = manager.apply_overrides(
			profile,
			browser_overrides=browser_overrides,
			llm_overrides=llm_overrides,
			agent_overrides=agent_overrides,
		)

	return profile


def list_profiles(profiles_file: Path | str | None = None) -> list[str]:
	"""List all available profile names.

	Args:
	    profiles_file: Optional path to profiles file.

	Returns:
	    List of profile names.
	"""
	manager = get_profile_manager(profiles_file=profiles_file)
	return manager.list_profiles()


def resolve_profile(name: str, profiles_file: Path | str | None = None) -> ResolvedProfile:
	"""Resolve a profile by name.

	Args:
	    name: Profile name.
	    profiles_file: Optional path to profiles file.

	Returns:
	    ResolvedProfile.
	"""
	manager = get_profile_manager(profiles_file=profiles_file)
	return manager.resolve_profile(name)
