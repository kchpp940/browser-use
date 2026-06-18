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
