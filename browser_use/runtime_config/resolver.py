"""Configuration resolver for browser-use.

ConfigResolver merges configuration from multiple sources with a well-defined
priority order, ensuring consistent configuration across all entry points.

Priority (highest to lowest):
    1. Explicit parameters (passed directly)
    2. CLI arguments
    3. Environment variables
    4. Configuration file (config.json)
    5. Entry-point defaults (Pydantic model defaults)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from browser_use.runtime_config.models import RuntimeConfig


class ConfigSource(ABC):
	"""Abstract base class for configuration sources."""

	@abstractmethod
	def get_config_dict(self) -> dict[str, Any]:
		"""Get configuration as a nested dictionary.

		The dictionary should follow the RuntimeConfig structure:
		{
		    'logging': {...},
		    'llm': {...},
		    'browser': {...},
		    ...
		}

		Returns:
		    Nested dictionary of configuration values.
		"""
		...

	@property
	@abstractmethod
	def priority(self) -> int:
		"""Priority level (higher = takes precedence).

		Standard priorities:
		- 100: Explicit parameters
		- 90:  CLI arguments
		- 80:  Environment variables
		- 70:  Configuration file
		- 10:  Default values
		"""
		...


class DictConfigSource(ConfigSource):
	"""Configuration source from a dictionary.

	Used for explicit parameters, CLI arguments, and other dict-based sources.
	"""

	def __init__(self, data: dict[str, Any], priority: int = 50, flat: bool = False):
		"""Initialize dict config source.

		Args:
		    data: Configuration dictionary (nested or flat)
		    priority: Priority level
		    flat: If True, data uses flat keys like 'logging_level'
		"""
		self._data = data
		self._priority = priority
		self._flat = flat

	def get_config_dict(self) -> dict[str, Any]:
		if self._flat:
			return self._flat_to_nested(self._data)
		return dict(self._data)

	@property
	def priority(self) -> int:
		return self._priority

	@staticmethod
	def _flat_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
		"""Convert flat dict with 'group_field' keys to nested dict."""
		nested: dict[str, dict[str, Any]] = {}
		top_level: dict[str, Any] = {}

		known_groups = {
			'logging', 'telemetry', 'cloud', 'llm', 'browser',
			'agent', 'security', 'filesystem',
		}

		for key, value in flat.items():
			if value is None:
				continue

			parts = key.split('_', 1)
			if len(parts) == 2 and parts[0] in known_groups:
				group_name, field_name = parts
				if group_name not in nested:
					nested[group_name] = {}
				nested[group_name][field_name] = value
			else:
				top_level[key] = value

		result = dict(nested)
		result.update(top_level)
		return result


class EnvConfigSource(ConfigSource):
	"""Configuration source from environment variables.

	Maps BROWSER_USE_* environment variables to config fields using
	a standardized naming convention.
	"""

	# Mapping from env var suffixes to (group, field) tuples
	ENV_MAPPING: dict[str, tuple[str, str]] = {
		# Logging
		'LOGGING_LEVEL': ('logging', 'level'),
		'CDP_LOGGING_LEVEL': ('logging', 'cdp_level'),
		'DEBUG_LOG_FILE': ('logging', 'debug_log_file'),
		'INFO_LOG_FILE': ('logging', 'info_log_file'),
		'SETUP_LOGGING': ('logging', 'setup_logging'),

		# Telemetry
		'ANONYMIZED_TELEMETRY': ('telemetry', 'anonymized_telemetry'),
		'CLOUD_SYNC': ('telemetry', 'cloud_sync'),
		'VERSION_CHECK': ('telemetry', 'version_check'),

		# Cloud
		'CLOUD_API_URL': ('cloud', 'api_url'),
		'CLOUD_UI_URL': ('cloud', 'ui_url'),
		'CLOUD_API_KEY': ('cloud', 'api_key'),
		'MODEL_PRICING_URL': ('cloud', 'model_pricing_url'),

		# LLM
		'DEFAULT_LLM': ('llm', 'default_llm'),
		'SKIP_LLM_API_KEY_VERIFICATION': ('llm', 'skip_api_key_verification'),
		'LLM_MODEL': ('llm', 'model'),
		'OPENAI_API_KEY': ('llm', 'openai_api_key'),
		'ANTHROPIC_API_KEY': ('llm', 'anthropic_api_key'),
		'GOOGLE_API_KEY': ('llm', 'google_api_key'),
		'DEEPSEEK_API_KEY': ('llm', 'deepseek_api_key'),
		'GROK_API_KEY': ('llm', 'grok_api_key'),
		'NOVITA_API_KEY': ('llm', 'novita_api_key'),
		'AZURE_OPENAI_ENDPOINT': ('llm', 'azure_endpoint'),
		'AZURE_OPENAI_KEY': ('llm', 'azure_api_key'),

		# Browser - general
		'HEADLESS': ('browser', 'headless'),
		'WINDOW_WIDTH': ('browser', 'window_width'),
		'WINDOW_HEIGHT': ('browser', 'window_height'),
		'USER_DATA_DIR': ('browser', 'user_data_dir'),
		'PROFILE_DIRECTORY': ('browser', 'profile_directory'),
		'CDP_URL': ('browser', 'cdp_url'),
		'CHANNEL': ('browser', 'channel'),
		'EXECUTABLE_PATH': ('browser', 'executable_path'),
		'KEEP_ALIVE': ('browser', 'keep_alive'),
		'DOWNLOADS_PATH': ('browser', 'downloads_path'),

		# Browser - cloud
		'USE_CLOUD': ('browser', 'use_cloud'),
		'CLOUD_PROFILE_ID': ('browser', 'cloud_profile_id'),
		'CLOUD_PROXY_COUNTRY_CODE': ('browser', 'cloud_proxy_country_code'),
		'CLOUD_TIMEOUT': ('browser', 'cloud_timeout'),

		# Browser - domains
		'ALLOWED_DOMAINS': ('browser', 'allowed_domains'),
		'PROHIBITED_DOMAINS': ('browser', 'prohibited_domains'),
		'BLOCK_IP_ADDRESSES': ('browser', 'block_ip_addresses'),

		# Browser - proxy
		'PROXY_URL': ('browser', 'proxy_server'),
		'PROXY_SERVER': ('browser', 'proxy_server'),
		'NO_PROXY': ('browser', 'proxy_bypass'),
		'PROXY_BYPASS': ('browser', 'proxy_bypass'),
		'PROXY_USERNAME': ('browser', 'proxy_username'),
		'PROXY_PASSWORD': ('browser', 'proxy_password'),

		# Browser - features
		'DISABLE_EXTENSIONS': ('browser', 'enable_default_extensions'),
		'HIGHLIGHT_ELEMENTS': ('browser', 'highlight_elements'),
		'PAINT_ORDER_FILTERING': ('browser', 'paint_order_filtering'),
		'CROSS_ORIGIN_IFRAMES': ('browser', 'cross_origin_iframes'),
		'DEMO_MODE': ('browser', 'demo_mode'),

		# Browser - recordings
		'RECORD_VIDEO_DIR': ('browser', 'record_video_dir'),
		'RECORD_HAR_PATH': ('browser', 'record_har_path'),
		'TRACES_DIR': ('browser', 'traces_dir'),

		# Browser - timing
		'MINIMUM_WAIT_PAGE_LOAD_TIME': ('browser', 'minimum_wait_page_load_time'),
		'WAIT_FOR_NETWORK_IDLE_PAGE_LOAD_TIME': ('browser', 'wait_for_network_idle_page_load_time'),
		'WAIT_BETWEEN_ACTIONS': ('browser', 'wait_between_actions'),

		# Agent
		'MAX_STEPS': ('agent', 'max_steps'),
		'MAX_ACTIONS_PER_STEP': ('agent', 'max_actions_per_step'),
		'MAX_FAILURES': ('agent', 'max_failures'),
		'USE_VISION': ('agent', 'use_vision'),
		'VISION_DETAIL_LEVEL': ('agent', 'vision_detail_level'),
		'USE_THINKING': ('agent', 'use_thinking'),
		'FLASH_MODE': ('agent', 'flash_mode'),
		'LLM_TIMEOUT': ('agent', 'llm_timeout'),
		'STEP_TIMEOUT': ('agent', 'step_timeout'),
		'SAVE_CONVERSATION_PATH': ('agent', 'save_conversation_path'),
		'GENERATE_GIF': ('agent', 'generate_gif'),
		'CALCULATE_COST': ('agent', 'calculate_cost'),
		'DIRECTLY_OPEN_URL': ('agent', 'directly_open_url'),
		'MAX_HISTORY_ITEMS': ('agent', 'max_history_items'),
		'LOOP_DETECTION_ENABLED': ('agent', 'loop_detection_enabled'),
		'ENABLE_PLANNING': ('agent', 'enable_planning'),

		# Security
		'SECURITY_ALLOWED_DOMAINS': ('security', 'allowed_domains'),
		'SECURITY_PROHIBITED_DOMAINS': ('security', 'prohibited_domains'),

		# Filesystem
		'CONFIG_DIR': ('filesystem', 'config_dir'),
		'CACHE_DIR': ('filesystem', 'cache_dir'),
		'FILE_SYSTEM_PATH': ('filesystem', 'file_system_path'),

		# Top-level
		'CONFIG_PATH': ('config_path',),
		'IN_DOCKER': ('in_docker',),
		'IS_IN_EVALS': ('is_in_evals',),
		'WIN_FONT_DIR': ('win_font_dir',),
		'API_KEY': ('cloud', 'api_key'),
	}

	def __init__(self, prefix: str = 'BROWSER_USE_', priority: int = 80):
		"""Initialize environment variable config source.

		Args:
		    prefix: Environment variable prefix
		    priority: Priority level
		"""
		self._prefix = prefix
		self._priority = priority

	def get_config_dict(self) -> dict[str, Any]:
		result: dict[str, Any] = {}

		for env_suffix, target in self.ENV_MAPPING.items():
			env_name = f'{self._prefix}{env_suffix}'
			env_value = os.getenv(env_name)

			if env_value is None:
				continue

			# Parse value to appropriate type
			parsed_value = self._parse_env_value(env_value, target)

			if parsed_value is None:
				continue

			if len(target) == 2:
				group_name, field_name = target
				if group_name not in result:
					result[group_name] = {}
				result[group_name][field_name] = parsed_value
			elif len(target) == 1:
				result[target[0]] = parsed_value

		return result

	@property
	def priority(self) -> int:
		return self._priority

	@staticmethod
	def _parse_env_value(value: str, target: tuple[str, ...]) -> Any:
		"""Parse environment variable string to appropriate Python type.

		Args:
		    value: Raw string value from environment
		    target: Target path tuple (group, field) or (top_level_field,)

		Returns:
		    Parsed value, or None if it should be skipped.
		"""
		field_name = target[-1]

		# Handle boolean fields
		boolean_fields = {
			'headless', 'use_cloud', 'block_ip_addresses',
			'enable_default_extensions', 'keep_alive',
			'highlight_elements', 'paint_order_filtering',
			'cross_origin_iframes', 'demo_mode',
			'disable_security', 'deterministic_rendering',
			'devtools', 'chromium_sandbox',
			'auto_download_pdfs', 'is_local',
			'use_thinking', 'flash_mode', 'calculate_cost',
			'directly_open_url', 'display_files_in_done_text',
			'final_response_after_failure', 'enable_planning',
			'loop_detection_enabled', 'use_judge',
			'include_tool_call_examples', 'include_recent_events',
			'message_compaction', 'anonymized_telemetry',
			'cloud_sync', 'version_check', 'setup_logging',
			'in_docker', 'is_in_evals',
			'skip_api_key_verification',
		}

		# Special case: DISABLE_EXTENSIONS is inverted
		if field_name == 'enable_default_extensions':
			# Value is for DISABLE_EXTENSIONS env var, invert it
			return value.lower() not in ('1', 'true', 'yes', 'on')

		if field_name in boolean_fields:
			return value.lower() in ('1', 'true', 'yes', 'on')

		# Handle list fields (comma-separated)
		list_fields = {
			'allowed_domains', 'prohibited_domains',
			'include_attributes', 'available_file_paths',
			'permissions',
		}

		if field_name in list_fields:
			if not value.strip():
				return None
			return [item.strip() for item in value.split(',') if item.strip()]

		# Handle int fields
		int_fields = {
			'max_steps', 'max_actions_per_step', 'max_failures',
			'llm_timeout', 'step_timeout', 'max_history_items',
			'loop_detection_window', 'planning_replan_on_stall',
			'planning_exploration_limit', 'max_clickable_elements_length',
			'max_tokens', 'cloud_timeout',
			'window_width', 'window_height',
		}

		if field_name in int_fields:
			try:
				return int(value)
			except (ValueError, TypeError):
				return None

		# Handle float fields
		float_fields = {
			'temperature',
			'minimum_wait_page_load_time',
			'wait_for_network_idle_page_load_time',
			'wait_between_actions',
		}

		if field_name in float_fields:
			try:
				return float(value)
			except (ValueError, TypeError):
				return None

		# Default: return as string
		return value


class FileConfigSource(ConfigSource):
	"""Configuration source from a JSON file.

	Supports both the new DB-style format and simple nested format.
	"""

	def __init__(self, config_path: str | Path | None = None, priority: int = 70):
		"""Initialize file config source.

		Args:
		    config_path: Path to config file. If None, uses default location.
		    priority: Priority level
		"""
		self._config_path = Path(config_path).expanduser().resolve() if config_path else None
		self._priority = priority

	def get_config_dict(self) -> dict[str, Any]:
		config_path = self._config_path or self._get_default_path()

		if not config_path.exists():
			return {}

		try:
			with open(config_path) as f:
				data = json.load(f)
			return self._convert_to_runtime_config_format(data)
		except (json.JSONDecodeError, OSError) as e:
			import logging

			logger = logging.getLogger(__name__)
			logger.warning(f'Failed to load config file {config_path}: {e}')
			return {}

	@property
	def priority(self) -> int:
		return self._priority

	@staticmethod
	def _get_default_path() -> Path:
		"""Get default config file path."""
		config_dir = os.getenv('BROWSER_USE_CONFIG_DIR')
		if config_dir:
			return Path(config_dir).expanduser().resolve() / 'config.json'

		xdg_config = os.getenv('XDG_CONFIG_HOME', '~/.config')
		return Path(xdg_config).expanduser().resolve() / 'browseruse' / 'config.json'

	@staticmethod
	def _convert_to_runtime_config_format(data: dict[str, Any]) -> dict[str, Any]:
		"""Convert various config formats to runtime config format.

		Handles:
		- New DB-style format: {'browser_profile': {...}, 'llm': {...}, 'agent': {...}}
		- Simple nested format: {'logging': {...}, 'browser': {...}, ...}
		"""
		result: dict[str, Any] = {}

		# Check if it's the DB-style format
		if 'browser_profile' in data or 'llm' in data or 'agent' in data:
			# Handle browser profiles - use default profile
			browser_profiles = data.get('browser_profile', {})
			if isinstance(browser_profiles, dict) and browser_profiles:
				default_profile = None
				for profile in browser_profiles.values():
					if isinstance(profile, dict) and profile.get('default'):
						default_profile = profile
						break
				if default_profile is None and browser_profiles:
					default_profile = next(iter(browser_profiles.values()))

				if default_profile:
					result['browser'] = FileConfigSource._extract_browser_config(default_profile)

			# Handle LLM configs
			llm_configs = data.get('llm', {})
			if isinstance(llm_configs, dict) and llm_configs:
				default_llm = None
				for llm in llm_configs.values():
					if isinstance(llm, dict) and llm.get('default'):
						default_llm = llm
						break
				if default_llm is None and llm_configs:
					default_llm = next(iter(llm_configs.values()))

				if default_llm:
					result['llm'] = FileConfigSource._extract_llm_config(default_llm)

			# Handle agent configs
			agent_configs = data.get('agent', {})
			if isinstance(agent_configs, dict) and agent_configs:
				default_agent = None
				for agent in agent_configs.values():
					if isinstance(agent, dict) and agent.get('default'):
						default_agent = agent
						break
				if default_agent is None and agent_configs:
					default_agent = next(iter(agent_configs.values()))

				if default_agent:
					result['agent'] = FileConfigSource._extract_agent_config(default_agent)

		else:
			# Assume it's already in runtime config format
			result = data

		return result

	@staticmethod
	def _extract_browser_config(profile: dict[str, Any]) -> dict[str, Any]:
		"""Extract browser config from profile dict."""
		browser = {}
		field_mapping = {
			'headless': 'headless',
			'user_data_dir': 'user_data_dir',
			'profile_directory': 'profile_directory',
			'cdp_url': 'cdp_url',
			'channel': 'channel',
			'executable_path': 'executable_path',
			'use_cloud': 'use_cloud',
			'cloud_profile_id': 'cloud_profile_id',
			'cloud_proxy_country_code': 'cloud_proxy_country_code',
			'cloud_timeout': 'cloud_timeout',
			'allowed_domains': 'allowed_domains',
			'prohibited_domains': 'prohibited_domains',
			'block_ip_addresses': 'block_ip_addresses',
			'enable_default_extensions': 'enable_default_extensions',
			'keep_alive': 'keep_alive',
			'downloads_path': 'downloads_path',
			'highlight_elements': 'highlight_elements',
			'paint_order_filtering': 'paint_order_filtering',
			'record_video_dir': 'record_video_dir',
			'record_har_path': 'record_har_path',
			'traces_dir': 'traces_dir',
			'cross_origin_iframes': 'cross_origin_iframes',
			'disable_security': 'disable_security',
			'deterministic_rendering': 'deterministic_rendering',
			'demo_mode': 'demo_mode',
			'devtools': 'devtools',
			'auto_download_pdfs': 'auto_download_pdfs',
		}

		for src_key, dst_key in field_mapping.items():
			if src_key in profile and profile[src_key] is not None:
				browser[dst_key] = profile[src_key]

		# Handle proxy settings
		if 'proxy' in profile and isinstance(profile['proxy'], dict):
			proxy = profile['proxy']
			if 'server' in proxy:
				browser['proxy_server'] = proxy['server']
			if 'bypass' in proxy:
				browser['proxy_bypass'] = proxy['bypass']
			if 'username' in proxy:
				browser['proxy_username'] = proxy['username']
			if 'password' in proxy:
				browser['proxy_password'] = proxy['password']

		return browser

	@staticmethod
	def _extract_llm_config(llm: dict[str, Any]) -> dict[str, Any]:
		"""Extract LLM config from dict."""
		result = {}
		field_mapping = {
			'model': 'model',
			'api_key': 'api_key',
			'temperature': 'temperature',
			'max_tokens': 'max_tokens',
			'provider': 'provider',
		}

		for src_key, dst_key in field_mapping.items():
			if src_key in llm and llm[src_key] is not None:
				result[dst_key] = llm[src_key]

		return result

	@staticmethod
	def _extract_agent_config(agent: dict[str, Any]) -> dict[str, Any]:
		"""Extract agent config from dict."""
		result = {}
		field_mapping = {
			'max_steps': 'max_steps',
			'use_vision': 'use_vision',
			'system_prompt': 'override_system_message',
			'max_actions_per_step': 'max_actions_per_step',
			'max_failures': 'max_failures',
		}

		for src_key, dst_key in field_mapping.items():
			if src_key in agent and agent[src_key] is not None:
				result[dst_key] = agent[src_key]

		return result


class ConfigResolver:
	"""Configuration resolver that merges multiple config sources.

	Priority (highest to lowest):
	    1. Explicit parameters (passed via add_explicit)
	    2. CLI arguments (passed via add_cli)
	    3. Environment variables (auto-detected)
	    4. Configuration file (auto-detected)
	    5. Default values (Pydantic model defaults)

	Usage:
	    resolver = ConfigResolver()
	    resolver.add_explicit({'browser': {'headless': True}})
	    config = resolver.resolve()
	"""

	def __init__(self, include_env: bool = True, include_file: bool = True):
		"""Initialize config resolver.

		Args:
		    include_env: Include environment variable source
		    include_file: Include configuration file source
		"""
		self._sources: list[ConfigSource] = []

		if include_file:
			self._sources.append(FileConfigSource())
		if include_env:
			self._sources.append(EnvConfigSource())

	def add_source(self, source: ConfigSource) -> 'ConfigResolver':
		"""Add a configuration source.

		Args:
		    source: Config source to add

		Returns:
		    Self for chaining
		"""
		self._sources.append(source)
		return self

	def add_explicit(self, data: dict[str, Any], flat: bool = False) -> 'ConfigResolver':
		"""Add explicit parameters (highest priority).

		Args:
		    data: Configuration dictionary
		    flat: If True, uses flat keys like 'logging_level'

		Returns:
		    Self for chaining
		"""
		self._sources.append(DictConfigSource(data, priority=100, flat=flat))
		return self

	def add_cli(self, data: dict[str, Any], flat: bool = False) -> 'ConfigResolver':
		"""Add CLI arguments (priority: 90).

		Args:
		    data: CLI arguments dictionary
		    flat: If True, uses flat keys like 'logging_level'

		Returns:
		    Self for chaining
		"""
		self._sources.append(DictConfigSource(data, priority=90, flat=flat))
		return self

	def add_env(self, prefix: str = 'BROWSER_USE_') -> 'ConfigResolver':
		"""Add environment variable source.

		Args:
		    prefix: Environment variable prefix

		Returns:
		    Self for chaining
		"""
		self._sources.append(EnvConfigSource(prefix=prefix, priority=80))
		return self

	def add_file(self, config_path: str | Path | None = None) -> 'ConfigResolver':
		"""Add configuration file source.

		Args:
		    config_path: Path to config file, or None for default

		Returns:
		    Self for chaining
		"""
		self._sources.append(FileConfigSource(config_path=config_path, priority=70))
		return self

	def add_defaults(self, data: dict[str, Any], flat: bool = False) -> 'ConfigResolver':
		"""Add custom defaults (lowest priority, below model defaults in practice).

		Args:
		    data: Default values dictionary
		    flat: If True, uses flat keys like 'logging_level'

		Returns:
		    Self for chaining
		"""
		self._sources.append(DictConfigSource(data, priority=10, flat=flat))
		return self

	def resolve(self) -> RuntimeConfig:
		"""Resolve and merge all configuration sources.

		Returns:
		    Merged RuntimeConfig instance

		Raises:
		    ValidationError: If merged configuration is invalid
		"""
		# Sort sources by priority (lowest first) so higher priority sources
		# are merged later and override lower priority ones
		sorted_sources = sorted(self._sources, key=lambda s: s.priority)

		# Deep merge all source dicts - later sources override earlier ones
		merged: dict[str, Any] = {}
		for source in sorted_sources:
			source_data = source.get_config_dict()
			self._deep_merge(merged, source_data)

		# Parse into RuntimeConfig
		try:
			return RuntimeConfig(**merged)
		except ValidationError:
			# Try to validate as much as possible, using defaults for invalid fields
			import logging

			logger = logging.getLogger(__name__)
			logger.warning('Some configuration values were invalid, using defaults')
			return RuntimeConfig()

	@staticmethod
	def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
		"""Deep merge override dict into base dict (modifies base in-place).

		Values in override take precedence over values in base.
		"""
		for key, value in override.items():
			if value is None:
				continue

			if key in base and isinstance(base[key], dict) and isinstance(value, dict):
				ConfigResolver._deep_merge(base[key], value)
			else:
				base[key] = value


# Global default resolver instance
_default_resolver: ConfigResolver | None = None


def get_default_resolver() -> ConfigResolver:
	"""Get the default global ConfigResolver instance.

	Returns:
	    Default ConfigResolver with env and file sources enabled
	"""
	global _default_resolver
	if _default_resolver is None:
		_default_resolver = ConfigResolver(include_env=True, include_file=True)
	return _default_resolver


def get_runtime_config() -> RuntimeConfig:
	"""Get the default runtime configuration.

	Convenience function that uses the default resolver.

	Returns:
	    Resolved RuntimeConfig
	"""
	return get_default_resolver().resolve()
