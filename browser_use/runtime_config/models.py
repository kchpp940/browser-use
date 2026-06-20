"""Runtime configuration data models.

All configuration is organized into logical groups, each represented by
a Pydantic model. The top-level RuntimeConfig aggregates all groups.

These models are type-safe and validated, ensuring configuration is always
consistent across all entry points.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
	"""Logging configuration."""

	model_config = ConfigDict(extra='forbid')

	level: str = Field(default='info', description='Logging level (debug, info, warning, error, critical, result)')
	cdp_level: str = Field(default='WARNING', description='CDP logging level')
	debug_log_file: str | None = Field(default=None, description='Path to debug log file')
	info_log_file: str | None = Field(default=None, description='Path to info log file')
	setup_logging: bool = Field(default=True, description='Whether to set up logging automatically')

	@field_validator('level')
	@classmethod
	def validate_log_level(cls, v: str) -> str:
		valid_levels = ['debug', 'info', 'warning', 'error', 'critical', 'result']
		if v.lower() not in valid_levels:
			raise ValueError(f'Invalid log level: {v}. Must be one of {valid_levels}')
		return v.lower()

	@property
	def python_level(self) -> int:
		"""Convert log level string to Python logging level integer."""
		level_map = {
			'debug': logging.DEBUG,
			'info': logging.INFO,
			'warning': logging.WARNING,
			'error': logging.ERROR,
			'critical': logging.CRITICAL,
			'result': 35,
		}
		return level_map.get(self.level.lower(), logging.INFO)


class TelemetryConfig(BaseModel):
	"""Telemetry configuration."""

	model_config = ConfigDict(extra='forbid')

	anonymized_telemetry: bool = Field(default=True, description='Enable anonymized telemetry')
	cloud_sync: bool | None = Field(default=None, description='Enable cloud sync (defaults to anonymized_telemetry)')
	version_check: bool = Field(default=True, description='Check for latest browser-use version')

	@property
	def cloud_sync_enabled(self) -> bool:
		"""Get cloud sync setting with fallback to anonymized_telemetry."""
		if self.cloud_sync is not None:
			return self.cloud_sync
		return self.anonymized_telemetry


class CloudConfig(BaseModel):
	"""Cloud browser configuration."""

	model_config = ConfigDict(extra='forbid')

	api_url: str = Field(default='https://api.browser-use.com', description='Cloud API URL')
	ui_url: str = Field(default='', description='Cloud UI URL')
	api_key: str | None = Field(default=None, description='Cloud API key')
	profile_id: str | None = Field(default=None, description='Cloud browser profile ID')
	proxy_country_code: str | None = Field(default=None, description='Cloud proxy country code')
	timeout: int | None = Field(default=None, description='Cloud browser session timeout in minutes')
	model_pricing_url: str = Field(default='', description='Model pricing URL')


class LLMConfig(BaseModel):
	"""LLM configuration."""

	model_config = ConfigDict(extra='forbid')

	provider: str | None = Field(default=None, description='LLM provider name')
	model: str | None = Field(default=None, description='Model name')
	api_key: str | None = Field(default=None, description='API key')
	api_base: str | None = Field(default=None, description='API base URL')
	temperature: float | None = Field(default=None, description='Sampling temperature')
	max_tokens: int | None = Field(default=None, description='Maximum tokens to generate')
	timeout: int | None = Field(default=None, description='LLM call timeout in seconds')
	default_llm: str = Field(default='', description='Default LLM name to use')
	skip_api_key_verification: bool = Field(default=False, description='Skip LLM API key verification')

	openai_api_key: str = Field(default='', description='OpenAI API key')
	anthropic_api_key: str = Field(default='', description='Anthropic API key')
	google_api_key: str = Field(default='', description='Google API key')
	deepseek_api_key: str = Field(default='', description='DeepSeek API key')
	grok_api_key: str = Field(default='', description='Grok API key')
	novita_api_key: str = Field(default='', description='Novita API key')
	azure_endpoint: str = Field(default='', description='Azure OpenAI endpoint')
	azure_api_key: str = Field(default='', description='Azure OpenAI API key')

	region: str | None = Field(default=None, description='AWS region for Bedrock')
	aws_sso_auth: bool = Field(default=False, description='Use AWS SSO auth for Bedrock')

	@property
	def resolved_api_key(self) -> str | None:
		"""Get the resolved API key based on provider or fallback."""
		if self.api_key:
			return self.api_key
		provider_lower = (self.provider or '').lower()
		if 'openai' in provider_lower:
			return self.openai_api_key or None
		elif 'anthropic' in provider_lower:
			return self.anthropic_api_key or None
		elif 'google' in provider_lower or 'gemini' in provider_lower:
			return self.google_api_key or None
		elif 'deepseek' in provider_lower:
			return self.deepseek_api_key or None
		return None


class BrowserConfig(BaseModel):
	"""Browser configuration.

	This mirrors the key settings from BrowserProfile but in a flat,
	serializable format suitable for config files and env vars.
	"""

	model_config = ConfigDict(extra='forbid')

	headless: bool | None = Field(default=None, description='Run browser in headless mode')
	window_width: int | None = Field(default=None, description='Browser window width')
	window_height: int | None = Field(default=None, description='Browser window height')
	user_data_dir: str | None = Field(default=None, description='Browser user data directory')
	profile_directory: str = Field(default='Default', description='Chrome profile subdirectory')
	cdp_url: str | None = Field(default=None, description='CDP URL for remote browser')
	channel: str | None = Field(default=None, description='Browser channel (chromium, chrome, etc.)')
	executable_path: str | None = Field(default=None, description='Path to browser executable')

	use_cloud: bool = Field(default=False, description='Use cloud browser')
	cloud_profile_id: str | None = Field(default=None, description='Cloud browser profile ID')
	cloud_proxy_country_code: str | None = Field(default=None, description='Cloud proxy country code')
	cloud_timeout: int | None = Field(default=None, description='Cloud browser timeout in minutes')

	allowed_domains: list[str] | None = Field(default=None, description='Allowed domains for navigation')
	prohibited_domains: list[str] | None = Field(default=None, description='Prohibited domains for navigation')
	block_ip_addresses: bool = Field(default=False, description='Block navigation to IP addresses')

	proxy_server: str | None = Field(default=None, description='Proxy server URL')
	proxy_bypass: str | None = Field(default=None, description='Proxy bypass list (comma-separated)')
	proxy_username: str | None = Field(default=None, description='Proxy username')
	proxy_password: str | None = Field(default=None, description='Proxy password')

	enable_default_extensions: bool = Field(default=True, description='Enable default browser extensions')
	keep_alive: bool | None = Field(default=None, description='Keep browser alive after agent run')
	downloads_path: str | None = Field(default=None, description='Downloads directory path')

	minimum_wait_page_load_time: float = Field(default=0.25, description='Minimum page load wait time')
	wait_for_network_idle_page_load_time: float = Field(default=0.5, description='Network idle wait time')
	wait_between_actions: float = Field(default=0.1, description='Wait time between actions')

	highlight_elements: bool = Field(default=True, description='Highlight interactive elements')
	paint_order_filtering: bool = Field(default=True, description='Enable paint order filtering')

	record_video_dir: str | None = Field(default=None, description='Video recording directory')
	record_har_path: str | None = Field(default=None, description='HAR recording file path')
	traces_dir: str | None = Field(default=None, description='Trace files directory')

	cross_origin_iframes: bool = Field(default=True, description='Enable cross-origin iframe support')

	disable_security: bool = Field(default=False, description='Disable browser security (not recommended)')
	deterministic_rendering: bool = Field(default=False, description='Enable deterministic rendering')

	devtools: bool = Field(default=False, description='Open DevTools automatically')
	chromium_sandbox: bool | None = Field(default=None, description='Enable Chromium sandbox')

	auto_download_pdfs: bool = Field(default=True, description='Automatically download PDFs')
	demo_mode: bool = Field(default=False, description='Enable demo mode')

	is_local: bool = Field(default=False, description='Whether this is a local browser instance')

	permissions: list[str] = Field(
		default_factory=lambda: ['clipboardReadWrite', 'notifications'],
		description='Browser permissions to grant',
	)


class AgentConfig(BaseModel):
	"""Agent configuration."""

	model_config = ConfigDict(extra='forbid')

	max_steps: int = Field(default=100, description='Maximum number of agent steps')
	max_actions_per_step: int = Field(default=5, description='Maximum actions per step')
	max_failures: int = Field(default=5, description='Maximum retries for failed steps')

	use_vision: bool | Literal['auto'] = Field(default='auto', description='Vision mode (auto, true, false)')
	vision_detail_level: Literal['auto', 'low', 'high'] = Field(default='auto', description='Screenshot detail level')
	use_thinking: bool = Field(default=True, description='Enable thinking mode')
	flash_mode: bool = Field(default=False, description='Enable flash mode (fast but less accurate)')

	llm_timeout: int | None = Field(default=None, description='LLM call timeout in seconds')
	step_timeout: int = Field(default=180, description='Timeout for each agent step in seconds')

	save_conversation_path: str | None = Field(default=None, description='Path to save conversation history')
	save_conversation_path_encoding: str = Field(default='utf-8', description='Encoding for saved conversations')

	generate_gif: bool | str = Field(default=False, description='Generate GIF of agent actions')
	calculate_cost: bool = Field(default=False, description='Calculate and track API costs')

	directly_open_url: bool = Field(default=True, description='Directly open URLs detected in task')
	display_files_in_done_text: bool = Field(default=True, description='Show file info in completion messages')

	final_response_after_failure: bool = Field(default=True, description='Force final response after max failures')
	max_history_items: int | None = Field(default=None, description='Maximum number of history items to keep')

	max_clickable_elements_length: int = Field(default=40000, description='Maximum clickable elements length')
	loop_detection_window: int = Field(default=20, description='Window size for loop detection')
	loop_detection_enabled: bool = Field(default=True, description='Enable loop detection')

	enable_planning: bool = Field(default=True, description='Enable planning capability')
	planning_replan_on_stall: int = Field(default=3, description='Replan after N stalled steps')
	planning_exploration_limit: int = Field(default=5, description='Exploration limit for planning')

	include_attributes: list[str] | None = Field(default=None, description='HTML attributes to include in analysis')
	source: str | None = Field(default=None, description='Source identifier for telemetry')

	use_judge: bool = Field(default=True, description='Use judge model for evaluation')
	include_tool_call_examples: bool = Field(default=False, description='Include tool call examples in prompt')
	include_recent_events: bool = Field(default=False, description='Include recent events in state')

	message_compaction: bool = Field(default=True, description='Enable message compaction')


class SecurityConfig(BaseModel):
	"""Security configuration."""

	model_config = ConfigDict(extra='forbid')

	allowed_domains: list[str] | None = Field(default=None, description='Allowed domains')
	prohibited_domains: list[str] | None = Field(default=None, description='Prohibited domains')
	block_ip_addresses: bool = Field(default=False, description='Block IP addresses')
	sensitive_data: dict[str, str] = Field(default_factory=dict, description='Sensitive data patterns')


class FilesystemConfig(BaseModel):
	"""Filesystem configuration."""

	model_config = ConfigDict(extra='forbid')

	available_file_paths: list[str] = Field(default_factory=list, description='Accessible file paths')
	downloads_path: str | None = Field(default=None, description='Downloads directory')
	file_system_path: str | None = Field(default=None, description='File system root path')
	config_dir: str | None = Field(default=None, description='Configuration directory')
	cache_dir: str | None = Field(default=None, description='Cache directory')

	@property
	def resolved_config_dir(self) -> Path:
		"""Get resolved config directory path."""
		if self.config_dir:
			return Path(self.config_dir).expanduser().resolve()
		return Path.home() / '.config' / 'browseruse'

	@property
	def resolved_cache_dir(self) -> Path:
		"""Get resolved cache directory path."""
		if self.cache_dir:
			return Path(self.cache_dir).expanduser().resolve()
		return Path.home() / '.cache' / 'browseruse'


class RuntimeConfig(BaseModel):
	"""Top-level runtime configuration.

	Aggregates all configuration groups. This is the single source of truth
	for runtime configuration across all entry points.
	"""

	model_config = ConfigDict(extra='forbid')

	logging: LoggingConfig = Field(default_factory=LoggingConfig)
	telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
	cloud: CloudConfig = Field(default_factory=CloudConfig)
	llm: LLMConfig = Field(default_factory=LLMConfig)
	browser: BrowserConfig = Field(default_factory=BrowserConfig)
	agent: AgentConfig = Field(default_factory=AgentConfig)
	security: SecurityConfig = Field(default_factory=SecurityConfig)
	filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)

	in_docker: bool | None = Field(default=None, description='Running in Docker container')
	is_in_evals: bool = Field(default=False, description='Running in evaluation mode')
	win_font_dir: str = Field(default='C:\\Windows\\Fonts', description='Windows font directory')

	config_path: str | None = Field(default=None, description='Path to configuration file')

	@property
	def is_docker(self) -> bool:
		"""Check if running in Docker with auto-detection fallback."""
		if self.in_docker is not None:
			return self.in_docker
		try:
			if Path('/.dockerenv').exists():
				return True
		except Exception:
			pass
		return False

	def to_flat_dict(self) -> dict[str, Any]:
		"""Convert nested config to a flat dictionary for env var style access.

		Returns:
		    Flat dict with keys like 'logging_level', 'browser_headless', etc.
		"""
		result: dict[str, Any] = {}

		for group_name in ['logging', 'telemetry', 'cloud', 'llm', 'browser', 'agent', 'security', 'filesystem']:
			group = getattr(self, group_name)
			for field_name, field_value in group.model_dump(exclude_none=False).items():
				key = f'{group_name}_{field_name}'
				result[key] = field_value

		for field_name in ['in_docker', 'is_in_evals', 'win_font_dir', 'config_path']:
			result[field_name] = getattr(self, field_name)

		return result

	@classmethod
	def from_flat_dict(cls, data: dict[str, Any]) -> RuntimeConfig:
		"""Create RuntimeConfig from a flat dictionary.

		Keys should be in 'group_field' format (e.g., 'logging_level').

		Args:
		    data: Flat dictionary of configuration values

		Returns:
		    Parsed RuntimeConfig instance
		"""
		groups: dict[str, dict[str, Any]] = {
			'logging': {},
			'telemetry': {},
			'cloud': {},
			'llm': {},
			'browser': {},
			'agent': {},
			'security': {},
			'filesystem': {},
		}
		top_level: dict[str, Any] = {}

		for key, value in data.items():
			if value is None:
				continue

			parts = key.split('_', 1)
			if len(parts) == 2 and parts[0] in groups:
				group_name, field_name = parts
				groups[group_name][field_name] = value
			else:
				top_level[key] = value

		return cls(
			logging=LoggingConfig(**groups['logging']),
			telemetry=TelemetryConfig(**groups['telemetry']),
			cloud=CloudConfig(**groups['cloud']),
			llm=LLMConfig(**groups['llm']),
			browser=BrowserConfig(**groups['browser']),
			agent=AgentConfig(**groups['agent']),
			security=SecurityConfig(**groups['security']),
			filesystem=FilesystemConfig(**groups['filesystem']),
			**top_level,
		)
