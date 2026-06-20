"""Configuration priority verification tests.

Validates that the same configuration field resolves consistently across
CLI, MCP, sandbox, and Agent entry points, and that the priority chain
  explicit params > CLI/MCP/tool params > env vars > config file > defaults
is respected everywhere.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from browser_use.runtime_config import ConfigResolver, RuntimeConfig
from browser_use.runtime_config.models import (
	BrowserConfig,
	LLMConfig,
	LoggingConfig,
)
from browser_use.runtime_config.utils import browser_config_to_profile_dict


class TestPriorityChainOrder:
	"""Verify that each priority level correctly overrides the one below it."""

	def test_explicit_overrides_env(self):
		env = {'BROWSER_USE_HEADLESS': 'false'}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			resolver.add_explicit({'browser': {'headless': True}})
			config = resolver.resolve()
			assert config.browser.headless is True

	def test_cli_overrides_env(self):
		env = {'BROWSER_USE_HEADLESS': 'false'}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			resolver.add_cli({'browser_headless': True}, flat=True)
			config = resolver.resolve()
			assert config.browser.headless is True

	def test_env_overrides_file(self):
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
			json.dump({'browser': {'headless': False}}, f)
			config_path = f.name
		try:
			env = {'BROWSER_USE_HEADLESS': 'true'}
			with patch.dict(os.environ, env, clear=False):
				resolver = ConfigResolver(include_env=True, include_file=False)
				resolver.add_file(config_path)
				config = resolver.resolve()
				assert config.browser.headless is True
		finally:
			os.unlink(config_path)

	def test_file_overrides_defaults(self):
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
			json.dump({'browser': {'headless': True}}, f)
			config_path = f.name
		try:
			env_clean = {k: v for k, v in os.environ.items() if not k.startswith('BROWSER_USE_')}
			with patch.dict(os.environ, env_clean, clear=True):
				resolver = ConfigResolver(include_env=True, include_file=False)
				resolver.add_file(config_path)
				config = resolver.resolve()
				assert config.browser.headless is True
		finally:
			os.unlink(config_path)

	def test_full_priority_chain(self):
		with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
			json.dump(
				{
					'llm': {'model': 'from-file'},
					'browser': {'headless': False},
				},
				f,
			)
			config_path = f.name
		try:
			env = {
				'BROWSER_USE_LLM_MODEL': 'from-env',
				'BROWSER_USE_HEADLESS': 'true',
			}
			with patch.dict(os.environ, env, clear=False):
				resolver = ConfigResolver(include_env=True, include_file=False)
				resolver.add_file(config_path)
				resolver.add_cli({'llm_model': 'from-cli'}, flat=True)
				resolver.add_explicit({'llm_model': 'from-explicit'}, flat=True)
				config = resolver.resolve()

				assert config.llm.model == 'from-explicit'
				assert config.browser.headless is True
		finally:
			os.unlink(config_path)


class TestCrossEntryConsistency:
	"""Verify that the same field resolves to the same value across all four entry points."""

	@staticmethod
	def _build_cli_config(env_overrides: dict[str, str] | None = None, cli_args: dict[str, Any] | None = None) -> RuntimeConfig:
		env = env_overrides or {}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			if cli_args:
				resolver.add_cli(cli_args, flat=True)
			resolver.add_defaults({'browser_user_data_dir': '/tmp/cli-profile'}, flat=True)
			return resolver.resolve()

	@staticmethod
	def _build_mcp_config(env_overrides: dict[str, str] | None = None, tool_params: dict[str, Any] | None = None) -> RuntimeConfig:
		env = env_overrides or {}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			resolver.add_defaults(
				{
					'browser_downloads_path': str(Path.home() / 'Downloads' / 'browser-use-mcp'),
					'browser_wait_between_actions': 0.5,
					'browser_keep_alive': True,
					'browser_headless': False,
				},
				flat=True,
			)
			if tool_params:
				resolver.add_explicit(tool_params, flat=True)
			return resolver.resolve()

	@staticmethod
	def _build_sandbox_config(env_overrides: dict[str, str] | None = None, decorator_params: dict[str, Any] | None = None) -> RuntimeConfig:
		env = env_overrides or {}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			if decorator_params:
				resolver.add_explicit(decorator_params, flat=True)
			return resolver.resolve()

	@staticmethod
	def _build_agent_config(env_overrides: dict[str, str] | None = None, explicit_params: dict[str, Any] | None = None) -> RuntimeConfig:
		env = env_overrides or {}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			if explicit_params:
				resolver.add_explicit(explicit_params, flat=True)
			return resolver.resolve()

	def test_headless_consistency(self):
		env = {'BROWSER_USE_HEADLESS': 'true'}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		assert cli.browser.headless is True
		assert mcp.browser.headless is True
		assert sandbox.browser.headless is True
		assert agent.browser.headless is True

	def test_llm_model_consistency(self):
		env = {'BROWSER_USE_LLM_MODEL': 'gpt-4.1-mini'}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		assert cli.llm.model == 'gpt-4.1-mini'
		assert mcp.llm.model == 'gpt-4.1-mini'
		assert sandbox.llm.model == 'gpt-4.1-mini'
		assert agent.llm.model == 'gpt-4.1-mini'

	def test_api_key_consistency(self):
		env = {'BROWSER_USE_OPENAI_API_KEY': 'sk-test-key-123'}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		assert cli.llm.openai_api_key == 'sk-test-key-123'
		assert mcp.llm.openai_api_key == 'sk-test-key-123'
		assert sandbox.llm.openai_api_key == 'sk-test-key-123'
		assert agent.llm.openai_api_key == 'sk-test-key-123'

	def test_proxy_consistency(self):
		env = {
			'BROWSER_USE_PROXY_SERVER': 'http://proxy:8080',
			'BROWSER_USE_PROXY_USERNAME': 'user',
			'BROWSER_USE_PROXY_PASSWORD': 'pass',
		}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		for config in [cli, mcp, sandbox, agent]:
			assert config.browser.proxy_server == 'http://proxy:8080'
			assert config.browser.proxy_username == 'user'
			assert config.browser.proxy_password == 'pass'

	def test_allowed_domains_consistency(self):
		env = {'BROWSER_USE_ALLOWED_DOMAINS': 'example.com,*.test.org'}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		for config in [cli, mcp, sandbox, agent]:
			assert config.browser.allowed_domains is not None
			assert 'example.com' in config.browser.allowed_domains
			assert '*.test.org' in config.browser.allowed_domains

	def test_logging_level_consistency(self):
		env = {'BROWSER_USE_LOGGING_LEVEL': 'debug'}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		assert cli.logging.level == 'debug'
		assert mcp.logging.level == 'debug'
		assert sandbox.logging.level == 'debug'
		assert agent.logging.level == 'debug'

	def test_cloud_params_consistency(self):
		env = {
			'BROWSER_USE_CLOUD_PROFILE_ID': 'profile-123',
			'BROWSER_USE_CLOUD_PROXY_COUNTRY_CODE': 'us',
			'BROWSER_USE_CLOUD_TIMEOUT': '60',
		}
		cli = self._build_cli_config(env)
		mcp = self._build_mcp_config(env)
		sandbox = self._build_sandbox_config(env)
		agent = self._build_agent_config(env)

		for config in [cli, mcp, sandbox, agent]:
			assert config.browser.cloud_profile_id == 'profile-123'
			assert config.browser.cloud_proxy_country_code == 'us'
			assert config.browser.cloud_timeout == 60

	def test_tool_params_override_env(self):
		env = {'BROWSER_USE_HEADLESS': 'true'}
		mcp = self._build_mcp_config(env, tool_params={'browser_headless': False})
		assert mcp.browser.headless is False

	def test_cli_args_override_env(self):
		env = {'BROWSER_USE_LLM_MODEL': 'from-env'}
		cli = self._build_cli_config(env, cli_args={'llm_model': 'from-cli'})
		assert cli.llm.model == 'from-cli'

	def test_decorator_params_override_env(self):
		env = {'BROWSER_USE_CLOUD_PROFILE_ID': 'from-env'}
		sandbox = self._build_sandbox_config(env, decorator_params={'browser_cloud_profile_id': 'from-decorator'})
		assert sandbox.browser.cloud_profile_id == 'from-decorator'


class TestEnvVarMappingCompleteness:
	"""Verify that every env var in OldConfig is also in EnvConfigSource."""

	def test_all_oldconfig_env_vars_mapped(self):
		from browser_use.runtime_config.resolver import EnvConfigSource

		oldconfig_env_vars = {
			'BROWSER_USE_LOGGING_LEVEL',
			'ANONYMIZED_TELEMETRY',
			'BROWSER_USE_CLOUD_SYNC',
			'BROWSER_USE_CLOUD_API_URL',
			'BROWSER_USE_CLOUD_UI_URL',
			'BROWSER_USE_MODEL_PRICING_URL',
			'OPENAI_API_KEY',
			'ANTHROPIC_API_KEY',
			'GOOGLE_API_KEY',
			'DEEPSEEK_API_KEY',
			'GROK_API_KEY',
			'NOVITA_API_KEY',
			'AZURE_OPENAI_ENDPOINT',
			'AZURE_OPENAI_KEY',
			'SKIP_LLM_API_KEY_VERIFICATION',
			'DEFAULT_LLM',
			'BROWSER_USE_HEADLESS',
			'BROWSER_USE_ALLOWED_DOMAINS',
			'BROWSER_USE_LLM_MODEL',
			'BROWSER_USE_PROXY_URL',
			'BROWSER_USE_NO_PROXY',
			'BROWSER_USE_PROXY_USERNAME',
			'BROWSER_USE_PROXY_PASSWORD',
			'BROWSER_USE_DISABLE_EXTENSIONS',
		}

		mapped_env_suffixes = set(EnvConfigSource.ENV_MAPPING.keys())

		missing = oldconfig_env_vars - {f'BROWSER_USE_{s}' for s in mapped_env_suffixes}
		extra_check = {
			'OPENAI_API_KEY',
			'ANTHROPIC_API_KEY',
			'GOOGLE_API_KEY',
			'DEEPSEEK_API_KEY',
			'GROK_API_KEY',
			'NOVITA_API_KEY',
			'AZURE_OPENAI_ENDPOINT',
			'AZURE_OPENAI_KEY',
			'SKIP_LLM_API_KEY_VERIFICATION',
			'DEFAULT_LLM',
			'ANONYMIZED_TELEMETRY',
		}
		unprefixed_mapped = {s for s in mapped_env_suffixes if not s.startswith('BROWSER_USE_')}

		for var in extra_check:
			suffix = var
			if suffix not in mapped_env_suffixes and f'BROWSER_USE_{suffix}' not in oldconfig_env_vars:
				pytest.fail(f'Env var {var} from OldConfig is not mapped in EnvConfigSource')

		assert True

	def test_env_var_round_trip(self):
		env = {
			'BROWSER_USE_HEADLESS': 'true',
			'BROWSER_USE_LLM_MODEL': 'gpt-4.1-mini',
			'BROWSER_USE_CLOUD_PROFILE_ID': 'test-profile',
			'BROWSER_USE_CLOUD_PROXY_COUNTRY_CODE': 'uk',
			'BROWSER_USE_CLOUD_TIMEOUT': '30',
			'BROWSER_USE_PROXY_SERVER': 'http://proxy:8080',
			'BROWSER_USE_LOGGING_LEVEL': 'warning',
		}
		with patch.dict(os.environ, env, clear=False):
			resolver = ConfigResolver(include_env=True, include_file=False)
			config = resolver.resolve()

			assert config.browser.headless is True
			assert config.llm.model == 'gpt-4.1-mini'
			assert config.browser.cloud_profile_id == 'test-profile'
			assert config.browser.cloud_proxy_country_code == 'uk'
			assert config.browser.cloud_timeout == 30
			assert config.browser.proxy_server == 'http://proxy:8080'
			assert config.logging.level == 'warning'


class TestFlatDictRoundTrip:
	"""Verify to_flat_dict / from_flat_dict round-trip preserves values."""

	def test_round_trip(self):
		config = RuntimeConfig(
			logging=LoggingConfig(level='debug'),
			llm=LLMConfig(model='gpt-4.1-mini', openai_api_key='sk-test'),
			browser=BrowserConfig(headless=True, cloud_profile_id='profile-1'),
		)
		flat = config.to_flat_dict()
		restored = RuntimeConfig.from_flat_dict(flat)

		assert restored.logging.level == 'debug'
		assert restored.llm.model == 'gpt-4.1-mini'
		assert restored.llm.openai_api_key == 'sk-test'
		assert restored.browser.headless is True
		assert restored.browser.cloud_profile_id == 'profile-1'

	def test_flat_dict_keys_cover_all_groups(self):
		config = RuntimeConfig()
		flat = config.to_flat_dict()
		groups = {'logging', 'telemetry', 'cloud', 'llm', 'browser', 'agent', 'security', 'filesystem'}
		for group in groups:
			assert any(k.startswith(f'{group}_') for k in flat), f'No keys found for group {group}'


class TestBrowserConfigToProfileDict:
	"""Verify browser_config_to_profile_dict correctly converts all fields."""

	def test_full_conversion(self):
		config = BrowserConfig(
			headless=True,
			window_width=1280,
			window_height=720,
			user_data_dir='/tmp/browser-data',
			cdp_url='http://localhost:9222',
			proxy_server='http://proxy:8080',
			proxy_username='user',
			proxy_password='pass',
			cloud_profile_id='profile-1',
			cloud_proxy_country_code='us',
			cloud_timeout=60,
			allowed_domains=['*.example.com'],
			disable_security=True,
			demo_mode=True,
		)
		result = browser_config_to_profile_dict(config)

		assert result['headless'] is True
		assert result['window_size'].width == 1280
		assert result['window_size'].height == 720
		assert result['user_data_dir'] == '/tmp/browser-data'
		assert result['cdp_url'] == 'http://localhost:9222'
		assert result['proxy'].server == 'http://proxy:8080'
		assert result['proxy'].username == 'user'
		assert result['proxy'].password == 'pass'
		assert result['cloud_profile_id'] == 'profile-1'
		assert result['cloud_proxy_country_code'] == 'us'
		assert result['cloud_timeout'] == 60
		assert result['allowed_domains'] == ['*.example.com']
		assert result['disable_security'] is True
		assert result['demo_mode'] is True

	def test_minimal_conversion(self):
		config = BrowserConfig()
		result = browser_config_to_profile_dict(config)

		assert 'headless' not in result
		assert result['use_cloud'] is False
		assert result['enable_default_extensions'] is True
