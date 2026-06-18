"""Data models for profile presets."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProfileLLMConfig(BaseModel):
	"""LLM configuration within a profile.

	The provider field determines which chat model class to instantiate.
	All other fields are passed as kwargs to the model constructor.
	"""

	model_config = ConfigDict(extra='allow', populate_by_name=True)

	provider: str | None = Field(
		default=None,
		description='LLM provider name (e.g., "openai", "anthropic", "google", "browser_use"). '
		'If not set, auto-detection based on available API keys is used.',
	)
	model: str | None = Field(default=None, description='Model name, e.g. "gpt-4.1-mini"')
	temperature: float | None = Field(default=None, description='Sampling temperature')
	api_key: str | None = Field(default=None, description='API key for the provider')
	api_base: str | None = Field(default=None, description='Custom API base URL')
	max_tokens: int | None = Field(default=None, description='Maximum tokens to generate')


class ProfileDefinition(BaseModel):
	"""A single named profile definition.

	Contains browser, LLM, and agent configuration sections.
	Can extend (inherit from) another profile by name.
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	description: str | None = Field(default=None, description='Human-readable description of this profile')
	extends: str | None = Field(default=None, description='Name of parent profile to inherit from')

	browser: dict[str, Any] = Field(
		default_factory=dict,
		description='BrowserProfile settings as a dictionary',
	)
	llm: ProfileLLMConfig = Field(
		default_factory=ProfileLLMConfig,
		description='LLM configuration',
	)
	agent: dict[str, Any] = Field(
		default_factory=dict,
		description='AgentSettings as a dictionary',
	)


class ResolvedProfile(BaseModel):
	"""A fully resolved profile with all inheritance applied.

	This is the result of loading and merging a profile chain.
	No extends field - all inheritance has been resolved.
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	name: str = Field(description='Name of the resolved profile')
	description: str | None = Field(default=None, description='Description of this profile')

	browser: dict[str, Any] = Field(
		default_factory=dict,
		description='Resolved browser settings',
	)
	llm: ProfileLLMConfig = Field(
		default_factory=ProfileLLMConfig,
		description='Resolved LLM configuration',
	)
	agent: dict[str, Any] = Field(
		default_factory=dict,
		description='Resolved agent settings',
	)

	def signature(self) -> str:
		"""Generate a stable hash signature of this profile's configuration.

		Used to verify that different entry points produce the same profile.
		"""
		config_dict = {
			'browser': dict(sorted(self.browser.items())),
			'llm': dict(sorted(self.llm.model_dump(exclude_none=True).items())),
			'agent': dict(sorted(self.agent.items())),
		}
		serialized = json.dumps(config_dict, sort_keys=True, default=str)
		return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


class EffectiveProfileConfig(BaseModel):
	"""The final, effective profile configuration after all overrides are applied.

	This is the single source of truth for profile configuration across
	all entry points (Python API, CLI, skill_cli, sandbox).

	Override priority (lowest to highest):
	1. Profile preset (from profiles.json)
	2. Environment variables
	3. Explicit CLI/API parameters
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	profile_name: str | None = Field(
		default=None,
		description='Name of the source profile preset, or None if no profile was used',
	)
	source: str = Field(
		default='defaults',
		description='Source of this config: "defaults", "profile", "env", "cli", "api"',
	)

	browser: dict[str, Any] = Field(
		default_factory=dict,
		description='Final effective browser settings',
	)
	llm: ProfileLLMConfig = Field(
		default_factory=ProfileLLMConfig,
		description='Final effective LLM configuration',
	)
	agent: dict[str, Any] = Field(
		default_factory=dict,
		description='Final effective agent settings',
	)

	def signature(self) -> str:
		"""Generate a stable hash signature of the final effective configuration.

		This signature can be used to verify that different entry points
		produce identical configuration for the same profile + overrides.
		"""
		config_dict = {
			'browser': dict(sorted(self.browser.items())),
			'llm': dict(sorted(self.llm.model_dump(exclude_none=True).items())),
			'agent': dict(sorted(self.agent.items())),
		}
		serialized = json.dumps(config_dict, sort_keys=True, default=str)
		return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]

	def has_browser_config(self) -> bool:
		"""Check if there are any browser settings configured."""
		return len(self.browser) > 0

	def has_llm_config(self) -> bool:
		"""Check if there are any LLM settings configured."""
		return self.llm.provider is not None or self.llm.model is not None

	def has_agent_config(self) -> bool:
		"""Check if there are any agent settings configured."""
		return len(self.agent) > 0


class ProfilesFile(BaseModel):
	"""Top-level structure of the profiles.json config file."""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	profiles: dict[str, ProfileDefinition] = Field(
		default_factory=dict,
		description='Map of profile name to profile definition',
	)
	default_profile: str | None = Field(
		default=None,
		description='Name of the default profile to use when none is specified',
	)
