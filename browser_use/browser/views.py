from dataclasses import dataclass, field
from typing import Any

from bubus import BaseEvent
from cdp_use.cdp.target import TargetID
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_serializer

from browser_use.browser.cloud.views import CloudBrowserParams
from browser_use.dom.views import DOMInteractedElement, SerializedDOMState

# Known placeholder image data for about:blank pages - a 4x4 white PNG
PLACEHOLDER_4PX_SCREENSHOT = (
	'iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAFElEQVR4nGP8//8/AwwwMSAB3BwAlm4DBfIlvvkAAAAASUVORK5CYII='
)


# Pydantic
class TabInfo(BaseModel):
	"""Represents information about a browser tab"""

	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		populate_by_name=True,
	)

	# Original fields
	url: str
	title: str
	target_id: TargetID = Field(serialization_alias='tab_id', validation_alias=AliasChoices('tab_id', 'target_id'))
	parent_target_id: TargetID | None = Field(
		default=None, serialization_alias='parent_tab_id', validation_alias=AliasChoices('parent_tab_id', 'parent_target_id')
	)  # parent page that contains this popup or cross-origin iframe

	@field_serializer('target_id')
	def serialize_target_id(self, target_id: TargetID, _info: Any) -> str:
		return target_id[-4:]

	@field_serializer('parent_target_id')
	def serialize_parent_target_id(self, parent_target_id: TargetID | None, _info: Any) -> str | None:
		return parent_target_id[-4:] if parent_target_id else None


class PageInfo(BaseModel):
	"""Comprehensive page size and scroll information"""

	# Current viewport dimensions
	viewport_width: int
	viewport_height: int

	# Total page dimensions
	page_width: int
	page_height: int

	# Current scroll position
	scroll_x: int
	scroll_y: int

	# Calculated scroll information
	pixels_above: int
	pixels_below: int
	pixels_left: int
	pixels_right: int

	# Page statistics are now computed dynamically instead of stored


@dataclass
class NetworkRequest:
	"""Information about a pending network request"""

	url: str
	method: str = 'GET'
	loading_duration_ms: float = 0.0  # How long this request has been loading (ms since request started, max 10s)
	resource_type: str | None = None  # e.g., 'Document', 'Stylesheet', 'Image', 'Script', 'XHR', 'Fetch'


@dataclass
class PaginationButton:
	"""Information about a pagination button detected on the page"""

	button_type: str  # 'next', 'prev', 'first', 'last', 'page_number'
	backend_node_id: int  # Backend node ID for clicking
	text: str  # Button text/label
	selector: str  # XPath or other selector to locate the element
	is_disabled: bool = False  # Whether the button appears disabled


@dataclass
class BrowserStateSummary:
	"""The summary of the browser's current state designed for an LLM to process"""

	# provided by SerializedDOMState:
	dom_state: SerializedDOMState

	url: str
	title: str
	tabs: list[TabInfo]
	screenshot: str | None = field(default=None, repr=False)
	page_info: PageInfo | None = None  # Enhanced page information

	# Keep legacy fields for backward compatibility
	pixels_above: int = 0
	pixels_below: int = 0
	browser_errors: list[str] = field(default_factory=list)
	is_pdf_viewer: bool = False  # Whether the current page is a PDF viewer
	recent_events: str | None = None  # Text summary of recent browser events
	pending_network_requests: list[NetworkRequest] = field(default_factory=list)  # Currently loading network requests
	pagination_buttons: list[PaginationButton] = field(default_factory=list)  # Detected pagination buttons
	closed_popup_messages: list[str] = field(default_factory=list)  # Messages from auto-closed JavaScript dialogs


@dataclass
class BrowserStateHistory:
	"""The summary of the browser's state at a past point in time to usse in LLM message history"""

	url: str
	title: str
	tabs: list[TabInfo]
	interacted_element: list[DOMInteractedElement | None] | list[None]
	screenshot_path: str | None = None

	def get_screenshot(self) -> str | None:
		"""Load screenshot from disk and return as base64 string"""
		if not self.screenshot_path:
			return None

		import base64
		from pathlib import Path

		path_obj = Path(self.screenshot_path)
		if not path_obj.exists():
			return None

		try:
			with open(path_obj, 'rb') as f:
				screenshot_data = f.read()
			return base64.b64encode(screenshot_data).decode('utf-8')
		except Exception:
			return None

	def to_dict(self) -> dict[str, Any]:
		data = {}
		data['tabs'] = [tab.model_dump() for tab in self.tabs]
		data['screenshot_path'] = self.screenshot_path
		data['interacted_element'] = [el.to_dict() if el else None for el in self.interacted_element]
		data['url'] = self.url
		data['title'] = self.title
		return data


class BrowserError(Exception):
	"""Browser error with structured memory for LLM context management.

	This exception class provides separate memory contexts for browser actions:
	- short_term_memory: Immediate context shown once to the LLM for the next action
	- long_term_memory: Persistent error information stored across steps
	"""

	message: str
	short_term_memory: str | None = None
	long_term_memory: str | None = None
	details: dict[str, Any] | None = None
	while_handling_event: BaseEvent[Any] | None = None

	def __init__(
		self,
		message: str,
		short_term_memory: str | None = None,
		long_term_memory: str | None = None,
		details: dict[str, Any] | None = None,
		event: BaseEvent[Any] | None = None,
	):
		"""Initialize a BrowserError with structured memory contexts.

		Args:
			message: Technical error message for logging and debugging
			short_term_memory: Context shown once to LLM (e.g., available actions, options)
			long_term_memory: Persistent error info stored in agent memory
			details: Additional metadata for debugging
			event: The browser event that triggered this error
		"""
		self.message = message
		self.short_term_memory = short_term_memory
		self.long_term_memory = long_term_memory
		self.details = details
		self.while_handling_event = event
		super().__init__(message)

	def __str__(self) -> str:
		parts = [self.message]
		if self.details:
			parts.append(f'({self.details})')
		if self.while_handling_event:
			parts.append(f'during: {self.while_handling_event}')
		return ' '.join(parts)


class URLNotAllowedError(BrowserError):
	"""Error raised when a URL is not allowed"""


class LLMEffectiveConfig(BaseModel):
	"""Unified, already-resolved LLM configuration after merging all sources.

	This is the single source of truth for LLM initialization across all entry points
	(Python API, CLI, TUI, skill_cli, beta agent, sandbox/cloud).

	Provider-specific parameters that don't have a dedicated field can be placed in
	``provider_params`` — build_llm() will forward them as kwargs to the constructor.
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	# Core identity
	provider: str | None = Field(default=None, description='LLM provider name, e.g. "browser-use", "openai", "anthropic"')
	model: str | None = Field(default=None, description='Model name string, e.g. "gpt-4.1-mini"')
	model_class: str | None = Field(
		default=None, description='Fully qualified class name, e.g. "browser_use.llm.browser_use.chat.ChatBrowserUse"'
	)

	# Common sampling params
	temperature: float | None = Field(default=None, description='Sampling temperature')
	max_tokens: int | None = Field(default=None, description='Maximum output tokens (alias: max_completion_tokens)')
	max_completion_tokens: int | None = Field(default=None, description='Maximum completion tokens')
	top_p: float | None = None
	frequency_penalty: float | None = None
	presence_penalty: float | None = None
	seed: int | None = None
	reasoning_effort: str | None = None

	# Client / transport params
	api_key: str | None = Field(default=None, description='API key (masked in logs)')
	base_url: str | None = Field(default=None, description='Custom API base URL')
	timeout: float | None = Field(default=None, description='LLM call timeout in seconds')
	max_retries: int | None = None
	organization: str | None = None
	project: str | None = None

	# Provider-specific catch-all — forwarded as **kwargs to the LLM constructor
	provider_params: dict[str, Any] = Field(
		default_factory=dict,
		description='Provider-specific kwargs forwarded directly to the LLM constructor',
	)

	def get_log_safe_dict(self) -> dict[str, Any]:
		"""Return a dict safe for logging (API key masked, provider_params redacted)."""
		d = self.model_dump(exclude_none=True)
		if d.get('api_key'):
			key = d['api_key']
			d['api_key'] = key[:4] + '...' + key[-4:] if len(key) > 8 else '***'
		if d.get('provider_params'):
			d['provider_params'] = f'<{len(d["provider_params"])} keys>'
		return d

	def to_llm_kwargs(self) -> dict[str, Any]:
		"""Build the kwargs dict for constructing the LLM client.

		Maps field aliases (e.g. max_tokens → max_completion_tokens for OpenAI-style clients)
		and merges provider_params on top.
		"""
		raw = self.model_dump(exclude_none=True, exclude={'provider', 'model_class', 'provider_params'})

		# Alias: max_tokens <-> max_completion_tokens (prefer explicit max_completion_tokens)
		if 'max_completion_tokens' in raw:
			raw['max_tokens'] = raw.pop('max_completion_tokens')

		# Merge provider-specific params (they win if keys collide)
		raw.update(self.provider_params or {})

		# Remove None values again after merging
		return {k: v for k, v in raw.items() if v is not None}


class EffectiveConfig(BaseModel):
	"""Unified, already-resolved configuration after merging ALL config sources.

	This is the single source of truth for browser + LLM initialization across all entry points.
	All consumers (BrowserSession, Agent LLM init, sandbox/cloud payload) must ONLY use this.

	Merge priority (highest -> lowest):
	1. direct_kwargs (Python API call-site parameters)
	2. cli_args (CLI / decorator arguments)
	3. Environment variables (BROWSER_USE_*, OPENAI_API_KEY, etc.)
	4. Config file (config.json)
	5. Pydantic model defaults
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	browser: dict[str, Any] = Field(
		default_factory=dict,
		description='Resolved BrowserProfile field dict — pass directly to BrowserProfile(**effective.browser)',
	)
	llm: LLMEffectiveConfig = Field(
		default_factory=LLMEffectiveConfig,
		description='Resolved LLM config — use to instantiate the correct ChatModel subclass',
	)
	agent: dict[str, Any] = Field(
		default_factory=dict,
		description='Resolved Agent settings — max_steps, use_vision, etc.',
	)

	def browser_profile_kwargs(self) -> dict[str, Any]:
		"""Return kwargs ready for BrowserProfile(**kwargs)."""
		return dict(self.browser)

	def get_config_signature(self) -> dict[str, Any]:
		"""Return a stable signature for consistency checks (logging, daemon config compare, cloud)."""
		sig: dict[str, Any] = {
			'browser': {},
			'llm': self.llm.get_log_safe_dict(),
		}

		# Only include a curated set of browser fields that actually affect the runtime
		from browser_use.browser.profile import ProxySettings

		browser_fields = [
			'headless',
			'cdp_url',
			'use_cloud',
			'user_data_dir',
			'profile_directory',
			'downloads_path',
			'storage_state',
			'allowed_domains',
			'prohibited_domains',
			'proxy',
			'window_size',
			'viewport',
			'device_scale_factor',
			'keep_alive',
			'auto_download_pdfs',
			'disable_security',
			'cloud_browser_params',
		]
		for f in browser_fields:
			val = self.browser.get(f)
			if val is None:
				continue
			if hasattr(val, 'model_dump'):
				sig['browser'][f] = val.model_dump()
			elif isinstance(val, ProxySettings):
				sig['browser'][f] = val.model_dump()
			else:
				sig['browser'][f] = val

		return sig

	def get_log_safe_dict(self) -> dict[str, Any]:
		"""Return the full effective config with secrets masked, for logging."""
		return {
			'browser': self.browser_profile_kwargs(),
			'llm': self.llm.get_log_safe_dict(),
			'agent': dict(self.agent),
		}

	@classmethod
	def from_config_sources(
		cls,
		direct_kwargs: dict[str, Any] | None = None,
		cli_args: dict[str, Any] | None = None,
		load_from_env: bool = True,
		load_from_config_file: bool = True,
	) -> 'EffectiveConfig':
		"""Create the unified EffectiveConfig by merging ALL config sources.

		This is the SINGLE factory used by every entry point to produce the one true
		configuration that BrowserSession, LLM initialisation, and the sandbox/cloud
		payload all consume.

		Priority (highest to lowest):
		1. direct_kwargs   — explicit Python API kwargs
		2. cli_args        — CLI / decorator arguments (normalised via _normalize_cli_args)
		3. env vars        — BROWSER_USE_*, OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
		4. config.json     — DB-style browser_profile / llm / agent entries
		5. Pydantic defaults
		"""
		from browser_use.browser.profile import BrowserProfile, ProxySettings
		from browser_use.config import CONFIG, FlatEnvConfig

		# ---------- Split direct_kwargs into browser/llm/agent buckets ----------
		direct_browser: dict[str, Any] = {}
		direct_llm: dict[str, Any] = {}
		direct_agent: dict[str, Any] = {}

		if direct_kwargs:
			# Known LLM field names (flat provider-style names accepted on the CLI)
			llm_fields = {
				'provider',
				'model',
				'llm_model',
				'temperature',
				'max_tokens',
				'api_key',
				'base_url',
				'llm_timeout',
			}
			# Known Agent top-level overrides
			agent_fields = {'max_steps', 'use_vision', 'flash_mode'}

			for k, v in direct_kwargs.items():
				if v is None:
					continue
				if k in llm_fields:
					# map flat aliases to canonical LLMEffectiveConfig field names
					if k == 'llm_model':
						direct_llm['model'] = v
					elif k == 'llm_timeout':
						direct_llm['timeout'] = v
					else:
						direct_llm[k] = v
				elif k in agent_fields:
					direct_agent[k] = v
				else:
					direct_browser[k] = v

		# ---------- Normalise CLI args (BrowserProfile._normalize_cli_args handles the names) ----------
		cli_browser: dict[str, Any] = {}
		cli_llm: dict[str, Any] = {}
		cli_agent: dict[str, Any] = {}

		if cli_args:
			# Route to the right bucket before normalising browser names
			llm_cli_aliases = {
				'model': 'model',
				'llm_model': 'model',
				'provider': 'provider',
				'temperature': 'temperature',
				'api_key': 'api_key',
				'llm_timeout': 'timeout',
			}
			agent_cli_aliases = {'max_steps': 'max_steps'}

			raw_browser: dict[str, Any] = {}
			for k, v in cli_args.items():
				if v is None:
					continue
				if k in llm_cli_aliases:
					cli_llm[llm_cli_aliases[k]] = v
				elif k in agent_cli_aliases:
					cli_agent[agent_cli_aliases[k]] = v
				else:
					raw_browser[k] = v

			cli_browser = BrowserProfile._normalize_cli_args(raw_browser)

		# ---------- Layer 4: config.json (lowest) ----------
		cfg_browser: dict[str, Any] = {}
		cfg_llm: dict[str, Any] = {}
		cfg_agent: dict[str, Any] = {}

		if load_from_config_file:
			try:
				config_data = CONFIG.load_config()
				# BrowserProfileEntry also has DB metadata — strip id/default/created_at
				raw_cfg_browser = dict(config_data.get('browser_profile', {}))
				browser_meta = {'id', 'default', 'created_at'}
				cfg_browser = {k: v for k, v in raw_cfg_browser.items() if k not in browser_meta}
				# LLMEntry has DB metadata fields (id, default, created_at) — strip them
				raw_cfg_llm = dict(config_data.get('llm', {}))
				llm_allowed = {'provider', 'model', 'temperature', 'max_tokens', 'api_key', 'base_url', 'timeout'}
				cfg_llm = {k: v for k, v in raw_cfg_llm.items() if k in llm_allowed}
				# AgentEntry also has DB metadata — strip it
				raw_cfg_agent = dict(config_data.get('agent', {}))
				agent_allowed = {'max_steps', 'use_vision', 'flash_mode', 'system_prompt'}
				cfg_agent = {k: v for k, v in raw_cfg_agent.items() if k in agent_allowed}
			except Exception as e:
				from browser_use.utils import logger

				logger.debug(f'[EffectiveConfig] Failed to load config file: {e}')

		# ---------- Layer 3: environment variables ----------
		env_browser: dict[str, Any] = {}
		env_llm: dict[str, Any] = {}
		env_agent: dict[str, Any] = {}

		if load_from_env:
			try:
				env_config = FlatEnvConfig()

				# Browser env vars
				if env_config.BROWSER_USE_HEADLESS is not None:
					env_browser['headless'] = env_config.BROWSER_USE_HEADLESS
				if env_config.BROWSER_USE_ALLOWED_DOMAINS:
					domains = [d.strip() for d in env_config.BROWSER_USE_ALLOWED_DOMAINS.split(',') if d.strip()]
					env_browser['allowed_domains'] = domains

				# Proxy env vars
				proxy_dict: dict[str, Any] = {}
				if env_config.BROWSER_USE_PROXY_URL:
					proxy_dict['server'] = env_config.BROWSER_USE_PROXY_URL
				if env_config.BROWSER_USE_NO_PROXY:
					proxy_dict['bypass'] = ','.join([d.strip() for d in env_config.BROWSER_USE_NO_PROXY.split(',') if d.strip()])
				if env_config.BROWSER_USE_PROXY_USERNAME:
					proxy_dict['username'] = env_config.BROWSER_USE_PROXY_USERNAME
				if env_config.BROWSER_USE_PROXY_PASSWORD:
					proxy_dict['password'] = env_config.BROWSER_USE_PROXY_PASSWORD
				if proxy_dict:
					env_browser['proxy'] = ProxySettings(**proxy_dict)

				if env_config.BROWSER_USE_DISABLE_EXTENSIONS is not None:
					env_browser['enable_default_extensions'] = not env_config.BROWSER_USE_DISABLE_EXTENSIONS

				# LLM env vars
				if env_config.BROWSER_USE_LLM_MODEL:
					env_llm['model'] = env_config.BROWSER_USE_LLM_MODEL
				if env_config.DEFAULT_LLM:
					env_llm.setdefault('model', env_config.DEFAULT_LLM)
				if env_config.OPENAI_API_KEY:
					env_llm['openai_api_key'] = env_config.OPENAI_API_KEY
				if env_config.ANTHROPIC_API_KEY:
					env_llm['anthropic_api_key'] = env_config.ANTHROPIC_API_KEY
				if env_config.GOOGLE_API_KEY:
					env_llm['google_api_key'] = env_config.GOOGLE_API_KEY
				if env_config.DEEPSEEK_API_KEY:
					env_llm['deepseek_api_key'] = env_config.DEEPSEEK_API_KEY
			except Exception as e:
				from browser_use.utils import logger

				logger.debug(f'[EffectiveConfig] Failed to load env config: {e}')

		# ---------- Merge: config.json < env < cli < direct ----------
		merged_browser: dict[str, Any] = {}
		merged_llm: dict[str, Any] = {}
		merged_agent: dict[str, Any] = {}

		# Browser
		merged_browser.update(cfg_browser)
		merged_browser.update(env_browser)
		merged_browser.update(cli_browser)
		merged_browser.update(direct_browser)

		# LLM
		merged_llm.update(cfg_llm)
		merged_llm.update(env_llm)
		merged_llm.update(cli_llm)
		merged_llm.update(direct_llm)

		# Agent
		merged_agent.update(cfg_agent)
		merged_agent.update(env_agent)
		merged_agent.update(cli_agent)
		merged_agent.update(direct_agent)

		# ---------- Post-process browser (proxy dict → ProxySettings, cloud params nesting) ----------
		if 'proxy' in merged_browser and isinstance(merged_browser['proxy'], dict):
			merged_browser['proxy'] = ProxySettings(**merged_browser['proxy'])

		cloud_params_keys = ['cloud_profile_id', 'cloud_proxy_country_code', 'cloud_timeout']
		cloud_params_dict: dict[str, Any] = {}
		for key in cloud_params_keys:
			if key in merged_browser:
				if key == 'cloud_profile_id':
					cloud_params_dict['profile_id'] = merged_browser.pop(key)
				elif key == 'cloud_proxy_country_code':
					cloud_params_dict['proxy_country_code'] = merged_browser.pop(key)
				elif key == 'cloud_timeout':
					cloud_params_dict['timeout'] = merged_browser.pop(key)

		use_cloud = merged_browser.get('use_cloud', False)
		if cloud_params_dict or use_cloud:
			existing = merged_browser.get('cloud_browser_params')
			if existing is None:
				merged_browser['cloud_browser_params'] = CloudBrowserParams(**cloud_params_dict)
			elif isinstance(existing, dict):
				existing.update(cloud_params_dict)
				merged_browser['cloud_browser_params'] = CloudBrowserParams(**existing)

		# ---------- Post-process LLM: resolve provider + api_key from env keys ----------
		# Extract provider-specific API keys stored as separate fields in merged_llm
		api_key = merged_llm.pop('api_key', None)
		provider_api_keys = {
			'openai': merged_llm.pop('openai_api_key', None),
			'anthropic': merged_llm.pop('anthropic_api_key', None),
			'google': merged_llm.pop('google_api_key', None),
			'deepseek': merged_llm.pop('deepseek_api_key', None),
			'mistral': merged_llm.pop('mistral_api_key', None),
			'browser-use': merged_llm.pop('browser_use_api_key', None) or merged_llm.pop('browser_use_key', None),
		}

		# Determine provider from model name if not explicit
		llm_model = merged_llm.get('model')
		llm_provider = merged_llm.get('provider')

		if not llm_provider and llm_model:
			lower = llm_model.lower()
			if lower.startswith(('bu-', 'browser-use', 'chatbrowseruse')):
				llm_provider = 'browser-use'
			elif lower.startswith('gpt') or 'openai' in lower:
				llm_provider = 'openai'
			elif lower.startswith('claude'):
				llm_provider = 'anthropic'
			elif lower.startswith('gemini'):
				llm_provider = 'google'
			elif lower.startswith('deepseek'):
				llm_provider = 'deepseek'
			elif lower.startswith('mistral') or lower.startswith('codestral') or lower.startswith('pixtral'):
				llm_provider = 'mistral'
			elif '/' in llm_model:
				# e.g. "anthropic/claude-sonnet-4-0"
				prefix = llm_model.split('/', 1)[0]
				if prefix in provider_api_keys:
					llm_provider = prefix
		merged_llm['provider'] = llm_provider

		# Resolve final api_key: explicit > provider-specific env > none
		if not api_key and llm_provider and provider_api_keys.get(llm_provider):
			api_key = provider_api_keys[llm_provider]
		if api_key:
			merged_llm['api_key'] = api_key

		# Default to ChatBrowserUse when nothing else is configured
		if not merged_llm.get('model') and not merged_llm.get('provider'):
			merged_llm['provider'] = 'browser-use'
			merged_llm['model'] = 'bu-latest'
			if not merged_llm.get('api_key'):
				import os

				merged_llm['api_key'] = os.getenv('BROWSER_USE_API_KEY')

		# Default temperature
		if 'temperature' not in merged_llm or merged_llm['temperature'] is None:
			merged_llm['temperature'] = 0.0

		return cls(
			browser=merged_browser,
			llm=LLMEffectiveConfig(**{k: v for k, v in merged_llm.items() if v is not None}),
			agent=merged_agent,
		)

	def build_llm(self):
		"""Instantiate the correct BaseChatModel subclass from this effective config.

		Uses the same dispatch logic as ``browser_use.llm.models.get_llm_by_name`` but
		runs only off the already-resolved EffectiveConfig so provider + model +
		api_key are guaranteed to match the logged signature.
		"""
		from browser_use.llm.anthropic.chat import ChatAnthropic
		from browser_use.llm.browser_use.chat import ChatBrowserUse
		from browser_use.llm.cerebras.chat import ChatCerebras
		from browser_use.llm.google.chat import ChatGoogle
		from browser_use.llm.mistral.chat import ChatMistral
		from browser_use.llm.openai.chat import ChatOpenAI

		provider = (self.llm.provider or 'browser-use').lower()
		model = self.llm.model or 'bu-latest'
		temperature = self.llm.temperature if self.llm.temperature is not None else 0.0
		api_key = self.llm.api_key
		base_url = self.llm.base_url
		timeout = self.llm.timeout

		common_kwargs: dict[str, Any] = {}
		if api_key is not None:
			common_kwargs['api_key'] = api_key
		if base_url is not None:
			common_kwargs['base_url'] = base_url
		if timeout is not None:
			common_kwargs['timeout'] = timeout

		if provider in ('browser-use', 'bu'):
			return ChatBrowserUse(model=model, temperature=temperature, **common_kwargs)
		elif provider == 'openai':
			return ChatOpenAI(model=model, temperature=temperature, **common_kwargs)
		elif provider == 'anthropic':
			return ChatAnthropic(model=model, temperature=temperature, **common_kwargs)
		elif provider == 'google':
			return ChatGoogle(model=model, temperature=temperature, **common_kwargs)
		elif provider == 'mistral':
			return ChatMistral(model=model, temperature=temperature, **common_kwargs)
		elif provider == 'cerebras':
			return ChatCerebras(model=model, temperature=temperature, **common_kwargs)
		else:
			# Fall back to ChatBrowserUse for unknown providers
			return ChatBrowserUse(model=model, temperature=temperature, **common_kwargs)
