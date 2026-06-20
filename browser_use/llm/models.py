"""
Convenient access to LLM models.

Usage:
    from browser_use import llm

    # Simple model access
    model = llm.azure_gpt_4_1_mini
    model = llm.openai_gpt_4o
    model = llm.google_gemini_2_5_pro
    model = llm.bu_latest  # or bu_1_0, bu_2_0
"""

import os
from typing import TYPE_CHECKING

# ChatBrowserUse uses httpx only (core dep), always available
from browser_use.llm.browser_use.chat import ChatBrowserUse

# --- Optional per-provider imports with friendly install hints ---
try:
	from browser_use.llm.openai.chat import ChatOpenAI

	OPENAI_AVAILABLE = True
except ImportError as _e:
	ChatOpenAI = None  # type: ignore
	OPENAI_AVAILABLE = False
	_OPENAI_MISSING = _e

try:
	from browser_use.llm.azure.chat import ChatAzureOpenAI

	AZURE_AVAILABLE = True
except ImportError as _e:
	ChatAzureOpenAI = None  # type: ignore
	AZURE_AVAILABLE = False
	_AZURE_MISSING = _e

try:
	from browser_use.llm.google.chat import ChatGoogle

	GOOGLE_AVAILABLE = True
except ImportError as _e:
	ChatGoogle = None  # type: ignore
	GOOGLE_AVAILABLE = False
	_GOOGLE_MISSING = _e

try:
	from browser_use.llm.mistral.chat import ChatMistral

	MISTRAL_AVAILABLE = True
except ImportError as _e:
	ChatMistral = None  # type: ignore
	MISTRAL_AVAILABLE = False
	_MISTRAL_MISSING = _e

try:
	from browser_use.llm.cerebras.chat import ChatCerebras

	CEREBRAS_AVAILABLE = True
except ImportError as _e:
	ChatCerebras = None  # type: ignore
	CEREBRAS_AVAILABLE = False
	_CEREBRAS_MISSING = _e

try:
	from browser_use.llm.anthropic.chat import ChatAnthropic

	ANTHROPIC_AVAILABLE = True
except ImportError as _e:
	ChatAnthropic = None  # type: ignore
	ANTHROPIC_AVAILABLE = False
	_ANTHROPIC_MISSING = _e

try:
	from browser_use.llm.oci_raw.chat import ChatOCIRaw

	OCI_AVAILABLE = True
except ImportError as _e:
	ChatOCIRaw = None  # type: ignore
	OCI_AVAILABLE = False
	_OCI_MISSING = _e

try:
	from browser_use.llm.groq.chat import ChatGroq

	GROQ_AVAILABLE = True
except ImportError as _e:
	ChatGroq = None  # type: ignore
	GROQ_AVAILABLE = False
	_GROQ_MISSING = _e

try:
	from browser_use.llm.ollama.chat import ChatOllama

	OLLAMA_AVAILABLE = True
except ImportError as _e:
	ChatOllama = None  # type: ignore
	OLLAMA_AVAILABLE = False
	_OLLAMA_MISSING = _e

try:
	from browser_use.llm.aws.chat_bedrock import ChatAWSBedrock
	from browser_use.llm.aws.chat_anthropic import ChatAnthropicBedrock

	AWS_AVAILABLE = True
except ImportError as _e:
	ChatAWSBedrock = None  # type: ignore
	ChatAnthropicBedrock = None  # type: ignore
	AWS_AVAILABLE = False
	_AWS_MISSING = _e

try:
	from browser_use.llm.deepseek.chat import ChatDeepSeek

	DEEPSEEK_AVAILABLE = True
except ImportError as _e:
	ChatDeepSeek = None  # type: ignore
	DEEPSEEK_AVAILABLE = False
	_DEEPSEEK_MISSING = _e

try:
	from browser_use.llm.openrouter.chat import ChatOpenRouter

	OPENROUTER_AVAILABLE = True
except ImportError as _e:
	ChatOpenRouter = None  # type: ignore
	OPENROUTER_AVAILABLE = False
	_OPENROUTER_MISSING = _e

try:
	from browser_use.llm.litellm.chat import ChatLiteLLM

	LITELLM_AVAILABLE = True
except ImportError as _e:
	ChatLiteLLM = None  # type: ignore
	LITELLM_AVAILABLE = False
	_LITELLM_MISSING = _e

try:
	from browser_use.llm.vercel.chat import ChatVercel

	VERCEL_AVAILABLE = True
except ImportError as _e:
	ChatVercel = None  # type: ignore
	VERCEL_AVAILABLE = False
	_VERCEL_MISSING = _e


def _require_provider(provider: str, extra: str, original_error: Exception | None):
	"""Raise a friendly ImportError directing the user to install the missing provider extra."""
	msg = (
		f'The {provider} LLM provider is not installed. '
		f'Install it with: `pip install "browser-use[{extra}]"` '
		f'or `uv pip install "browser-use[{extra}]"`'
	)
	if original_error is not None:
		msg += f'\nOriginal error: {original_error}'
	raise ImportError(msg)

if TYPE_CHECKING:
	from browser_use.llm.base import BaseChatModel

# Type stubs for IDE autocomplete
openai_gpt_4o: 'BaseChatModel'
openai_gpt_4o_mini: 'BaseChatModel'
openai_gpt_4_1_mini: 'BaseChatModel'
openai_o1: 'BaseChatModel'
openai_o1_mini: 'BaseChatModel'
openai_o1_pro: 'BaseChatModel'
openai_o3: 'BaseChatModel'
openai_o3_mini: 'BaseChatModel'
openai_o3_pro: 'BaseChatModel'
openai_o4_mini: 'BaseChatModel'
openai_gpt_5: 'BaseChatModel'
openai_gpt_5_mini: 'BaseChatModel'
openai_gpt_5_nano: 'BaseChatModel'

azure_gpt_4o: 'BaseChatModel'
azure_gpt_4o_mini: 'BaseChatModel'
azure_gpt_4_1_mini: 'BaseChatModel'
azure_o1: 'BaseChatModel'
azure_o1_mini: 'BaseChatModel'
azure_o1_pro: 'BaseChatModel'
azure_o3: 'BaseChatModel'
azure_o3_mini: 'BaseChatModel'
azure_o3_pro: 'BaseChatModel'
azure_gpt_5: 'BaseChatModel'
azure_gpt_5_mini: 'BaseChatModel'

google_gemini_2_0_flash: 'BaseChatModel'
google_gemini_2_0_pro: 'BaseChatModel'
google_gemini_2_5_pro: 'BaseChatModel'
google_gemini_2_5_flash: 'BaseChatModel'
google_gemini_2_5_flash_lite: 'BaseChatModel'
mistral_large: 'BaseChatModel'
mistral_medium: 'BaseChatModel'
mistral_small: 'BaseChatModel'
codestral: 'BaseChatModel'
pixtral_large: 'BaseChatModel'

anthropic_claude_sonnet_4_0: 'BaseChatModel'
anthropic_claude_fable_5: 'BaseChatModel'
anthropic_claude_3_5_sonnet_latest: 'BaseChatModel'
anthropic_claude_3_5_haiku_latest: 'BaseChatModel'

cerebras_llama3_1_8b: 'BaseChatModel'
cerebras_llama3_3_70b: 'BaseChatModel'
cerebras_gpt_oss_120b: 'BaseChatModel'
cerebras_llama_4_scout_17b_16e_instruct: 'BaseChatModel'
cerebras_llama_4_maverick_17b_128e_instruct: 'BaseChatModel'
cerebras_qwen_3_32b: 'BaseChatModel'
cerebras_qwen_3_235b_a22b_instruct_2507: 'BaseChatModel'
cerebras_qwen_3_235b_a22b_thinking_2507: 'BaseChatModel'
cerebras_qwen_3_coder_480b: 'BaseChatModel'

bu_latest: 'BaseChatModel'
bu_1_0: 'BaseChatModel'
bu_2_0: 'BaseChatModel'


def get_llm_by_name(model_name: str):
	"""
	Factory function to create LLM instances from string names with API keys from environment.

	Args:
	    model_name: String name like 'azure_gpt_4_1_mini', 'openai_gpt_4o', etc.

	Returns:
	    LLM instance with API keys from environment variables

	Raises:
	    ValueError: If model_name is not recognized
	"""
	if not model_name:
		raise ValueError('Model name cannot be empty')

	# Handle top-level Mistral aliases without provider prefix
	mistral_aliases = {
		'mistral_large': 'mistral-large-latest',
		'mistral_medium': 'mistral-medium-latest',
		'mistral_small': 'mistral-small-latest',
		'codestral': 'codestral-latest',
		'pixtral_large': 'pixtral-large-latest',
	}
	if model_name in mistral_aliases:
		api_key = os.getenv('MISTRAL_API_KEY')
		base_url = os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai/v1')
		return ChatMistral(model=mistral_aliases[model_name], api_key=api_key, base_url=base_url)

	# Parse model name
	parts = model_name.split('_', 1)
	if len(parts) < 2:
		raise ValueError(f"Invalid model name format: '{model_name}'. Expected format: 'provider_model_name'")

	provider = parts[0]
	model_part = parts[1]

	# Convert underscores back to dots/dashes for actual model names
	if 'gpt_4_1_mini' in model_part:
		model = model_part.replace('gpt_4_1_mini', 'gpt-4.1-mini')
	elif 'gpt_4o_mini' in model_part:
		model = model_part.replace('gpt_4o_mini', 'gpt-4o-mini')
	elif 'gpt_4o' in model_part:
		model = model_part.replace('gpt_4o', 'gpt-4o')
	elif 'gemini_2_0' in model_part:
		model = model_part.replace('gemini_2_0', 'gemini-2.0').replace('_', '-')
	elif 'gemini_2_5' in model_part:
		model = model_part.replace('gemini_2_5', 'gemini-2.5').replace('_', '-')
	elif 'llama3_1' in model_part:
		model = model_part.replace('llama3_1', 'llama3.1').replace('_', '-')
	elif 'llama3_3' in model_part:
		model = model_part.replace('llama3_3', 'llama-3.3').replace('_', '-')
	elif 'llama_4_scout' in model_part:
		model = model_part.replace('llama_4_scout', 'llama-4-scout').replace('_', '-')
	elif 'llama_4_maverick' in model_part:
		model = model_part.replace('llama_4_maverick', 'llama-4-maverick').replace('_', '-')
	elif 'gpt_oss_120b' in model_part:
		model = model_part.replace('gpt_oss_120b', 'gpt-oss-120b')
	elif 'qwen_3_32b' in model_part:
		model = model_part.replace('qwen_3_32b', 'qwen-3-32b')
	elif 'qwen_3_235b_a22b_instruct' in model_part:
		if model_part.endswith('_2507'):
			model = model_part.replace('qwen_3_235b_a22b_instruct_2507', 'qwen-3-235b-a22b-instruct-2507')
		else:
			model = model_part.replace('qwen_3_235b_a22b_instruct', 'qwen-3-235b-a22b-instruct-2507')
	elif 'qwen_3_235b_a22b_thinking' in model_part:
		if model_part.endswith('_2507'):
			model = model_part.replace('qwen_3_235b_a22b_thinking_2507', 'qwen-3-235b-a22b-thinking-2507')
		else:
			model = model_part.replace('qwen_3_235b_a22b_thinking', 'qwen-3-235b-a22b-thinking-2507')
	elif 'qwen_3_coder_480b' in model_part:
		model = model_part.replace('qwen_3_coder_480b', 'qwen-3-coder-480b')
	else:
		model = model_part.replace('_', '-')

	# OpenAI Models
	if provider == 'openai':
		if not OPENAI_AVAILABLE:
			_require_provider('OpenAI', 'llm-openai', _OPENAI_MISSING)
		api_key = os.getenv('OPENAI_API_KEY')
		return ChatOpenAI(model=model, api_key=api_key)  # type: ignore

	# Azure OpenAI Models
	elif provider == 'azure':
		if not AZURE_AVAILABLE:
			_require_provider('Azure OpenAI', 'llm-openai', _AZURE_MISSING)
		api_key = os.getenv('AZURE_OPENAI_KEY') or os.getenv('AZURE_OPENAI_API_KEY')
		azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
		return ChatAzureOpenAI(model=model, api_key=api_key, azure_endpoint=azure_endpoint)  # type: ignore

	# Google Models
	elif provider == 'google':
		if not GOOGLE_AVAILABLE:
			_require_provider('Google Gemini', 'llm-google', _GOOGLE_MISSING)
		api_key = os.getenv('GOOGLE_API_KEY')
		return ChatGoogle(model=model, api_key=api_key)  # type: ignore

	# Anthropic Models
	elif provider == 'anthropic':
		if not ANTHROPIC_AVAILABLE:
			_require_provider('Anthropic Claude', 'llm-anthropic', _ANTHROPIC_MISSING)
		api_key = os.getenv('ANTHROPIC_API_KEY')
		return ChatAnthropic(model=model, api_key=api_key)  # type: ignore

	# Mistral Models
	elif provider == 'mistral':
		if not MISTRAL_AVAILABLE:
			_require_provider('Mistral', 'llm-openai', _MISTRAL_MISSING)
		api_key = os.getenv('MISTRAL_API_KEY')
		base_url = os.getenv('MISTRAL_BASE_URL', 'https://api.mistral.ai/v1')
		mistral_map = {
			'large': 'mistral-large-latest',
			'medium': 'mistral-medium-latest',
			'small': 'mistral-small-latest',
			'codestral': 'codestral-latest',
			'pixtral-large': 'pixtral-large-latest',
		}
		normalized_model_part = model_part.replace('_', '-')
		resolved_model = mistral_map.get(normalized_model_part, model.replace('_', '-'))
		return ChatMistral(model=resolved_model, api_key=api_key, base_url=base_url)  # type: ignore

	# OCI Models
	elif provider == 'oci':
		if not OCI_AVAILABLE:
			_require_provider('OCI', 'llm-oci', _OCI_MISSING)
		raise ValueError('OCI models require manual configuration. Use ChatOCIRaw directly with your OCI credentials.')

	# Cerebras Models
	elif provider == 'cerebras':
		if not CEREBRAS_AVAILABLE:
			_require_provider('Cerebras', 'llm-openai', _CEREBRAS_MISSING)
		api_key = os.getenv('CEREBRAS_API_KEY')
		return ChatCerebras(model=model, api_key=api_key)  # type: ignore

	# Browser Use Models
	elif provider == 'bu':
		model = f'bu-{model_part.replace("_", "-")}'
		api_key = os.getenv('BROWSER_USE_API_KEY')
		return ChatBrowserUse(model=model, api_key=api_key)

	else:
		available_providers = ['openai', 'azure', 'google', 'anthropic', 'mistral', 'oci', 'cerebras', 'bu']
		raise ValueError(f"Unknown provider: '{provider}'. Available providers: {', '.join(available_providers)}")


# Pre-configured model instances (lazy loaded via __getattr__)
def __getattr__(name: str) -> 'BaseChatModel':
	"""Create model instances on demand with API keys from environment."""
	# Handle chat classes first
	if name == 'ChatOpenAI':
		if not OPENAI_AVAILABLE:
			_require_provider('OpenAI', 'llm-openai', _OPENAI_MISSING)
		return ChatOpenAI  # type: ignore
	elif name == 'ChatAzureOpenAI':
		if not AZURE_AVAILABLE:
			_require_provider('Azure OpenAI', 'llm-openai', _AZURE_MISSING)
		return ChatAzureOpenAI  # type: ignore
	elif name == 'ChatGoogle':
		if not GOOGLE_AVAILABLE:
			_require_provider('Google Gemini', 'llm-google', _GOOGLE_MISSING)
		return ChatGoogle  # type: ignore
	elif name == 'ChatAnthropic':
		if not ANTHROPIC_AVAILABLE:
			_require_provider('Anthropic Claude', 'llm-anthropic', _ANTHROPIC_MISSING)
		return ChatAnthropic  # type: ignore
	elif name == 'ChatMistral':
		if not MISTRAL_AVAILABLE:
			_require_provider('Mistral', 'llm-openai', _MISTRAL_MISSING)
		return ChatMistral  # type: ignore
	elif name == 'ChatOCIRaw':
		if not OCI_AVAILABLE:
			_require_provider('OCI', 'llm-oci', _OCI_MISSING)
		return ChatOCIRaw  # type: ignore
	elif name == 'ChatCerebras':
		if not CEREBRAS_AVAILABLE:
			_require_provider('Cerebras', 'llm-openai', _CEREBRAS_MISSING)
		return ChatCerebras  # type: ignore
	elif name == 'ChatBrowserUse':
		return ChatBrowserUse  # type: ignore

	# Handle model instances - these are the main use case
	try:
		return get_llm_by_name(name)
	except ValueError:
		raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all classes and preconfigured instances, conditionally including optional providers
__all__ = [
	'ChatBrowserUse',
]

if OPENAI_AVAILABLE:
	__all__.append('ChatOpenAI')
if AZURE_AVAILABLE:
	__all__.append('ChatAzureOpenAI')
if GOOGLE_AVAILABLE:
	__all__.append('ChatGoogle')
if ANTHROPIC_AVAILABLE:
	__all__.append('ChatAnthropic')
if MISTRAL_AVAILABLE:
	__all__.append('ChatMistral')
if CEREBRAS_AVAILABLE:
	__all__.append('ChatCerebras')
if OCI_AVAILABLE:
	__all__.append('ChatOCIRaw')

__all__ += [
	'get_llm_by_name',
	# OpenAI instances - created on demand
	'openai_gpt_4o',
	'openai_gpt_4o_mini',
	'openai_gpt_4_1_mini',
	'openai_o1',
	'openai_o1_mini',
	'openai_o1_pro',
	'openai_o3',
	'openai_o3_mini',
	'openai_o3_pro',
	'openai_o4_mini',
	'openai_gpt_5',
	'openai_gpt_5_mini',
	'openai_gpt_5_nano',
	# Azure instances - created on demand
	'azure_gpt_4o',
	'azure_gpt_4o_mini',
	'azure_gpt_4_1_mini',
	'azure_o1',
	'azure_o1_mini',
	'azure_o1_pro',
	'azure_o3',
	'azure_o3_mini',
	'azure_o3_pro',
	'azure_gpt_5',
	'azure_gpt_5_mini',
	# Google instances - created on demand
	'google_gemini_2_0_flash',
	'google_gemini_2_0_pro',
	'google_gemini_2_5_pro',
	'google_gemini_2_5_flash',
	'google_gemini_2_5_flash_lite',
	# Anthropic instances - created on demand
	'anthropic_claude_sonnet_4_0',
	'anthropic_claude_fable_5',
	'anthropic_claude_3_5_sonnet_latest',
	'anthropic_claude_3_5_haiku_latest',
	# Mistral instances - created on demand
	'mistral_large',
	'mistral_medium',
	'mistral_small',
	'codestral',
	'pixtral_large',
	# Cerebras instances - created on demand
	'cerebras_llama3_1_8b',
	'cerebras_llama3_3_70b',
	'cerebras_gpt_oss_120b',
	'cerebras_llama_4_scout_17b_16e_instruct',
	'cerebras_llama_4_maverick_17b_128e_instruct',
	'cerebras_qwen_3_32b',
	'cerebras_qwen_3_235b_a22b_instruct_2507',
	'cerebras_qwen_3_235b_a22b_thinking_2507',
	'cerebras_qwen_3_coder_480b',
	# Browser Use instances - created on demand
	'bu_latest',
	'bu_1_0',
	'bu_2_0',
]

# NOTE: OCI backend is optional. The try/except ImportError and conditional __all__ are required
# so this module can be imported without browser-use[oci] installed.
