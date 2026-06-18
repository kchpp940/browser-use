from enum import Enum
from typing import Any

from pydantic import BaseModel


class StructuredOutputMethod(str, Enum):
	JSON_SCHEMA = 'json_schema'
	TOOL_CALLING = 'tool_calling'
	PROMPT_TEXT = 'prompt_text'


class ProviderCapabilities(BaseModel):
	structured_output: StructuredOutputMethod = StructuredOutputMethod.JSON_SCHEMA
	structured_output_fallback: StructuredOutputMethod | None = None
	supports_vision: bool = True
	supports_thinking: bool = False
	supports_json_schema_response_format: bool = True
	supports_tool_calling: bool = True
	supports_image_input: bool = True
	supports_streaming: bool = True
	max_retries: int = 5
	retryable_status_codes: list[int] = [429, 500, 502, 503, 504]

	needs_schema_in_prompt: bool = False

	prompt_format: str = 'openai'

	def get_structured_output_method(self) -> StructuredOutputMethod:
		if self.supports_json_schema_response_format:
			return StructuredOutputMethod.JSON_SCHEMA
		if self.supports_tool_calling:
			return StructuredOutputMethod.TOOL_CALLING
		return StructuredOutputMethod.PROMPT_TEXT

	def get_fallback_method(self) -> StructuredOutputMethod | None:
		return self.structured_output_fallback

	def is_deepseek(self) -> bool:
		return False

	def is_xai_no_vision(self) -> bool:
		return False


_DEFAULT_PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
	'openai': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': True,
		'supports_thinking': True,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 5,
		'prompt_format': 'openai',
	},
	'anthropic': {
		'structured_output': StructuredOutputMethod.TOOL_CALLING,
		'structured_output_fallback': StructuredOutputMethod.PROMPT_TEXT,
		'supports_vision': True,
		'supports_thinking': True,
		'supports_json_schema_response_format': False,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 10,
		'prompt_format': 'anthropic',
	},
	'google': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'structured_output_fallback': StructuredOutputMethod.PROMPT_TEXT,
		'supports_vision': True,
		'supports_thinking': True,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 5,
		'prompt_format': 'google',
	},
	'deepseek': {
		'structured_output': StructuredOutputMethod.TOOL_CALLING,
		'structured_output_fallback': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': False,
		'supports_thinking': False,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': False,
		'max_retries': 5,
		'prompt_format': 'openai',
	},
	'groq': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'structured_output_fallback': StructuredOutputMethod.TOOL_CALLING,
		'supports_vision': False,
		'supports_thinking': False,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': False,
		'max_retries': 10,
		'prompt_format': 'openai',
	},
	'ollama': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': True,
		'supports_thinking': False,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 3,
		'prompt_format': 'ollama',
	},
	'mistral': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': False,
		'supports_thinking': False,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': False,
		'max_retries': 5,
		'prompt_format': 'openai',
	},
	'cerebras': {
		'structured_output': StructuredOutputMethod.PROMPT_TEXT,
		'supports_vision': False,
		'supports_thinking': False,
		'supports_json_schema_response_format': False,
		'supports_tool_calling': False,
		'supports_image_input': False,
		'max_retries': 5,
		'prompt_format': 'openai',
	},
	'openrouter': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': True,
		'supports_thinking': False,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 10,
		'prompt_format': 'openai',
	},
	'vercel': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'structured_output_fallback': StructuredOutputMethod.PROMPT_TEXT,
		'supports_vision': True,
		'supports_thinking': False,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 5,
		'prompt_format': 'openai',
	},
	'azure': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': True,
		'supports_thinking': True,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 5,
		'prompt_format': 'openai',
	},
	'oci-raw': {
		'structured_output': StructuredOutputMethod.PROMPT_TEXT,
		'supports_vision': False,
		'supports_thinking': False,
		'supports_json_schema_response_format': False,
		'supports_tool_calling': False,
		'supports_image_input': False,
		'max_retries': 3,
		'prompt_format': 'oci',
	},
	'browser-use': {
		'structured_output': StructuredOutputMethod.JSON_SCHEMA,
		'supports_vision': True,
		'supports_thinking': True,
		'supports_json_schema_response_format': True,
		'supports_tool_calling': True,
		'supports_image_input': True,
		'max_retries': 5,
		'prompt_format': 'browser_use',
	},
}


def get_default_capabilities(provider: str) -> ProviderCapabilities:
	kwargs = _DEFAULT_PROVIDER_CAPABILITIES.get(provider, {})
	return ProviderCapabilities(**kwargs)
