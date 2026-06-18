import json
import logging
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


class StructuredOutputMethod(str, Enum):
	JSON_SCHEMA = 'json_schema'
	TOOL_CALLING = 'tool_calling'
	PROMPT_TEXT = 'prompt_text'


def _extract_balanced(text: str, start_idx: int, open_char: str, close_char: str) -> str | None:
	"""Extract a balanced bracket expression starting at start_idx."""
	depth = 0
	i = start_idx
	while i < len(text):
		ch = text[i]
		if ch == open_char:
			depth += 1
		elif ch == close_char:
			depth -= 1
			if depth == 0:
				return text[start_idx : i + 1]
		i += 1
	return None


def extract_json_candidates(text: str) -> list[str]:
	"""
	Extract potential JSON objects/arrays from a text response.

	Tries multiple strategies in order:
	1. The full stripped text
	2. Content inside markdown code blocks (```json ... ``` or ``` ... ```)
	3. Balanced {...} or [...] expressions found in the text
	"""
	import re

	candidates: list[str] = []
	stripped = text.strip()
	if stripped:
		candidates.append(stripped)

	# Extract all markdown code blocks (anywhere in the text)
	fence_pattern = re.compile(r'```(?:json)?\s*\n(.*?)```', re.DOTALL | re.IGNORECASE)
	for match in fence_pattern.finditer(text):
		code_content = match.group(1).strip()
		if code_content:
			candidates.append(code_content)

	# Extract balanced {...} and [...] expressions
	for open_char, close_char in (('{', '}'), ('[', ']')):
		search_start = 0
		while True:
			start_idx = text.find(open_char, search_start)
			if start_idx == -1:
				break
			balanced = _extract_balanced(text, start_idx, open_char, close_char)
			if balanced is not None:
				candidates.append(balanced)
				search_start = start_idx + len(balanced)
			else:
				search_start = start_idx + 1

	return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def parse_structured_output_from_text(text: str, output_format: type[T]) -> T | None:
	"""
	Try to parse a Pydantic model from a text response by extracting JSON candidates.

	Returns the parsed model instance, or None if no candidate could be parsed.
	"""
	for candidate in extract_json_candidates(text):
		try:
			return output_format.model_validate_json(candidate)
		except Exception:
			try:
				return output_format.model_validate(json.loads(candidate))
			except Exception:
				continue
	return None


def build_prompt_text_schema_instruction(output_format: type[BaseModel]) -> str:
	"""
	Build the schema instruction text to append to a prompt when using
	PROMPT_TEXT structured output fallback.
	"""
	from browser_use.llm.schema import SchemaOptimizer

	schema = SchemaOptimizer.create_optimized_json_schema(output_format)
	schema_json = json.dumps(schema, ensure_ascii=False)
	return f'\n\nPlease respond ONLY with a valid JSON object matching this schema (no extra commentary, no markdown code blocks, no explanation text):\n{schema_json}'


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
		"""
		Return the primary structured output method.

		Precedence:
		1. The explicitly declared ``structured_output`` field on the capabilities.
		2. JSON_SCHEMA if ``supports_json_schema_response_format`` is True.
		3. TOOL_CALLING if ``supports_tool_calling`` is True.
		4. PROMPT_TEXT as the final fallback.
		"""
		if self.structured_output:
			return self.structured_output
		if self.supports_json_schema_response_format:
			return StructuredOutputMethod.JSON_SCHEMA
		if self.supports_tool_calling:
			return StructuredOutputMethod.TOOL_CALLING
		return StructuredOutputMethod.PROMPT_TEXT

	def get_fallback_method(self) -> StructuredOutputMethod | None:
		return self.structured_output_fallback

	def get_structured_output_strategy_chain(self) -> list[StructuredOutputMethod]:
		"""
		Return an ordered list of structured output strategies to try, including fallbacks.

		The list is de-duplicated while preserving order. For example:
		- OpenAI: [JSON_SCHEMA]
		- Anthropic: [TOOL_CALLING, PROMPT_TEXT]
		- DeepSeek: [TOOL_CALLING, JSON_SCHEMA]
		- Cerebras: [PROMPT_TEXT]
		"""
		chain: list[StructuredOutputMethod] = []
		primary = self.get_structured_output_method()
		chain.append(primary)
		fallback = self.get_fallback_method()
		if fallback is not None and fallback not in chain:
			chain.append(fallback)
		if StructuredOutputMethod.PROMPT_TEXT not in chain:
			chain.append(StructuredOutputMethod.PROMPT_TEXT)
		return chain

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
