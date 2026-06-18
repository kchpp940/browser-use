"""
ChatLiteLLM - LiteLLM chat model wrapper.

Requires the `litellm` package to be installed separately:
    pip install litellm

Note: litellm is NOT included as a dependency of browser-use.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.capabilities import (
	ProviderCapabilities,
	StructuredOutputMethod,
	build_prompt_text_schema_instruction,
	get_default_capabilities,
	parse_structured_output_from_text,
)
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage, ContentPartTextParam
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

from .serializer import LiteLLMMessageSerializer

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatLiteLLM(BaseChatModel):
	model: str
	api_key: str | None = None
	api_base: str | None = None
	temperature: float | None = 0.0
	max_tokens: int | None = 4096
	max_retries: int = 3
	metadata: dict[str, Any] | None = None

	_provider_name: str = field(default='', init=False, repr=False)
	_clean_model: str = field(default='', init=False, repr=False)

	def __post_init__(self) -> None:
		"""Resolve provider info from the model string via litellm."""
		try:
			from litellm import get_llm_provider  # type: ignore[reportMissingImports]

			self._clean_model, self._provider_name, _, _ = get_llm_provider(self.model)
		except Exception:
			if '/' in self.model:
				self._provider_name, self._clean_model = self.model.split('/', 1)
			else:
				self._provider_name = 'openai'
				self._clean_model = self.model

		logger.debug(
			'ChatLiteLLM initialized: model=%s, provider=%s, clean=%s, api_base=%s',
			self.model,
			self._provider_name,
			self._clean_model,
			self.api_base or '(default)',
		)

	@property
	def provider(self) -> str:
		return self._provider_name or 'litellm'

	@property
	def capabilities(self) -> ProviderCapabilities:
		provider_key = self._provider_name or 'litellm'
		if provider_key in ('openai', 'azure'):
			return get_default_capabilities(provider_key)
		return ProviderCapabilities(
			structured_output=StructuredOutputMethod.JSON_SCHEMA,
			supports_vision=True,
			supports_thinking=False,
			supports_json_schema_response_format=True,
			supports_tool_calling=True,
			supports_image_input=True,
			max_retries=self.max_retries,
			prompt_format='openai',
		)

	@property
	def name(self) -> str:
		return self._clean_model or self.model

	@staticmethod
	def _parse_usage(response: Any) -> ChatInvokeUsage | None:
		"""Extract token usage from a litellm response."""
		usage = getattr(response, 'usage', None)
		if usage is None:
			return None

		prompt_tokens = getattr(usage, 'prompt_tokens', 0) or 0
		completion_tokens = getattr(usage, 'completion_tokens', 0) or 0

		prompt_cached = getattr(usage, 'cache_read_input_tokens', None)
		cache_creation = getattr(usage, 'cache_creation_input_tokens', None)

		if prompt_cached is None:
			details = getattr(usage, 'prompt_tokens_details', None)
			if details:
				prompt_cached = getattr(details, 'cached_tokens', None)

		return ChatInvokeUsage(
			prompt_tokens=prompt_tokens,
			prompt_cached_tokens=int(prompt_cached) if prompt_cached is not None else None,
			prompt_cache_creation_tokens=int(cache_creation) if cache_creation is not None else None,
			prompt_image_tokens=None,
			completion_tokens=completion_tokens,
			total_tokens=prompt_tokens + completion_tokens,
		)

	@overload
	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: None = None,
		structured_output_method: StructuredOutputMethod | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T],
		structured_output_method: StructuredOutputMethod | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T]: ...

	def _build_base_params(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
		params: dict[str, Any] = {
			'model': self.model,
			'messages': messages,
			'num_retries': self.max_retries,
		}
		if self.temperature is not None:
			params['temperature'] = self.temperature
		if self.max_tokens is not None:
			params['max_tokens'] = self.max_tokens
		if self.api_key:
			params['api_key'] = self.api_key
		if self.api_base:
			params['api_base'] = self.api_base
		if self.metadata:
			params['metadata'] = self.metadata
		return params

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		structured_output_method: StructuredOutputMethod | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		from litellm import acompletion  # type: ignore[reportMissingImports]
		from litellm.exceptions import APIConnectionError, APIError, RateLimitError, Timeout  # type: ignore[reportMissingImports]
		from litellm.types.utils import ModelResponse  # type: ignore[reportMissingImports]

		base_litellm_messages = LiteLLMMessageSerializer.serialize(messages)

		try:
			if output_format is None:
				params = self._build_base_params(base_litellm_messages)
				raw_response = await acompletion(**params)
				assert isinstance(raw_response, ModelResponse), f'Expected ModelResponse, got {type(raw_response)}'
				response: ModelResponse = raw_response

				choice = response.choices[0] if response.choices else None
				if choice is None:
					raise ModelProviderError(
						message='Empty response: no choices returned by the model',
						status_code=502,
						model=self.name,
					)
				content = choice.message.content or ''
				usage = self._parse_usage(response)
				stop_reason = choice.finish_reason
				thinking: str | None = None
				reasoning = getattr(choice.message, 'reasoning_content', None)
				if reasoning:
					thinking = str(reasoning)
				return ChatInvokeCompletion(
					completion=content,
					thinking=thinking,
					usage=usage,
					stop_reason=stop_reason,
				)

			else:
				strategy_chain = self.capabilities.get_structured_output_strategy_chain()
				strategy = structured_output_method if structured_output_method is not None else strategy_chain[0]

				if strategy == StructuredOutputMethod.JSON_SCHEMA:
					schema = SchemaOptimizer.create_optimized_json_schema(output_format)
					params = self._build_base_params(base_litellm_messages)
					params['response_format'] = {
						'type': 'json_schema',
						'json_schema': {
							'name': output_format.__name__,
							'strict': True,
							'schema': schema,
						},
					}
					raw_response = await acompletion(**params)
					assert isinstance(raw_response, ModelResponse)
					choice = raw_response.choices[0] if raw_response.choices else None
					if choice is None or not (choice.message.content or ''):
						raise ValueError('Empty response for JSON_SCHEMA strategy')
					content = choice.message.content or ''
					usage = self._parse_usage(raw_response)
					stop_reason = choice.finish_reason
					thinking: str | None = None
					reasoning = getattr(choice.message, 'reasoning_content', None)
					if reasoning:
						thinking = str(reasoning)
					parsed = output_format.model_validate_json(content)
					return ChatInvokeCompletion(
						completion=parsed,
						thinking=thinking,
						usage=usage,
						stop_reason=stop_reason,
					)

				elif strategy == StructuredOutputMethod.TOOL_CALLING:
					schema = SchemaOptimizer.create_optimized_json_schema(output_format)
					tool_spec = {
						'type': 'function',
						'function': {
							'name': output_format.__name__,
							'description': f'Extract information in the format of {output_format.__name__}',
							'strict': True,
							'parameters': schema,
						},
					}
					params = self._build_base_params(base_litellm_messages)
					params['tools'] = [tool_spec]
					params['tool_choice'] = {'type': 'function', 'function': {'name': output_format.__name__}}
					raw_response = await acompletion(**params)
					assert isinstance(raw_response, ModelResponse)
					choice = raw_response.choices[0] if raw_response.choices else None
					if choice is None:
						raise ValueError('Empty response for TOOL_CALLING strategy')
					msg = choice.message
					usage = self._parse_usage(raw_response)
					stop_reason = choice.finish_reason
					thinking: str | None = None
					reasoning = getattr(msg, 'reasoning_content', None)
					if reasoning:
						thinking = str(reasoning)
					tool_calls = getattr(msg, 'tool_calls', None)
					if tool_calls and len(tool_calls) > 0:
						args_str = getattr(tool_calls[0].function, 'arguments', '')
						parsed = output_format.model_validate_json(args_str)
						return ChatInvokeCompletion(
							completion=parsed,
							thinking=thinking,
							usage=usage,
							stop_reason=stop_reason,
						)
					raise ValueError('No tool calls found in response')

				elif strategy == StructuredOutputMethod.PROMPT_TEXT:
					modified_messages = [m.model_copy(deep=True) for m in messages]
					instruction_added = False
					if modified_messages and isinstance(modified_messages[-1].content, str):
						modified_messages[-1].content += build_prompt_text_schema_instruction(output_format)
						instruction_added = True
					elif modified_messages and isinstance(modified_messages[-1].content, list):
						modified_messages[-1].content.append(
							ContentPartTextParam(text=build_prompt_text_schema_instruction(output_format))
						)
						instruction_added = True
					if not instruction_added and modified_messages and isinstance(modified_messages[0].content, str):
						modified_messages[0].content += build_prompt_text_schema_instruction(output_format)

					fallback_litellm_messages = LiteLLMMessageSerializer.serialize(modified_messages)
					params = self._build_base_params(fallback_litellm_messages)
					raw_response = await acompletion(**params)
					assert isinstance(raw_response, ModelResponse)
					choice = raw_response.choices[0] if raw_response.choices else None
					if choice is None or not (choice.message.content or ''):
						raise ValueError('Empty response for PROMPT_TEXT strategy')
					content = choice.message.content or ''
					usage = self._parse_usage(raw_response)
					stop_reason = choice.finish_reason
					thinking: str | None = None
					reasoning = getattr(choice.message, 'reasoning_content', None)
					if reasoning:
						thinking = str(reasoning)
					parsed = parse_structured_output_from_text(content, output_format)
					if parsed is not None:
						return ChatInvokeCompletion(
							completion=parsed,
							thinking=thinking,
							usage=usage,
							stop_reason=stop_reason,
						)
					raise ValueError('Failed to parse structured output from prompt text response')

				else:
					raise ValueError(f'Unsupported structured_output_method: {strategy}')

		except RateLimitError as e:
			raise ModelRateLimitError(message=str(e), model=self.name) from e
		except Timeout as e:
			raise ModelProviderError(message=f'Request timed out: {e}', model=self.name) from e
		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
		except APIError as e:
			status = getattr(e, 'status_code', 502) or 502
			raise ModelProviderError(message=str(e), status_code=status, model=self.name) from e
		except ModelProviderError:
			raise
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
