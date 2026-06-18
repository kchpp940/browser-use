import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.shared_params.response_format_json_schema import (
	JSONSchema,
	ResponseFormatJSONSchema,
)
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
from browser_use.llm.openrouter.serializer import OpenRouterMessageSerializer
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOpenRouter(BaseChatModel):
	"""
	A wrapper around OpenRouter's chat API, which provides access to various LLM models
	through a unified OpenAI-compatible interface.

	This class implements the BaseChatModel protocol for OpenRouter's API.
	"""

	# Model configuration
	model: str

	# Model params
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	# Client initialization parameters
	api_key: str | None = None
	http_referer: str | None = None  # OpenRouter specific parameter for tracking
	base_url: str | httpx.URL = 'https://openrouter.ai/api/v1'
	timeout: float | httpx.Timeout | None = None
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None
	http_client: httpx.AsyncClient | None = None
	_strict_response_validation: bool = False
	extra_body: dict[str, Any] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'openrouter'

	@property
	def capabilities(self) -> ProviderCapabilities:
		return get_default_capabilities('openrouter')

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		# Define base client params
		base_params = {
			'api_key': self.api_key,
			'base_url': self.base_url,
			'timeout': self.timeout,
			'max_retries': self.max_retries,
			'default_headers': self.default_headers,
			'default_query': self.default_query,
			'_strict_response_validation': self._strict_response_validation,
			'top_p': self.top_p,
			'seed': self.seed,
		}

		# Create client_params dict with non-None values
		client_params = {k: v for k, v in base_params.items() if v is not None}

		# Add http_client if provided
		if self.http_client is not None:
			client_params['http_client'] = self.http_client

		return client_params

	def get_client(self) -> AsyncOpenAI:
		"""
		Returns an AsyncOpenAI client configured for OpenRouter.

		Returns:
		    AsyncOpenAI: An instance of the AsyncOpenAI client with OpenRouter base URL.
		"""
		if not hasattr(self, '_client'):
			client_params = self._get_client_params()
			self._client = AsyncOpenAI(**client_params)
		return self._client

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		"""Extract usage information from the OpenRouter response."""
		if response.usage is None:
			return None

		prompt_details = getattr(response.usage, 'prompt_tokens_details', None)
		cached_tokens = prompt_details.cached_tokens if prompt_details else None

		return ChatInvokeUsage(
			prompt_tokens=response.usage.prompt_tokens,
			prompt_cached_tokens=cached_tokens,
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
			# Completion
			completion_tokens=response.usage.completion_tokens,
			total_tokens=response.usage.total_tokens,
		)

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	def _build_model_params(self, extra_headers: dict[str, str]) -> dict[str, Any]:
		model_params: dict[str, Any] = {'extra_headers': extra_headers}
		if self.temperature is not None:
			model_params['temperature'] = self.temperature
		if self.top_p is not None:
			model_params['top_p'] = self.top_p
		if self.seed is not None:
			model_params['seed'] = self.seed
		if self.extra_body:
			model_params.update(self.extra_body)
		return model_params

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the model with the given messages through OpenRouter.

		Args:
		    messages: List of chat messages
		    output_format: Optional Pydantic model class for structured output

		Returns:
		    Either a string response or an instance of output_format
		"""
		openrouter_messages = OpenRouterMessageSerializer.serialize_messages(messages)

		extra_headers = {}
		if self.http_referer:
			extra_headers['HTTP-Referer'] = self.http_referer

		model_params = self._build_model_params(extra_headers)

		try:
			if output_format is None:
				response = await self.get_client().chat.completions.create(
					model=self.model,
					messages=openrouter_messages,
					**model_params,
				)

				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=response.choices[0].message.content or '',
					usage=usage,
				)

			else:
				strategy_chain = self.capabilities.get_structured_output_strategy_chain()
				last_error: Exception | None = None

				for strategy in strategy_chain:
					try:
						if strategy == StructuredOutputMethod.JSON_SCHEMA:
							schema = SchemaOptimizer.create_optimized_json_schema(output_format)
							response_format_schema: JSONSchema = {
								'name': 'agent_output',
								'strict': True,
								'schema': schema,
							}

							response = await self.get_client().chat.completions.create(
								model=self.model,
								messages=openrouter_messages,
								response_format=ResponseFormatJSONSchema(
									json_schema=response_format_schema,
									type='json_schema',
								),
								**model_params,
							)

							if response.choices[0].message.content is None:
								raise ValueError('Empty content in JSON schema response')

							usage = self._get_usage(response)
							parsed = output_format.model_validate_json(response.choices[0].message.content)
							return ChatInvokeCompletion(
								completion=parsed,
								usage=usage,
							)

						elif strategy == StructuredOutputMethod.PROMPT_TEXT:
							modified_messages = [m.model_copy(deep=True) for m in messages]
							instruction_added = False
							if modified_messages and isinstance(modified_messages[0].content, str):
								modified_messages[0].content += build_prompt_text_schema_instruction(output_format)
								instruction_added = True
							elif modified_messages and isinstance(modified_messages[0].content, list):
								modified_messages[0].content.append(
									ContentPartTextParam(text=build_prompt_text_schema_instruction(output_format))
								)
								instruction_added = True
							if not instruction_added and modified_messages and isinstance(modified_messages[-1].content, str):
								modified_messages[-1].content += build_prompt_text_schema_instruction(output_format)
								instruction_added = True

							modified_openrouter_messages = OpenRouterMessageSerializer.serialize_messages(modified_messages)

							response = await self.get_client().chat.completions.create(
								model=self.model,
								messages=modified_openrouter_messages,
								**model_params,
							)

							usage = self._get_usage(response)
							content = response.choices[0].message.content or ''
							parsed = parse_structured_output_from_text(content, output_format)
							if parsed is not None:
								return ChatInvokeCompletion(
									completion=parsed,
									usage=usage,
								)
							raise ValueError('Failed to parse structured output from prompt text response')

					except Exception as e:
						last_error = e
						logger.debug(f'OpenRouter structured output strategy {strategy} failed: {e}')
						continue

				if last_error is not None:
					raise ModelProviderError(
						message=f'All structured output strategies failed. Last error: {last_error}',
						model=self.name,
					) from last_error
				raise ModelProviderError(
					message='No valid structured output strategy available for OpenRouter',
					model=self.name,
				)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e

		except APIConnectionError as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
