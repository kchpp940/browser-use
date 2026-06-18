import logging
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, overload

from groq import (
	APIError,
	APIResponseValidationError,
	APIStatusError,
	AsyncGroq,
	NotGiven,
	RateLimitError,
	Timeout,
)
from groq.types.chat import ChatCompletion, ChatCompletionToolChoiceOptionParam, ChatCompletionToolParam
from groq.types.chat.completion_create_params import (
	ResponseFormatResponseFormatJsonSchema,
	ResponseFormatResponseFormatJsonSchemaJsonSchema,
)
from httpx import URL
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel, ChatInvokeCompletion
from browser_use.llm.capabilities import (
	ProviderCapabilities,
	StructuredOutputMethod,
	build_prompt_text_schema_instruction,
	get_default_capabilities,
	parse_structured_output_from_text,
)
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.groq.parser import try_parse_groq_failed_generation
from browser_use.llm.groq.serializer import GroqMessageSerializer
from browser_use.llm.messages import BaseMessage
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeUsage

GroqVerifiedModels = Literal[
	'meta-llama/llama-4-maverick-17b-128e-instruct',
	'meta-llama/llama-4-scout-17b-16e-instruct',
	'qwen/qwen3-32b',
	'moonshotai/kimi-k2-instruct',
	'openai/gpt-oss-20b',
	'openai/gpt-oss-120b',
]

JsonSchemaModels = [
	'meta-llama/llama-4-maverick-17b-128e-instruct',
	'meta-llama/llama-4-scout-17b-16e-instruct',
	'openai/gpt-oss-20b',
	'openai/gpt-oss-120b',
]

ToolCallingModels = [
	'moonshotai/kimi-k2-instruct',
]

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class ChatGroq(BaseChatModel):
	"""
	A wrapper around AsyncGroq that implements the BaseLLM protocol.
	"""

	# Model configuration
	model: GroqVerifiedModels | str

	# Model params
	temperature: float | None = None
	service_tier: Literal['auto', 'on_demand', 'flex'] | None = None
	top_p: float | None = None
	seed: int | None = None

	# Client initialization parameters
	api_key: str | None = None
	base_url: str | URL | None = None
	timeout: float | Timeout | NotGiven | None = None
	max_retries: int = 10  # Increase default retries for automation reliability

	def get_client(self) -> AsyncGroq:
		return AsyncGroq(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries)

	@property
	def provider(self) -> str:
		return 'groq'

	@property
	def capabilities(self) -> ProviderCapabilities:
		base = get_default_capabilities('groq')
		if self.model in ToolCallingModels:
			return base.model_copy(
				update={
					'structured_output': StructuredOutputMethod.TOOL_CALLING,
					'structured_output_fallback': StructuredOutputMethod.JSON_SCHEMA,
				}
			)
		return base

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		usage = (
			ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				completion_tokens=response.usage.completion_tokens,
				total_tokens=response.usage.total_tokens,
				prompt_cached_tokens=None,  # Groq doesn't support cached tokens
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
			)
			if response.usage is not None
			else None
		)
		return usage

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

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		structured_output_method: StructuredOutputMethod | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		groq_messages = GroqMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				return await self._invoke_regular_completion(groq_messages)
			else:
				return await self._invoke_structured_output(
					groq_messages, output_format, messages, structured_output_method
				)

		except RateLimitError as e:
			raise ModelRateLimitError(message=e.response.text, status_code=e.response.status_code, model=self.name) from e

		except APIResponseValidationError as e:
			raise ModelProviderError(message=e.response.text, status_code=e.response.status_code, model=self.name) from e

		except APIStatusError as e:
			if output_format is None:
				raise ModelProviderError(message=e.response.text, status_code=e.response.status_code, model=self.name) from e
			else:
				try:
					logger.debug(f'Groq failed generation: {e.response.text}; fallback to manual parsing')

					parsed_response = try_parse_groq_failed_generation(e, output_format)

					logger.debug('Manual error parsing successful ✅')

					return ChatInvokeCompletion(
						completion=parsed_response,
						usage=None,  # because this is a hacky way to get the outputs
						# TODO: @groq needs to fix their parsers and validators
					)
				except Exception as _:
					raise ModelProviderError(message=str(e), status_code=e.response.status_code, model=self.name) from e

		except APIError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e

	async def _invoke_regular_completion(self, groq_messages) -> ChatInvokeCompletion[str]:
		"""Handle regular completion without structured output."""
		chat_completion = await self.get_client().chat.completions.create(
			messages=groq_messages,
			model=self.model,
			service_tier=self.service_tier,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
		)
		usage = self._get_usage(chat_completion)
		return ChatInvokeCompletion(
			completion=chat_completion.choices[0].message.content or '',
			usage=usage,
		)

	async def _invoke_structured_output(
		self,
		groq_messages,
		output_format: type[T],
		original_messages: list[BaseMessage],
		structured_output_method: StructuredOutputMethod | None = None,
	) -> ChatInvokeCompletion[T]:
		"""Handle structured output using capabilities-driven strategy chain."""
		schema = SchemaOptimizer.create_optimized_json_schema(output_format)

		if structured_output_method is None:
			strategy = self.capabilities.get_structured_output_strategy_chain()[0]
		else:
			strategy = structured_output_method

		if strategy == StructuredOutputMethod.TOOL_CALLING:
			response = await self._invoke_with_tool_calling(groq_messages, output_format, schema)
			if not response.choices[0].message.content:
				raise ValueError('No content in tool calling response')
			parsed_response = output_format.model_validate_json(response.choices[0].message.content)
			usage = self._get_usage(response)
			return ChatInvokeCompletion(
				completion=parsed_response,
				usage=usage,
			)

		elif strategy == StructuredOutputMethod.JSON_SCHEMA:
			response = await self._invoke_with_json_schema(groq_messages, output_format, schema)
			if not response.choices[0].message.content:
				raise ValueError('No content in JSON schema response')
			parsed_response = output_format.model_validate_json(response.choices[0].message.content)
			usage = self._get_usage(response)
			return ChatInvokeCompletion(
				completion=parsed_response,
				usage=usage,
			)

		elif strategy == StructuredOutputMethod.PROMPT_TEXT:
			modified_messages = [m.model_copy(deep=True) for m in original_messages]
			if modified_messages and isinstance(modified_messages[-1].content, str):
				modified_messages[-1].content += build_prompt_text_schema_instruction(output_format)
			fallback_groq_messages = GroqMessageSerializer.serialize_messages(modified_messages)
			response = await self.get_client().chat.completions.create(
				model=self.model,
				messages=fallback_groq_messages,
				temperature=self.temperature,
				top_p=self.top_p,
				seed=self.seed,
				service_tier=self.service_tier,
			)
			content = response.choices[0].message.content
			if not content:
				raise ValueError('No content in prompt text fallback')
			parsed = parse_structured_output_from_text(content, output_format)
			if parsed is not None:
				usage = self._get_usage(response)
				return ChatInvokeCompletion(
					completion=parsed,
					usage=usage,
				)
			raise ValueError(f'Failed to parse prompt text fallback: {content[:200]}')

		else:
			raise ValueError(f'Unsupported structured output strategy: {strategy}')

	async def _invoke_with_tool_calling(self, groq_messages, output_format: type[T], schema) -> ChatCompletion:
		"""Handle structured output using tool calling."""
		tool = ChatCompletionToolParam(
			function={
				'name': output_format.__name__,
				'description': f'Extract information in the format of {output_format.__name__}',
				'parameters': schema,
			},
			type='function',
		)
		tool_choice: ChatCompletionToolChoiceOptionParam = 'required'

		return await self.get_client().chat.completions.create(
			model=self.model,
			messages=groq_messages,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
			tools=[tool],
			tool_choice=tool_choice,
			service_tier=self.service_tier,
		)

	async def _invoke_with_json_schema(self, groq_messages, output_format: type[T], schema) -> ChatCompletion:
		"""Handle structured output using JSON schema."""
		return await self.get_client().chat.completions.create(
			model=self.model,
			messages=groq_messages,
			temperature=self.temperature,
			top_p=self.top_p,
			seed=self.seed,
			response_format=ResponseFormatResponseFormatJsonSchema(
				json_schema=ResponseFormatResponseFormatJsonSchemaJsonSchema(
					name=output_format.__name__,
					description='Model output schema',
					schema=schema,
				),
				type='json_schema',
			),
			service_tier=self.service_tier,
		)
