import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

from anthropic import (
	APIConnectionError,
	APIStatusError,
	AsyncAnthropicBedrock,
	RateLimitError,
	omit,
)
from anthropic.types import CacheControlEphemeralParam, Message, ToolParam
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_choice_tool_param import ToolChoiceToolParam
from pydantic import BaseModel

from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
from browser_use.llm.aws.chat_bedrock import ChatAWSBedrock
from browser_use.llm.capabilities import (
	ProviderCapabilities,
	StructuredOutputMethod,
	build_prompt_text_schema_instruction,
	parse_structured_output_from_text,
)
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage, ContentPartTextParam
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
	from boto3.session import Session  # pyright: ignore


T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatAnthropicBedrock(ChatAWSBedrock):
	"""
	AWS Bedrock Anthropic Claude chat model.

	This is a convenience class that provides Claude-specific defaults
	for the AWS Bedrock service. It inherits all functionality from
	ChatAWSBedrock but sets Anthropic Claude as the default model.
	"""

	# Anthropic Claude specific defaults
	model: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
	max_tokens: int = 8192
	temperature: float | None = None
	top_p: float | None = None
	top_k: int | None = None
	stop_sequences: list[str] | None = None
	seed: int | None = None

	# AWS credentials and configuration
	aws_access_key: str | None = None
	aws_secret_key: str | None = None
	aws_session_token: str | None = None
	aws_region: str | None = None
	session: 'Session | None' = None

	# Client initialization parameters
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None

	@property
	def provider(self) -> str:
		return 'anthropic_bedrock'

	@property
	def capabilities(self) -> ProviderCapabilities:
		return ProviderCapabilities(
			structured_output=StructuredOutputMethod.TOOL_CALLING,
			structured_output_fallback=StructuredOutputMethod.PROMPT_TEXT,
			supports_vision=True,
			supports_thinking=False,
			supports_json_schema_response_format=False,
			supports_tool_calling=True,
			supports_image_input=True,
			max_retries=10,
			prompt_format='anthropic',
		)

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for Bedrock."""
		client_params: dict[str, Any] = {}

		if self.session:
			credentials = self.session.get_credentials()
			client_params.update(
				{
					'aws_access_key': credentials.access_key,
					'aws_secret_key': credentials.secret_key,
					'aws_session_token': credentials.token,
					'aws_region': self.session.region_name,
				}
			)
		else:
			# Use individual credentials
			if self.aws_access_key:
				client_params['aws_access_key'] = self.aws_access_key
			if self.aws_secret_key:
				client_params['aws_secret_key'] = self.aws_secret_key
			if self.aws_region:
				client_params['aws_region'] = self.aws_region
			if self.aws_session_token:
				client_params['aws_session_token'] = self.aws_session_token

		# Add optional parameters
		if self.max_retries:
			client_params['max_retries'] = self.max_retries
		if self.default_headers:
			client_params['default_headers'] = self.default_headers
		if self.default_query:
			client_params['default_query'] = self.default_query

		return client_params

	def _get_client_params_for_invoke(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for invoke."""
		client_params = {}

		if self.temperature is not None:
			client_params['temperature'] = self.temperature
		if self.max_tokens is not None:
			client_params['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			client_params['top_p'] = self.top_p
		if self.top_k is not None:
			client_params['top_k'] = self.top_k
		if self.seed is not None:
			client_params['seed'] = self.seed
		if self.stop_sequences is not None:
			client_params['stop_sequences'] = self.stop_sequences

		return client_params

	def get_client(self) -> AsyncAnthropicBedrock:
		"""
		Returns an AsyncAnthropicBedrock client.

		Returns:
			AsyncAnthropicBedrock: An instance of the AsyncAnthropicBedrock client.
		"""
		client_params = self._get_client_params()
		return AsyncAnthropicBedrock(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_cache_creation_tokens(self, response: Message) -> tuple[int | None, int | None]:
		cache_creation = getattr(response.usage, 'cache_creation', None)
		if cache_creation is None:
			return None, None
		return (
			getattr(cache_creation, 'ephemeral_5m_input_tokens', None),
			getattr(cache_creation, 'ephemeral_1h_input_tokens', None),
		)

	def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
		"""Extract usage information from the response."""
		cache_creation_5m_tokens, cache_creation_1h_tokens = self._get_cache_creation_tokens(response)
		usage = ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens
			+ (
				response.usage.cache_read_input_tokens or 0
			),  # Total tokens in Anthropic are a bit fucked, you have to add cached tokens to the prompt tokens
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.input_tokens + response.usage.output_tokens,
			prompt_cached_tokens=response.usage.cache_read_input_tokens,
			prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
			prompt_cache_creation_5m_tokens=cache_creation_5m_tokens,
			prompt_cache_creation_1h_tokens=cache_creation_1h_tokens,
			prompt_image_tokens=None,
		)
		return usage

	def _extract_text_from_response(self, response: Message) -> str:
		first_content = response.content[0]
		if isinstance(first_content, TextBlock):
			return first_content.text
		elif hasattr(first_content, 'type') and first_content.type == 'text':
			return getattr(first_content, 'text', '')
		return str(first_content)

	def _parse_tool_use_input(self, content_block: Any, output_format: type[T]) -> T:
		import json as _json

		try:
			return output_format.model_validate(content_block.input)
		except Exception:
			_input = content_block.input
			if isinstance(_input, str):
				_input = _json.loads(_input)
			elif isinstance(_input, dict):
				for key, value in _input.items():
					if isinstance(value, str) and value.startswith(('[', '{')):
						try:
							_input[key] = _json.loads(value)
						except _json.JSONDecodeError:
							cleaned = value.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
							try:
								_input[key] = _json.loads(cleaned)
							except _json.JSONDecodeError:
								pass
			else:
				raise
			return output_format.model_validate(_input)

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
		anthropic_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					system=system_prompt or omit,
					**self._get_client_params_for_invoke(),
				)
				usage = self._get_usage(response)
				response_text = self._extract_text_from_response(response)
				return ChatInvokeCompletion(completion=response_text, usage=usage)

			else:
				strategy_chain = self.capabilities.get_structured_output_strategy_chain()
				strategy = structured_output_method if structured_output_method is not None else strategy_chain[0]

				if strategy == StructuredOutputMethod.TOOL_CALLING:
					tool_name = output_format.__name__
					schema = SchemaOptimizer.create_optimized_json_schema(output_format)
					if 'title' in schema:
						del schema['title']

					tool = ToolParam(
						name=tool_name,
						description=f'Extract information in the format of {tool_name}',
						input_schema=schema,
						cache_control=CacheControlEphemeralParam(type='ephemeral'),
					)
					tool_choice = ToolChoiceToolParam(type='tool', name=tool_name)

					response = await self.get_client().messages.create(
						model=self.model,
						messages=anthropic_messages,
						tools=[tool],
						system=system_prompt or omit,
						tool_choice=tool_choice,
						**self._get_client_params_for_invoke(),
					)
					usage = self._get_usage(response)

					for content_block in response.content:
						if hasattr(content_block, 'type') and content_block.type == 'tool_use':
							parsed = self._parse_tool_use_input(content_block, output_format)
							return ChatInvokeCompletion(completion=parsed, usage=usage)
					raise ValueError('Expected tool use in response but none found')

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

					fallback_msgs, fallback_system = AnthropicMessageSerializer.serialize_messages(modified_messages)
					response = await self.get_client().messages.create(
						model=self.model,
						messages=fallback_msgs,
						system=fallback_system or omit,
						**self._get_client_params_for_invoke(),
					)
					usage = self._get_usage(response)
					response_text = self._extract_text_from_response(response)
					parsed = parse_structured_output_from_text(response_text, output_format)
					if parsed is not None:
						return ChatInvokeCompletion(completion=parsed, usage=usage)
					raise ValueError('Failed to parse structured output from prompt text response')

				else:
					raise ValueError(f'Unsupported structured output strategy: {strategy}')

		except APIConnectionError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except ModelProviderError:
			raise
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
