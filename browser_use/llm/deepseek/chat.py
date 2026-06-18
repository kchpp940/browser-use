from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from openai import (
	APIConnectionError,
	APIError,
	APIStatusError,
	APITimeoutError,
	AsyncOpenAI,
	RateLimitError,
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
from browser_use.llm.deepseek.serializer import DeepSeekMessageSerializer
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.schema import SchemaOptimizer
from browser_use.llm.views import ChatInvokeCompletion

T = TypeVar('T', bound=BaseModel)

logger = logging.getLogger(__name__)


@dataclass
class ChatDeepSeek(BaseChatModel):
	"""DeepSeek /chat/completions wrapper (OpenAI-compatible)."""

	model: str = 'deepseek-chat'

	# Generation parameters
	max_tokens: int | None = None
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None

	# Connection parameters
	api_key: str | None = None
	base_url: str | httpx.URL | None = 'https://api.deepseek.com/v1'
	timeout: float | httpx.Timeout | None = None
	client_params: dict[str, Any] | None = None

	@property
	def provider(self) -> str:
		return 'deepseek'

	@property
	def capabilities(self) -> ProviderCapabilities:
		return get_default_capabilities('deepseek')

	def _client(self) -> AsyncOpenAI:
		return AsyncOpenAI(
			api_key=self.api_key,
			base_url=self.base_url,
			timeout=self.timeout,
			**(self.client_params or {}),
		)

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: None = None,
		tools: list[dict[str, Any]] | None = None,
		stop: list[str] | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T],
		tools: list[dict[str, Any]] | None = None,
		stop: list[str] | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		tools: list[dict[str, Any]] | None = None,
		stop: list[str] | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		DeepSeek ainvoke supports:
		1. Regular text/multi-turn conversation
		2. Function Calling
		3. JSON Output (response_format)
		4. Conversation prefix continuation (beta, prefix, stop)
		"""
		client = self._client()
		ds_messages = DeepSeekMessageSerializer.serialize_messages(messages)
		common: dict[str, Any] = {}

		if self.temperature is not None:
			common['temperature'] = self.temperature
		if self.max_tokens is not None:
			common['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			common['top_p'] = self.top_p
		if self.seed is not None:
			common['seed'] = self.seed

		# Beta conversation prefix continuation (see official documentation)
		if self.base_url and str(self.base_url).endswith('/beta'):
			if ds_messages and isinstance(ds_messages[-1], dict) and ds_messages[-1].get('role') == 'assistant':
				ds_messages[-1]['prefix'] = True
			if stop:
				common['stop'] = stop

		# ① Regular multi-turn conversation/text output
		if output_format is None and not tools:
			try:
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=ds_messages,  # type: ignore
					**common,
				)
				return ChatInvokeCompletion(
					completion=resp.choices[0].message.content or '',
					usage=None,
				)
			except RateLimitError as e:
				raise ModelRateLimitError(str(e), model=self.name) from e
			except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
				raise ModelProviderError(str(e), model=self.name) from e
			except Exception as e:
				raise ModelProviderError(str(e), model=self.name) from e

		# ② If explicit tools are passed, use them regardless of output_format
		if tools:
			try:
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=ds_messages,  # type: ignore
					tools=tools,  # type: ignore
					**common,
				)
				msg = resp.choices[0].message
				if msg.tool_calls:
					raw_args = msg.tool_calls[0].function.arguments
					if isinstance(raw_args, str):
						parsed = json.loads(raw_args)
					else:
						parsed = raw_args
					if output_format is not None:
						return ChatInvokeCompletion(
							completion=output_format.model_validate(parsed),
							usage=None,
						)
					return ChatInvokeCompletion(
						completion=parsed,
						usage=None,
					)
				raise ValueError('Expected tool_calls in response but got none')
			except RateLimitError as e:
				raise ModelRateLimitError(str(e), model=self.name) from e
			except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
				raise ModelProviderError(str(e), model=self.name) from e
			except Exception as e:
				raise ModelProviderError(str(e), model=self.name) from e

		# ③ Structured output with output_format — use capabilities-driven strategy chain
		if output_format is None:
			raise ModelProviderError('No valid ainvoke execution path: output_format is None and no tools provided', model=self.name)
		_output_format: type[T] = output_format
		strategy_chain = self.capabilities.get_structured_output_strategy_chain()
		last_error: Exception | None = None

		for strategy in strategy_chain:
			try:
				if strategy == StructuredOutputMethod.TOOL_CALLING:
					tool_name = _output_format.__name__
					schema = SchemaOptimizer.create_optimized_json_schema(_output_format)
					schema.pop('title', None)
					call_tools = [
						{
							'type': 'function',
							'function': {
								'name': tool_name,
								'description': f'Return a JSON object of type {tool_name}',
								'parameters': schema,
							},
						}
					]
					tool_choice = {'type': 'function', 'function': {'name': tool_name}}
					resp = await client.chat.completions.create(  # type: ignore
						model=self.model,
						messages=ds_messages,  # type: ignore
						tools=call_tools,  # type: ignore
						tool_choice=tool_choice,  # type: ignore
						**common,
					)
					msg = resp.choices[0].message
					if not msg.tool_calls:
						raise ValueError('Expected tool_calls in response but got none')
					raw_args = msg.tool_calls[0].function.arguments
					if isinstance(raw_args, str):
						parsed = json.loads(raw_args)
					else:
						parsed = raw_args
					return ChatInvokeCompletion(
						completion=_output_format.model_validate(parsed),
						usage=None,
					)

				elif strategy == StructuredOutputMethod.JSON_SCHEMA:
					resp = await client.chat.completions.create(  # type: ignore
						model=self.model,
						messages=ds_messages,  # type: ignore
						response_format={'type': 'json_object'},
						**common,
					)
					content = resp.choices[0].message.content
					if not content:
						raise ModelProviderError('Empty JSON content in DeepSeek response', model=self.name)
					parsed = parse_structured_output_from_text(content, _output_format)
					if parsed is not None:
						return ChatInvokeCompletion(
							completion=parsed,
							usage=None,
						)
					raise ValueError(f'Failed to parse JSON schema response: {content[:200]}')

				elif strategy == StructuredOutputMethod.PROMPT_TEXT:
					modified_messages = [m.model_copy(deep=True) for m in messages]
					if modified_messages and isinstance(modified_messages[-1].content, str):
						modified_messages[-1].content += build_prompt_text_schema_instruction(_output_format)
					fallback_ds_messages = DeepSeekMessageSerializer.serialize_messages(modified_messages)
					resp = await client.chat.completions.create(  # type: ignore
						model=self.model,
						messages=fallback_ds_messages,  # type: ignore
						**common,
					)
					content = resp.choices[0].message.content
					if not content:
						raise ValueError('Empty response in prompt text fallback')
					parsed = parse_structured_output_from_text(content, _output_format)
					if parsed is not None:
						return ChatInvokeCompletion(
							completion=parsed,
							usage=None,
						)
					raise ValueError(f'Failed to parse prompt text fallback: {content[:200]}')

			except Exception as e:
				last_error = e
				logger.debug(f'DeepSeek structured output strategy {strategy} failed: {e}')
				continue

		if last_error is not None:
			raise ModelProviderError(
				message=f'All structured output strategies failed. Last error: {last_error}',
				model=self.name,
			) from last_error
		raise ModelProviderError('No valid ainvoke execution path for DeepSeek LLM', model=self.name)
