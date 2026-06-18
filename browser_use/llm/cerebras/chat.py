from __future__ import annotations

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
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.capabilities import (
	ProviderCapabilities,
	StructuredOutputMethod,
	build_prompt_text_schema_instruction,
	get_default_capabilities,
	parse_structured_output_from_text,
)
from browser_use.llm.cerebras.serializer import CerebrasMessageSerializer
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatCerebras(BaseChatModel):
	"""Cerebras inference wrapper (OpenAI-compatible)."""

	model: str = 'llama3.1-8b'

	# Generation parameters
	max_tokens: int | None = 4096
	temperature: float | None = 0.2
	top_p: float | None = None
	seed: int | None = None

	# Connection parameters
	api_key: str | None = None
	base_url: str | httpx.URL | None = 'https://api.cerebras.ai/v1'
	timeout: float | httpx.Timeout | None = None
	client_params: dict[str, Any] | None = None

	@property
	def provider(self) -> str:
		return 'cerebras'

	@property
	def capabilities(self) -> ProviderCapabilities:
		return get_default_capabilities('cerebras')

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

	def _get_usage(self, response: ChatCompletion) -> ChatInvokeUsage | None:
		if response.usage is not None:
			usage = ChatInvokeUsage(
				prompt_tokens=response.usage.prompt_tokens,
				prompt_cached_tokens=None,
				prompt_cache_creation_tokens=None,
				prompt_image_tokens=None,
				completion_tokens=response.usage.completion_tokens,
				total_tokens=response.usage.total_tokens,
			)
		else:
			usage = None
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
		"""
		Cerebras ainvoke supports:
		1. Regular text/multi-turn conversation
		2. JSON Output (response_format)
		"""
		client = self._client()
		cerebras_messages = CerebrasMessageSerializer.serialize_messages(messages)
		common: dict[str, Any] = {}

		if self.temperature is not None:
			common['temperature'] = self.temperature
		if self.max_tokens is not None:
			common['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			common['top_p'] = self.top_p
		if self.seed is not None:
			common['seed'] = self.seed

		# ① Regular multi-turn conversation/text output
		if output_format is None:
			try:
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=cerebras_messages,  # type: ignore
					**common,
				)
				usage = self._get_usage(resp)
				return ChatInvokeCompletion(
					completion=resp.choices[0].message.content or '',
					usage=usage,
				)
			except RateLimitError as e:
				raise ModelRateLimitError(str(e), model=self.name) from e
			except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
				raise ModelProviderError(str(e), model=self.name) from e
			except Exception as e:
				raise ModelProviderError(str(e), model=self.name) from e

		# ② Structured output — use capabilities-driven strategy chain
		strategy_chain = self.capabilities.get_structured_output_strategy_chain()
		strategy = structured_output_method if structured_output_method is not None else strategy_chain[0]

		try:
			if strategy == StructuredOutputMethod.PROMPT_TEXT:
				modified_messages = [m.model_copy(deep=True) for m in messages]
				if modified_messages and isinstance(modified_messages[-1].content, str):
					modified_messages[-1].content += build_prompt_text_schema_instruction(output_format)
				fallback_cerebras_messages = CerebrasMessageSerializer.serialize_messages(modified_messages)
				resp = await client.chat.completions.create(  # type: ignore
					model=self.model,
					messages=fallback_cerebras_messages,  # type: ignore
					**common,
				)
				content = resp.choices[0].message.content
				if not content:
					raise ModelProviderError('Empty JSON content in Cerebras response', model=self.name)
				usage = self._get_usage(resp)
				parsed = parse_structured_output_from_text(content, output_format)
				if parsed is not None:
					return ChatInvokeCompletion(
						completion=parsed,
						usage=usage,
					)
				raise ValueError(f'Failed to parse JSON from text: {content[:200]}')

			elif strategy in (StructuredOutputMethod.JSON_SCHEMA, StructuredOutputMethod.TOOL_CALLING):
				raise ValueError(f'Strategy {strategy} is not supported by Cerebras provider')

			else:
				raise ValueError(f'Unsupported structured output strategy: {strategy}')

		except RateLimitError as e:
			raise ModelRateLimitError(str(e), model=self.name) from e
		except (APIError, APIConnectionError, APITimeoutError, APIStatusError) as e:
			raise ModelProviderError(str(e), model=self.name) from e
		except Exception as e:
			raise ModelProviderError(str(e), model=self.name) from e
