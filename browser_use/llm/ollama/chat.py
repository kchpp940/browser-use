import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from ollama import AsyncClient as OllamaAsyncClient
from ollama import Options
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.capabilities import (
	ProviderCapabilities,
	StructuredOutputMethod,
	build_prompt_text_schema_instruction,
	get_default_capabilities,
	parse_structured_output_from_text,
)
from browser_use.llm.exceptions import ModelProviderError
from browser_use.llm.messages import BaseMessage, ContentPartTextParam
from browser_use.llm.ollama.serializer import OllamaMessageSerializer
from browser_use.llm.views import ChatInvokeCompletion

logger = logging.getLogger(__name__)
T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatOllama(BaseChatModel):
	"""
	A wrapper around Ollama's chat model.
	"""

	model: str

	# # Model params
	# TODO (matic): Why is this commented out?
	# temperature: float | None = None

	# Client initialization parameters
	host: str | None = None
	timeout: float | httpx.Timeout | None = None
	client_params: dict[str, Any] | None = None
	ollama_options: Mapping[str, Any] | Options | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'ollama'

	@property
	def capabilities(self) -> ProviderCapabilities:
		return get_default_capabilities('ollama')

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary."""
		return {
			'host': self.host,
			'timeout': self.timeout,
			'client_params': self.client_params,
		}

	def get_client(self) -> OllamaAsyncClient:
		"""
		Returns an OllamaAsyncClient client.
		"""
		return OllamaAsyncClient(host=self.host, timeout=self.timeout, **self.client_params or {})

	@property
	def name(self) -> str:
		return self.model

	@overload
	async def ainvoke(
		self, messages: list[BaseMessage], output_format: None = None, **kwargs: Any
	) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T], **kwargs: Any) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None, **kwargs: Any
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		try:
			if output_format is None:
				ollama_messages = OllamaMessageSerializer.serialize_messages(messages)
				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
					options=self.ollama_options,
				)
				return ChatInvokeCompletion(completion=response.message.content or '', usage=None)

			else:
				strategy_chain = self.capabilities.get_structured_output_strategy_chain()
				last_error: Exception | None = None

				for strategy in strategy_chain:
					try:
						if strategy == StructuredOutputMethod.JSON_SCHEMA:
							schema = output_format.model_json_schema()
							ollama_messages = OllamaMessageSerializer.serialize_messages(messages)
							response = await self.get_client().chat(
								model=self.model,
								messages=ollama_messages,
								format=schema,
								options=self.ollama_options,
							)
							completion_text = response.message.content or ''
							parsed = output_format.model_validate_json(completion_text)
							return ChatInvokeCompletion(completion=parsed, usage=None)

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

							ollama_messages = OllamaMessageSerializer.serialize_messages(modified_messages)
							response = await self.get_client().chat(
								model=self.model,
								messages=ollama_messages,
								options=self.ollama_options,
							)
							completion_text = response.message.content or ''
							parsed = parse_structured_output_from_text(completion_text, output_format)
							if parsed is not None:
								return ChatInvokeCompletion(completion=parsed, usage=None)
							raise ValueError('Failed to parse structured output from prompt text response')

					except Exception as e:
						last_error = e
						logger.debug(f'Ollama structured output strategy {strategy} failed: {e}')
						continue

				if last_error is not None:
					raise ModelProviderError(
						message=f'All structured output strategies failed. Last error: {last_error}',
						model=self.name,
					) from last_error
				raise ModelProviderError(
					message='No valid structured output strategy available for Ollama',
					model=self.name,
				)

		except ModelProviderError:
			raise
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
