from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, overload

import httpx
from ollama import AsyncClient as OllamaAsyncClient
from ollama import Options
from pydantic import BaseModel

from browser_use.llm.base import BaseChatModel
from browser_use.llm.exceptions import ModelParseError, ModelProviderError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.ollama.serializer import OllamaMessageSerializer
from browser_use.llm.parser import AgentOutputParser, NormalizedLLMResponse
from browser_use.llm.views import ChatInvokeCompletion

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
		ollama_messages = OllamaMessageSerializer.serialize_messages(messages)

		try:
			if output_format is None:
				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
					options=self.ollama_options,
				)

				return ChatInvokeCompletion(completion=response.message.content or '', usage=None)
			else:
				schema = output_format.model_json_schema()

				response = await self.get_client().chat(
					model=self.model,
					messages=ollama_messages,
					format=schema,
					options=self.ollama_options,
				)

				raw_text = response.message.content or ''

				# Extract tool_calls if present in Ollama format
				from browser_use.llm.parser import NormalizedToolCall

				tool_calls_list: list[NormalizedToolCall] = []
				msg_tool_calls = getattr(response.message, 'tool_calls', None)
				if isinstance(msg_tool_calls, list):
					for tc in msg_tool_calls:
						fn = getattr(tc, 'function', None)
						if fn is not None:
							tool_calls_list.append(
								NormalizedToolCall(
									id=getattr(tc, 'id', None),
									name=getattr(fn, 'name', ''),
									arguments=getattr(fn, 'arguments', ''),
								)
							)

				normalized = NormalizedLLMResponse(
					raw_text=raw_text,
					tool_calls=tool_calls_list,
				)
				try:
					parsed = AgentOutputParser(output_format).parse(normalized)
				except ModelParseError as e:
					e.model = self.name
					raise

				return ChatInvokeCompletion(completion=parsed, usage=None)

		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
