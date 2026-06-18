import logging
from dataclasses import dataclass
from os import getenv
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import BaseModel

from browser_use.llm.aws.serializer import AWSBedrockMessageSerializer
from browser_use.llm.base import BaseChatModel
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
	from boto3 import client as AwsClient  # type: ignore
	from boto3.session import Session  # type: ignore

T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatAWSBedrock(BaseChatModel):
	"""
	AWS Bedrock chat model supporting multiple providers (Anthropic, Meta, etc.).

	This class provides access to various models via AWS Bedrock,
	supporting both text generation and structured output via tool calling.

	To use this model, you need to either:
	1. Set the following environment variables:
	   - AWS_ACCESS_KEY_ID
	   - AWS_SECRET_ACCESS_KEY
	   - AWS_SESSION_TOKEN (only required when using temporary credentials)
	   - AWS_REGION
	2. Or provide a boto3 Session object
	3. Or use AWS SSO authentication
	"""

	# Model configuration
	model: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
	max_tokens: int | None = 4096
	temperature: float | None = None
	top_p: float | None = None
	seed: int | None = None
	stop_sequences: list[str] | None = None

	# AWS credentials and configuration
	aws_access_key_id: str | None = None
	aws_secret_access_key: str | None = None
	aws_session_token: str | None = None
	aws_region: str | None = None
	aws_sso_auth: bool = False
	session: 'Session | None' = None

	# Request parameters
	request_params: dict[str, Any] | None = None

	# Static
	@property
	def provider(self) -> str:
		return 'aws_bedrock'

	@property
	def capabilities(self) -> ProviderCapabilities:
		return ProviderCapabilities(
			structured_output=StructuredOutputMethod.TOOL_CALLING,
			supports_vision=True,
			supports_thinking=False,
			supports_json_schema_response_format=False,
			supports_tool_calling=True,
			supports_image_input=True,
			max_retries=3,
			prompt_format='aws_bedrock',
		)

	def _get_client(self) -> 'AwsClient':  # type: ignore
		"""Get the AWS Bedrock client."""
		try:
			from boto3 import client as AwsClient  # type: ignore
		except ImportError:
			raise ImportError(
				'`boto3` not installed. Please install using `pip install browser-use[aws] or pip install browser-use[all]`'
			)

		if self.session:
			return self.session.client('bedrock-runtime')

		# Get credentials from environment or instance parameters
		access_key = self.aws_access_key_id or getenv('AWS_ACCESS_KEY_ID')
		secret_key = self.aws_secret_access_key or getenv('AWS_SECRET_ACCESS_KEY')
		session_token = self.aws_session_token or getenv('AWS_SESSION_TOKEN')
		region = self.aws_region or getenv('AWS_REGION') or getenv('AWS_DEFAULT_REGION')

		if self.aws_sso_auth:
			return AwsClient(service_name='bedrock-runtime', region_name=region)
		else:
			if not access_key or not secret_key:
				raise ModelProviderError(
					message='AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables (and AWS_SESSION_TOKEN if using temporary credentials) or provide a boto3 session.',
					model=self.name,
				)

			return AwsClient(
				service_name='bedrock-runtime',
				region_name=region,
				aws_access_key_id=access_key,
				aws_secret_access_key=secret_key,
				aws_session_token=session_token,
			)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_inference_config(self) -> dict[str, Any]:
		"""Get the inference configuration for the request."""
		config = {}
		if self.max_tokens is not None:
			config['maxTokens'] = self.max_tokens
		if self.temperature is not None:
			config['temperature'] = self.temperature
		if self.top_p is not None:
			config['topP'] = self.top_p
		if self.stop_sequences is not None:
			config['stopSequences'] = self.stop_sequences
		if self.seed is not None:
			config['seed'] = self.seed
		return config

	def _format_tools_for_request(self, output_format: type[BaseModel]) -> list[dict[str, Any]]:
		"""Format a Pydantic model as a tool for structured output."""
		schema = SchemaOptimizer.create_optimized_json_schema(output_format)

		return [
			{
				'toolSpec': {
					'name': f'extract_{output_format.__name__.lower()}',
					'description': f'Extract information in the format of {output_format.__name__}',
					'inputSchema': {'json': schema},
				}
			}
		]

	def _get_usage(self, response: dict[str, Any]) -> ChatInvokeUsage | None:
		"""Extract usage information from the response."""
		if 'usage' not in response:
			return None

		usage_data = response['usage']
		return ChatInvokeUsage(
			prompt_tokens=usage_data.get('inputTokens', 0),
			completion_tokens=usage_data.get('outputTokens', 0),
			total_tokens=usage_data.get('totalTokens', 0),
			prompt_cached_tokens=None,  # Bedrock doesn't provide this
			prompt_cache_creation_tokens=None,
			prompt_image_tokens=None,
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

	def _extract_text_from_content(self, content: list[dict[str, Any]]) -> str:
		text_content = []
		for item in content:
			if 'text' in item:
				text_content.append(item['text'])
		return '\n'.join(text_content) if text_content else ''

	def _extract_tool_use_from_content(self, content: list[dict[str, Any]], output_format: type[T]) -> T | None:
		import json as _json

		for item in content:
			if 'toolUse' in item:
				tool_use = item['toolUse']
				tool_input = tool_use.get('input', {})
				try:
					return output_format.model_validate(tool_input)
				except Exception:
					if isinstance(tool_input, str):
						try:
							data = _json.loads(tool_input)
							return output_format.model_validate(data)
						except _json.JSONDecodeError:
							pass
		return None

	async def ainvoke(
		self,
		messages: list[BaseMessage],
		output_format: type[T] | None = None,
		structured_output_method: StructuredOutputMethod | None = None,
		**kwargs: Any,
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		"""
		Invoke the AWS Bedrock model with the given messages.

		Args:
			messages: List of chat messages
			output_format: Optional Pydantic model class for structured output
			structured_output_method: Optional structured output method to use

		Returns:
			Either a string response or an instance of output_format
		"""
		try:
			from botocore.exceptions import ClientError  # type: ignore
		except ImportError:
			raise ImportError(
				'`boto3` not installed. Please install using `pip install browser-use[aws] or pip install browser-use[all]`'
			)

		try:
			if output_format is None:
				bedrock_messages, system_message = AWSBedrockMessageSerializer.serialize_messages(messages)
				body: dict[str, Any] = {}
				if system_message:
					body['system'] = system_message
				inference_config = self._get_inference_config()
				if inference_config:
					body['inferenceConfig'] = inference_config
				if self.request_params:
					body.update(self.request_params)
				body = {k: v for k, v in body.items() if v is not None}

				client = self._get_client()
				response = client.converse(modelId=self.model, messages=bedrock_messages, **body)
				usage = self._get_usage(response)

				response_text = ''
				if 'output' in response and 'message' in response['output']:
					content = response['output']['message'].get('content', [])
					response_text = self._extract_text_from_content(content)

				return ChatInvokeCompletion(completion=response_text, usage=usage)

			else:
				strategy_chain = self.capabilities.get_structured_output_strategy_chain()
				strategy = structured_output_method if structured_output_method is not None else strategy_chain[0]

				if strategy == StructuredOutputMethod.TOOL_CALLING:
					bedrock_messages, system_message = AWSBedrockMessageSerializer.serialize_messages(messages)
					tools = self._format_tools_for_request(output_format)
					body = {'toolConfig': {'tools': tools}}
					if system_message:
						body['system'] = system_message
					inference_config = self._get_inference_config()
					if inference_config:
						body['inferenceConfig'] = inference_config
					if self.request_params:
						body.update(self.request_params)
					body = {k: v for k, v in body.items() if v is not None}

					client = self._get_client()
					response = client.converse(modelId=self.model, messages=bedrock_messages, **body)
					usage = self._get_usage(response)

					if 'output' in response and 'message' in response['output']:
						content = response['output']['message'].get('content', [])
						parsed = self._extract_tool_use_from_content(content, output_format)
						if parsed is not None:
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

					bedrock_messages, system_message = AWSBedrockMessageSerializer.serialize_messages(modified_messages)
					body: dict[str, Any] = {}
					if system_message:
						body['system'] = system_message
					inference_config = self._get_inference_config()
					if inference_config:
						body['inferenceConfig'] = inference_config
					if self.request_params:
						body.update(self.request_params)
					body = {k: v for k, v in body.items() if v is not None}

					client = self._get_client()
					response = client.converse(modelId=self.model, messages=bedrock_messages, **body)
					usage = self._get_usage(response)

					response_text = ''
					if 'output' in response and 'message' in response['output']:
						content = response['output']['message'].get('content', [])
						response_text = self._extract_text_from_content(content)

					parsed = parse_structured_output_from_text(response_text, output_format)
					if parsed is not None:
						return ChatInvokeCompletion(completion=parsed, usage=usage)
					raise ValueError('Failed to parse structured output from prompt text response')

				else:
					raise ValueError(f'Unsupported structured output strategy: {strategy}')

		except ClientError as e:
			error_code = e.response.get('Error', {}).get('Code', 'Unknown')
			error_message = e.response.get('Error', {}).get('Message', str(e))

			if error_code in ['ThrottlingException', 'TooManyRequestsException']:
				raise ModelRateLimitError(message=error_message, model=self.name) from e
			else:
				raise ModelProviderError(message=error_message, model=self.name) from e
		except ModelProviderError:
			raise
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
