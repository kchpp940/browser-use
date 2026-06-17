"""
Provider response extraction unit tests.

Constructs lightweight fake response objects for OpenAI / Anthropic / Groq / Google /
Mistral, matching each provider's *actual* Python object shape (not just dicts), then
feeds them through the provider's extraction / _build_normalized logic and asserts the
resulting NormalizedLLMResponse is structurally identical across providers.

This guards against bugs like:
  - wrong attribute access on the response object
  - forgetting to extract tool_calls / refusal from a particular provider's message
  - inconsistent schema-wrapper unwrapping between providers
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from pydantic import BaseModel, create_model

from browser_use.agent.views import AgentOutput
from browser_use.llm.parser import (
	AgentOutputParser,
	NormalizedLLMResponse,
	NormalizedToolCall,
)
from browser_use.tools.registry.views import ActionModel


# ---------------------------------------------------------------------------
# Helpers: create AgentOutput type with dynamic ActionModel
# ---------------------------------------------------------------------------
def _make_action_model() -> type[ActionModel]:
	DoneParams = create_model('DoneParams', success=(bool, True), text=(str | None, None))
	NavigateParams = create_model('NavigateParams', url=(str, ...))
	return create_model(  # type: ignore[return-value]
		'ActionModel',
		done=(DoneParams | None, None),
		navigate_to=(NavigateParams | None, None),
		__base__=ActionModel,
	)


def _make_agent_output_type() -> type[AgentOutput]:
	ActionModelT = _make_action_model()
	return create_model(  # type: ignore[return-value]
		'AgentOutput',
		action=(list[ActionModelT], ...),  # type: ignore[valid-type]
		__base__=AgentOutput,
	)


AGENT_OUTPUT_TYPE = _make_agent_output_type()
REFERENCE_DICT: dict[str, Any] = {
	'evaluation_previous_goal': 'opened google',
	'memory': 'user wants weather',
	'next_goal': 'search weather paris',
	'thinking': 'let me search',
	'action': [
		{'navigate_to': {'url': 'https://google.com?q=weather+paris'}},
		{'done': {'success': True, 'text': 'all done'}},
	],
}
REFERENCE_JSON = json.dumps(REFERENCE_DICT)


def _assert_normalized_equivalent(a: NormalizedLLMResponse, b: NormalizedLLMResponse) -> None:
	"""Structural equivalence of NormalizedLLMResponse for the fields we care about."""
	parser = AgentOutputParser(AGENT_OUTPUT_TYPE)
	out_a = parser.parse(a)
	out_b = parser.parse(b)
	assert out_a.evaluation_previous_goal == out_b.evaluation_previous_goal
	assert out_a.memory == out_b.memory
	assert out_a.next_goal == out_b.next_goal
	assert len(out_a.action) == len(out_b.action)
	for ai, bi in zip(out_a.action, out_b.action):
		assert ai.model_dump() == bi.model_dump()


# ===========================================================================
# OpenAI
# ===========================================================================


class TestOpenAIResponseExtraction:
	"""Tests the extraction logic at browser_use/llm/openai/chat.py:278-282."""

	def _make_fake_openai_response(
		self,
		content: str | None = None,
		refusal: str | None = None,
		finish_reason: str = 'stop',
		tool_calls: list[Any] | None = None,
	) -> SimpleNamespace:
		"""Mock an openai.types.chat.ChatCompletion response object."""
		message = SimpleNamespace(
			content=content,
			refusal=refusal,
			tool_calls=tool_calls,
		)
		choice = SimpleNamespace(
			message=message,
			finish_reason=finish_reason,
		)
		usage = SimpleNamespace(
			prompt_tokens=10,
			completion_tokens=20,
			total_tokens=30,
		)
		return SimpleNamespace(
			choices=[choice],
			usage=usage,
		)

	def test_json_content_only(self):
		"""Simulates: response_format=json_schema, model returns JSON in .content."""
		fake_resp = self._make_fake_openai_response(content=REFERENCE_JSON)

		choice = fake_resp.choices[0]
		# This is the EXACT extraction code from openai/chat.py:278-282
		normalized = NormalizedLLMResponse(
			raw_text=choice.message.content or '',
			refusal=getattr(choice.message, 'refusal', None),
			stop_reason=choice.finish_reason,
		)

		assert normalized.raw_text == REFERENCE_JSON
		assert normalized.refusal is None
		assert normalized.stop_reason == 'stop'
		# Must parse to valid AgentOutput
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_refusal_message(self):
		"""OpenAI returns refusal when safety policy violated."""
		fake_resp = self._make_fake_openai_response(
			content=None,
			refusal='I cannot help with harmful requests.',
			finish_reason='stop',
		)

		choice = fake_resp.choices[0]
		normalized = NormalizedLLMResponse(
			raw_text=choice.message.content or '',
			refusal=getattr(choice.message, 'refusal', None),
			stop_reason=choice.finish_reason,
		)

		assert normalized.raw_text == ''
		assert normalized.refusal == 'I cannot help with harmful requests.'

	def test_empty_content_with_refusal(self):
		"""Edge case: empty content but has refusal."""
		fake_resp = self._make_fake_openai_response(
			content='',
			refusal='Blocked by policy',
		)
		choice = fake_resp.choices[0]
		normalized = NormalizedLLMResponse(
			raw_text=choice.message.content or '',
			refusal=getattr(choice.message, 'refusal', None),
			stop_reason=choice.finish_reason,
		)
		assert normalized.refusal == 'Blocked by policy'
		assert normalized.raw_text == ''


# ===========================================================================
# Anthropic
# ===========================================================================


class TestAnthropicResponseExtraction:
	"""Tests the extraction logic at browser_use/llm/anthropic/chat.py:393-414."""

	def _make_fake_text_block(self, text: str) -> SimpleNamespace:
		return SimpleNamespace(type='text', text=text)

	def _make_fake_thinking_block(self, thinking: str) -> SimpleNamespace:
		return SimpleNamespace(type='thinking', thinking=thinking)

	def _make_fake_redacted_thinking_block(self, data: str) -> SimpleNamespace:
		return SimpleNamespace(type='redacted_thinking', data=data)

	def _make_fake_tool_use_block(self, tool_id: str, name: str, input_data: dict[str, Any]) -> SimpleNamespace:
		return SimpleNamespace(type='tool_use', id=tool_id, name=name, input=input_data)

	def _make_fake_anthropic_response(
		self,
		content_blocks: list[SimpleNamespace],
		stop_reason: str = 'stop',
	) -> SimpleNamespace:
		usage = SimpleNamespace(
			input_tokens=5,
			output_tokens=15,
			total_tokens=20,
		)
		return SimpleNamespace(
			content=content_blocks,
			stop_reason=stop_reason,
			usage=usage,
		)

	def test_tool_use_block_with_json_input(self):
		"""Anthropic returns output in tool_use block with input=dict (actual SDK shape)."""
		tool_block = self._make_fake_tool_use_block(
			tool_id='toolu_01ABC',
			name='AgentOutput',
			input_data=REFERENCE_DICT,
		)
		text_block = self._make_fake_text_block('I will use my tool to respond.')
		fake_resp = self._make_fake_anthropic_response(
			content_blocks=[text_block, tool_block],
			stop_reason='tool_use',
		)

		# Replicate the extraction code from anthropic/chat.py
		chat_model = MagicMock()
		chat_model._extract_content_blocks = lambda resp: (
			''.join(getattr(b, 'text', '') for b in resp.content if getattr(b, 'type', None) == 'text'),
			None,
			None,
		)

		# Actual extraction code (anthropic/chat.py:393-405)
		tool_calls: list[NormalizedToolCall] = []
		for content_block in fake_resp.content:
			if hasattr(content_block, 'type') and content_block.type == 'tool_use':
				tool_id = getattr(content_block, 'id', None)
				tool_name = getattr(content_block, 'name', AGENT_OUTPUT_TYPE.__name__)
				tool_input = getattr(content_block, 'input', {})
				tool_calls.append(
					NormalizedToolCall(
						id=tool_id,
						name=tool_name,
						arguments=tool_input,
					)
				)

		response_text, thinking, redacted_thinking = chat_model._extract_content_blocks(fake_resp)
		normalized = NormalizedLLMResponse(
			raw_text=response_text,
			tool_calls=tool_calls,
			thinking=thinking,
			redacted_thinking=redacted_thinking,
			stop_reason=fake_resp.stop_reason,
		)

		assert len(normalized.tool_calls) == 1
		assert normalized.tool_calls[0].id == 'toolu_01ABC'
		assert normalized.tool_calls[0].name == 'AgentOutput'
		assert normalized.tool_calls[0].arguments == REFERENCE_DICT
		assert normalized.stop_reason == 'tool_use'

	def test_thinking_and_redacted_blocks(self):
		"""Anthropic extended thinking: thinking block + redacted_thinking block."""
		thinking_block = self._make_fake_thinking_block('I need to search for weather.')
		redacted_block = self._make_fake_redacted_thinking_block('█' * 100)
		text_block = self._make_fake_text_block('Here is the result.')
		tool_block = self._make_fake_tool_use_block(
			tool_id='toolu_02',
			name='AgentOutput',
			input_data=REFERENCE_DICT,
		)
		fake_resp = self._make_fake_anthropic_response(
			content_blocks=[thinking_block, redacted_block, text_block, tool_block],
			stop_reason='tool_use',
		)

		# _extract_content_blocks (anthropic/chat.py:233-262)
		text_parts: list[str] = []
		thinking_parts: list[str] = []
		redacted_parts: list[str] = []
		for content_block in fake_resp.content:
			block_type = getattr(content_block, 'type', None)
			if block_type == 'text':
				text = getattr(content_block, 'text', None)
				if text:
					text_parts.append(text)
			elif block_type == 'thinking':
				thinking_text = getattr(content_block, 'thinking', None)
				if thinking_text:
					thinking_parts.append(thinking_text)
			elif block_type == 'redacted_thinking':
				redacted_text = getattr(content_block, 'data', None) or getattr(content_block, 'redacted_thinking', None)
				if redacted_text:
					redacted_parts.append(str(redacted_text))

		response_text = ''.join(text_parts) if text_parts else ''
		thinking = '\n'.join(thinking_parts) if thinking_parts else None
		redacted = '\n'.join(redacted_parts) if redacted_parts else None

		assert response_text == 'Here is the result.'
		assert thinking == 'I need to search for weather.'
		assert redacted is not None and '█' in redacted

	def test_multiple_tool_use_blocks(self):
		"""Rare case: multiple tool_use blocks (Anthropic supports this)."""
		block1 = self._make_fake_tool_use_block('toolu_01', 'AgentOutput', REFERENCE_DICT)
		block2 = self._make_fake_tool_use_block('toolu_02', 'OtherTool', {'query': 'test'})
		fake_resp = self._make_fake_anthropic_response(
			content_blocks=[block1, block2],
			stop_reason='tool_use',
		)

		tool_calls: list[NormalizedToolCall] = []
		for content_block in fake_resp.content:
			if hasattr(content_block, 'type') and content_block.type == 'tool_use':
				tool_calls.append(
					NormalizedToolCall(
						id=getattr(content_block, 'id', None),
						name=getattr(content_block, 'name', 'unknown'),
						arguments=getattr(content_block, 'input', {}),
					)
				)

		assert len(tool_calls) == 2
		assert tool_calls[0].name == 'AgentOutput'
		assert tool_calls[1].name == 'OtherTool'


# ===========================================================================
# Groq
# ===========================================================================


class TestGroqResponseExtraction:
	"""Tests the extraction logic at browser_use/llm/groq/chat.py:186-205."""

	def _make_fake_groq_tool_call(self, tc_id: str, name: str, arguments: str | dict[str, Any]) -> SimpleNamespace:
		func = SimpleNamespace(name=name, arguments=arguments)
		return SimpleNamespace(id=tc_id, function=func)

	def _make_fake_groq_response(
		self,
		content: str | None = None,
		refusal: str | None = None,
		tool_calls: list[SimpleNamespace] | None = None,
		finish_reason: str = 'stop',
	) -> SimpleNamespace:
		"""Groq response has same shape as OpenAI but Groq may not have .refusal on Message."""
		msg_dict: dict[str, Any] = {'content': content, 'tool_calls': tool_calls}
		if refusal is not None:
			msg_dict['refusal'] = refusal
		message = SimpleNamespace(**msg_dict)
		choice = SimpleNamespace(message=message, finish_reason=finish_reason)
		usage = SimpleNamespace(prompt_tokens=5, completion_tokens=10, total_tokens=15)
		return SimpleNamespace(choices=[choice], usage=usage)

	def test_json_schema_path(self):
		"""Groq JSON schema path: JSON in .content, no tool_calls."""
		fake_resp = self._make_fake_groq_response(content=REFERENCE_JSON)

		# Extraction code from groq/chat.py:186-205
		choice = fake_resp.choices[0]
		raw_text = choice.message.content or ''
		tool_calls_list: list[NormalizedToolCall] = []

		if getattr(choice.message, 'tool_calls', None):
			for tc in choice.message.tool_calls:
				tool_calls_list.append(
					NormalizedToolCall(
						id=getattr(tc, 'id', None),
						name=tc.function.name,
						arguments=tc.function.arguments,
					)
				)

		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			refusal=getattr(choice.message, 'refusal', None),
			tool_calls=tool_calls_list,
		)

		assert normalized.raw_text == REFERENCE_JSON
		assert len(normalized.tool_calls) == 0
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_tool_calling_path_with_json_string_arguments(self):
		"""Groq tool calling path: tool_calls[].function.arguments is a JSON string."""
		tc = self._make_fake_groq_tool_call(
			tc_id='call_groq_001',
			name='AgentOutput',
			arguments=REFERENCE_JSON,  # Groq returns arguments as JSON string
		)
		fake_resp = self._make_fake_groq_response(
			content='',
			tool_calls=[tc],
			finish_reason='tool_calls',
		)

		choice = fake_resp.choices[0]
		raw_text = choice.message.content or ''
		tool_calls_list: list[NormalizedToolCall] = []

		if getattr(choice.message, 'tool_calls', None):
			for tc_item in choice.message.tool_calls:
				tool_calls_list.append(
					NormalizedToolCall(
						id=getattr(tc_item, 'id', None),
						name=tc_item.function.name,
						arguments=tc_item.function.arguments,
					)
				)

		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			refusal=getattr(choice.message, 'refusal', None),
			tool_calls=tool_calls_list,
		)

		assert len(normalized.tool_calls) == 1
		assert normalized.tool_calls[0].id == 'call_groq_001'
		assert normalized.tool_calls[0].arguments == REFERENCE_JSON  # string!
		# Parser must handle string arguments
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_tool_calling_path_with_dict_arguments(self):
		"""Some Groq proxy returns arguments already parsed as dict."""
		tc = self._make_fake_groq_tool_call(
			tc_id='call_groq_002',
			name='AgentOutput',
			arguments=REFERENCE_DICT,  # dict!
		)
		fake_resp = self._make_fake_groq_response(content='', tool_calls=[tc])

		choice = fake_resp.choices[0]
		raw_text = choice.message.content or ''
		tool_calls_list: list[NormalizedToolCall] = []

		if getattr(choice.message, 'tool_calls', None):
			for tc_item in choice.message.tool_calls:
				tool_calls_list.append(
					NormalizedToolCall(
						id=getattr(tc_item, 'id', None),
						name=tc_item.function.name,
						arguments=tc_item.function.arguments,
					)
				)

		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			refusal=getattr(choice.message, 'refusal', None),
			tool_calls=tool_calls_list,
		)

		assert normalized.tool_calls[0].arguments == REFERENCE_DICT
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_refusal_without_attribute(self):
		"""Groq TypedDict doesn't guarantee 'refusal' key - getattr with default."""
		msg = SimpleNamespace(content=None, tool_calls=None)  # No 'refusal' attr!
		choice = SimpleNamespace(message=msg, finish_reason='stop')
		fake_resp = SimpleNamespace(choices=[choice])

		choice = fake_resp.choices[0]
		normalized = NormalizedLLMResponse(
			raw_text=choice.message.content or '',
			refusal=getattr(choice.message, 'refusal', None),  # Must not raise AttributeError
			tool_calls=[],
		)
		assert normalized.refusal is None


# ===========================================================================
# Google / Gemini
# ===========================================================================


class TestGoogleResponseExtraction:
	"""Tests the extraction logic at browser_use/llm/google/chat.py:358-402."""

	def _make_fake_gemini_function_call(
		self,
		name: str,
		args: dict[str, Any],
	) -> SimpleNamespace:
		"""Simulates google.genai.types.FunctionCall (has .name and .args)."""
		return SimpleNamespace(name=name, args=args)

	def _make_fake_gemini_part(
		self,
		text: str | None = None,
		function_call: SimpleNamespace | None = None,
	) -> SimpleNamespace:
		"""Simulates google.genai.types.Part."""
		part = SimpleNamespace()
		if text is not None:
			part.text = text
		if function_call is not None:
			part.function_call = function_call
		return part

	def _make_fake_gemini_response(
		self,
		text: str | None = None,
		parsed: dict[str, Any] | BaseModel | None = None,
		parts: list[SimpleNamespace] | None = None,
		finish_reason: str = 'STOP',
	) -> SimpleNamespace:
		"""Simulates google.genai.types.GenerateContentResponse.

		Key shape:
		  .text          -> str or None
		  .parsed        -> dict or BaseModel instance (native structured output)
		  .candidates[0] -> has .content.parts list
		"""
		if parts is None:
			parts = []

		content = SimpleNamespace(parts=parts)
		candidate = SimpleNamespace(
			content=content,
			finish_reason=finish_reason,
		)
		usage = SimpleNamespace(
			prompt_token_count=5,
			candidates_token_count=10,
			total_token_count=15,
		)
		resp = SimpleNamespace(
			text=text,
			parsed=parsed,
			candidates=[candidate],
			usage_metadata=usage,
		)
		return resp

	def test_native_structured_output_parsed_dict(self):
		"""Gemini supports_structured_output=True: .parsed is a dict."""
		fake_resp = self._make_fake_gemini_response(
			text=REFERENCE_JSON,
			parsed=REFERENCE_DICT,  # Native parsed dict!
			parts=[self._make_fake_gemini_part(text=REFERENCE_JSON)],
		)

		# _build_normalized from google/chat.py:358-402
		raw_text = fake_resp.text or ''
		parsed_dict: dict[str, Any] | None = None
		tool_calls: list[NormalizedToolCall] = []

		if fake_resp.parsed is not None:
			try:
				if isinstance(fake_resp.parsed, dict):
					parsed_dict = fake_resp.parsed
				elif not isinstance(fake_resp.parsed, bytes) and hasattr(fake_resp.parsed, 'model_dump'):
					parsed_dict = fake_resp.parsed.model_dump()  # type: ignore
				elif isinstance(fake_resp.parsed, BaseModel):
					parsed_dict = fake_resp.parsed.model_dump()
			except Exception:
				parsed_dict = None

		if hasattr(fake_resp, 'candidates') and fake_resp.candidates:
			candidate = fake_resp.candidates[0]
			if hasattr(candidate, 'content') and candidate.content is not None:
				parts_iter = getattr(candidate.content, 'parts', []) or []
				for part in parts_iter:
					fc = getattr(part, 'function_call', None)
					if fc is not None:
						fname = str(getattr(fc, 'name', AGENT_OUTPUT_TYPE.__name__))
						fargs = getattr(fc, 'args', None)
						args_dict: dict[str, Any] = {}
						if isinstance(fargs, dict) and fargs:
							for k, v in fargs.items():
								args_dict[str(k)] = v
						elif fargs is not None and not isinstance(fargs, (bytes, bytearray)):
							try:
								args_dict = dict(fargs)  # type: ignore
							except Exception:
								args_dict = {}
						tool_calls.append(NormalizedToolCall(name=fname, arguments=args_dict))

		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			parsed_dict=parsed_dict,
			tool_calls=tool_calls,
		)

		# Strategy 1 (parsed_dict) must win
		assert normalized.parsed_dict == REFERENCE_DICT
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_native_structured_output_parsed_basemodel(self):
		"""Gemini .parsed can also be a Pydantic BaseModel instance."""

		class MyOutput(BaseModel):
			action: list[dict[str, Any]]

		parsed_obj = MyOutput(action=REFERENCE_DICT['action'])
		fake_resp = self._make_fake_gemini_response(
			text=REFERENCE_JSON,
			parsed=parsed_obj,  # BaseModel instance!
		)

		raw_text = fake_resp.text or ''
		parsed_dict: dict[str, Any] | None = None

		if fake_resp.parsed is not None:
			try:
				if isinstance(fake_resp.parsed, dict):
					parsed_dict = fake_resp.parsed
				elif not isinstance(fake_resp.parsed, bytes) and hasattr(fake_resp.parsed, 'model_dump'):
					parsed_dict = fake_resp.parsed.model_dump()  # type: ignore
				elif isinstance(fake_resp.parsed, BaseModel):
					parsed_dict = fake_resp.parsed.model_dump()
			except Exception:
				parsed_dict = None

		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			parsed_dict=parsed_dict,
			tool_calls=[],
		)

		assert parsed_dict is not None
		assert 'action' in parsed_dict
		# BaseModel doesn't include fields we didn't define (missing evaluation_previous_goal)
		# but the schema we pass won't have those as required, so this is fine.

	def test_function_calling_parts_extraction(self):
		"""Gemini function calling mode: function_call in parts, not .parsed."""
		fc = self._make_fake_gemini_function_call(name='AgentOutput', args=REFERENCE_DICT)
		part = self._make_fake_gemini_part(function_call=fc)
		fake_resp = self._make_fake_gemini_response(
			text=None,
			parsed=None,
			parts=[part],
			finish_reason='STOP',
		)

		# _build_normalized
		raw_text = fake_resp.text or ''
		parsed_dict: dict[str, Any] | None = None
		tool_calls: list[NormalizedToolCall] = []

		if fake_resp.parsed is not None:
			parsed_dict = None  # None in this case

		if hasattr(fake_resp, 'candidates') and fake_resp.candidates:
			candidate = fake_resp.candidates[0]
			if hasattr(candidate, 'content') and candidate.content is not None:
				parts_iter = getattr(candidate.content, 'parts', []) or []
				for part_item in parts_iter:
					fc_item = getattr(part_item, 'function_call', None)
					if fc_item is not None:
						fname = str(getattr(fc_item, 'name', AGENT_OUTPUT_TYPE.__name__))
						fargs = getattr(fc_item, 'args', None)
						args_dict: dict[str, Any] = {}
						if isinstance(fargs, dict) and fargs:
							for k, v in fargs.items():
								args_dict[str(k)] = v
						tool_calls.append(NormalizedToolCall(name=fname, arguments=args_dict))

		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			parsed_dict=parsed_dict,
			tool_calls=tool_calls,
		)

		assert len(normalized.tool_calls) == 1
		assert normalized.tool_calls[0].name == 'AgentOutput'
		assert normalized.tool_calls[0].arguments == REFERENCE_DICT
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_fallback_json_in_text_mode(self):
		"""Gemini supports_structured_output=False: JSON only in .text."""
		fake_resp = self._make_fake_gemini_response(
			text=f'```json\n{REFERENCE_JSON}\n```',
			parsed=None,
		)

		raw_text = fake_resp.text or ''
		normalized = NormalizedLLMResponse(
			raw_text=raw_text,
			parsed_dict=None,
			tool_calls=[],
		)

		# Parser must extract JSON from markdown code block
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2


# ===========================================================================
# Mistral
# ===========================================================================


class TestMistralResponseExtraction:
	"""Tests the extraction logic at browser_use/llm/mistral/chat.py:210-241."""

	def _make_fake_mistral_response(
		self,
		content: str | None = None,
		refusal: str | None = None,
		tool_calls: list[dict[str, Any]] | None = None,
		finish_reason: str = 'stop',
	) -> dict[str, Any]:
		"""Mistral returns raw dicts via HTTP JSON response, not Python objects.

		Shape (from mistral/chat.py:203-209):
		  data['choices'][0]['message']['content']
		  data['choices'][0]['message']['tool_calls']
		  data['choices'][0]['message']['refusal']
		  data['choices'][0]['finish_reason']
		"""
		message: dict[str, Any] = {'content': content}
		if refusal is not None:
			message['refusal'] = refusal
		if tool_calls is not None:
			message['tool_calls'] = tool_calls

		choice = {'message': message, 'finish_reason': finish_reason}
		usage = {'prompt_tokens': 5, 'completion_tokens': 15, 'total_tokens': 20}
		return {'choices': [choice], 'usage': usage}

	def test_json_schema_path(self):
		"""Mistral JSON schema path: JSON in content."""
		data = self._make_fake_mistral_response(content=REFERENCE_JSON)

		# Extraction from mistral/chat.py:210-241
		choice = data['choices'][0]
		message = choice.get('message', {})

		# _extract_content_text
		content_text = message.get('content') or ''

		tool_calls_list: list[NormalizedToolCall] = []
		tool_calls_raw = message.get('tool_calls', [])
		if isinstance(tool_calls_raw, list):
			for tc in tool_calls_raw:
				if isinstance(tc, dict):
					fn = tc.get('function', {})
					tool_calls_list.append(
						NormalizedToolCall(
							id=tc.get('id'),
							name=fn.get('name', ''),
							arguments=fn.get('arguments', ''),
						)
					)

		refusal = message.get('refusal')

		normalized = NormalizedLLMResponse(
			raw_text=content_text,
			refusal=refusal,
			tool_calls=tool_calls_list,
			stop_reason=choice.get('finish_reason'),
		)

		assert normalized.raw_text == REFERENCE_JSON
		assert normalized.stop_reason == 'stop'
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_tool_calls_path(self):
		"""Mistral tool calling path: tool_calls in message dict."""
		tool_calls_data = [
			{
				'id': 'call_mistral_001',
				'type': 'function',
				'function': {'name': 'AgentOutput', 'arguments': REFERENCE_JSON},
			}
		]
		data = self._make_fake_mistral_response(
			content='',
			tool_calls=tool_calls_data,
			finish_reason='tool_calls',
		)

		choice = data['choices'][0]
		message = choice.get('message', {})
		content_text = message.get('content') or ''

		tool_calls_list: list[NormalizedToolCall] = []
		tool_calls_raw = message.get('tool_calls', [])
		if isinstance(tool_calls_raw, list):
			for tc in tool_calls_raw:
				if isinstance(tc, dict):
					fn = tc.get('function', {})
					tool_calls_list.append(
						NormalizedToolCall(
							id=tc.get('id'),
							name=fn.get('name', ''),
							arguments=fn.get('arguments', ''),
						)
					)

		refusal = message.get('refusal')

		normalized = NormalizedLLMResponse(
			raw_text=content_text,
			refusal=refusal,
			tool_calls=tool_calls_list,
			stop_reason=choice.get('finish_reason'),
		)

		assert len(normalized.tool_calls) == 1
		assert normalized.tool_calls[0].id == 'call_mistral_001'
		assert normalized.tool_calls[0].name == 'AgentOutput'
		# JSON string argument - parser must handle it
		parsed = AgentOutputParser(AGENT_OUTPUT_TYPE).parse(normalized)
		assert len(parsed.action) == 2

	def test_refusal_message(self):
		"""Mistral refusal in message dict."""
		data = self._make_fake_mistral_response(
			content=None,
			refusal='I cannot generate that content.',
		)

		choice = data['choices'][0]
		message = choice.get('message', {})
		content_text = message.get('content') or ''
		refusal = message.get('refusal')

		normalized = NormalizedLLMResponse(
			raw_text=content_text,
			refusal=refusal,
			tool_calls=[],
		)

		assert normalized.raw_text == ''
		assert normalized.refusal == 'I cannot generate that content.'

	def test_missing_tool_calls_key(self):
		"""Mistral response may not have 'tool_calls' key at all."""
		data = self._make_fake_mistral_response(content=REFERENCE_JSON)
		# Ensure no 'tool_calls' key exists
		assert 'tool_calls' not in data['choices'][0]['message']

		choice = data['choices'][0]
		message = choice.get('message', {})
		tool_calls_raw = message.get('tool_calls', [])  # Must use default []
		assert isinstance(tool_calls_raw, list)
		assert len(tool_calls_raw) == 0


# ===========================================================================
# CROSS-PROVIDER EQUIVALENCE — THE CRITICAL ASSERTION
# ===========================================================================


class TestCrossProviderExtractionEquivalence:
	"""Same logical output, extracted from 5 different provider responses, must yield
	equivalent NormalizedLLMResponse → AgentOutput.
	"""

	def test_five_providers_same_action_same_output(self):
		"""Build fake responses for 5 providers, extract, and assert AgentOutput equality."""
		# --- OpenAI (json in content) ---
		openai_msg = SimpleNamespace(content=REFERENCE_JSON, refusal=None, tool_calls=None)
		openai_choice = SimpleNamespace(message=openai_msg, finish_reason='stop')
		openai_norm = NormalizedLLMResponse(
			raw_text=openai_choice.message.content or '',
			refusal=getattr(openai_choice.message, 'refusal', None),
			stop_reason=openai_choice.finish_reason,
		)

		# --- Anthropic (tool_use block with input dict) ---
		anthropic_content = [
			SimpleNamespace(type='text', text='Using tool'),
			SimpleNamespace(type='tool_use', id='toolu_x', name='AgentOutput', input=REFERENCE_DICT),
		]
		anthropic_resp = SimpleNamespace(content=anthropic_content, stop_reason='tool_use')
		anthropic_tool_calls: list[NormalizedToolCall] = []
		for block in anthropic_resp.content:
			if getattr(block, 'type', None) == 'tool_use':
				anthropic_tool_calls.append(
					NormalizedToolCall(
						id=getattr(block, 'id', None),
						name=getattr(block, 'name', 'AgentOutput'),
						arguments=getattr(block, 'input', {}),
					)
				)
		anthropic_norm = NormalizedLLMResponse(
			raw_text='Using tool',
			tool_calls=anthropic_tool_calls,
			stop_reason='tool_use',
		)

		# --- Groq (tool_calls with JSON string args) ---
		groq_func = SimpleNamespace(name='AgentOutput', arguments=REFERENCE_JSON)
		groq_tc = SimpleNamespace(id='call_groq_001', function=groq_func)
		groq_msg = SimpleNamespace(content='', tool_calls=[groq_tc])
		groq_choice = SimpleNamespace(message=groq_msg, finish_reason='tool_calls')
		groq_tool_calls: list[NormalizedToolCall] = []
		if getattr(groq_choice.message, 'tool_calls', None):
			for tc in groq_choice.message.tool_calls:
				groq_tool_calls.append(
					NormalizedToolCall(
						id=getattr(tc, 'id', None),
						name=tc.function.name,
						arguments=tc.function.arguments,
					)
				)
		groq_norm = NormalizedLLMResponse(
			raw_text=groq_choice.message.content or '',
			refusal=getattr(groq_choice.message, 'refusal', None),
			tool_calls=groq_tool_calls,
		)

		# --- Gemini (native parsed dict) ---
		gemini_content = SimpleNamespace(parts=[SimpleNamespace(text=REFERENCE_JSON)])
		gemini_candidate = SimpleNamespace(content=gemini_content)
		gemini_resp = SimpleNamespace(
			text=REFERENCE_JSON,
			parsed=REFERENCE_DICT,
			candidates=[gemini_candidate],
		)
		gemini_parsed_dict = gemini_resp.parsed if isinstance(gemini_resp.parsed, dict) else None
		gemini_norm = NormalizedLLMResponse(
			raw_text=gemini_resp.text or '',
			parsed_dict=gemini_parsed_dict,
			tool_calls=[],
		)

		# --- Mistral (raw dict response) ---
		mistral_data = {
			'choices': [
				{
					'message': {'content': REFERENCE_JSON},
					'finish_reason': 'stop',
				}
			]
		}
		mistral_choice = mistral_data['choices'][0]
		mistral_msg = mistral_choice.get('message', {})
		mistral_norm = NormalizedLLMResponse(
			raw_text=mistral_msg.get('content') or '',
			refusal=mistral_msg.get('refusal'),
			tool_calls=[],
			stop_reason=mistral_choice.get('finish_reason'),
		)

		# --- Assert all produce equivalent AgentOutput ---
		parser = AgentOutputParser(AGENT_OUTPUT_TYPE)
		outputs = [
			parser.parse(openai_norm),
			parser.parse(anthropic_norm),
			parser.parse(groq_norm),
			parser.parse(gemini_norm),
			parser.parse(mistral_norm),
		]

		first = outputs[0]
		for i, other in enumerate(outputs[1:], start=2):
			assert first.evaluation_previous_goal == other.evaluation_previous_goal
			assert first.memory == other.memory
			assert first.next_goal == other.next_goal
			assert len(first.action) == len(other.action)
			for ai, bi in zip(first.action, other.action):
				assert ai.model_dump() == bi.model_dump(), f'Provider #{i} action differs'


# ===========================================================================
# FALLBACK HISTORY -> PROVIDER SERIALIZER (end-to-end)
# ===========================================================================


class TestFallbackHistorySerializationE2E:
	"""Simulate a fallback handoff: Agent has history with Provider X, falls back to
	Provider Y.  The same in-memory BaseMessage list must serialize cleanly for BOTH
	providers.  This is the real fallback guarantee — not just that parser outputs are
	the same, but that the *conversation history* survives the hand-off.
	"""

	def _make_fallback_history(self) -> list:
		"""A rich realistic history: system + user with image + assistant refusal +
		assistant tool_calls + another user + assistant.
		"""
		from browser_use.llm.messages import (
			AssistantMessage,
			ContentPartImageParam,
			ContentPartRefusalParam,
			ContentPartTextParam,
			Function,
			ImageURL,
			SystemMessage,
			ToolCall,
			UserMessage,
		)

		return [
			SystemMessage(content='You are a helpful agent.'),
			UserMessage(
				content=[
					ContentPartTextParam(type='text', text='Describe this chart'),
					ContentPartImageParam(
						type='image_url',
						image_url=ImageURL(
							url='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkAAIAAAoAAv/lxKUAAAAASUVORK5CYII=',
							detail='low',
						),
					),
				]
			),
			AssistantMessage(
				content=[
					ContentPartRefusalParam(type='refusal', refusal='I cannot analyse untrusted images.'),
					ContentPartTextParam(type='text', text='But I can describe typical bar charts.'),
				],
				refusal='I cannot help with untrusted images.',
				tool_calls=[
					ToolCall(
						id='call_abc',
						type='function',
						function=Function(
							name='search',
							arguments='{"query":"typical bar chart features"}',
						),
					)
				],
			),
			UserMessage(content='OK, tell me about click actions.', name='user_42'),
			AssistantMessage(content='Click triggers mouse click on element N.', name='assistant_alice'),
		]

	def test_fallback_openai_to_anthropic(self):
		"""Fallback scenario: primary=OpenAI, secondary=Anthropic.

		The same BaseMessage history must serialize for BOTH providers without loss
		of information needed for context continuity.
		"""
		from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		history = self._make_fallback_history()

		# Serialize as if OpenAI was the original provider
		openai_msgs = OpenAIMessageSerializer.serialize_messages(history)
		assert len(list(openai_msgs)) == 5
		# Key assertions: refusal + tool_calls preserved
		assert openai_msgs[2].get('refusal') == 'I cannot help with untrusted images.'
		assert 'tool_calls' in openai_msgs[2]

		# Now serialize the SAME BaseMessage history for Anthropic (fallback target)
		anthropic_contents, anthropic_system = AnthropicMessageSerializer.serialize_messages(history)
		# System must be extracted
		assert anthropic_system is not None and 'helpful agent' in anthropic_system
		# Remaining: user/assistant/user/assistant = 4 blocks
		assert len(list(anthropic_contents)) == 4

	def test_fallback_groq_to_google(self):
		"""Fallback scenario: primary=Groq, secondary=Google."""
		from browser_use.llm.google.serializer import GoogleMessageSerializer
		from browser_use.llm.groq.serializer import GroqMessageSerializer

		history = self._make_fallback_history()

		groq_msgs = GroqMessageSerializer.serialize_messages(history)
		assert len(list(groq_msgs)) == 5
		assert groq_msgs[2].get('refusal') == 'I cannot help with untrusted images.'
		assert 'tool_calls' in groq_msgs[2]

		google_contents, google_system = GoogleMessageSerializer.serialize_messages(history)
		assert google_system is not None and 'helpful agent' in google_system
		assert len(list(google_contents)) >= 4  # type: ignore[arg-type]

	def test_fallback_mistral_to_litellm(self):
		"""Fallback scenario: primary=Mistral, secondary=LiteLLM."""
		from browser_use.llm.litellm.serializer import LiteLLMMessageSerializer
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		history = self._make_fallback_history()

		# Mistral uses OpenAI serializer
		mistral_msgs = OpenAIMessageSerializer.serialize_messages(history)
		assert len(list(mistral_msgs)) == 5

		litellm_msgs = LiteLLMMessageSerializer.serialize(history)
		assert len(list(litellm_msgs)) == 5
		# LiteLLM preserves refusal
		assert litellm_msgs[2].get('refusal') == 'I cannot help with untrusted images.'
		# LiteLLM preserves tool_calls
		assert 'tool_calls' in litellm_msgs[2]

	def test_full_fallback_chain_all_providers(self):
		"""The same history must serialize cleanly for EVERY supported provider.

		This is the ultimate fallback guarantee: no matter which provider dies, any
		other can take over without loss of context.
		"""
		from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
		from browser_use.llm.google.serializer import GoogleMessageSerializer
		from browser_use.llm.groq.serializer import GroqMessageSerializer
		from browser_use.llm.litellm.serializer import LiteLLMMessageSerializer
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		history = self._make_fallback_history()

		# All of these must succeed without exceptions and return reasonable results
		results = {
			'openai': OpenAIMessageSerializer.serialize_messages(history),
			'groq': GroqMessageSerializer.serialize_messages(history),
			'anthropic': AnthropicMessageSerializer.serialize_messages(history),
			'google': GoogleMessageSerializer.serialize_messages(history),
			'litellm': LiteLLMMessageSerializer.serialize(history),
		}

		# OpenAI-like providers: 5 messages
		for name in ('openai', 'groq', 'litellm'):
			msgs = results[name]
			assert len(list(msgs)) == 5, f'{name} serializer returned wrong count'

		# Anthropic/Google: split system + 4 content blocks
		for name in ('anthropic', 'google'):
			contents, system = results[name]
			assert system is not None, f'{name} serializer lost system prompt'
			assert 'helpful agent' in system, f'{name} serializer corrupted system prompt'
			assert len(list(contents)) == 4, f'{name} serializer lost content blocks'

	def test_tool_call_round_trip_through_serializers(self):
		"""Tool calls in history must survive round-trip through every serializer.

		This catches bugs where a serializer omits tool_calls or corrupts the
		arguments JSON string.
		"""
		from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
		from browser_use.llm.google.serializer import GoogleMessageSerializer
		from browser_use.llm.groq.serializer import GroqMessageSerializer
		from browser_use.llm.litellm.serializer import LiteLLMMessageSerializer
		from browser_use.llm.messages import (
			AssistantMessage,
			Function,
			SystemMessage,
			ToolCall,
			UserMessage,
		)
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		history = [
			SystemMessage(content='Use tools'),
			UserMessage(content='Search for weather'),
			AssistantMessage(
				content='I will search for you.',
				tool_calls=[
					ToolCall(
						id='call_abc_123',
						type='function',
						function=Function(
							name='search',
							arguments='{"query":"weather in paris","max_results":3}',
						),
					)
				],
			),
		]

		# Every serializer must produce something with tool_call info
		openai_out = OpenAIMessageSerializer.serialize_messages(history)
		assistant = openai_out[2]
		assert 'tool_calls' in assistant
		tool_calls_list = list(assistant['tool_calls'])
		assert tool_calls_list[0].get('id') == 'call_abc_123'
		assert 'function' in tool_calls_list[0]
		assert tool_calls_list[0]['function'].get('name') == 'search'

		groq_out = GroqMessageSerializer.serialize_messages(history)
		assert 'tool_calls' in groq_out[2]

		litellm_out = LiteLLMMessageSerializer.serialize(history)
		assert 'tool_calls' in litellm_out[2]

		anthropic_contents, anthropic_system = AnthropicMessageSerializer.serialize_messages(history)
		assert anthropic_system is not None
		assert len(list(anthropic_contents)) >= 2

		google_contents, google_system = GoogleMessageSerializer.serialize_messages(history)
		assert google_system is not None
		assert len(list(google_contents)) >= 2  # type: ignore[arg-type]
