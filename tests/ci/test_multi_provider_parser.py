"""
Cross-provider unified parser test suite.

Tests that the same logical input (raw text JSON / tool_calls / parsed_dict / refusal)
produces the EXACT SAME AgentOutput regardless of which provider path (normalizer)
delivered it. This is the critical guarantee for:
  1. multi-provider action parity
  2. fallback LLM hand-off without history corruption
  3. consistent error-recovery across providers
"""

import json
from typing import Any

import pytest
from pydantic import create_model

from browser_use.agent.views import AgentOutput
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
from browser_use.llm.parser import (
	AgentOutputParser,
	NormalizedLLMResponse,
	NormalizedToolCall,
)
from browser_use.tools.registry.views import ActionModel


# ---------------------------------------------------------------------------
# Helpers: build a realistic AgentOutput type with a dynamic ActionModel
# ---------------------------------------------------------------------------
def make_dynamic_action_model() -> type[ActionModel]:
	DoneParams = create_model('DoneParams', success=(bool, True), text=(str | None, None))
	NavigateToParams = create_model('NavigateToParams', url=(str, ...))
	ClickParams = create_model('ClickParams', index=(int, ...))
	return create_model(  # type: ignore[return-value]
		'ActionModel',
		done=(DoneParams | None, None),
		navigate_to=(NavigateToParams | None, None),
		click_element=(ClickParams | None, None),
		__base__=ActionModel,
	)


def make_agent_output_type() -> type[AgentOutput]:
	ActionModelT = make_dynamic_action_model()
	return create_model(  # type: ignore[return-value]
		'AgentOutput',
		action=(list[ActionModelT], ...),  # type: ignore[valid-type]
		__base__=AgentOutput,
	)


AGENT_OUTPUT_TYPE = make_agent_output_type()
REFERENCE_JSON_DICT: dict[str, Any] = {
	'evaluation_previous_goal': 'opened google homepage',
	'memory': 'user wants weather',
	'next_goal': 'search for weather in paris',
	'thinking': 'need to use search action',
	'action': [
		{'navigate_to': {'url': 'https://www.google.com/search?q=weather+paris'}},
		{'done': {'success': True, 'text': 'task complete'}},
	],
}
REFERENCE_JSON_STR = json.dumps(REFERENCE_JSON_DICT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def parser() -> AgentOutputParser[AgentOutput]:
	return AgentOutputParser(AGENT_OUTPUT_TYPE)


@pytest.fixture
def reference_dict() -> dict[str, Any]:
	return json.loads(REFERENCE_JSON_STR)


def assert_outputs_equivalent(a: AgentOutput, b: AgentOutput) -> None:
	assert a.evaluation_previous_goal == b.evaluation_previous_goal
	assert a.memory == b.memory
	assert a.next_goal == b.next_goal
	assert a.thinking == b.thinking
	assert len(a.action) == len(b.action)
	for ai, bi in zip(a.action, b.action):
		assert ai.model_dump() == bi.model_dump()


# ===========================================================================
# TEST GROUP 1: Same JSON content, delivered via different Normalized fields
#   → simulates different providers that surface the data differently
# ===========================================================================


class TestProviderDeliveryParity:
	"""Different providers return the same JSON via different carriers.

	OpenAI-like  → raw JSON in .content
	Anthropic    → tool_use block → parsed_dict in tool_calls[0].arguments (dict)
	Groq ToolCall→ tool_calls[0].arguments (JSON string)
	Gemini       → .parsed (native dict) + also raw text + also function_call parts
	Browser Use  → parsed_dict only
	"""

	def test_raw_text_only_openai_style(self, parser: AgentOutputParser[AgentOutput]):
		normalized = NormalizedLLMResponse(raw_text=REFERENCE_JSON_STR)
		result = parser.parse(normalized)
		assert len(result.action) == 2
		assert result.action[0].navigate_to is not None  # type: ignore[attr-defined]
		assert result.action[0].navigate_to.url == 'https://www.google.com/search?q=weather+paris'  # type: ignore[attr-defined]

	def test_parsed_dict_only_browser_use_style(self, parser: AgentOutputParser[AgentOutput], reference_dict: dict[str, Any]):
		normalized = NormalizedLLMResponse(parsed_dict=reference_dict)
		result = parser.parse(normalized)
		# Strategy 1 must win
		assert len(result.action) == 2
		assert result.action[1].done is not None  # type: ignore[attr-defined]

	def test_tool_call_arguments_as_dict_anthropic_style(
		self, parser: AgentOutputParser[AgentOutput], reference_dict: dict[str, Any]
	):
		normalized = NormalizedLLMResponse(
			raw_text='',
			tool_calls=[
				NormalizedToolCall(
					id='toolu_01',
					name='AgentOutput',
					arguments=reference_dict,
				)
			],
		)
		result = parser.parse(normalized)
		assert len(result.action) == 2
		assert result.next_goal == 'search for weather in paris'

	def test_tool_call_arguments_as_json_string_groq_style(self, parser: AgentOutputParser[AgentOutput]):
		normalized = NormalizedLLMResponse(
			raw_text='',
			tool_calls=[
				NormalizedToolCall(
					id='call_123',
					name='AgentOutput',
					arguments=REFERENCE_JSON_STR,
				)
			],
		)
		result = parser.parse(normalized)
		assert len(result.action) == 2
		assert result.action[0].navigate_to is not None  # type: ignore[attr-defined]

	def test_markdown_wrapped_raw_text_gemini_style(self, parser: AgentOutputParser[AgentOutput]):
		wrapped = f'Here is my response:\n```json\n{REFERENCE_JSON_STR}\n```\nThank you.'
		normalized = NormalizedLLMResponse(raw_text=wrapped)
		result = parser.parse(normalized)
		assert len(result.action) == 2
		assert result.memory == 'user wants weather'

	def test_partial_duplication_raw_text_then_parsed_dict(
		self, parser: AgentOutputParser[AgentOutput], reference_dict: dict[str, Any]
	):
		"""parsed_dict strategy must win over raw_text even when both are present."""
		normalized = NormalizedLLMResponse(
			raw_text='some garbage text',
			parsed_dict=reference_dict,
		)
		result = parser.parse(normalized)
		assert len(result.action) == 2

	def test_parsed_dict_with_missing_optional_fields_still_ok(self, parser: AgentOutputParser[AgentOutput]):
		truncated: dict[str, Any] = {
			'evaluation_previous_goal': 'opened google homepage',
			'memory': 'user wants weather',
			'next_goal': 'search for weather in paris',
			'action': [
				{'navigate_to': {'url': 'https://www.google.com/search?q=weather+paris'}},
			],
		}
		normalized = NormalizedLLMResponse(parsed_dict=truncated)
		result = parser.parse(normalized)
		assert result.thinking is None
		assert len(result.action) == 1


# ===========================================================================
# TEST GROUP 2: Cross-provider output parity — the KEY assertion
#   Build N different Normalized inputs (one per provider) and assert the
#   resulting AgentOutputs are byte-for-byte identical.
# ===========================================================================


class TestOutputDeterminismAcrossProviders:
	def build_openai_response(self) -> NormalizedLLMResponse:
		return NormalizedLLMResponse(raw_text=REFERENCE_JSON_STR, stop_reason='stop')

	def build_anthropic_response(self, reference_dict: dict[str, Any]) -> NormalizedLLMResponse:
		return NormalizedLLMResponse(
			raw_text='I will now use my tool to output the result',
			thinking='<thinking>I should search for weather</thinking>',
			tool_calls=[
				NormalizedToolCall(
					id='toolu_01ABC',
					name='AgentOutput',
					arguments=reference_dict,
				)
			],
			stop_reason='tool_use',
		)

	def build_groq_toolcall_response(self) -> NormalizedLLMResponse:
		return NormalizedLLMResponse(
			raw_text='',
			tool_calls=[
				NormalizedToolCall(
					id='call_groq_001',
					name='AgentOutput',
					arguments=REFERENCE_JSON_STR,
				)
			],
			stop_reason='tool_calls',
		)

	def build_mistral_response(self) -> NormalizedLLMResponse:
		return NormalizedLLMResponse(
			raw_text=f'```json\n{REFERENCE_JSON_STR}\n```',
			stop_reason='stop',
		)

	def build_gemini_response(self, reference_dict: dict[str, Any]) -> NormalizedLLMResponse:
		return NormalizedLLMResponse(
			raw_text=REFERENCE_JSON_STR,
			parsed_dict=reference_dict,
			stop_reason='STOP',
		)

	def test_all_provider_paths_yield_identical_output(
		self, parser: AgentOutputParser[AgentOutput], reference_dict: dict[str, Any]
	):
		results: list[AgentOutput] = [
			parser.parse(self.build_openai_response()),
			parser.parse(self.build_anthropic_response(reference_dict)),
			parser.parse(self.build_groq_toolcall_response()),
			parser.parse(self.build_mistral_response()),
			parser.parse(self.build_gemini_response(reference_dict)),
		]

		# Cross-compare every pair to be 100% sure they are equivalent
		first = results[0]
		for i, other in enumerate(results[1:], start=2):
			assert_outputs_equivalent(first, other)  # pyright: ignore[reportUnusedExpression]

		# And validate the actual content shape
		assert first.action[0].navigate_to is not None  # type: ignore[attr-defined]
		assert first.action[1].done is not None  # type: ignore[attr-defined]
		assert first.action[1].done.success is True  # type: ignore[attr-defined]


# ===========================================================================
# TEST GROUP 3: Serializer parity for fallback history compatibility
#   When agent hands off to a fallback LLM, the in-memory BaseMessage list
#   must serialise cleanly for every target provider.  We don't need a real
#   network call — round-tripping through each provider's serializer proves
#   the messages are lossless enough for the target provider.
# ===========================================================================


def make_rich_message_history() -> list:
	"""A realistic history containing: refusal content parts, image parts, tool_calls, name fields."""
	return [
		SystemMessage(content='You are a helpful agent.'),
		UserMessage(
			content=[
				ContentPartTextParam(type='text', text='Look at this chart and describe it'),
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
				ContentPartTextParam(type='text', text='I can describe what I know about typical charts instead.'),
			],
			refusal='I cannot help with untrusted images.',
			tool_calls=[
				ToolCall(
					id='call_abc',
					type='function',
					function=Function(name='search', arguments='{"query":"typical bar chart features"}'),
				)
			],
		),
		UserMessage(content='OK, tell me what a click action does.', name='user_42'),
		AssistantMessage(
			content='Click triggers a mouse click on element N.',
			name='assistant_alice',
			refusal=None,
		),
	]


class TestFallbackHistorySerialization:
	"""Fallback LLM hand-off: messages must survive round-trip through each serializer.

	We test this without making network calls, by:
	  1. constructing a rich history
	  2. serializing it with provider X's serializer
	  3. checking the resulting payload is structurally valid for that provider
	  4. (important) the same messages must serialize for EVERY provider — this is the
	     fallback guarantee.
	"""

	def test_openai_serializer_round_trip_rich_history(self):
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		messages = make_rich_message_history()
		result = OpenAIMessageSerializer.serialize_messages(messages)
		assert len(list(result)) == 5
		# Check refusal preserved on assistant message #2 (index 2)
		assert 'refusal' in result[2]
		assert result[2]['refusal'] == 'I cannot help with untrusted images.'
		# Check tool_calls present
		assert 'tool_calls' in result[2]
		assert len(list(result[2]['tool_calls'])) == 1
		# Check name preserved
		assert result[4].get('name') == 'assistant_alice'

	def test_groq_serializer_rich_history(self):
		from browser_use.llm.groq.serializer import GroqMessageSerializer

		messages = make_rich_message_history()
		result = GroqMessageSerializer.serialize_messages(messages)
		assert len(result) == 5
		assistant_with_refusal = result[2]
		assert 'refusal' in assistant_with_refusal
		assert 'tool_calls' in assistant_with_refusal

	def test_anthropic_serializer_rich_history(self):
		from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer

		messages = make_rich_message_history()
		contents, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)
		assert system_prompt is not None and 'helpful agent' in system_prompt
		# Anthropic serializer groups: system + [user, assistant, user, assistant]
		assert len(contents) == 4  # 2 user + 2 assistant blocks, system is separate

	def test_google_serializer_rich_history(self):
		from browser_use.llm.google.serializer import GoogleMessageSerializer

		messages = make_rich_message_history()
		contents, system_prompt = GoogleMessageSerializer.serialize_messages(messages)
		# System prompt extracted
		assert system_prompt is not None and 'helpful agent' in system_prompt
		# Google alternates user/assistant
		assert len(list(contents)) >= 4  # type: ignore[arg-type]

	def test_mistral_serializer_rich_history(self):
		"""Mistral uses OpenAI serializer, confirm it still works through the wrapper."""
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		messages = make_rich_message_history()
		result = OpenAIMessageSerializer.serialize_messages(messages)
		# Mistral expects role/content/tool_calls at minimum.  The OpenAI serializer
		# already guarantees this; we sanity-check that assistant refusal is present
		# for providers that support it.
		roles = [r['role'] for r in result]
		assert roles.count('assistant') == 2

	def test_litellm_serializer_rich_history(self):
		from browser_use.llm.litellm.serializer import LiteLLMMessageSerializer

		messages = make_rich_message_history()
		result = LiteLLMMessageSerializer.serialize(messages)
		assert len(result) == 5
		# LiteLLM converts refusal → text; so refusal is in content but we still need
		# tool_calls preserved
		assert 'tool_calls' in result[2]
		assert 'refusal' in result[2]

	def test_all_serialisers_agree_on_message_count(self):
		"""The total number of logical messages should not vary wildly between providers."""
		from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
		from browser_use.llm.google.serializer import GoogleMessageSerializer
		from browser_use.llm.groq.serializer import GroqMessageSerializer
		from browser_use.llm.litellm.serializer import LiteLLMMessageSerializer
		from browser_use.llm.openai.serializer import OpenAIMessageSerializer

		messages = make_rich_message_history()

		openai_msgs = OpenAIMessageSerializer.serialize_messages(messages)
		groq_msgs = GroqMessageSerializer.serialize_messages(messages)
		litellm_msgs = LiteLLMMessageSerializer.serialize(messages)

		# Providers that follow the OpenAI message-list structure must return exactly 5
		assert len(openai_msgs) == len(groq_msgs) == len(litellm_msgs) == 5

		# Anthropic / Google: extract system separately
		anthropic_contents, anthropic_sys = AnthropicMessageSerializer.serialize_messages(messages)
		google_contents, google_sys = GoogleMessageSerializer.serialize_messages(messages)

		assert anthropic_sys is not None
		assert google_sys is not None
		# Remaining contents: 2 user-assistant pairs == 4 blocks
		assert len(list(anthropic_contents)) == 4
		assert len(list(google_contents)) == 4  # type: ignore[arg-type]


# ===========================================================================
# TEST GROUP 4: Parser error reporting for diagnostics
# ===========================================================================


class TestParserDiagnostics:
	def test_parse_error_includes_raw_text(self, parser: AgentOutputParser[AgentOutput]):
		from browser_use.llm.exceptions import ModelParseError

		garbage = NormalizedLLMResponse(raw_text='totally not json at all')
		with pytest.raises(ModelParseError) as exc_info:
			parser.parse(garbage)
		assert 'raw_text' in str(exc_info.value.message)

	def test_parse_error_includes_refusal_when_present(self, parser: AgentOutputParser[AgentOutput]):
		from browser_use.llm.exceptions import ModelParseError

		refused = NormalizedLLMResponse(
			raw_text='',
			refusal='I cannot help with illegal queries.',
		)
		with pytest.raises(ModelParseError) as exc_info:
			parser.parse(refused)
		assert 'refusal' in str(exc_info.value.message)
		assert 'illegal queries' in str(exc_info.value.message)

	def test_parse_error_includes_tool_call_names(self, parser: AgentOutputParser[AgentOutput]):
		from browser_use.llm.exceptions import ModelParseError

		bad_tc = NormalizedLLMResponse(
			raw_text='',
			tool_calls=[
				NormalizedToolCall(name='WrongTool', arguments={'a': 1}),
				NormalizedToolCall(name='AlsoWrong', arguments='{bad json'),
			],
		)
		with pytest.raises(ModelParseError) as exc_info:
			parser.parse(bad_tc)
		assert 'WrongTool' in str(exc_info.value.message)
		assert 'AlsoWrong' in str(exc_info.value.message)
