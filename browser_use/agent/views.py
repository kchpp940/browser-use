from __future__ import annotations

import hashlib
import json
import logging
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator
from typing_extensions import TypeVar
from uuid_extensions import uuid7str

from browser_use.agent.message_manager.views import MessageManagerState
from browser_use.browser.views import BrowserStateHistory
from browser_use.dom.views import DEFAULT_INCLUDE_ATTRIBUTES, DOMInteractedElement, DOMSelectorMap

# from browser_use.dom.history_tree_processor.service import (
# 	DOMElementNode,
# 	DOMHistoryElement,
# 	HistoryTreeProcessor,
# )
# from browser_use.dom.views import SelectorMap
from browser_use.filesystem.file_system import FileSystemState
from browser_use.llm.base import BaseChatModel
from browser_use.tokens.views import UsageSummary
from browser_use.tools.registry.views import ActionModel
from browser_use.utils import collect_sensitive_data_values, redact_sensitive_string

logger = logging.getLogger(__name__)


class MessageCompactionSettings(BaseModel):
	"""Summarizes older history into a compact memory block to reduce prompt size."""

	enabled: bool = True
	compact_every_n_steps: int = 25
	trigger_char_count: int | None = None  # Min char floor; set via trigger_token_count if preferred
	trigger_token_count: int | None = None  # Alternative to trigger_char_count (~4 chars/token)
	chars_per_token: float = 4.0
	keep_last_items: int = 6
	summary_max_chars: int = 6000
	include_read_state: bool = False
	compaction_llm: BaseChatModel | None = None

	@model_validator(mode='after')
	def _resolve_trigger_threshold(self) -> MessageCompactionSettings:
		if self.trigger_char_count is not None and self.trigger_token_count is not None:
			raise ValueError('Set trigger_char_count or trigger_token_count, not both.')
		if self.trigger_token_count is not None:
			self.trigger_char_count = int(self.trigger_token_count * self.chars_per_token)
		elif self.trigger_char_count is None:
			self.trigger_char_count = 40000  # ~10k tokens
		return self


class AgentSettings(BaseModel):
	"""Configuration options for the Agent"""

	use_vision: bool | Literal['auto'] = True
	vision_detail_level: Literal['auto', 'low', 'high'] = 'auto'
	save_conversation_path: str | Path | None = None
	save_conversation_path_encoding: str | None = 'utf-8'
	max_failures: int = 5
	generate_gif: bool | str = False
	override_system_message: str | None = None
	extend_system_message: str | None = None
	include_attributes: list[str] | None = DEFAULT_INCLUDE_ATTRIBUTES
	max_actions_per_step: int = 5
	use_thinking: bool = True
	flash_mode: bool = False  # If enabled, disables evaluation_previous_goal and next_goal, and sets use_thinking = False
	use_judge: bool = True
	ground_truth: str | None = None  # Ground truth answer or criteria for judge validation
	max_history_items: int | None = None
	message_compaction: MessageCompactionSettings | None = None
	enable_planning: bool = True
	planning_replan_on_stall: int = 3  # consecutive failures before replan nudge; 0 = disabled
	planning_exploration_limit: int = 5  # steps without a plan before nudge; 0 = disabled

	page_extraction_llm: BaseChatModel | None = None
	calculate_cost: bool = False
	include_tool_call_examples: bool = False
	llm_timeout: int = 60  # Timeout in seconds for LLM calls (auto-detected: 30s for gemini, 90s for o3, 60s default)
	step_timeout: int = 180  # Timeout in seconds for each step
	final_response_after_failure: bool = True  # If True, attempt one final recovery call after max_failures

	# Loop detection settings
	loop_detection_window: int = 20  # Rolling window size for action similarity tracking
	loop_detection_enabled: bool = True  # Whether to enable loop detection nudges
	max_clickable_elements_length: int = 40000  # Max characters for clickable elements in prompt


class PageFingerprint(BaseModel):
	"""Lightweight fingerprint of the browser page state."""

	model_config = ConfigDict(frozen=True)

	url: str
	element_count: int
	text_hash: str  # First 16 chars of SHA-256 of the DOM text representation

	@staticmethod
	def from_browser_state(url: str, dom_text: str, element_count: int) -> PageFingerprint:
		text_hash = hashlib.sha256(dom_text.encode('utf-8', errors='replace')).hexdigest()[:16]
		return PageFingerprint(url=url, element_count=element_count, text_hash=text_hash)


def _normalize_action_for_hash(action_name: str, params: dict[str, Any]) -> str:
	"""Normalize action parameters for similarity hashing.

	For search actions: strip minor keyword variations by sorting tokens.
	For click actions: hash by element type + rough text content, ignoring index.
	For navigate: hash by URL domain only.
	For others: hash by action_name + sorted params.
	"""
	if action_name == 'search':
		query = str(params.get('query', ''))
		# Normalize search: lowercase, sort tokens, collapse whitespace
		tokens = sorted(set(re.sub(r'[^\w\s]', ' ', query.lower()).split()))
		engine = params.get('engine', 'google')
		return f'search|{engine}|{"|".join(tokens)}'

	if action_name in ('click', 'input'):
		# For element-interaction actions, we only use the index (element identity).
		# Two clicks on the same element index are the same action.
		index = params.get('index')
		if action_name == 'input':
			text = str(params.get('text', ''))
			# Normalize input text: lowercase, strip whitespace
			return f'input|{index}|{text.strip().lower()}'
		return f'click|{index}'

	if action_name == 'navigate':
		url = str(params.get('url', ''))
		# Hash by full URL — navigating to different paths is genuine exploration,
		# only repeated navigation to the exact same URL is a loop signal.
		return f'navigate|{url}'

	if action_name == 'scroll':
		direction = 'down' if params.get('down', True) else 'up'
		index = params.get('index')
		return f'scroll|{direction}|{index}'

	# Default: hash by action name + sorted params (excluding None values)
	filtered = {k: v for k, v in sorted(params.items()) if v is not None}
	return f'{action_name}|{json.dumps(filtered, sort_keys=True, default=str)}'


def compute_action_hash(action_name: str, params: dict[str, Any]) -> str:
	"""Compute a stable hash string for an action based on type + normalized parameters."""
	normalized = _normalize_action_for_hash(action_name, params)
	return hashlib.sha256(normalized.encode('utf-8')).hexdigest()[:12]


class ActionLoopDetector(BaseModel):
	"""Tracks action repetition and page stagnation to detect behavioral loops.

	This is a soft detection system — it generates context messages for the LLM
	but never blocks actions. The agent can still repeat if it wants to.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	# Rolling window of recent action hashes
	window_size: int = 20
	recent_action_hashes: list[str] = Field(default_factory=list)

	# Page fingerprint tracking for stagnation detection
	recent_page_fingerprints: list[PageFingerprint] = Field(default_factory=list)

	# Current repetition state
	max_repetition_count: int = 0  # Highest count of any single hash in the window
	most_repeated_hash: str | None = None
	consecutive_stagnant_pages: int = 0  # How many consecutive steps had the same page fingerprint

	def record_action(self, action_name: str, params: dict[str, Any]) -> None:
		"""Record an action and update repetition statistics."""
		h = compute_action_hash(action_name, params)
		self.recent_action_hashes.append(h)
		# Trim to window size
		if len(self.recent_action_hashes) > self.window_size:
			self.recent_action_hashes = self.recent_action_hashes[-self.window_size :]
		self._update_repetition_stats()

	def record_page_state(self, url: str, dom_text: str, element_count: int) -> None:
		"""Record the current page fingerprint and update stagnation count."""
		fp = PageFingerprint.from_browser_state(url, dom_text, element_count)
		if self.recent_page_fingerprints and self.recent_page_fingerprints[-1] == fp:
			self.consecutive_stagnant_pages += 1
		else:
			self.consecutive_stagnant_pages = 0
		self.recent_page_fingerprints.append(fp)
		# Keep only last few fingerprints (no need for a large window)
		if len(self.recent_page_fingerprints) > 5:
			self.recent_page_fingerprints = self.recent_page_fingerprints[-5:]

	def _update_repetition_stats(self) -> None:
		"""Recompute max_repetition_count from the current window."""
		if not self.recent_action_hashes:
			self.max_repetition_count = 0
			self.most_repeated_hash = None
			return
		counts: dict[str, int] = {}
		for h in self.recent_action_hashes:
			counts[h] = counts.get(h, 0) + 1
		self.most_repeated_hash = max(counts, key=lambda k: counts[k])
		self.max_repetition_count = counts[self.most_repeated_hash]

	def get_nudge_message(self) -> str | None:
		"""Return an escalating awareness nudge based on repetition severity, or None if no loop detected."""
		messages: list[str] = []

		# Action repetition nudges (escalating at 5, 8, 12)
		if self.max_repetition_count >= 12:
			messages.append(
				f'Heads up: you have repeated a similar action {self.max_repetition_count} times '
				f'in the last {len(self.recent_action_hashes)} actions. '
				'If you are making progress with each repetition, keep going. '
				'If not, a different approach might get you there faster.'
			)
		elif self.max_repetition_count >= 8:
			messages.append(
				f'Heads up: you have repeated a similar action {self.max_repetition_count} times '
				f'in the last {len(self.recent_action_hashes)} actions. '
				'Are you still making progress with each attempt? '
				'If so, carry on. Otherwise, it might be worth trying a different approach.'
			)
		elif self.max_repetition_count >= 5:
			messages.append(
				f'Heads up: you have repeated a similar action {self.max_repetition_count} times '
				f'in the last {len(self.recent_action_hashes)} actions. '
				'If this is intentional and making progress, carry on. '
				'If not, it might be worth reconsidering your approach.'
			)

		# Page stagnation nudge
		if self.consecutive_stagnant_pages >= 5:
			messages.append(
				f'The page content has not changed across {self.consecutive_stagnant_pages} consecutive actions. '
				'Your actions might not be having the intended effect. '
				'It could be worth trying a different element or approach.'
			)

		if messages:
			return '\n\n'.join(messages)
		return None


class AgentState(BaseModel):
	"""Holds all state information for an Agent"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	agent_id: str = Field(default_factory=uuid7str)
	n_steps: int = 1
	consecutive_failures: int = 0
	last_result: list[ActionResult] | None = None
	plan: list[PlanItem] | None = None
	current_plan_item_index: int = 0
	plan_generation_step: int | None = None
	last_model_output: AgentOutput | None = None

	# Pause/resume state (kept serialisable for checkpointing)
	paused: bool = False
	stopped: bool = False
	session_initialized: bool = False  # Track if session events have been dispatched
	follow_up_task: bool = False  # Track if the agent is a follow-up task

	message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)
	file_system_state: FileSystemState | None = None

	# Loop detection state
	loop_detector: ActionLoopDetector = Field(default_factory=ActionLoopDetector)


@dataclass
class AgentStepInfo:
	step_number: int
	max_steps: int

	def is_last_step(self) -> bool:
		"""Check if this is the last step"""
		return self.step_number >= self.max_steps - 1


class JudgementResult(BaseModel):
	"""LLM judgement of agent trace"""

	reasoning: str | None = Field(default=None, description='Explanation of the judgement')
	verdict: bool = Field(description='Whether the trace was successful or not')
	failure_reason: str | None = Field(
		default=None,
		description='Max 5 sentences explanation of why the task was not completed successfully in case of failure. If verdict is true, use an empty string.',
	)
	impossible_task: bool = Field(
		default=False,
		description='True if the task was impossible to complete due to vague instructions, broken website, inaccessible links, missing login credentials, or other insurmountable obstacles',
	)
	reached_captcha: bool = Field(
		default=False,
		description='True if the agent encountered captcha challenges during task execution',
	)


class ActionResult(BaseModel):
	"""Result of executing an action"""

	# For done action
	is_done: bool | None = False
	success: bool | None = None

	# For trace judgement
	judgement: JudgementResult | None = None

	# Error handling - always include in long term memory
	error: str | None = None

	# Files
	attachments: list[str] | None = None  # Files to display in the done message

	# Images (base64 encoded) - separate from text content for efficient handling
	images: list[dict[str, Any]] | None = None  # [{"name": "file.jpg", "data": "base64_string"}]

	# Always include in long term memory
	long_term_memory: str | None = None  # Memory of this action

	# if update_only_read_state is True we add the extracted_content to the agent context only once for the next step
	# if update_only_read_state is False we add the extracted_content to the agent long term memory if no long_term_memory is provided
	extracted_content: str | None = None
	include_extracted_content_only_once: bool = False  # Whether the extracted content should be used to update the read_state

	# Metadata for observability (e.g., click coordinates)
	metadata: dict | None = None

	# Deprecated
	include_in_memory: bool = False  # whether to include in extracted_content inside long_term_memory

	@model_validator(mode='after')
	def validate_success_requires_done(self):
		"""Ensure success=True can only be set when is_done=True"""
		if self.success is True and self.is_done is not True:
			raise ValueError(
				'success=True can only be set when is_done=True. '
				'For regular actions that succeed, leave success as None. '
				'Use success=False only for actions that fail.'
			)
		return self


class RerunSummaryAction(BaseModel):
	"""AI-generated summary for rerun completion"""

	summary: str = Field(description='Summary of what happened during the rerun')
	success: bool = Field(description='Whether the rerun completed successfully based on visual inspection')
	completion_status: Literal['complete', 'partial', 'failed'] = Field(
		description='Status of rerun completion: complete (all steps succeeded), partial (some steps succeeded), failed (task did not complete)'
	)


class StepMetadata(BaseModel):
	"""Metadata for a single step including timing and token information"""

	step_start_time: float
	step_end_time: float
	step_number: int
	step_interval: float | None = None

	@property
	def duration_seconds(self) -> float:
		"""Calculate step duration in seconds"""
		return self.step_end_time - self.step_start_time


class PlanItem(BaseModel):
	text: str
	status: Literal['pending', 'current', 'done', 'skipped'] = 'pending'


class AgentBrain(BaseModel):
	thinking: str | None = None
	evaluation_previous_goal: str
	memory: str
	next_goal: str


class AgentOutput(BaseModel):
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	thinking: str | None = None
	evaluation_previous_goal: str | None = None
	memory: str | None = None
	next_goal: str | None = None
	current_plan_item: int | None = None
	plan_update: list[str] | None = None
	action: list[ActionModel] = Field(
		...,
		json_schema_extra={'min_items': 1},  # Ensure at least one action is provided
	)

	@classmethod
	def model_json_schema(cls, **kwargs):
		schema = super().model_json_schema(**kwargs)
		schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
		return schema

	@property
	def current_state(self) -> AgentBrain:
		"""For backward compatibility - returns an AgentBrain with the flattened properties"""
		return AgentBrain(
			thinking=self.thinking,
			evaluation_previous_goal=self.evaluation_previous_goal if self.evaluation_previous_goal else '',
			memory=self.memory if self.memory else '',
			next_goal=self.next_goal if self.next_goal else '',
		)

	@staticmethod
	def type_with_custom_actions(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions"""

		model_ = create_model(
			'AgentOutput',
			__base__=AgentOutput,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutput.__module__,
		)
		return model_

	@staticmethod
	def type_with_custom_actions_no_thinking(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions and exclude thinking field"""

		class AgentOutputNoThinking(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				del schema['properties']['thinking']
				schema['required'] = ['evaluation_previous_goal', 'memory', 'next_goal', 'action']
				return schema

		model = create_model(
			'AgentOutput',
			__base__=AgentOutputNoThinking,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutputNoThinking.__module__,
		)

		return model

	@staticmethod
	def type_with_custom_actions_flash_mode(custom_actions: type[ActionModel]) -> type[AgentOutput]:
		"""Extend actions with custom actions for flash mode - memory and action fields only"""

		class AgentOutputFlashMode(AgentOutput):
			@classmethod
			def model_json_schema(cls, **kwargs):
				schema = super().model_json_schema(**kwargs)
				# Remove thinking, evaluation_previous_goal, next_goal, and plan fields
				del schema['properties']['thinking']
				del schema['properties']['evaluation_previous_goal']
				del schema['properties']['next_goal']
				schema['properties'].pop('current_plan_item', None)
				schema['properties'].pop('plan_update', None)
				# Update required fields to only include remaining properties
				schema['required'] = ['memory', 'action']
				return schema

		model = create_model(
			'AgentOutput',
			__base__=AgentOutputFlashMode,
			action=(
				list[custom_actions],  # type: ignore
				Field(..., json_schema_extra={'min_items': 1}),
			),
			__module__=AgentOutputFlashMode.__module__,
		)

		return model


class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: list[ActionResult]
	state: BrowserStateHistory
	metadata: StepMetadata | None = None
	state_message: str | None = None

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

	@staticmethod
	def get_interacted_element(model_output: AgentOutput, selector_map: DOMSelectorMap) -> list[DOMInteractedElement | None]:
		elements = []
		for action in model_output.action:
			index = action.get_index()
			if index is not None and index in selector_map:
				el = selector_map[index]
				elements.append(DOMInteractedElement.load_from_enhanced_dom_tree(el))
			else:
				elements.append(None)
		return elements

	def _filter_sensitive_data_from_string(self, value: str, sensitive_data: dict[str, str | dict[str, str]] | None) -> str:
		"""Filter out sensitive data from a string value"""
		if not sensitive_data:
			return value

		sensitive_values = collect_sensitive_data_values(sensitive_data)

		# If there are no valid sensitive data entries, just return the original value
		if not sensitive_values:
			return value

		return redact_sensitive_string(value, sensitive_values)

	def _filter_sensitive_data_from_dict(
		self, data: dict[str, Any], sensitive_data: dict[str, str | dict[str, str]] | None
	) -> dict[str, Any]:
		"""Recursively filter sensitive data from a dictionary"""
		if not sensitive_data:
			return data

		filtered_data = {}
		for key, value in data.items():
			if isinstance(value, str):
				filtered_data[key] = self._filter_sensitive_data_from_string(value, sensitive_data)
			elif isinstance(value, dict):
				filtered_data[key] = self._filter_sensitive_data_from_dict(value, sensitive_data)
			elif isinstance(value, list):
				filtered_data[key] = [
					self._filter_sensitive_data_from_string(item, sensitive_data)
					if isinstance(item, str)
					else self._filter_sensitive_data_from_dict(item, sensitive_data)
					if isinstance(item, dict)
					else item
					for item in value
				]
			else:
				filtered_data[key] = value
		return filtered_data

	def model_dump(self, sensitive_data: dict[str, str | dict[str, str]] | None = None, **kwargs) -> dict[str, Any]:
		"""Custom serialization handling circular references and filtering sensitive data"""

		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = [action.model_dump(exclude_none=True, mode='json') for action in self.model_output.action]

			# Filter sensitive data only from input action parameters if sensitive_data is provided
			if sensitive_data:
				action_dump = [
					self._filter_sensitive_data_from_dict(action, sensitive_data) if 'input' in action else action
					for action in action_dump
				]

			model_output_dump = {
				'evaluation_previous_goal': self.model_output.evaluation_previous_goal,
				'memory': self.model_output.memory,
				'next_goal': self.model_output.next_goal,
				'action': action_dump,  # This preserves the actual action data
			}
			# Only include thinking if it's present
			if self.model_output.thinking is not None:
				model_output_dump['thinking'] = self.model_output.thinking
			if self.model_output.current_plan_item is not None:
				model_output_dump['current_plan_item'] = self.model_output.current_plan_item
			if self.model_output.plan_update is not None:
				model_output_dump['plan_update'] = self.model_output.plan_update

		# Handle result serialization - don't filter ActionResult data
		# as it should contain meaningful information for the agent
		result_dump = [r.model_dump(exclude_none=True, mode='json') for r in self.result]

		return {
			'model_output': model_output_dump,
			'result': result_dump,
			'state': self.state.to_dict(),
			'metadata': self.metadata.model_dump() if self.metadata else None,
			'state_message': self.state_message,
		}


AgentStructuredOutput = TypeVar('AgentStructuredOutput', bound=BaseModel)


class AgentHistoryList(BaseModel, Generic[AgentStructuredOutput]):
	"""List of AgentHistory messages, i.e. the history of the agent's actions and thoughts."""

	history: list[AgentHistory]
	usage: UsageSummary | None = None

	_output_model_schema: type[AgentStructuredOutput] | None = None

	def total_duration_seconds(self) -> float:
		"""Get total duration of all steps in seconds"""
		total = 0.0
		for h in self.history:
			if h.metadata:
				total += h.metadata.duration_seconds
		return total

	def __len__(self) -> int:
		"""Return the number of history items"""
		return len(self.history)

	def __str__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

	def add_item(self, history_item: AgentHistory) -> None:
		"""Add a history item to the list"""
		self.history.append(history_item)

	def __repr__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return self.__str__()

	def save_to_file(self, filepath: str | Path, sensitive_data: dict[str, str | dict[str, str]] | None = None) -> None:
		"""Save history to JSON file with proper serialization and optional sensitive data filtering"""
		try:
			Path(filepath).parent.mkdir(parents=True, exist_ok=True)
			data = self.model_dump(sensitive_data=sensitive_data)
			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2, ensure_ascii=False)
		except Exception as e:
			raise e

	# def save_as_playwright_script(
	# 	self,
	# 	output_path: str | Path,
	# 	sensitive_data_keys: list[str] | None = None,
	# 	browser_config: BrowserConfig | None = None,
	# 	context_config: BrowserContextConfig | None = None,
	# ) -> None:
	# 	"""
	# 	Generates a Playwright script based on the agent's history and saves it to a file.
	# 	Args:
	# 		output_path: The path where the generated Python script will be saved.
	# 		sensitive_data_keys: A list of keys used as placeholders for sensitive data
	# 							 (e.g., ['username_placeholder', 'password_placeholder']).
	# 							 These will be loaded from environment variables in the
	# 							 generated script.
	# 		browser_config: Configuration of the original Browser instance.
	# 		context_config: Configuration of the original BrowserContext instance.
	# 	"""
	# 	from browser_use.agent.playwright_script_generator import PlaywrightScriptGenerator

	# 	try:
	# 		serialized_history = self.model_dump()['history']
	# 		generator = PlaywrightScriptGenerator(serialized_history, sensitive_data_keys, browser_config, context_config)

	# 		script_content = generator.generate_script_content()
	# 		path_obj = Path(output_path)
	# 		path_obj.parent.mkdir(parents=True, exist_ok=True)
	# 		with open(path_obj, 'w', encoding='utf-8') as f:
	# 			f.write(script_content)
	# 	except Exception as e:
	# 		raise e

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Custom serialization that properly uses AgentHistory's model_dump"""
		return {
			'history': [h.model_dump(**kwargs) for h in self.history],
		}

	@classmethod
	def load_from_dict(cls, data: dict[str, Any], output_model: type[AgentOutput]) -> AgentHistoryList:
		# loop through history and validate output_model actions to enrich with custom actions
		for h in data.get('history', []):
			# Use .get() to avoid KeyError on incomplete or legacy history entries
			model_output = h.get('model_output')
			if model_output:
				if isinstance(model_output, dict):
					h['model_output'] = output_model.model_validate(model_output)
				else:
					h['model_output'] = None
			state = h.get('state') or {}
			if 'interacted_element' not in state:
				state['interacted_element'] = None
				h['state'] = state

		history = cls.model_validate(data)
		return history

	@classmethod
	def load_from_file(cls, filepath: str | Path, output_model: type[AgentOutput]) -> AgentHistoryList:
		"""Load history from JSON file"""
		with open(filepath, encoding='utf-8') as f:
			data = json.load(f)
		return cls.load_from_dict(data, output_model)

	def last_action(self) -> None | dict:
		"""Last action in history"""
		if self.history and self.history[-1].model_output:
			return self.history[-1].model_output.action[-1].model_dump(exclude_none=True, mode='json')
		return None

	def errors(self) -> list[str | None]:
		"""Get all errors from history, with None for steps without errors"""
		errors = []
		for h in self.history:
			step_errors = [r.error for r in h.result if r.error]

			# each step can have only one error
			errors.append(step_errors[0] if step_errors else None)
		return errors

	def final_result(self) -> None | str:
		"""Final result from history"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.extracted_content:
				return last_result.extracted_content
		return None

	def is_done(self) -> bool:
		"""Check if the agent is done"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			return last_result.is_done is True
		return False

	def is_successful(self) -> bool | None:
		"""Check if the agent completed successfully - the agent decides in the last step if it was successful or not. None if not done yet."""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.is_done is True:
				return last_result.success
		return None

	def has_errors(self) -> bool:
		"""Check if the agent has any non-None errors"""
		return any(error is not None for error in self.errors())

	def judgement(self) -> dict | None:
		"""Get the judgement result as a dictionary if it exists"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.judgement:
				return last_result.judgement.model_dump()
		return None

	def is_judged(self) -> bool:
		"""Check if the agent trace has been judged"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			return last_result.judgement is not None
		return False

	def is_validated(self) -> bool | None:
		"""Check if the judge validated the agent execution (verdict is True). Returns None if not judged yet."""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.judgement:
				return last_result.judgement.verdict
		return None

	def urls(self) -> list[str | None]:
		"""Get all unique URLs from history"""
		return [h.state.url if h.state.url is not None else None for h in self.history]

	def screenshot_paths(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
		"""Get all screenshot paths from history"""
		if n_last == 0:
			return []
		if n_last is None:
			if return_none_if_not_screenshot:
				return [h.state.screenshot_path if h.state.screenshot_path is not None else None for h in self.history]
			else:
				return [h.state.screenshot_path for h in self.history if h.state.screenshot_path is not None]
		else:
			if return_none_if_not_screenshot:
				return [h.state.screenshot_path if h.state.screenshot_path is not None else None for h in self.history[-n_last:]]
			else:
				return [h.state.screenshot_path for h in self.history[-n_last:] if h.state.screenshot_path is not None]

	def screenshots(self, n_last: int | None = None, return_none_if_not_screenshot: bool = True) -> list[str | None]:
		"""Get all screenshots from history as base64 strings"""
		if n_last == 0:
			return []

		history_items = self.history if n_last is None else self.history[-n_last:]
		screenshots = []

		for item in history_items:
			screenshot_b64 = item.state.get_screenshot()
			if screenshot_b64:
				screenshots.append(screenshot_b64)
			else:
				if return_none_if_not_screenshot:
					screenshots.append(None)
				# If return_none_if_not_screenshot is False, we skip None values

		return screenshots

	def action_names(self) -> list[str]:
		"""Get all action names from history"""
		action_names = []
		for action in self.model_actions():
			actions = list(action.keys())
			if actions:
				action_names.append(actions[0])
		return action_names

	def model_thoughts(self) -> list[AgentBrain]:
		"""Get all thoughts from history"""
		return [h.model_output.current_state for h in self.history if h.model_output]

	def model_outputs(self) -> list[AgentOutput]:
		"""Get all model outputs from history"""
		return [h.model_output for h in self.history if h.model_output]

	# get all actions with params
	def model_actions(self) -> list[dict]:
		"""Get all actions from history"""
		outputs = []

		for h in self.history:
			if h.model_output:
				# Guard against None interacted_element before zipping
				interacted_elements = h.state.interacted_element or [None] * len(h.model_output.action)
				for action, interacted_element in zip(h.model_output.action, interacted_elements):
					output = action.model_dump(exclude_none=True, mode='json')
					output['interacted_element'] = interacted_element
					outputs.append(output)
		return outputs

	def action_history(self) -> list[list[dict]]:
		"""Get truncated action history with only essential fields"""
		step_outputs = []

		for h in self.history:
			step_actions = []
			if h.model_output:
				# Guard against None interacted_element before zipping
				interacted_elements = h.state.interacted_element or [None] * len(h.model_output.action)
				# Zip actions with interacted elements and results
				for action, interacted_element, result in zip(h.model_output.action, interacted_elements, h.result):
					action_output = action.model_dump(exclude_none=True, mode='json')
					action_output['interacted_element'] = interacted_element
					# Only keep long_term_memory from result
					action_output['result'] = result.long_term_memory if result and result.long_term_memory else None
					step_actions.append(action_output)
			step_outputs.append(step_actions)

		return step_outputs

	def action_results(self) -> list[ActionResult]:
		"""Get all results from history"""
		results = []
		for h in self.history:
			results.extend([r for r in h.result if r])
		return results

	def extracted_content(self) -> list[str]:
		"""Get all extracted content from history"""
		content = []
		for h in self.history:
			content.extend([r.extracted_content for r in h.result if r.extracted_content])
		return content

	def model_actions_filtered(self, include: list[str] | None = None) -> list[dict]:
		"""Get all model actions from history as JSON"""
		if include is None:
			include = []
		outputs = self.model_actions()
		result = []
		for o in outputs:
			for i in include:
				if i == list(o.keys())[0]:
					result.append(o)
		return result

	def number_of_steps(self) -> int:
		"""Get the number of steps in the history"""
		return len(self.history)

	def agent_steps(self) -> list[str]:
		"""Format agent history as readable step descriptions for judge evaluation."""
		steps = []

		# Iterate through history items (each is an AgentHistory)
		for i, h in enumerate(self.history):
			step_text = f'Step {i + 1}:\n'

			# Get actions from model_output
			if h.model_output and h.model_output.action:
				# Use model_dump with mode='json' to serialize enums properly
				actions_list = [action.model_dump(exclude_none=True, mode='json') for action in h.model_output.action]
				action_json = json.dumps(actions_list, indent=1)
				step_text += f'Actions: {action_json}\n'

			# Get results (already a list[ActionResult] in h.result)
			if h.result:
				for j, result in enumerate(h.result):
					if result.extracted_content:
						content = str(result.extracted_content)
						step_text += f'Result {j + 1}: {content}\n'

					if result.error:
						error = str(result.error)
						step_text += f'Error {j + 1}: {error}\n'

			steps.append(step_text)

		return steps

	@property
	def structured_output(self) -> AgentStructuredOutput | None:
		"""Get the structured output from the history

		Returns:
			The structured output if both final_result and _output_model_schema are available,
			otherwise None
		"""
		final_result = self.final_result()
		if final_result is not None and self._output_model_schema is not None:
			return self._output_model_schema.model_validate_json(final_result)

		return None

	def get_structured_output(self, output_model: type[AgentStructuredOutput]) -> AgentStructuredOutput | None:
		"""Get the structured output from history, parsing with the provided schema.

		Use this method when accessing structured output from sandbox execution,
		since the _output_model_schema private attribute is not preserved during serialization.

		Args:
			output_model: The Pydantic model class to parse the output with

		Returns:
			The parsed structured output, or None if no final result exists
		"""
		final_result = self.final_result()
		if final_result is not None:
			return output_model.model_validate_json(final_result)
		return None


class AgentError:
	"""Container for agent error handling"""

	VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
	RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
	NO_VALID_ACTION = 'No valid action found'

	@staticmethod
	def format_error(error: Exception, include_trace: bool = False) -> str:
		"""Format error message based on error type and optionally include trace"""
		message = ''
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		# Lazy import to avoid loading openai SDK (~800ms) at module level
		from openai import RateLimitError

		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR

		# Handle LLM response validation errors from llm_use
		error_str = str(error)
		if 'LLM response missing required fields' in error_str or 'Expected format: AgentOutput' in error_str:
			# Extract the main error message without the huge stacktrace
			lines = error_str.split('\n')
			main_error = lines[0] if lines else error_str

			# Provide a clearer error message
			helpful_msg = f'{main_error}\n\nThe previous response had an invalid output structure. Please stick to the required output format. \n\n'

			if include_trace:
				helpful_msg += f'\n\nFull stacktrace:\n{traceback.format_exc()}'

			return helpful_msg

		if include_trace:
			return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
		return f'{str(error)}'


class DetectedVariable(BaseModel):
	"""A detected variable in agent history"""

	name: str
	original_value: str
	type: str = 'string'
	format: str | None = None


class VariableMetadata(BaseModel):
	"""Metadata about detected variables in history"""

	detected_variables: dict[str, DetectedVariable] = Field(default_factory=dict)


TaskStatus = Literal[
	'success',
	'partial_success',
	'failed',
	'cancelled',
	'max_steps_exceeded',
	'running',
]


class ToolExecutionResult(BaseModel):
	"""**Deprecated:** Use ``RuntimeExecutionResult`` with ``ResultAssembler.from_single_action()`` instead.

	This model is kept for backwards compatibility but should not be used in new code.
	All single-step browser operations (MCP tools, skill_cli commands) now use
	``RuntimeExecutionResult`` with exactly one step, assembled via::

	    result = ResultAssembler.from_single_action(
	        tool_name='navigate',
	        message='Navigated to: https://example.com',
	        data={'url': 'https://example.com'},
	        entry_point='mcp_tool',
	    )

	.. deprecated::
	    Use :class:`RuntimeExecutionResult` via :meth:`ResultAssembler.from_single_action`.
	"""

	tool_name: str = Field(description='Name of the tool that was executed')
	success: bool = Field(default=True, description='Whether the action succeeded')
	message: str = Field(default='', description='Human-readable description of the outcome')
	data: dict[str, Any] = Field(default_factory=dict, description='Structured key-value data for programmatic consumers')
	error: str | None = Field(default=None, description='Error message if the action failed')
	url: str | None = Field(default=None, description='Current URL after the action (if applicable)')
	screenshot_path: str | None = Field(default=None, description='Path to screenshot if one was captured')
	output_files: list[str] = Field(default_factory=list, description='Paths to files produced by this action')
	duration_seconds: float | None = Field(default=None, description='How long the action took')

	def to_mcp_contents(self) -> list[dict[str, Any]]:
		"""Convert to MCP content dicts (text + optional structured JSON).

		Returns a list of dicts with ``type`` and ``text`` keys.
		The MCP server wrapper converts these to ``TextContent`` objects.
		Format:
		  1) Human-readable message
		  2) ``--- STRUCTURED RESULT (JSON) ---`` header + full JSON  (only when data/error present)
		"""
		contents: list[dict[str, Any]] = []
		contents.append({'type': 'text', 'text': self.message or ('Success' if self.success else f'Error: {self.error}')})

		if self.data or self.error or not self.success:
			structured_json = self.model_dump_json(indent=2)
			contents.append(
				{
					'type': 'text',
					'text': f'--- STRUCTURED RESULT (JSON) ---\n{structured_json}',
				}
			)
		return contents

	def to_daemon_response(self, req_id: str = '') -> dict:
		"""Convert to daemon socket response dict.

		Preserves backwards compatibility with existing ``{success, data, error}`` format
		while embedding the full RuntimeExecutionResult as ``_structured`` in data.

		.. deprecated::
		    Use :meth:`RuntimeExecutionResult.to_daemon_response` instead.
		"""
		result_data: dict[str, Any] = dict(self.data)
		result_data['_structured'] = self.model_dump(exclude={'data'})
		return {
			'id': req_id,
			'success': self.success,
			'data': result_data,
			**({'error': self.error} if self.error else {}),
		}


class OutputFile(BaseModel):
	"""Information about a file produced during execution."""

	path: str = Field(description='Absolute or relative path to the file')
	size_bytes: int | None = Field(default=None, description='File size in bytes if available')
	description: str | None = Field(default=None, description='Optional description of the file contents')
	mime_type: str | None = Field(default=None, description='MIME type if known')

	@model_validator(mode='before')
	@classmethod
	def from_string(cls, v: Any) -> Any:
		"""Allow constructing from a plain path string for convenience."""
		if isinstance(v, str):
			return {'path': v}
		return v


class StepExecutionResult(BaseModel):
	"""Structured result for a single agent step."""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	step_number: int = Field(description='1-indexed step number')
	duration_seconds: float | None = Field(default=None, description='How long this step took')
	status: Literal['success', 'failed', 'skipped'] = Field(default='success', description='Step-level status')
	action_names: list[str] = Field(default_factory=list, description='Names of actions executed in this step')
	extracted_content: str | None = Field(default=None, description='Content extracted during this step')
	error: str | None = Field(default=None, description='Error message if step failed')
	url: str | None = Field(default=None, description='URL at the end of this step')
	screenshot_path: str | None = Field(default=None, description='Path to screenshot captured for this step')
	model_thinking: str | None = Field(default=None, description='LLM thinking/rationale for this step')
	interacted_elements: list[DOMInteractedElement | None] = Field(default_factory=list)
	attachments: list[str] = Field(default_factory=list, description='Files produced in this step')
	metadata: dict[str, Any] = Field(default_factory=dict, description='Additional step metadata')


class RuntimeExecutionResult(BaseModel, Generic[AgentStructuredOutput]):
	"""Unified structured execution result.

	This is the single authoritative result object produced by ALL
	execution entry points in the browser-use library:

	1. **Agent API** — :meth:`Agent.run_with_result() <browser_use.agent.service.Agent.run_with_result>`
	   Full agent task execution via Python API.

	2. **MCP Agent Tool** — ``retry_with_browser_use_agent``
	   Full agent execution via MCP server.

	3. **MCP Browser Tools** — All direct browser action tools
	   (``browser_navigate``, ``browser_click``, etc.) assembled via
	   :meth:`ResultAssembler.from_single_action`.

	4. **CLI / TUI** — The interactive browser-use CLI
	   (``BrowserUseApp.run_task``) uses :meth:`Agent.run_with_result`.

	5. **skill_cli** — The skill CLI daemon commands
	   (navigate, click, etc.) assembled via
	   :meth:`ResultAssembler.from_single_action`.

	6. **Task Templates** — Any task executed through the CLI
	   template system (``--template`` flag) or programmatic task
	   execution, all going through :meth:`Agent.run_with_result`.

	All entry points assemble and return this object.
	CLI may format fields for human consumption, but must NOT lose the structured data.
	MCP tools return a dual-format response (human-readable text + JSON serialization).
	Python API consumers get the full typed object.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	task_id: str = Field(description='Unique identifier for this task run')
	task: str = Field(description='Original user task description')
	status: TaskStatus = Field(description='Final task status')
	success: bool | None = Field(default=None, description='Binary success flag (None if not yet done)')

	final_text: str | None = Field(
		default=None,
		description='Human-readable final message produced by the done action or exit handler',
	)
	structured_output: AgentStructuredOutput | None = Field(
		default=None,
		description='Parsed structured output when output_model_schema was provided',
	)
	structured_output_json: str | None = Field(
		default=None,
		description='Raw JSON string of structured output (for wire transmission / MCP)',
	)

	output_files: list[OutputFile] = Field(
		default_factory=list,
		description='All files produced during the run (CSVs, PDFs, screenshots, GIFs, etc.)',
	)
	visited_urls: list[str] = Field(
		default_factory=list,
		description='Deduplicated list of URLs visited, in order of first appearance',
	)
	screenshot_paths: list[str] = Field(
		default_factory=list,
		description='Screenshot files captured during the run',
	)

	steps_completed: int = Field(default=0, description='Number of steps actually executed')
	max_steps: int = Field(default=0, description='Maximum steps allowed')
	total_duration_seconds: float = Field(default=0.0, description='Total wall-clock duration')

	error: str | None = Field(
		default=None,
		description='Fatal error that terminated the run, if any (distinct from step-level recoverable errors)',
	)
	step_errors: list[str] = Field(
		default_factory=list,
		description='Recoverable errors encountered during individual steps',
	)
	consecutive_failures: int = Field(
		default=0,
		description='How many consecutive steps ended in error at termination',
	)

	judgement: JudgementResult | None = Field(
		default=None,
		description='Judge LLM verdict if use_judge was enabled',
	)

	token_usage: UsageSummary | None = Field(
		default=None,
		description='Token consumption and cost data',
	)

	steps: list[StepExecutionResult] = Field(
		default_factory=list,
		description='Per-step execution results (optional; populated for richer observability)',
	)

	raw_history: AgentHistoryList[AgentStructuredOutput] | None = Field(
		default=None,
		description='Reference to the full AgentHistoryList for callers that need it (Python API only; excluded from JSON serialization)',
		json_schema_extra={'exclude_from_json': True},
	)

	metadata: dict[str, Any] = Field(
		default_factory=dict,
		description='Free-form metadata: entry_point ("agent" / "mcp" / "cli" / "skill"), model name, proxy country, etc.',
	)

	def to_human_readable(self) -> str:
		"""Format the result as a human-readable string for CLI / logging.

		The structured object itself remains available to programmatic callers.
		"""
		lines: list[str] = []

		lines.append(f'📋 Task: {self.task}')
		lines.append(f'   Task ID: {self.task_id}')

		status_icon = {
			'success': '✅',
			'partial_success': '⚠️',
			'failed': '❌',
			'cancelled': '🛑',
			'max_steps_exceeded': '⏰',
			'running': '⏳',
		}.get(self.status, '❔')
		lines.append(f'{status_icon} Task status: {self.status}')
		if self.success is not None:
			lines.append(f'   Success flag: {self.success}')

		lines.append(f'   Steps: {self.steps_completed}/{self.max_steps}')
		lines.append(f'   Duration: {self.total_duration_seconds:.1f}s')

		if self.final_text:
			lines.append('')
			lines.append('📝 Final result:')
			for line in str(self.final_text).splitlines():
				lines.append(f'   {line}')

		if self.visited_urls:
			lines.append('')
			lines.append(f'🔗 URLs visited ({len(self.visited_urls)}):')
			for url in self.visited_urls[:10]:
				lines.append(f'   - {url}')
			if len(self.visited_urls) > 10:
				lines.append(f'   ... and {len(self.visited_urls) - 10} more')

		if self.output_files:
			lines.append('')
			lines.append(f'📎 Output files ({len(self.output_files)}):')
			for f in self.output_files:
				size = f' ({f.size_bytes} bytes)' if f.size_bytes else ''
				desc = f' — {f.description}' if f.description else ''
				lines.append(f'   - {f.path}{size}{desc}')

		if self.error:
			lines.append('')
			lines.append(f'💥 Fatal error: {self.error}')

		if self.step_errors:
			lines.append('')
			lines.append(f'⚠️  Step errors ({len(self.step_errors)}):')
			for i, err in enumerate(self.step_errors[-5:]):
				lines.append(f'   {i + 1}. {err[:200]}')
			if len(self.step_errors) > 5:
				lines.append(f'   ... {len(self.step_errors) - 5} earlier errors omitted')

		if self.token_usage and self.token_usage.total_tokens:
			lines.append('')
			lines.append(
				f'💰 Tokens: {self.token_usage.total_tokens:,} '
				f'(input {self.token_usage.total_prompt_tokens or 0:,}, '
				f'output {self.token_usage.total_completion_tokens or 0:,})'
			)
			if self.token_usage.total_cost is not None:
				lines.append(f'   Estimated cost: ${self.token_usage.total_cost:.4f}')

		if self.judgement:
			lines.append('')
			judge_icon = '✅' if self.judgement.verdict else '❌'
			lines.append(f'🔍 Judge verdict: {judge_icon} {"PASS" if self.judgement.verdict else "FAIL"}')
			if self.judgement.failure_reason:
				lines.append(f'   Reason: {self.judgement.failure_reason}')
			if self.judgement.impossible_task:
				lines.append('   Flagged: impossible task')
			if self.judgement.reached_captcha:
				lines.append('   Flagged: reached captcha')

		return '\n'.join(lines)

	@property
	def tool_name(self) -> str | None:
		"""Convenience accessor: get tool_name from metadata for single-action results."""
		return self.metadata.get('tool_name')

	def to_mcp_contents(self) -> list[dict[str, Any]]:
		"""Convert to MCP content dicts (text + optional structured JSON + screenshot).

		The first element is always the human-readable text message.
		If there is structured data or an error, a second element with the
		JSON serialization is appended.
		"""
		contents: list[dict[str, Any]] = []
		human_text = self.final_text or self.to_human_readable()
		contents.append({'type': 'text', 'text': human_text})
		always_include_keys = {'entry_point', 'tool_name'}
		extra_keys = set(self.metadata.keys()) - always_include_keys if self.metadata else set()
		has_data = bool(extra_keys)
		if self.error or has_data or not self.success:
			structured_json = self.model_dump_json(indent=2, exclude={'raw_history'})
			contents.append({'type': 'text', 'text': f'--- STRUCTURED RESULT (JSON) ---\n{structured_json}'})
		return contents

	def to_daemon_response(self, req_id: str = '') -> dict[str, Any]:
		"""Convert to daemon socket response dict.

		Keeps backwards compatibility with the {success, data, error} protocol,
		but adds a _structured key containing the full RuntimeExecutionResult.
		"""
		data: dict[str, Any] = {}
		for step in self.steps:
			if step.metadata and 'data' in step.metadata:
				data.update(step.metadata['data'])
		data['_structured'] = self.model_dump(exclude={'raw_history'})
		if self.visited_urls:
			data.setdefault('url', self.visited_urls[-1])
		if self.steps and self.steps[0].screenshot_path:
			data.setdefault('screenshot_path', self.steps[0].screenshot_path)
		result: dict[str, Any] = {
			'id': req_id,
			'success': bool(self.success),
			'data': data,
		}
		if self.error:
			result['error'] = self.error
		return result

	def model_dump(self, **kwargs) -> dict[str, Any]:
		"""Serialize, excluding the raw_history reference (not JSON-serialisable)."""
		kwargs.setdefault('exclude', {'raw_history'})
		kwargs.setdefault('exclude_none', False)
		return super().model_dump(**kwargs)

	def model_dump_json(self, **kwargs) -> str:
		"""Serialize to JSON, excluding raw_history."""
		kwargs.setdefault('exclude', {'raw_history'})
		kwargs.setdefault('exclude_none', False)
		return super().model_dump_json(**kwargs)


class ResultAssembler:
	"""Assembles RuntimeExecutionResult from various data sources.

	Usage:
	    result = (
	        ResultAssembler(task_id=..., task=...)
	        .from_history(history)
	        .with_entry_point("agent")
	        .with_max_steps(max_steps)
	        .with_fatal_error(str(e) if failed else None)
	        .with_total_duration(end - start)
	        .assemble()
	    )
	"""

	def __init__(self, task_id: str, task: str):
		self._task_id = task_id
		self._task = task
		self._history: AgentHistoryList | None = None
		self._max_steps: int = 0
		self._total_duration: float = 0.0
		self._fatal_error: str | None = None
		self._entry_point: str = 'unknown'
		self._extra_output_files: list[OutputFile] = []
		self._extra_metadata: dict[str, Any] = {}
		self._include_steps: bool = True
		self._output_model_schema: type[BaseModel] | None = None

	def from_history(self, history: AgentHistoryList) -> ResultAssembler:
		self._history = history
		return self

	def with_max_steps(self, max_steps: int) -> ResultAssembler:
		self._max_steps = max_steps
		return self

	def with_total_duration(self, duration_seconds: float) -> ResultAssembler:
		self._total_duration = duration_seconds
		return self

	def with_fatal_error(self, error: str | None) -> ResultAssembler:
		self._fatal_error = error
		return self

	def with_entry_point(self, entry_point: str) -> ResultAssembler:
		self._entry_point = entry_point
		return self

	def with_output_files(self, files: list[str | OutputFile]) -> ResultAssembler:
		for f in files:
			if isinstance(f, OutputFile):
				self._extra_output_files.append(f)
			else:
				self._extra_output_files.append(OutputFile(path=str(f)))
		return self

	def with_metadata(self, **meta: Any) -> ResultAssembler:
		self._extra_metadata.update(meta)
		return self

	def with_output_model_schema(self, schema: type[AgentStructuredOutput] | None) -> ResultAssembler:
		self._output_model_schema = schema
		return self

	def include_steps(self, include: bool = True) -> ResultAssembler:
		self._include_steps = include
		return self

	@staticmethod
	def from_single_action(
		*,
		tool_name: str,
		message: str | None = None,
		data: dict[str, Any] | None = None,
		error: str | None = None,
		url: str | None = None,
		screenshot_path: str | None = None,
		screenshot_b64: str | None = None,
		output_files: list[str] | None = None,
		duration_seconds: float | None = None,
		task_id: str | None = None,
		entry_point: str = 'mcp_tool',
	) -> RuntimeExecutionResult[BaseModel]:
		"""Assemble a RuntimeExecutionResult for a single atomic browser action.

		This replaces the separate ToolExecutionResult model by representing
		single-tool invocations as a RuntimeExecutionResult with exactly one step.

		Args:
		    tool_name: Name of the tool that was executed (e.g., "navigate", "click")
		    message: Human-readable description of the outcome
		    data: Structured key-value data for programmatic consumers
		    error: Error message if the action failed
		    url: Current URL after the action (if applicable)
		    screenshot_path: Path to screenshot if one was captured
		    screenshot_b64: Base64-encoded screenshot (stored in step metadata)
		    output_files: Paths to files produced by this action
		    duration_seconds: How long the action took
		    task_id: Unique identifier for this task run (auto-generated if None)
		    entry_point: Where this execution originated ("mcp_tool", "skill_cli", etc.)

		Returns:
		    RuntimeExecutionResult with a single StepExecutionResult in steps
		"""
		import uuid

		success = error is None
		status: TaskStatus = 'success' if success else 'failed'
		actual_task_id = task_id or f'single-{tool_name}-{uuid.uuid4().hex[:8]}'
		task_desc = message or f'{tool_name}: {data}' if data else tool_name

		step_metadata: dict[str, Any] = {'tool_name': tool_name}
		if data:
			step_metadata['data'] = data
		if screenshot_b64:
			step_metadata['screenshot_b64'] = screenshot_b64

		step = StepExecutionResult(
			step_number=1,
			duration_seconds=duration_seconds,
			status='success' if success else 'failed',
			action_names=[tool_name],
			extracted_content=message,
			error=error,
			url=url,
			screenshot_path=screenshot_path,
			attachments=list(output_files or []),
			metadata=step_metadata,
		)

		result_files: list[OutputFile] = [OutputFile(path=f) for f in (output_files or [])]

		return RuntimeExecutionResult[BaseModel](
			task_id=actual_task_id,
			task=task_desc,
			status=status,
			success=success,
			final_text=message,
			output_files=result_files,
			visited_urls=[url] if url else [],
			screenshot_paths=[screenshot_path] if screenshot_path else [],
			steps_completed=1,
			max_steps=1,
			total_duration_seconds=round(duration_seconds or 0.0, 3),
			error=error,
			step_errors=[error] if error else [],
			consecutive_failures=0 if success else 1,
			steps=[step],
			metadata={'entry_point': entry_point, 'tool_name': tool_name, **(data or {})},
		)

	@staticmethod
	def _dedupe_preserve_order(items: list[str | None]) -> list[str]:
		seen: set[str] = set()
		result: list[str] = []
		for item in items:
			if item and item not in seen:
				seen.add(item)
				result.append(item)
		return result

	def assemble(self) -> RuntimeExecutionResult:
		history = self._history
		if history is None:
			history = AgentHistoryList(history=[], usage=None)

		steps_completed = len(history.history)
		has_steps = steps_completed > 0

		is_done = history.is_done() if has_steps else False
		is_successful = history.is_successful() if has_steps else None
		final_text = history.final_result() if has_steps else None
		judgement_obj = None
		if has_steps and history.is_judged():
			judgement_obj = history.history[-1].result[-1].judgement if history.history[-1].result else None

		visited_urls = self._dedupe_preserve_order(history.urls() if has_steps else [])
		screenshot_paths = [p for p in history.screenshot_paths(return_none_if_not_screenshot=False) if p]

		step_errors_raw: list[str] = []
		output_files_set: dict[str, OutputFile] = {}

		for step_idx, h in enumerate(history.history):
			for r in h.result:
				if r.error:
					step_errors_raw.append(r.error)
				if r.attachments:
					for att in r.attachments:
						if att not in output_files_set:
							output_files_set[att] = OutputFile(path=att)

		# Merge with extra output files passed directly
		for ef in self._extra_output_files:
			if ef.path not in output_files_set:
				output_files_set[ef.path] = ef
		output_files = list(output_files_set.values())

		# GIF generation path may exist in metadata
		gif_path: str | None = self._extra_metadata.pop('_generated_gif_path', None)
		if gif_path and gif_path not in {f.path for f in output_files}:
			output_files.append(OutputFile(path=gif_path, description='Agent action animation'))

		# Determine status
		if self._fatal_error:
			err_lower = self._fatal_error.lower()
			if 'KeyboardInterrupt' in self._fatal_error or 'Cancelled' in self._fatal_error or 'SIGINT' in self._fatal_error:
				status: TaskStatus = 'cancelled'
			elif (
				('max' in err_lower and ('step' in err_lower or 'failures' in err_lower))
				or 'consecutive failures' in err_lower
				or 'stopped due to' in err_lower
				or 'maximum steps' in err_lower
			):
				status = 'max_steps_exceeded'
			elif is_done and is_successful is True:
				status = 'partial_success'
			else:
				status = 'failed'
		elif is_done:
			if is_successful is True:
				status = 'success'
			elif is_successful is False:
				status = 'failed'
			else:
				status = 'partial_success'
		elif steps_completed >= self._max_steps > 0:
			status = 'max_steps_exceeded'
		else:
			status = 'running'

		# Consecutive failures at end of run
		consecutive_failures = 0
		for h in reversed(history.history):
			if any(r.error for r in h.result):
				consecutive_failures += 1
			else:
				break

		# Structured output parsing
		structured_output_obj: BaseModel | None = None
		structured_output_json_str: str | None = None
		if final_text and self._output_model_schema is not None:
			try:
				structured_output_obj = self._output_model_schema.model_validate_json(final_text)
				structured_output_json_str = final_text
			except (ValidationError, json.JSONDecodeError, TypeError):
				structured_output_obj = None
		elif final_text is not None and history._output_model_schema is not None:
			try:
				structured_output_obj = history._output_model_schema.model_validate_json(final_text)
				structured_output_json_str = final_text
			except (ValidationError, json.JSONDecodeError, TypeError):
				structured_output_obj = None

		# Build per-step results if requested
		steps_list: list[StepExecutionResult] = []
		if self._include_steps:
			for i, h in enumerate(history.history):
				duration = h.metadata.duration_seconds if h.metadata else None
				has_error = any(r.error for r in h.result)
				action_names: list[str] = []
				extracted_parts: list[str] = []
				attachments_this: list[str] = []
				if h.model_output:
					for a in h.model_output.action:
						data = a.model_dump(exclude_unset=True, mode='json')
						if data:
							action_names.append(next(iter(data.keys())))
				for r in h.result:
					if r.extracted_content:
						extracted_parts.append(r.extracted_content)
					if r.attachments:
						attachments_this.extend(r.attachments)
				err_msg = next((r.error for r in h.result if r.error), None)
				thinking = (
					h.model_output.current_state.thinking if h.model_output and h.model_output.current_state.thinking else None
				)
				steps_list.append(
					StepExecutionResult(
						step_number=i + 1,
						duration_seconds=duration,
						status='failed' if has_error else 'success',
						action_names=action_names,
						extracted_content='\n\n'.join(extracted_parts) if extracted_parts else None,
						error=err_msg,
						url=h.state.url,
						screenshot_path=h.state.screenshot_path,
						model_thinking=thinking,
						interacted_elements=list(h.state.interacted_element or []),
						attachments=attachments_this,
						metadata=h.metadata.model_dump() if h.metadata else {},
					)
				)

		total_duration = self._total_duration
		if total_duration <= 0 and steps_list:
			total_duration = sum(s.duration_seconds or 0.0 for s in steps_list)
		elif total_duration <= 0:
			total_duration = history.total_duration_seconds()

		# Determine binary success flag based on final status (consistent override)
		success_flag: bool | None
		if status == 'success' or status == 'partial_success':
			success_flag = True
		elif status == 'running':
			success_flag = None  # still in progress
		else:
			success_flag = False

		# Build metadata dict (entry_point first, then extras merged)
		metadata: dict[str, Any] = {'entry_point': self._entry_point}
		metadata.update(self._extra_metadata)

		return RuntimeExecutionResult(
			task_id=self._task_id,
			task=self._task,
			status=status,
			success=success_flag,
			final_text=final_text,
			structured_output=structured_output_obj,
			structured_output_json=structured_output_json_str,
			output_files=output_files,
			visited_urls=visited_urls,
			screenshot_paths=screenshot_paths,
			steps_completed=steps_completed,
			max_steps=self._max_steps,
			total_duration_seconds=round(total_duration, 3),
			error=self._fatal_error,
			step_errors=step_errors_raw,
			consecutive_failures=consecutive_failures,
			judgement=judgement_obj,
			token_usage=history.usage,
			steps=steps_list,
			raw_history=history,
			metadata=metadata,
		)


class ResultSerializer:
	"""Unified serializer for RuntimeExecutionResult — single entry point for all output formats.

	All entry points (Python API, CLI, MCP, template scripts) should use this class
	to convert a ``RuntimeExecutionResult`` into whatever format their caller needs.
	This avoids ad-hoc ``to_human_readable`` / ``model_dump_json`` calls scattered
	across the codebase and ensures consistent serialization behaviour.

	Output modes:
	    - ``"text"``      — human-readable summary (CLI default, logs)
	    - ``"json"``      — compact JSON (programmatic consumers, ``--json`` flag)
	    - ``"json_pretty"`` — indented JSON (debugging, trace dumps)
	    - ``"both"``      — text summary followed by ``--- JSON ---`` separator (MCP dual-format)

	Usage::

	    serializer = ResultSerializer(result)
	    print(serializer.to_text())                    # CLI human-readable
	    print(serializer.to_json(indent=2))            # Python API
	    for item in serializer.to_mcp_contents():      # MCP server
	        ...
	    click.echo(serializer.output(mode='json'))     # CLI --json flag
	"""

	def __init__(self, result: RuntimeExecutionResult[Any]):
		self._result = result

	# ------------------------------------------------------------------
	# Core formatters
	# ------------------------------------------------------------------

	def to_text(self) -> str:
		"""Human-readable summary (equivalent to RuntimeExecutionResult.to_human_readable)."""
		return self._result.to_human_readable()

	def to_json(self, indent: int | None = None, include_raw_history: bool = False) -> str:
		"""Serialize to JSON string.

		Args:
		    indent: If set, pretty-print with this many spaces per level.
		    include_raw_history: Include the raw AgentHistoryList (very large; off by default).
		"""
		exclude: set[str] = set() if include_raw_history else {'raw_history'}
		return self._result.model_dump_json(indent=indent, exclude=exclude)

	def to_dict(self, include_raw_history: bool = False) -> dict[str, Any]:
		"""Serialize to dict."""
		exclude: set[str] = set() if include_raw_history else {'raw_history'}
		return self._result.model_dump(exclude=exclude)

	# ------------------------------------------------------------------
	# Protocol-specific output
	# ------------------------------------------------------------------

	def to_mcp_contents(self) -> list[dict[str, Any]]:
		"""MCP dual-format output: human text + structured JSON block when non-trivial.

		Matches the contract used by the MCP server's direct browser tools.
		"""
		return self._result.to_mcp_contents()

	def to_daemon_response(self, req_id: str = '') -> dict[str, Any]:
		"""skill_cli daemon socket response: {id, success, data, error?} with _structured key."""
		return self._result.to_daemon_response(req_id=req_id)

	def to_exit_code(self) -> int:
		"""Unix-style exit code: 0 on success, 1 on failure, 130 on cancellation."""
		s = self._result.status
		if s in ('success', 'partial_success'):
			return 0
		if s == 'cancelled':
			return 130
		return 1

	# ------------------------------------------------------------------
	# High-level dispatcher used by CLI / template scripts
	# ------------------------------------------------------------------

	def output(
		self,
		mode: Literal['text', 'json', 'json_pretty', 'both'] = 'text',
		exit_on_failure: bool = False,
	) -> str:
		"""Produce output string based on the selected mode.

		Args:
		    mode: Output format selector.
		    exit_on_failure: If True and result status is not success/partial_success,
		        raise ``SystemExit`` with the appropriate exit code *after* producing
		        the output string — callers should print/save the string before
		        the exception propagates.

		Returns:
		    Formatted output string ready for stdout or file writing.
		"""
		if mode == 'text':
			out = self.to_text()
		elif mode == 'json':
			out = self.to_json()
		elif mode == 'json_pretty':
			out = self.to_json(indent=2)
		elif mode == 'both':
			text_part = self.to_text()
			json_part = self.to_json(indent=2)
			out = f'{text_part}\n\n--- STRUCTURED RESULT (JSON) ---\n{json_part}'
		else:
			raise ValueError(f'Unknown output mode: {mode}')

		if exit_on_failure:
			code = self.to_exit_code()
			if code != 0:
				raise SystemExit(code)

		return out
