"""Structured trace export for replayable agent debugging.

When enabled via ``Agent(trace_dir=...)``, the :class:`TraceService` collects
a rich snapshot after every step and writes a single JSON file that can be
loaded by downstream tooling to reconstruct what the agent saw, thought and did.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from browser_use.utils import collect_sensitive_data_values, redact_sensitive_string

logger = logging.getLogger(__name__)

_SENSITIVE_KEY_PATTERNS = re.compile(
	r'(api[_-]?key|token|secret|password|cookie|auth|session[_-]?id|storage[_-]?state)',
	re.IGNORECASE,
)


def _sanitize_dict(data: dict[str, Any], sensitive_data: dict[str, str | dict[str, str]] | None) -> dict[str, Any]:
	"""Recursively redact sensitive values and strip keys that look like secrets."""
	sensitive_values = collect_sensitive_data_values(sensitive_data)
	return _sanitize_dict_inner(data, sensitive_values)


def _sanitize_dict_inner(data: dict[str, Any], sensitive_values: dict[str, str]) -> dict[str, Any]:
	out: dict[str, Any] = {}
	for key, value in data.items():
		if _SENSITIVE_KEY_PATTERNS.search(key):
			out[key] = '<redacted>'
			continue
		if isinstance(value, str):
			out[key] = redact_sensitive_string(value, sensitive_values)
		elif isinstance(value, dict):
			out[key] = _sanitize_dict_inner(value, sensitive_values)
		elif isinstance(value, list):
			out[key] = [
				_sanitize_dict_inner(item, sensitive_values)
				if isinstance(item, dict)
				else redact_sensitive_string(item, sensitive_values)
				if isinstance(item, str)
				else item
				for item in value
			]
		else:
			out[key] = value
	return out


class TraceAction(BaseModel):
	"""A single action executed by the agent in a step."""

	name: str
	params: dict[str, Any] = Field(default_factory=dict)


class TraceActionResult(BaseModel):
	"""Result of a single action."""

	extracted_content: str | None = None
	error: str | None = None
	success: bool | None = None
	is_done: bool | None = None
	long_term_memory: str | None = None
	attachments: list[str] | None = None


class TraceBrowserEvent(BaseModel):
	"""A browser event that occurred during a step."""

	event_type: str
	data: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
	"""Complete snapshot of one agent step for offline replay."""

	step_number: int
	timestamp: float

	goal: str | None = None
	evaluation_previous_goal: str | None = None
	thinking: str | None = None
	memory: str | None = None

	actions: list[TraceAction] = Field(default_factory=list)
	action_results: list[TraceActionResult] = Field(default_factory=list)

	url: str | None = None
	title: str | None = None
	tabs: list[dict[str, Any]] = Field(default_factory=list)
	dom_interactive_summary: str | None = None
	screenshot_path: str | None = None

	downloaded_files: list[str] = Field(default_factory=list)
	browser_events: list[TraceBrowserEvent] = Field(default_factory=list)

	duration_seconds: float | None = None
	error: str | None = None


class TraceFile(BaseModel):
	"""Top-level trace container that is written to disk."""

	agent_id: str
	task: str
	model: str
	trace_version: str = '1.0'
	created_at: float = Field(default_factory=time.time)
	steps: list[TraceStep] = Field(default_factory=list)

	def save(self, path: Path) -> None:
		path.parent.mkdir(parents=True, exist_ok=True)
		with open(path, 'w', encoding='utf-8') as f:
			f.write(self.model_dump_json(indent=2))

	@classmethod
	def load(cls, path: str | Path) -> TraceFile:
		with open(path, encoding='utf-8') as f:
			data = json.load(f)
		return cls.model_validate(data)


@dataclass
class _PendingStepState:
	"""Mutable state collected between step_start and step_end calls."""

	step_number: int = 0
	start_time: float = 0.0
	initial_url: str | None = None
	initial_title: str | None = None
	initial_tabs: list[dict[str, Any]] = field(default_factory=list)
	initial_dom_summary: str | None = None
	initial_screenshot_path: str | None = None
	initial_downloaded_files: set[str] = field(default_factory=set)
	events: list[TraceBrowserEvent] = field(default_factory=list)


class TraceService:
	"""Collects step-level data from an :class:`Agent` and writes a trace file.

    The service uses an explicit two-phase lifecycle per step:

    1. :meth:`step_start` — called at the beginning of a step, captures the
       initial page state (URL, title, DOM summary, screenshot, current
       download list) and begins buffering browser events for this step.
    2. :meth:`step_end` — called after actions are executed, merges LLM
       output, action results, new download deltas, buffered events and any
       step error into a final :class:`TraceStep` and appends it to the trace.

    The trace is written incrementally after each step so that it is always
    available even if the process crashes.
    """

	def __init__(
		self,
		trace_dir: str | Path,
		agent_id: str,
		task: str,
		model: str,
		sensitive_data: dict[str, str | dict[str, str]] | None = None,
	) -> None:
		self.trace_dir = Path(trace_dir)
		self.sensitive_data = sensitive_data
		self.trace_file = TraceFile(
			agent_id=agent_id,
			task=self._sanitize_string(task),
			model=model,
		)
		self._trace_path = self.trace_dir / f'trace_{agent_id[:8]}.json'
		self._pending: _PendingStepState | None = None

	# ------------------------------------------------------------------
	# Event bus integration
	# ------------------------------------------------------------------

	def register_event_bus(self, event_bus: Any) -> None:
		"""Subscribe to all browser events and route them to the active step."""
		event_bus.on('*', self._on_browser_event)

	def _on_browser_event(self, event: Any) -> None:
		event_name = event.__class__.__name__
		try:
			data = event.model_dump(exclude_unset=True) if hasattr(event, 'model_dump') else {}
		except Exception:
			data = {}
		data = _sanitize_dict(data, self.sensitive_data)
		trace_event = TraceBrowserEvent(event_type=event_name, data=data)
		if self._pending is not None:
			self._pending.events.append(trace_event)

	# ------------------------------------------------------------------
	# Helpers
	# ------------------------------------------------------------------

	def _sanitize_string(self, value: str) -> str:
		sensitive_values = collect_sensitive_data_values(self.sensitive_data)
		return redact_sensitive_string(value, sensitive_values)

	def _sanitize_actions(self, actions: list[dict[str, Any]]) -> list[TraceAction]:
		result: list[TraceAction] = []
		for action_dict in actions:
			action_name = next(iter(action_dict.keys()), 'unknown')
			params = action_dict.get(action_name, {})
			if not isinstance(params, dict):
				params = {}
			params = _sanitize_dict(params, self.sensitive_data)
			result.append(TraceAction(name=action_name, params=params))
		return result

	# ------------------------------------------------------------------
	# Step lifecycle
	# ------------------------------------------------------------------

	def step_start(
		self,
		step_number: int,
		browser_state_summary: Any | None,
		downloaded_files: list[str] | None = None,
	) -> None:
		"""Mark the start of a new step.

        Captures the page state the agent is about to observe and establishes
        the baseline for browser events and file downloads for this step.
        """
		start_time = time.time()

		url = None
		title = None
		tabs: list[dict[str, Any]] = []
		dom_interactive_summary = None
		screenshot_path = None

		if browser_state_summary is not None:
			url = browser_state_summary.url
			title = browser_state_summary.title
			try:
				tabs = [tab.model_dump() for tab in browser_state_summary.tabs]
			except Exception:
				tabs = []
			if browser_state_summary.dom_state:
				try:
					dom_interactive_summary = browser_state_summary.dom_state.llm_representation()
				except Exception:
					dom_interactive_summary = None
			if hasattr(browser_state_summary, 'screenshot_path'):
				screenshot_path = browser_state_summary.screenshot_path

		self._pending = _PendingStepState(
			step_number=step_number,
			start_time=start_time,
			initial_url=url,
			initial_title=title,
			initial_tabs=tabs,
			initial_dom_summary=dom_interactive_summary,
			initial_screenshot_path=screenshot_path,
			initial_downloaded_files=set(downloaded_files or []),
			events=[],
		)

	def step_end(
		self,
		model_output: Any | None,
		action_results: list[Any] | None,
		browser_state_summary: Any | None,
		screenshot_path: str | None,
		downloaded_files: list[str],
		error: str | None = None,
	) -> None:
		"""Finalize the current step and write it to the trace file.

        Merges the data captured at :meth:`step_start` with the LLM decision,
        action execution results, post-step page state, download deltas and
        buffered browser events.
        """
		if self._pending is None:
			logger.warning('step_end called without matching step_start, skipping')
			return

		pending = self._pending
		self._pending = None
		end_time = time.time()

		# --- Extract LLM decision data -----------------------------------
		goal = None
		evaluation_previous_goal = None
		thinking = None
		memory = None
		actions: list[TraceAction] = []
		action_results_trace: list[TraceActionResult] = []

		if model_output is not None:
			goal = getattr(model_output, 'next_goal', None)
			evaluation_previous_goal = getattr(model_output, 'evaluation_previous_goal', None)
			thinking = getattr(model_output, 'thinking', None)
			memory = getattr(model_output, 'memory', None)

			model_actions = getattr(model_output, 'action', None)
			if model_actions:
				action_dicts = [a.model_dump(exclude_unset=True, mode='json') for a in model_actions]
				actions = self._sanitize_actions(action_dicts)

		if action_results:
			for r in action_results:
				action_results_trace.append(
					TraceActionResult(
						extracted_content=self._sanitize_string(r.extracted_content) if r.extracted_content else None,
						error=self._sanitize_string(r.error) if r.error else None,
						success=r.success,
						is_done=r.is_done,
						long_term_memory=self._sanitize_string(r.long_term_memory) if r.long_term_memory else None,
						attachments=r.attachments,
					)
				)

		# --- Resolve final page state ------------------------------------
		# Prefer the initial values (captured when the agent observed the
		# page before making a decision), but fall back to the summary
		# passed at step_end if step_start didn't capture them.
		url = pending.initial_url
		title = pending.initial_title
		tabs = pending.initial_tabs
		dom_interactive_summary = pending.initial_dom_summary
		final_screenshot_path = screenshot_path or pending.initial_screenshot_path

		if (url is None or title is None or dom_interactive_summary is None) and browser_state_summary is not None:
			if url is None:
				url = browser_state_summary.url
			if title is None:
				title = browser_state_summary.title
			if not tabs:
				try:
					tabs = [tab.model_dump() for tab in browser_state_summary.tabs]
				except Exception:
					tabs = []
			if dom_interactive_summary is None and browser_state_summary.dom_state:
				try:
					dom_interactive_summary = browser_state_summary.dom_state.llm_representation()
				except Exception:
					dom_interactive_summary = None

		# --- Compute download delta (only new files since step_start) ----
		current_downloads = set(downloaded_files)
		new_downloads = sorted(current_downloads - pending.initial_downloaded_files)

		# --- Assemble and write step -------------------------------------
		browser_events = pending.events

		step = TraceStep(
			step_number=pending.step_number,
			timestamp=end_time,
			goal=self._sanitize_string(goal) if goal else None,
			evaluation_previous_goal=self._sanitize_string(evaluation_previous_goal) if evaluation_previous_goal else None,
			thinking=self._sanitize_string(thinking) if thinking else None,
			memory=self._sanitize_string(memory) if memory else None,
			actions=actions,
			action_results=action_results_trace,
			url=url,
			title=title,
			tabs=tabs,
			dom_interactive_summary=dom_interactive_summary,
			screenshot_path=final_screenshot_path,
			downloaded_files=new_downloads,
			browser_events=browser_events,
			duration_seconds=end_time - pending.start_time,
			error=self._sanitize_string(error) if error else None,
		)

		self.trace_file.steps.append(step)
		self._flush()

	# ------------------------------------------------------------------
	# Persistence
	# ------------------------------------------------------------------

	def _flush(self) -> None:
		"""Write the current trace to disk."""
		try:
			self.trace_file.save(self._trace_path)
		except Exception as e:
			logger.warning(f'Failed to write trace file: {e}')

	def finalize(self) -> Path:
		"""Write final trace and return the path."""
		if self._pending is not None:
			self.step_end(
				model_output=None,
				action_results=None,
				browser_state_summary=None,
				screenshot_path=None,
				downloaded_files=[],
				error='Agent run terminated before step completed',
			)
		self._flush()
		return self._trace_path
