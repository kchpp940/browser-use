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


class TraceService:
	"""Collects step-level data from an :class:`Agent` and writes a trace file.

    Usage::

        agent = Agent(task="...", llm=..., trace_dir="./traces")
        await agent.run()

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
		self._step_events: list[TraceBrowserEvent] = []

	def register_event_bus(self, event_bus: Any) -> None:
		"""Subscribe to all browser events for the current step."""
		event_bus.on('*', self._on_browser_event)

	def _on_browser_event(self, event: Any) -> None:
		event_name = event.__class__.__name__
		try:
			data = event.model_dump(exclude_unset=True) if hasattr(event, 'model_dump') else {}
		except Exception:
			data = {}
		data = _sanitize_dict(data, self.sensitive_data)
		self._step_events.append(TraceBrowserEvent(event_type=event_name, data=data))

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

	def record_step(
		self,
		step_number: int,
		model_output: Any | None,
		action_results: list[Any] | None,
		browser_state_summary: Any | None,
		screenshot_path: str | None,
		downloaded_files: list[str],
		step_start_time: float,
		step_end_time: float,
		error: str | None = None,
	) -> None:
		"""Collect data for a completed step and append to the trace."""
		goal = None
		evaluation_previous_goal = None
		thinking = None
		memory = None
		actions: list[TraceAction] = []
		action_results_trace: list[TraceActionResult] = []

		if model_output is not None:
			goal = model_output.next_goal
			evaluation_previous_goal = model_output.evaluation_previous_goal
			thinking = getattr(model_output, 'thinking', None)
			memory = model_output.memory

			if model_output.action:
				action_dicts = [a.model_dump(exclude_unset=True, mode='json') for a in model_output.action]
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

		url = None
		title = None
		tabs: list[dict[str, Any]] = []
		dom_interactive_summary = None

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

		browser_events = self._step_events
		self._step_events = []

		step = TraceStep(
			step_number=step_number,
			timestamp=step_end_time,
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
			screenshot_path=screenshot_path,
			downloaded_files=downloaded_files,
			browser_events=browser_events,
			duration_seconds=step_end_time - step_start_time if step_start_time else None,
			error=self._sanitize_string(error) if error else None,
		)

		self.trace_file.steps.append(step)
		self._flush()

	def _flush(self) -> None:
		"""Write the current trace to disk."""
		try:
			self.trace_file.save(self._trace_path)
		except Exception as e:
			logger.warning(f'Failed to write trace file: {e}')

	def finalize(self) -> Path:
		"""Write final trace and return the path."""
		self._flush()
		return self._trace_path
