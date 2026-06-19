from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

from bubus import EventBus

from browser_use.agent.cloud_events import CreateAgentStepEvent
from browser_use.agent.views import (
	ActionResult,
	AgentHistory,
	AgentHistoryList,
	AgentOutput,
	AgentState,
	BrowserStateHistory,
	StepExecutionContext,
	StepExecutionResult,
	StepMetadata,
)
from browser_use.browser.views import BrowserStateSummary

if TYPE_CHECKING:
	from browser_use.agent.message_manager.service import MessageManager
	from browser_use.screenshots.service import ScreenshotService


class FileSystemSaver(Protocol):
	def save_file_system_state(self) -> None: ...


class DemoModeLogger(Protocol):
	async def _demo_mode_log(self, message: str, level: str = 'info', metadata: dict[str, Any] | None = None) -> None: ...


class StepLifecycleRecorder:
	"""Owns **all** side-effects that persist a step's result.

	Responsibilities (and **only** these):
	1. Sync ``state.last_model_output`` / ``last_result`` for the next step.
	2. Update ``state.consecutive_failures``.
	3. Build :class:`StepMetadata` and write the :class:`AgentHistory` item.
	4. Log step-completion summary + ``done()`` final-result banner.
	5. Persist file-system state.
	6. Dispatch the ``CreateAgentStepEvent`` cloud event.
	7. Bump ``state.n_steps``.

	None of the phase methods (``_get_next_action``, ``_execute_actions``,
	``_post_process``, ``_handle_step_error``) should perform any of these
	operations — they must go through :meth:`commit` instead.
	"""

	def __init__(
		self,
		state: AgentState,
		history: AgentHistoryList,
		eventbus: EventBus,
		logger: logging.Logger,
		screenshot_service: ScreenshotService,
		message_manager: MessageManager,
		fs_saver: FileSystemSaver,
		demo_logger: DemoModeLogger,
		agent_ref: Any,
	) -> None:
		self._state = state
		self._history = history
		self._eventbus = eventbus
		self._logger = logger
		self._screenshot_service = screenshot_service
		self._message_manager = message_manager
		self._fs_saver = fs_saver
		self._demo_logger = demo_logger
		self._agent_ref = agent_ref

	async def commit(self, ctx: StepExecutionContext, result: StepExecutionResult) -> None:
		"""Single point of persistence for a step's result.

		Takes the fully-populated :class:`StepExecutionResult` and performs
		*all* side-effects atomically.  Callers should never write to
		``state.last_*``, ``state.consecutive_failures``, ``state.n_steps``,
		or ``history`` directly.
		"""
		bss = ctx.browser_state_summary

		self._sync_cross_step_state(result)

		self._update_consecutive_failures(result)

		if result.is_empty and bss is None:
			return

		await self._write_history(ctx, result, bss)

		await self._log_step_summary(ctx, result)

		self._log_done_banner(result)

		self._persist_file_system()

		self._dispatch_step_event(result, bss)

		self._state.n_steps += 1

	def _sync_cross_step_state(self, result: StepExecutionResult) -> None:
		self._state.last_model_output = result.model_output
		self._state.last_result = result.action_results if result.action_results else None

	def _update_consecutive_failures(self, result: StepExecutionResult) -> None:
		error_for_counting = result.has_action_error or (
			result.error is not None and not result.action_results
		)

		if error_for_counting:
			self._state.consecutive_failures += 1
			self._logger.debug(
				f'🔄 Step {self._state.n_steps}: Consecutive failures: {self._state.consecutive_failures}'
			)
		elif result.action_results or result.model_output is not None:
			if self._state.consecutive_failures > 0:
				self._state.consecutive_failures = 0
				self._logger.debug(
					f'🔄 Step {self._state.n_steps}: Consecutive failures reset to: {self._state.consecutive_failures}'
				)

	async def _write_history(
		self,
		ctx: StepExecutionContext,
		result: StepExecutionResult,
		bss: BrowserStateSummary | None,
	) -> None:
		if bss is None or result.is_empty:
			return

		step_end_time = time.time()
		step_interval = None
		if len(self._history.history) > 0:
			last_history_item = self._history.history[-1]
			if last_history_item.metadata:
				previous_end_time = last_history_item.metadata.step_end_time
				previous_start_time = last_history_item.metadata.step_start_time
				step_interval = max(0, previous_end_time - previous_start_time)

		metadata = StepMetadata(
			step_number=self._state.n_steps,
			step_start_time=ctx.step_start_time,
			step_end_time=step_end_time,
			step_interval=step_interval,
		)

		await self._make_history_item(
			model_output=result.model_output,
			browser_state_summary=bss,
			result=result.action_results,
			metadata=metadata,
			state_message=self._message_manager.last_state_message_text,
		)

	async def _make_history_item(
		self,
		model_output: AgentOutput | None,
		browser_state_summary: BrowserStateSummary,
		result: list[ActionResult],
		metadata: StepMetadata | None = None,
		state_message: str | None = None,
	) -> None:
		if model_output:
			interacted_elements = AgentHistory.get_interacted_element(
				model_output, browser_state_summary.dom_state.selector_map
			)
		else:
			interacted_elements = [None]

		screenshot_path = None
		if browser_state_summary.screenshot:
			self._logger.debug(
				f'📸 Storing screenshot for step {self._state.n_steps}, screenshot length: {len(browser_state_summary.screenshot)}'
			)
			screenshot_path = await self._screenshot_service.store_screenshot(
				browser_state_summary.screenshot, self._state.n_steps
			)
			self._logger.debug(f'📸 Screenshot stored at: {screenshot_path}')
		else:
			self._logger.debug(f'📸 No screenshot in browser_state_summary for step {self._state.n_steps}')

		state_history = BrowserStateHistory(
			url=browser_state_summary.url,
			title=browser_state_summary.title,
			tabs=browser_state_summary.tabs,
			interacted_element=interacted_elements,
			screenshot_path=screenshot_path,
		)

		history_item = AgentHistory(
			model_output=model_output,
			result=result,
			state=state_history,
			metadata=metadata,
			state_message=state_message,
		)

		self._history.add_item(history_item)

	async def _log_step_summary(self, ctx: StepExecutionContext, result: StepExecutionResult) -> None:
		summary_message = self._build_step_summary_message(ctx.step_start_time, result.action_results)
		if summary_message:
			await self._demo_logger._demo_mode_log(summary_message, 'info', {'step': self._state.n_steps})

	def _build_step_summary_message(self, step_start_time: float, result: list[ActionResult]) -> str | None:
		if not result:
			return None

		step_duration = time.time() - step_start_time
		action_count = len(result)
		success_count = sum(1 for r in result if not r.error)
		failure_count = action_count - success_count

		success_indicator = f'✅ {success_count}' if success_count > 0 else ''
		failure_indicator = f'❌ {failure_count}' if failure_count > 0 else ''
		status_parts = [part for part in [success_indicator, failure_indicator] if part]
		status_str = ' | '.join(status_parts) if status_parts else '✅ 0'

		message = (
			f'📍 Step {self._state.n_steps}: Ran {action_count} action{"" if action_count == 1 else "s"} '
			f'in {step_duration:.2f}s: {status_str}'
		)
		self._logger.debug(message)
		return message

	def _log_done_banner(self, result: StepExecutionResult) -> None:
		if not result.is_done:
			return

		last = result.action_results[-1]
		if last.success:
			self._logger.info(f'\n📄 \033[32m Final Result:\033[0m \n{last.extracted_content}\n\n')
		else:
			self._logger.info(f'\n📄 \033[31m Final Result:\033[0m \n{last.extracted_content}\n\n')
		if last.attachments:
			total_attachments = len(last.attachments)
			for i, file_path in enumerate(last.attachments):
				self._logger.info(f'👉 Attachment {i + 1 if total_attachments > 1 else ""}: {file_path}')

	def _persist_file_system(self) -> None:
		self._fs_saver.save_file_system_state()

	def _dispatch_step_event(self, result: StepExecutionResult, bss: BrowserStateSummary | None) -> None:
		if bss is None or not result.model_output:
			return

		actions_data = []
		if result.model_output.action:
			for action in result.model_output.action:
				action_dict = action.model_dump() if hasattr(action, 'model_dump') else {}
				actions_data.append(action_dict)

		step_event = CreateAgentStepEvent.from_agent_step(
			self._agent_ref,
			result.model_output,
			result.action_results,
			actions_data,
			bss,
		)
		self._eventbus.dispatch(step_event)
