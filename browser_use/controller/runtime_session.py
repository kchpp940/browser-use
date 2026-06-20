from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
	from browser_use.agent.service import Agent, AgentHookFunc
	from browser_use.agent.views import AgentHistoryList, AgentStepInfo

logger = logging.getLogger(__name__)


class RuntimeSessionController:
	"""Unified runtime lifecycle controller for Agent tasks.

	Centralizes all lifecycle management that was previously duplicated across
	Python API, legacy CLI, skill_cli daemon, MCP server, and sandbox/cloud
	entry points. Manages:

	- Task startup (signal handlers, session/task events, browser session)
	- Step execution loop with unified timeout and cancellation
	- Exception normalization (KeyboardInterrupt, TimeoutError, connection errors)
	- File system state persistence
	- Event bus lifecycle
	- Browser/session cleanup
	- Telemetry capture
	- GIF generation
	- Final result submission

	All entry points should go through this controller instead of maintaining
	their own try/except/finally, timeout, and cleanup logic.
	"""

	def __init__(self, agent: Agent):
		self.agent = agent
		self._agent_run_error: str | None = None
		self._force_exit_telemetry_logged = False
		self._should_delay_close = False
		self._signal_handler: Any = None
		self._max_steps: int = 500
		self._on_step_start: AgentHookFunc | None = None
		self._on_step_end: AgentHookFunc | None = None

	async def run(
		self,
		max_steps: int = 500,
		on_step_start: AgentHookFunc | None = None,
		on_step_end: AgentHookFunc | None = None,
	) -> AgentHistoryList:
		"""Execute the full agent lifecycle and return the final history.

		This is the single authoritative entry point for running an Agent.
		All other entry points (CLI, MCP, daemon, sandbox) should delegate to this.
		"""
		self._max_steps = max_steps
		self._on_step_start = on_step_start
		self._on_step_end = on_step_end
		self._agent_run_error = None
		self._force_exit_telemetry_logged = False
		self._should_delay_close = False

		self._setup_signal_handlers(max_steps)

		try:
			return await self._execute_lifecycle(max_steps, on_step_start, on_step_end)
		except KeyboardInterrupt:
			return await self._handle_keyboard_interrupt()
		except Exception as e:
			return await self._handle_generic_exception(e)
		finally:
			await self._cleanup(max_steps)

	def _setup_signal_handlers(self, max_steps: int) -> None:
		"""Register signal handlers for pause/resume and force-exit telemetry."""
		loop = asyncio.get_event_loop()

		from browser_use.utils import SignalHandler

		def on_force_exit_log_telemetry():
			self.agent._log_agent_event(max_steps=max_steps, agent_run_error='SIGINT: Cancelled by user')
			if hasattr(self.agent, 'telemetry') and self.agent.telemetry:
				self.agent.telemetry.flush()
			self._force_exit_telemetry_logged = True

		self._signal_handler = SignalHandler(
			loop=loop,
			pause_callback=self.agent.pause,
			resume_callback=self.agent.resume,
			custom_exit_callback=on_force_exit_log_telemetry,
			exit_on_second_int=True,
			disabled=not self.agent.enable_signal_handler,
		)
		self._signal_handler.register()

	async def _execute_lifecycle(
		self,
		max_steps: int,
		on_step_start: AgentHookFunc | None,
		on_step_end: AgentHookFunc | None,
	) -> AgentHistoryList:
		"""Execute the main agent lifecycle: init -> step loop -> finalize."""
		await self._startup_phase()
		await self._initial_actions_phase()
		await self._main_step_loop(max_steps, on_step_start, on_step_end)
		return await self._finalize_success(max_steps)

	async def _startup_phase(self) -> None:
		"""Phase 1: Log startup, dispatch events, start browser, register skills."""
		from browser_use.agent.cloud_events import CreateAgentSessionEvent, CreateAgentTaskEvent

		await self.agent._log_agent_run()

		self.agent.logger.debug(
			f'🔧 Agent setup: Agent Session ID {self.agent.session_id[-4:]}, Task ID {self.agent.task_id[-4:]}, '
			f'Browser Session ID {self.agent.browser_session.id[-4:] if self.agent.browser_session else "None"} '
			f'{"(connecting via CDP)" if (self.agent.browser_session and self.agent.browser_session.cdp_url) else "(launching local browser)"}'
		)

		self.agent._session_start_time = time.time()
		self.agent._task_start_time = self.agent._session_start_time

		if not self.agent.state.session_initialized:
			self.agent.logger.debug('📡 Dispatching CreateAgentSessionEvent...')
			self.agent.eventbus.dispatch(CreateAgentSessionEvent.from_agent(self.agent))
			self.agent.state.session_initialized = True

		self.agent.logger.debug('📡 Dispatching CreateAgentTaskEvent...')
		self.agent.eventbus.dispatch(CreateAgentTaskEvent.from_agent(self.agent))

		self.agent._log_first_step_startup()
		await self.agent.browser_session.start()

		if self.agent._demo_mode_enabled:
			await self.agent._demo_mode_log(f'Started task: {self.agent.task}', 'info', {'tag': 'task'})
			await self.agent._demo_mode_log(
				'Demo mode active - follow the side panel for live thoughts and actions.',
				'info',
				{'tag': 'status'},
			)

		await self.agent._register_skills_as_actions()

	async def _initial_actions_phase(self) -> None:
		"""Phase 2: Execute initial actions (URL navigation etc.) with step timeout."""
		from browser_use.agent.views import ActionResult

		try:
			await asyncio.wait_for(
				self.agent._execute_initial_actions(),
				timeout=self.agent.settings.step_timeout,
			)
		except InterruptedError:
			pass
		except TimeoutError:
			initial_timeout_msg = (
				f'Initial actions timed out after {self.agent.settings.step_timeout}s '
				f'(browser may be unresponsive). Proceeding to main execution loop.'
			)
			self.agent.logger.error(f'⏰ {initial_timeout_msg}')
			self.agent.state.last_result = [ActionResult(error=initial_timeout_msg)]
			self.agent.state.consecutive_failures += 1

	async def _main_step_loop(
		self,
		max_steps: int,
		on_step_start: AgentHookFunc | None,
		on_step_end: AgentHookFunc | None,
	) -> None:
		"""Phase 3: Main step execution loop with pause/stop checks."""
		from browser_use.agent.views import AgentStepInfo

		self.agent.logger.debug(
			f'🔄 Starting main execution loop with max {max_steps} steps (currently at step {self.agent.state.n_steps})...'
		)

		while self.agent.state.n_steps <= max_steps:
			current_step = self.agent.state.n_steps - 1

			if self.agent.state.paused:
				self.agent.logger.debug(f'⏸️ Step {self.agent.state.n_steps}: Agent paused, waiting to resume...')
				await self.agent._external_pause_event.wait()
				self._signal_handler.reset()

			if (self.agent.state.consecutive_failures) >= self.agent.settings.max_failures + int(
				self.agent.settings.final_response_after_failure
			):
				self.agent.logger.error(f'❌ Stopping due to {self.agent.settings.max_failures} consecutive failures')
				self._agent_run_error = f'Stopped due to {self.agent.settings.max_failures} consecutive failures'
				break

			if self.agent.state.stopped:
				self.agent.logger.info('🛑 Agent stopped')
				self._agent_run_error = 'Agent stopped programmatically'
				break

			step_info = AgentStepInfo(step_number=current_step, max_steps=max_steps)
			is_done = await self._execute_single_step(current_step, max_steps, step_info, on_step_start, on_step_end)

			if is_done:
				if self.agent._demo_mode_enabled and self.agent.history.history:
					final_result_text = self.agent.history.final_result() or 'Task completed'
					await self.agent._demo_mode_log(f'Final Result: {final_result_text}', 'success', {'tag': 'task'})
				self._should_delay_close = True
				break
		else:
			self._handle_max_steps_exceeded(max_steps)

	async def _execute_single_step(
		self,
		current_step: int,
		max_steps: int,
		step_info: AgentStepInfo,
		on_step_start: AgentHookFunc | None,
		on_step_end: AgentHookFunc | None,
	) -> bool:
		"""Execute a single step with timeout wrapper and callback hooks.

		Returns True if the agent reports task completion, False otherwise.
		"""
		if on_step_start is not None:
			await on_step_start(self.agent)

		await self.agent._demo_mode_log(
			f'Starting step {current_step + 1}/{max_steps}',
			'info',
			{'step': current_step + 1, 'total_steps': max_steps},
		)

		self.agent.logger.debug(f'🚶 Starting step {current_step + 1}/{max_steps}...')

		try:
			await asyncio.wait_for(
				self.agent.step(step_info),
				timeout=self.agent.settings.step_timeout,
			)
			self.agent.logger.debug(f'✅ Completed step {current_step + 1}/{max_steps}')
		except TimeoutError:
			from browser_use.agent.views import ActionResult

			error_msg = f'Step {current_step + 1} timed out after {self.agent.settings.step_timeout} seconds'
			self.agent.logger.error(f'⏰ {error_msg}')
			await self.agent._demo_mode_log(error_msg, 'error', {'step': current_step + 1})
			self.agent.state.consecutive_failures += 1
			self.agent.state.last_result = [ActionResult(error=error_msg)]
			if self.agent.state.n_steps == current_step + 1:
				self.agent.state.n_steps += 1

		if on_step_end is not None:
			await on_step_end(self.agent)

		if self.agent.history.is_done():
			await self.agent.log_completion()
			if self.agent.settings.use_judge:
				await self.agent._judge_and_log()
			if self.agent.register_done_callback:
				import inspect

				if inspect.iscoroutinefunction(self.agent.register_done_callback):
					await self.agent.register_done_callback(self.agent.history)
				else:
					self.agent.register_done_callback(self.agent.history)
			return True

		return False

	def _handle_max_steps_exceeded(self, max_steps: int) -> None:
		"""Handle the case where agent exhausts all steps without completing."""
		from browser_use.agent.views import ActionResult, AgentHistory, BrowserStateHistory

		self._agent_run_error = 'Failed to complete task in maximum steps'

		self.agent.history.add_item(
			AgentHistory(
				model_output=None,
				result=[ActionResult(error=self._agent_run_error, include_in_memory=True)],
				state=BrowserStateHistory(
					url='',
					title='',
					tabs=[],
					interacted_element=[],
					screenshot_path=None,
				),
				metadata=None,
			)
		)

		self.agent.logger.info(f'❌ {self._agent_run_error}')

	async def _finalize_success(self, max_steps: int) -> AgentHistoryList:
		"""Phase 4: After successful execution - attach usage, output schema."""
		self.agent.history.usage = await self.agent.token_cost_service.get_usage_summary()

		if self.agent.history._output_model_schema is None and self.agent.output_model_schema is not None:
			self.agent.history._output_model_schema = self.agent.output_model_schema

		return self.agent.history

	async def _handle_keyboard_interrupt(self) -> AgentHistoryList:
		"""Handle KeyboardInterrupt - normalize and return current history."""
		self.agent.logger.debug('Got KeyboardInterrupt during execution, returning current history')
		self._agent_run_error = 'KeyboardInterrupt'
		self.agent.history.usage = await self.agent.token_cost_service.get_usage_summary()
		return self.agent.history

	async def _handle_generic_exception(self, e: Exception) -> AgentHistoryList:
		"""Handle all other exceptions - log and re-raise."""
		self.agent.logger.error(f'Agent run failed with exception: {e}', exc_info=True)
		self._agent_run_error = str(e)
		raise e

	async def _cleanup(self, max_steps: int) -> None:
		"""Phase 5: Unified cleanup - always runs in finally block.

		Handles:
		- Demo mode delay
		- Token cost logging
		- Signal handler unregistration
		- Telemetry capture
		- UpdateAgentTaskEvent dispatch
		- GIF generation
		- Final outcome messages
		- Event bus stop
		- Agent resource close
		"""
		from browser_use.agent.cloud_events import CreateAgentOutputFileEvent, UpdateAgentTaskEvent
		from browser_use.browser.events import _get_timeout

		if self._should_delay_close and self.agent._demo_mode_enabled and self._agent_run_error is None:
			await asyncio.sleep(30)

		if self._agent_run_error:
			await self.agent._demo_mode_log(f'Agent stopped: {self._agent_run_error}', 'error', {'tag': 'run'})

		await self.agent.token_cost_service.log_usage_summary()

		if self._signal_handler is not None:
			self._signal_handler.unregister()

		if not self._force_exit_telemetry_logged:
			try:
				self.agent._log_agent_event(max_steps=max_steps, agent_run_error=self._agent_run_error)
			except Exception as log_e:
				self.agent.logger.error(f'Failed to log telemetry event: {log_e}', exc_info=True)
		else:
			self.agent.logger.debug('Telemetry for force exit (SIGINT) was logged by custom exit callback.')

		self.agent.eventbus.dispatch(UpdateAgentTaskEvent.from_agent(self.agent))

		if self.agent.settings.generate_gif:
			output_path: str = 'agent_history.gif'
			if isinstance(self.agent.settings.generate_gif, str):
				output_path = self.agent.settings.generate_gif

			from browser_use.agent.gif import create_history_gif

			create_history_gif(task=self.agent.task, history=self.agent.history, output_path=output_path)

			if Path(output_path).exists():
				output_event = await CreateAgentOutputFileEvent.from_agent_and_file(self.agent, output_path)
				self.agent.eventbus.dispatch(output_event)

		self.agent._log_final_outcome_messages()

		await self.agent.eventbus.stop(clear=True, timeout=_get_timeout('TIMEOUT_AgentEventBusStop', 3.0))

		await self.agent.close()
