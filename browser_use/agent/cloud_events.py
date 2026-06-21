import base64
import os
from datetime import datetime, timezone
from pathlib import Path

import anyio
from bubus import BaseEvent
from pydantic import Field, field_validator
from uuid_extensions import uuid7str

from browser_use.observability_runtime import (
	ErrorInfo,
	EventSeverity,
	EventSource,
	EventType,
	OutputFile,
	RuntimeEvent,
)

MAX_STRING_LENGTH = 500000  # 100K chars ~ 25k tokens should be enough
MAX_URL_LENGTH = 100000
MAX_TASK_LENGTH = 100000
MAX_COMMENT_LENGTH = 2000
MAX_FILE_CONTENT_SIZE = 50 * 1024 * 1024  # 50MB


def _ensure_utc(dt: datetime | None) -> datetime | None:
	if dt is None:
		return None
	if dt.tzinfo is None:
		return dt.replace(tzinfo=timezone.utc)
	return dt.astimezone(timezone.utc)


class UpdateAgentTaskEvent(BaseEvent):
	# Required fields for identification
	id: str  # The task ID to update
	user_id: str = Field(max_length=255)  # For authorization
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup

	# Optional fields that can be updated
	stopped: bool | None = None
	paused: bool | None = None
	done_output: str | None = Field(None, max_length=MAX_STRING_LENGTH)
	finished_at: datetime | None = None
	agent_state: dict | None = None
	user_feedback_type: str | None = Field(None, max_length=10)  # UserFeedbackType enum value as string
	user_comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	gif_url: str | None = Field(None, max_length=MAX_URL_LENGTH)

	@classmethod
	def from_agent(cls, agent) -> 'UpdateAgentTaskEvent':
		"""Create an UpdateAgentTaskEvent from an Agent instance"""
		if not hasattr(agent, '_task_start_time'):
			raise ValueError('Agent must have _task_start_time attribute')

		done_output = agent.history.final_result() if agent.history else None
		if done_output and len(done_output) > MAX_STRING_LENGTH:
			done_output = done_output[:MAX_STRING_LENGTH]
		return cls(
			id=str(agent.task_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			stopped=agent.state.stopped if hasattr(agent.state, 'stopped') else False,
			paused=agent.state.paused if hasattr(agent.state, 'paused') else False,
			done_output=done_output,
			finished_at=datetime.now(timezone.utc) if agent.history and agent.history.is_done() else None,
			agent_state=agent.state.model_dump() if hasattr(agent.state, 'model_dump') else {},
			user_feedback_type=None,
			user_comment=None,
			gif_url=None,
			# user_feedback_type and user_comment would be set by the API/frontend
			# gif_url would be set after GIF generation if needed
		)

	def to_runtime_event(self) -> RuntimeEvent:
		"""Convert this cloud event to a unified RuntimeEvent.

		All canonical linking fields (task_id / session_id / step / error /
		output_files / source) come from a single adapter — no hand-built
		payloads in the cloud sync path.
		"""
		error_info: ErrorInfo | None = None
		if self.done_output and self.stopped and not self.paused:
			error_info = ErrorInfo(error_type='TaskStopped', error_message='Task was stopped')

		return RuntimeEvent(
			source=EventSource.AGENT,
			event_type=EventType.TASK_END,
			severity=EventSeverity.ERROR if error_info else EventSeverity.INFO,
			task_id=self.id,
			message='Agent task update',
			error=error_info,
			data={
				'stopped': self.stopped,
				'paused': self.paused,
				'done_output': self.done_output,
				'agent_state': self.agent_state,
				'user_feedback_type': self.user_feedback_type,
				'user_comment': self.user_comment,
				'gif_url': self.gif_url,
			},
			timestamp=_ensure_utc(self.finished_at) or datetime.now(timezone.utc),
		)

	def to_cloud_event_dict(self) -> dict:
		"""Serialize to a cloud-compatible payload via RuntimeEvent adapter."""
		return self.to_runtime_event().to_cloud_event_dict()


class CreateAgentOutputFileEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	task_id: str
	file_name: str = Field(max_length=255)
	file_content: str | None = None  # Base64 encoded file content
	content_type: str | None = Field(None, max_length=100)  # MIME type for file uploads
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

	@field_validator('file_content')
	@classmethod
	def validate_file_size(cls, v: str | None) -> str | None:
		"""Validate base64 file content size."""
		if v is None:
			return v
		# Remove data URL prefix if present
		if ',' in v:
			v = v.split(',')[1]
		# Estimate decoded size (base64 is ~33% larger)
		estimated_size = len(v) * 3 / 4
		if estimated_size > MAX_FILE_CONTENT_SIZE:
			raise ValueError(f'File content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

	@classmethod
	async def from_agent_and_file(cls, agent, output_path: str) -> 'CreateAgentOutputFileEvent':
		"""Create a CreateAgentOutputFileEvent from a file path"""

		gif_path = Path(output_path)
		if not gif_path.exists():
			raise FileNotFoundError(f'File not found: {output_path}')

		gif_size = os.path.getsize(gif_path)

		# Read GIF content for base64 encoding if needed
		gif_content = None
		if gif_size < 50 * 1024 * 1024:  # Only read if < 50MB
			async with await anyio.open_file(gif_path, 'rb') as f:
				gif_bytes = await f.read()
				gif_content = base64.b64encode(gif_bytes).decode('utf-8')

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			task_id=str(agent.task_id),
			file_name=gif_path.name,
			file_content=gif_content,  # Base64 encoded
			content_type='image/gif',
		)

	def to_runtime_event(self) -> RuntimeEvent:
		"""Convert to a unified RuntimeEvent with output_files via adapter."""
		return RuntimeEvent(
			source=EventSource.AGENT,
			event_type=EventType.OUTPUT_FILE,
			severity=EventSeverity.INFO,
			task_id=self.task_id,
			message=f'Output file: {self.file_name}',
			output_files=[
				OutputFile(
					name=self.file_name,
					path=None,
					size_bytes=None,
					mime_type=self.content_type,
				)
			],
			data={
				'file_name': self.file_name,
				'content_type': self.content_type,
			},
			timestamp=_ensure_utc(self.created_at) or datetime.now(timezone.utc),
		)

	def to_cloud_event_dict(self) -> dict:
		"""Serialize to a cloud-compatible payload via RuntimeEvent adapter."""
		return self.to_runtime_event().to_cloud_event_dict()


class CreateAgentStepEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)  # Added for authorization checks
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	agent_task_id: str
	step: int
	evaluation_previous_goal: str = Field(max_length=MAX_STRING_LENGTH)
	memory: str = Field(max_length=MAX_STRING_LENGTH)
	next_goal: str = Field(max_length=MAX_STRING_LENGTH)
	actions: list[dict]
	screenshot_url: str | None = Field(None, max_length=MAX_FILE_CONTENT_SIZE)  # ~50MB for base64 images
	url: str = Field(default='', max_length=MAX_URL_LENGTH)

	@field_validator('screenshot_url')
	@classmethod
	def validate_screenshot_size(cls, v: str | None) -> str | None:
		"""Validate screenshot URL or base64 content size."""
		if v is None or not v.startswith('data:'):
			return v
		# It's base64 data, check size
		if ',' in v:
			base64_part = v.split(',')[1]
			estimated_size = len(base64_part) * 3 / 4
			if estimated_size > MAX_FILE_CONTENT_SIZE:
				raise ValueError(f'Screenshot content exceeds maximum size of {MAX_FILE_CONTENT_SIZE / 1024 / 1024}MB')
		return v

	@classmethod
	def from_agent_step(
		cls, agent, model_output, result: list, actions_data: list[dict], browser_state_summary
	) -> 'CreateAgentStepEvent':
		"""Create a CreateAgentStepEvent from agent step data"""
		# Get first action details if available
		first_action = model_output.action[0] if model_output.action else None

		# Extract current state from model output
		current_state = model_output.current_state if hasattr(model_output, 'current_state') else None

		# Capture screenshot as base64 data URL if available
		screenshot_url = None
		if browser_state_summary.screenshot:
			screenshot_url = f'data:image/png;base64,{browser_state_summary.screenshot}'
			import logging

			logger = logging.getLogger(__name__)
			logger.debug(f'📸 Including screenshot in CreateAgentStepEvent, length: {len(browser_state_summary.screenshot)}')
		else:
			import logging

			logger = logging.getLogger(__name__)
			logger.debug('📸 No screenshot in browser_state_summary for CreateAgentStepEvent')

		return cls(
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			agent_task_id=str(agent.task_id),
			step=agent.state.n_steps,
			evaluation_previous_goal=current_state.evaluation_previous_goal if current_state else '',
			memory=current_state.memory if current_state else '',
			next_goal=current_state.next_goal if current_state else '',
			actions=actions_data,  # List of action dicts
			url=browser_state_summary.url,
			screenshot_url=screenshot_url,
		)

	def to_runtime_event(self) -> RuntimeEvent:
		"""Convert to a unified RuntimeEvent with step via adapter."""
		return RuntimeEvent(
			source=EventSource.AGENT,
			event_type=EventType.STEP_END,
			severity=EventSeverity.INFO,
			task_id=self.agent_task_id,
			step=self.step,
			message=f'Step {self.step}: {self.next_goal}',
			data={
				'evaluation_previous_goal': self.evaluation_previous_goal,
				'memory': self.memory,
				'next_goal': self.next_goal,
				'actions': self.actions,
				'url': self.url,
				'screenshot_url': self.screenshot_url,
			},
			timestamp=_ensure_utc(self.created_at) or datetime.now(timezone.utc),
		)

	def to_cloud_event_dict(self) -> dict:
		"""Serialize to a cloud-compatible payload via RuntimeEvent adapter."""
		return self.to_runtime_event().to_cloud_event_dict()


class CreateAgentTaskEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)  # Added for authorization checks
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	agent_session_id: str
	llm_model: str = Field(max_length=200)  # LLMModel enum value as string
	stopped: bool = False
	paused: bool = False
	task: str = Field(max_length=MAX_TASK_LENGTH)
	done_output: str | None = Field(None, max_length=MAX_STRING_LENGTH)
	scheduled_task_id: str | None = None
	started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
	finished_at: datetime | None = None
	agent_state: dict = Field(default_factory=dict)
	user_feedback_type: str | None = Field(None, max_length=10)  # UserFeedbackType enum value as string
	user_comment: str | None = Field(None, max_length=MAX_COMMENT_LENGTH)
	gif_url: str | None = Field(None, max_length=MAX_URL_LENGTH)

	@classmethod
	def from_agent(cls, agent) -> 'CreateAgentTaskEvent':
		"""Create a CreateAgentTaskEvent from an Agent instance"""
		return cls(
			id=str(agent.task_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			agent_session_id=str(agent.session_id),
			task=agent.task,
			llm_model=agent.llm.model_name,
			agent_state=agent.state.model_dump() if hasattr(agent.state, 'model_dump') else {},
			stopped=False,
			paused=False,
			done_output=None,
			started_at=datetime.fromtimestamp(agent._task_start_time, tz=timezone.utc),
			finished_at=None,
			user_feedback_type=None,
			user_comment=None,
			gif_url=None,
		)

	def to_runtime_event(self) -> RuntimeEvent:
		"""Convert to a unified RuntimeEvent with task_id/session_id via adapter."""
		error_info: ErrorInfo | None = None
		if self.done_output and self.stopped:
			error_info = ErrorInfo(error_type='TaskStopped', error_message='Task was stopped')

		return RuntimeEvent(
			source=EventSource.AGENT,
			event_type=EventType.TASK_START if not self.finished_at else EventType.TASK_END,
			severity=EventSeverity.ERROR if error_info else EventSeverity.INFO,
			task_id=self.id,
			session_id=self.agent_session_id,
			message=self.task,
			error=error_info,
			data={
				'llm_model': self.llm_model,
				'stopped': self.stopped,
				'paused': self.paused,
				'agent_state': self.agent_state,
				'scheduled_task_id': self.scheduled_task_id,
				'done_output': self.done_output,
				'user_feedback_type': self.user_feedback_type,
				'user_comment': self.user_comment,
				'gif_url': self.gif_url,
			},
			timestamp=_ensure_utc(self.started_at) or datetime.now(timezone.utc),
			duration_ms=(
				(self.finished_at - self.started_at).total_seconds() * 1000.0 if self.finished_at and self.started_at else None
			),
		)

	def to_cloud_event_dict(self) -> dict:
		"""Serialize to a cloud-compatible payload via RuntimeEvent adapter."""
		return self.to_runtime_event().to_cloud_event_dict()


class CreateAgentSessionEvent(BaseEvent):
	# Model fields
	id: str = Field(default_factory=uuid7str)
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)  # Device ID for auth lookup
	browser_session_id: str = Field(max_length=255)
	browser_session_live_url: str = Field(max_length=MAX_URL_LENGTH)
	browser_session_cdp_url: str = Field(max_length=MAX_URL_LENGTH)
	browser_session_stopped: bool = False
	browser_session_stopped_at: datetime | None = None
	is_source_api: bool | None = None
	browser_state: dict = Field(default_factory=dict)
	browser_session_data: dict | None = None

	@classmethod
	def from_agent(cls, agent) -> 'CreateAgentSessionEvent':
		"""Create a CreateAgentSessionEvent from an Agent instance"""
		return cls(
			id=str(agent.session_id),
			user_id='',  # To be filled by cloud handler
			device_id=agent.cloud_sync.auth_client.device_id
			if hasattr(agent, 'cloud_sync') and agent.cloud_sync and agent.cloud_sync.auth_client
			else None,
			browser_session_id=agent.browser_session.id,
			browser_session_live_url='',  # To be filled by cloud handler
			browser_session_cdp_url='',  # To be filled by cloud handler
			browser_state={
				'viewport': agent.browser_profile.viewport if agent.browser_profile else {'width': 1280, 'height': 720},
				'user_agent': agent.browser_profile.user_agent if agent.browser_profile else None,
				'headless': agent.browser_profile.headless if agent.browser_profile else True,
				'initial_url': None,  # Will be updated during execution
				'final_url': None,  # Will be updated during execution
				'total_pages_visited': 0,  # Will be updated during execution
				'session_duration_seconds': 0,  # Will be updated during execution
			},
			browser_session_data={
				'cookies': [],
				'secrets': {},
				# TODO: send secrets safely so tasks can be replayed on cloud seamlessly
				# 'secrets': dict(agent.sensitive_data) if agent.sensitive_data else {},
				'allowed_domains': agent.browser_profile.allowed_domains if agent.browser_profile else [],
			},
		)

	def to_runtime_event(self) -> RuntimeEvent:
		"""Convert to a unified RuntimeEvent with session_id via adapter."""
		return RuntimeEvent(
			source=EventSource.BROWSER,
			event_type=EventType.SESSION_START,
			severity=EventSeverity.INFO,
			session_id=self.browser_session_id,
			message='Browser session started',
			data={
				'browser_session_live_url': self.browser_session_live_url,
				'browser_session_cdp_url': self.browser_session_cdp_url,
				'browser_session_stopped': self.browser_session_stopped,
				'is_source_api': self.is_source_api,
				'browser_state': self.browser_state,
				'browser_session_data': self.browser_session_data,
			},
			timestamp=datetime.now(timezone.utc),
		)

	def to_cloud_event_dict(self) -> dict:
		"""Serialize to a cloud-compatible payload via RuntimeEvent adapter."""
		return self.to_runtime_event().to_cloud_event_dict()


class UpdateAgentSessionEvent(BaseEvent):
	"""Event to update an existing agent session"""

	# Model fields
	id: str  # Session ID to update
	user_id: str = Field(max_length=255)
	device_id: str | None = Field(None, max_length=255)
	browser_session_stopped: bool | None = None
	browser_session_stopped_at: datetime | None = None
	end_reason: str | None = Field(None, max_length=100)  # Why the session ended

	def to_runtime_event(self) -> RuntimeEvent:
		"""Convert to a unified RuntimeEvent with session_id via adapter."""
		return RuntimeEvent(
			source=EventSource.BROWSER,
			event_type=EventType.SESSION_END if self.browser_session_stopped else EventType.SESSION_UPDATE,
			severity=EventSeverity.INFO,
			session_id=self.id,
			message=f'Session update: {self.end_reason or "updated"}',
			error=ErrorInfo(error_type=self.end_reason, error_message=self.end_reason) if self.end_reason else None,
			data={
				'browser_session_stopped': self.browser_session_stopped,
				'end_reason': self.end_reason,
			},
			timestamp=_ensure_utc(self.browser_session_stopped_at) or datetime.now(timezone.utc),
		)

	def to_cloud_event_dict(self) -> dict:
		"""Serialize to a cloud-compatible payload via RuntimeEvent adapter."""
		return self.to_runtime_event().to_cloud_event_dict()
