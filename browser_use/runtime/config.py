from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
	provider: Literal['openai', 'anthropic', 'google', 'browser_use', 'aws_bedrock', 'auto'] = 'auto'
	model: str | None = None
	temperature: float = 0.0
	api_key: str | None = None
	base_url: str | None = None
	aws_region: str | None = None
	aws_sso_auth: bool = False


class BrowserSessionConfig(BaseModel):
	headless: bool | None = None
	headed: bool = False
	keep_alive: bool = True
	user_data_dir: str | None = None
	profile_directory: str | None = None
	executable_path: str | None = None
	cdp_url: str | None = None
	window_width: int | None = None
	window_height: int | None = None
	allowed_domains: list[str] | None = None
	prohibited_domains: list[str] | None = None
	wait_between_actions: float | None = None
	device_scale_factor: float | None = None
	disable_security: bool = False
	proxy_server: str | None = None
	proxy_username: str | None = None
	proxy_password: str | None = None
	proxy_bypass: str | None = None
	use_cloud: bool = False
	cloud_profile_id: str | None = None
	cloud_proxy_country_code: str | None = None
	cloud_timeout: int | None = None
	downloads_path: str | None = None
	ignore_https_errors: bool = False
	is_mobile: bool | None = None
	file_system_path: str | None = None

	@property
	def is_real_chrome(self) -> bool:
		return self.executable_path is not None or self.profile_directory is not None


class TaskConfig(BaseModel):
	task: str
	llm: LLMConfig = Field(default_factory=LLMConfig)
	browser: BrowserSessionConfig = Field(default_factory=BrowserSessionConfig)
	agent_settings: dict[str, Any] = Field(default_factory=dict)
	source: str | None = None
	max_steps: int = 100
	use_vision: bool | Literal['auto'] = True


class RunResult(BaseModel):
	success: bool = True
	exit_code: int = 0
	final_result: str | None = None
	steps: int = 0
	duration_seconds: float = 0.0
	urls_visited: list[str] = Field(default_factory=list)
	errors: list[str | None] = Field(default_factory=list)
	is_done: bool = False

	@classmethod
	def from_agent_history(cls, history: Any, duration: float = 0.0) -> RunResult:
		final = history.final_result()
		is_successful = history.is_successful()
		errors = history.errors()
		urls = [str(u) for u in history.urls() if u is not None]
		num_steps = history.number_of_steps()
		return cls(
			success=is_successful is not False,
			exit_code=0 if is_successful is not False else 1,
			final_result=final,
			steps=num_steps,
			duration_seconds=duration or history.total_duration_seconds(),
			urls_visited=urls,
			errors=errors,
			is_done=history.is_done(),
		)

	@classmethod
	def from_error(cls, error: str | Exception, duration: float = 0.0) -> RunResult:
		msg = str(error)
		return cls(
			success=False,
			exit_code=1,
			final_result=None,
			duration_seconds=duration,
			errors=[msg],
			is_done=True,
		)

	def format_text(self) -> str:
		parts: list[str] = []
		parts.append(f'Task completed in {self.steps} steps')
		parts.append(f'Success: {self.success}')
		if self.final_result:
			parts.append(f'\nFinal result:\n{self.final_result}')
		if self.errors:
			filtered = [e for e in self.errors if e is not None]
			if filtered:
				import json

				parts.append(f'\nErrors encountered:\n{json.dumps(filtered, indent=2)}')
		if self.urls_visited:
			parts.append(f'\nURLs visited: {", ".join(self.urls_visited)}')
		return '\n'.join(parts)
