"""Pydantic models for task templates.

Defines the data structures for reusable browser automation task templates,
including variables, browser profiles, tool restrictions, output file rules,
and execution results.
"""

import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TemplateVariableType(str, enum.Enum):
	"""Supported types for template variables."""

	STRING = 'string'
	INTEGER = 'integer'
	FLOAT = 'float'
	BOOLEAN = 'boolean'
	LIST = 'list'
	DICT = 'dict'


class TemplateVariable(BaseModel):
	"""Definition of a template variable.

	Each variable has a name, type, optional default value, and description
	that helps users understand what the variable controls.
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	type: TemplateVariableType = Field(
		default=TemplateVariableType.STRING,
		description='The data type of the variable',
	)
	default: Any | None = Field(
		default=None,
		description='Default value if not provided at runtime',
	)
	description: str = Field(
		default='',
		description='Human-readable description of what this variable controls',
	)
	required: bool = Field(
		default=False,
		description='If True, the variable must be provided at runtime',
	)
	choices: list[Any] | None = Field(
		default=None,
		description='If set, restricts values to this list of choices',
	)


class OutputFileRule(BaseModel):
	"""Rule for capturing output files from template execution.

	Output files can use the same {{ variable }} substitution as prompts.
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	pattern: str = Field(
		description='File path or glob pattern for files to capture. '
		'Supports {{ variable }} substitution.',
	)
	description: str = Field(
		default='',
		description='Human-readable description of what this file contains',
	)
	required: bool = Field(
		default=False,
		description='If True, execution fails if this file is not produced',
	)


class LLMProviderConfig(BaseModel):
	"""Default LLM provider configuration for a template.

	Users can override these at runtime, but the template provides sensible
	defaults for the task.
	"""

	model_config = ConfigDict(extra='allow', populate_by_name=True)

	model: str | None = Field(
		default=None,
		description='Model name (e.g., "gpt-4.1-mini", "claude-sonnet-4-0")',
	)
	provider: str | None = Field(
		default=None,
		description='Provider name: "openai", "anthropic", "google", "browser_use", etc. '
		'If omitted, the model name is used to infer the provider.',
	)
	base_url: str | None = Field(
		default=None,
		description='Custom API base URL for the provider',
	)
	temperature: float | None = Field(
		default=None,
		description='Sampling temperature (0.0 = deterministic, 1.0 = creative)',
		ge=0.0,
		le=2.0,
	)
	api_key_env: str | None = Field(
		default=None,
		description='Environment variable name for the API key. '
		'If omitted, uses the provider\'s default env var.',
	)


class TaskTemplateStatus(str, enum.Enum):
	"""Status of a template execution."""

	SUCCESS = 'success'
	FAILED = 'failed'
	PARTIAL = 'partial'
	RUNNING = 'running'


class TaskTemplate(BaseModel):
	"""A reusable browser automation task template.

	Templates encapsulate everything needed to run a specific automation task
	repeatedly: the prompt (with variable placeholders), browser configuration,
	LLM settings, allowed tools, and output file rules.

	Variables use Jinja2-style double-brace syntax: {{ variable_name }}
	"""

	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	name: str = Field(
		description='Unique name for the template (kebab-case or snake_case recommended)',
		pattern=r'^[a-zA-Z][a-zA-Z0-9_-]*$',
	)
	description: str = Field(
		default='',
		description='Human-readable description of what this template does',
	)
	prompt_template: str = Field(
		description='The task prompt with {{ variable }} placeholders for dynamic parts',
	)
	variables: dict[str, TemplateVariable] = Field(
		default_factory=dict,
		description='Definitions of all variables referenced in prompt_template, output_files, etc.',
	)
	browser_profile: dict[str, Any] | None = Field(
		default=None,
		description='Default BrowserProfile settings as a dict (headless, user_data_dir, etc.). '
		'Supports {{ variable }} substitution in string values.',
	)
	llm: LLMProviderConfig | None = Field(
		default=None,
		description='Default LLM configuration for running this template',
	)
	default_tools: list[str] | None = Field(
		default=None,
		description='List of tool names the agent is allowed to use. '
		'None means all default tools are available. '
		'Use exclude_tools to remove specific tools instead.',
	)
	exclude_tools: list[str] = Field(
		default_factory=list,
		description='List of tool names to exclude from the default tool set.',
	)
	output_files: list[OutputFileRule] = Field(
		default_factory=list,
		description='Rules for capturing output files produced during execution.',
	)
	max_steps: int = Field(
		default=100,
		description='Maximum number of agent steps for this template',
		ge=1,
	)
	use_vision: bool | str = Field(
		default='auto',
		description='Whether to use vision (True/False/"auto")',
	)
	tags: list[str] = Field(
		default_factory=list,
		description='Arbitrary tags for categorizing and filtering templates',
	)
	version: str = Field(
		default='1.0.0',
		description='Semantic version for the template',
	)
	author: str = Field(
		default='',
		description='Optional author name or contact info',
	)

	def get_required_variables(self) -> list[str]:
		"""Return names of variables marked as required."""
		return [name for name, var in self.variables.items() if var.required]


class TaskTemplateExecutionResult(BaseModel):
	"""Structured result of executing a task template.

	Unlike raw agent logs, this provides a machine-readable summary of
	what happened: overall status, extracted content, output files, and
	failure information if something went wrong.
	"""

	model_config = ConfigDict(extra='allow', populate_by_name=True)

	template_name: str = Field(
		description='Name of the template that was executed',
	)
	status: TaskTemplateStatus = Field(
		description='Overall execution status',
	)
	num_steps: int = Field(
		default=0,
		description='Number of agent steps executed',
		ge=0,
	)
	duration_seconds: float = Field(
		default=0.0,
		description='Total wall-clock execution time in seconds',
		ge=0.0,
	)
	variables_used: dict[str, Any] = Field(
		default_factory=dict,
		description='Actual variable values used for this execution (with defaults applied)',
	)
	extracted_content: str | None = Field(
		default=None,
		description='Final extracted content from the agent, if any',
	)
	structured_output: Any | None = Field(
		default=None,
		description='Parsed structured output if the template uses a Pydantic output schema',
	)
	output_files: list[str] = Field(
		default_factory=list,
		description='Absolute paths to all output files captured according to output_files rules',
	)
	missing_output_files: list[str] = Field(
		default_factory=list,
		description='Patterns from output_files rules that were marked as required but not found',
	)
	urls_visited: list[str] = Field(
		default_factory=list,
		description='All unique URLs visited during execution',
	)
	error: str | None = Field(
		default=None,
		description='Error message if status is FAILED',
	)
	error_type: str | None = Field(
		default=None,
		description='Exception class name if status is FAILED due to an exception',
	)
	success: bool = Field(
		default=False,
		description='Convenience boolean: True if status == SUCCESS',
	)

	@classmethod
	def from_agent_history(
		cls,
		template_name: str,
		history: Any,
		duration_seconds: float,
		variables_used: dict[str, Any],
		output_files: list[str],
		missing_output_files: list[str],
	) -> 'TaskTemplateExecutionResult':
		"""Build a result object from an AgentHistoryList instance."""
		is_successful = bool(history.is_successful())
		final_result = history.final_result()

		urls = []
		try:
			raw_urls = history.urls()
			seen = set()
			for u in raw_urls:
				if u and str(u) not in seen:
					seen.add(str(u))
					urls.append(str(u))
		except Exception:
			pass

		structured = None
		try:
			structured = getattr(history, 'structured_output', None)
		except Exception:
			pass

		return cls(
			template_name=template_name,
			status=TaskTemplateStatus.SUCCESS if is_successful else TaskTemplateStatus.FAILED,
			num_steps=len(history.history) if hasattr(history, 'history') else 0,
			duration_seconds=duration_seconds,
			variables_used=variables_used,
			extracted_content=final_result if final_result else None,
			structured_output=structured,
			output_files=output_files,
			missing_output_files=missing_output_files,
			urls_visited=urls,
			error=None if is_successful else 'Agent did not complete successfully',
			success=is_successful,
		)
