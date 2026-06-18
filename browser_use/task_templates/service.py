"""Template storage, management, and execution service.

Provides TemplateManager for saving/loading/listing templates from disk,
plus the run-time engine that injects variables, creates agents, executes
templates, and produces structured TaskTemplateExecutionResult objects.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from browser_use.task_templates.views import (
	LLMProviderConfig,
	OutputFileRule,
	TaskTemplate,
	TaskTemplateExecutionResult,
	TaskTemplateStatus,
	TemplateVariable,
	TemplateVariableType,
)

logger = logging.getLogger(__name__)

_VAR_PATTERN = re.compile(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}')

DEFAULT_TEMPLATES_DIRNAME = 'templates'


def _get_default_templates_dir() -> Path:
	"""Return the default directory for storing user templates.

	Resolution order:
	1. $BROWSER_USE_TEMPLATES_DIR (environment variable)
	2. $BROWSER_USE_HOME/templates  (via env or ~/.browser-use)
	3. ~/.config/browseruse/templates
	"""
	env_dir = os.environ.get('BROWSER_USE_TEMPLATES_DIR')
	if env_dir:
		p = Path(env_dir).expanduser()
		p.mkdir(parents=True, exist_ok=True)
		return p

	home = os.environ.get('BROWSER_USE_HOME')
	if home:
		p = Path(home).expanduser() / DEFAULT_TEMPLATES_DIRNAME
		p.mkdir(parents=True, exist_ok=True)
		return p

	xdg = os.environ.get('XDG_CONFIG_HOME', '~/.config')
	p = Path(xdg).expanduser() / 'browseruse' / DEFAULT_TEMPLATES_DIRNAME
	p.mkdir(parents=True, exist_ok=True)
	return p


def substitute_variables(template_str: str, variables: dict[str, Any]) -> str:
	"""Replace {{ var_name }} placeholders with values from variables.

	Values are converted to strings. Missing variables raise KeyError.
	"""

	def _replace(match: re.Match) -> str:
		var_name = match.group(1)
		if var_name not in variables:
			raise KeyError(f'Variable {{{{ {var_name} }}}} is not defined')
		return str(variables[var_name])

	return _VAR_PATTERN.sub(_replace, template_str)


def _coerce_value(raw: Any, var_type: TemplateVariableType) -> Any:
	"""Attempt to coerce a raw value to the declared variable type."""
	if raw is None:
		return None

	if var_type == TemplateVariableType.STRING:
		return str(raw)
	if var_type == TemplateVariableType.INTEGER:
		return int(raw)
	if var_type == TemplateVariableType.FLOAT:
		return float(raw)
	if var_type == TemplateVariableType.BOOLEAN:
		if isinstance(raw, bool):
			return raw
		if isinstance(raw, (int, float)):
			return bool(raw)
		if isinstance(raw, str):
			return raw.lower() in ('true', '1', 'yes', 'y', 'on')
		return bool(raw)
	if var_type == TemplateVariableType.LIST:
		if isinstance(raw, list):
			return raw
		if isinstance(raw, str):
			try:
				parsed = json.loads(raw)
				if isinstance(parsed, list):
					return parsed
			except (json.JSONDecodeError, TypeError):
				pass
			return [s.strip() for s in raw.split(',') if s.strip()]
		return list(raw)
	if var_type == TemplateVariableType.DICT:
		if isinstance(raw, dict):
			return raw
		if isinstance(raw, str):
			parsed = json.loads(raw)
			if isinstance(parsed, dict):
				return parsed
			raise ValueError(f'Variable string is not a JSON object: {raw!r}')
		return dict(raw)
	return raw


def resolve_template_variables(
	template: TaskTemplate, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
	"""Resolve final variable values from template defaults + user overrides.

	Applies defaults, type coercion, choice validation, and required-checks.
	Raises ValueError for missing required variables or invalid choices.
	"""
	result: dict[str, Any] = {}
	overrides = overrides or {}

	for name, var_def in template.variables.items():
		raw = overrides[name] if name in overrides else var_def.default

		if raw is None and var_def.required:
			raise ValueError(f'Required variable {name!r} is not provided and has no default')

		if raw is not None:
			coerced = _coerce_value(raw, var_def.type)
			if var_def.choices and coerced not in var_def.choices:
				raise ValueError(
					f'Variable {name!r} value {coerced!r} is not one of the allowed choices: {var_def.choices}'
				)
			result[name] = coerced
		else:
			result[name] = None

	unknown = set(overrides.keys()) - set(template.variables.keys())
	if unknown:
		logger.warning(f'Ignoring unknown variables for template {template.name!r}: {sorted(unknown)}')

	return result


class TemplateManager:
	"""Manages template storage and execution.

	Templates are stored as JSON files in a templates directory. Each template
	is one file named <template-name>.json.
	"""

	def __init__(self, templates_dir: Path | str | None = None):
		self._dir = Path(templates_dir) if templates_dir else _get_default_templates_dir()
		self._dir.mkdir(parents=True, exist_ok=True)
		self._cache: dict[str, TaskTemplate] = {}

	@property
	def templates_dir(self) -> Path:
		return self._dir

	# ------------------------------------------------------------------
	# Storage operations
	# ------------------------------------------------------------------

	def _path_for(self, name: str) -> Path:
		return self._dir / f'{name}.json'

	def list_names(self) -> list[str]:
		"""Return sorted list of template names available in storage."""
		names = []
		for f in self._dir.glob('*.json'):
			names.append(f.stem)
		return sorted(names)

	def list_templates(self) -> list[TaskTemplate]:
		"""Load and return all templates from storage."""
		result = []
		for name in self.list_names():
			try:
				tpl = self.load(name)
				result.append(tpl)
			except Exception as e:
				logger.warning(f'Skipping corrupt template {name!r}: {e}')
		return result

	def exists(self, name: str) -> bool:
		return self._path_for(name).exists()

	def load(self, name: str) -> TaskTemplate:
		"""Load a template by name. Raises FileNotFoundError if missing."""
		if name in self._cache:
			return self._cache[name]

		path = self._path_for(name)
		if not path.exists():
			raise FileNotFoundError(f'Template {name!r} not found in {self._dir}')

		try:
			data = json.loads(path.read_text(encoding='utf-8'))
			tpl = TaskTemplate.model_validate(data)
		except (json.JSONDecodeError, ValidationError) as e:
			raise ValueError(f'Failed to parse template {name!r}: {e}') from e

		self._cache[name] = tpl
		return tpl

	def save(self, template: TaskTemplate, overwrite: bool = True) -> Path:
		"""Persist a template to storage. Returns the written file path."""
		path = self._path_for(template.name)
		if path.exists() and not overwrite:
			raise FileExistsError(f'Template {template.name!r} already exists (pass overwrite=True to replace)')

		data = template.model_dump(mode='json', indent=2) if False else json.dumps(
			template.model_dump(mode='json'), indent=2, ensure_ascii=False
		)
		path.write_text(data + '\n', encoding='utf-8')
		self._cache[template.name] = template
		return path

	def delete(self, name: str) -> bool:
		"""Delete a template by name. Returns True if something was deleted."""
		path = self._path_for(name)
		if not path.exists():
			return False
		path.unlink()
		self._cache.pop(name, None)
		return True

	def create_from_dict(self, data: dict[str, Any], overwrite: bool = False) -> TaskTemplate:
		"""Validate raw dict as a TaskTemplate and save it."""
		tpl = TaskTemplate.model_validate(data)
		self.save(tpl, overwrite=overwrite)
		return tpl

	# ------------------------------------------------------------------
	# Variable substitution helpers
	# ------------------------------------------------------------------

	def render_prompt(self, template: TaskTemplate, variables: dict[str, Any]) -> str:
		"""Fully render the prompt_template with resolved variables."""
		resolved = resolve_template_variables(template, variables)
		return substitute_variables(template.prompt_template, resolved)

	# ------------------------------------------------------------------
	# Output file collection
	# ------------------------------------------------------------------

	def collect_output_files(
		self, template: TaskTemplate, variables: dict[str, Any], base_dir: Path | str | None = None
	) -> tuple[list[str], list[str]]:
		"""Collect actual files matching the template's output_files rules.

		Returns (found_files, missing_required_patterns).
		"""
		found: list[str] = []
		missing: list[str] = []
		resolved = resolve_template_variables(template, variables)
		base = Path(base_dir) if base_dir else Path.cwd()

		for rule in template.output_files:
			pattern = substitute_variables(rule.pattern, resolved)
			# Support both absolute and relative patterns
			if os.path.isabs(pattern):
				matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
			else:
				matches = [p for p in base.glob(pattern) if p.is_file()]
				# Also try recursive glob for patterns like **/*.csv
				if not matches and ('**' in pattern or '*' in pattern):
					matches = [p for p in base.rglob(pattern.lstrip('./')) if p.is_file()]

			if matches:
				found.extend(str(p.resolve()) for p in matches)
			elif rule.required:
				missing.append(rule.pattern)

		return found, missing

	# ------------------------------------------------------------------
	# Execution (runtime engine)
	# ------------------------------------------------------------------

	async def run(
		self,
		template_name: str,
		variables: dict[str, Any] | None = None,
		*,
		overrides: dict[str, Any] | None = None,
		browser_session: Any | None = None,
		llm: Any | None = None,
		tools: Any | None = None,
		max_steps: int | None = None,
	) -> TaskTemplateExecutionResult:
		"""Load and execute a template, returning a structured result.

		Parameters
		----------
		template_name:
			Name of the saved template to execute.
		variables:
			User-provided variable values (dict mapping name -> value).
		overrides:
			Optional additional template-level overrides (browser_profile, etc.).
		browser_session:
			Optional pre-created BrowserSession. If not provided, one is
			created from the template's browser_profile settings.
		llm:
			Optional pre-created LLM instance. If not provided, one is
			created from the template's llm config.
		tools:
			Optional Tools instance. If not provided, one is created using
			the template's default_tools / exclude_tools settings.
		max_steps:
			Override the template's default max_steps.
		"""
		template = self.load(template_name)
		return await self.run_template(
			template,
			variables=variables,
			overrides=overrides,
			browser_session=browser_session,
			llm=llm,
			tools=tools,
			max_steps=max_steps,
		)

	async def run_template(
		self,
		template: TaskTemplate,
		variables: dict[str, Any] | None = None,
		*,
		overrides: dict[str, Any] | None = None,
		browser_session: Any | None = None,
		llm: Any | None = None,
		tools: Any | None = None,
		max_steps: int | None = None,
	) -> TaskTemplateExecutionResult:
		"""Execute a template object directly."""
		start = time.time()

		try:
			resolved_vars = resolve_template_variables(template, variables)
		except (ValueError, KeyError) as e:
			return TaskTemplateExecutionResult(
				template_name=template.name,
				status=TaskTemplateStatus.FAILED,
				error=f'Variable resolution failed: {e}',
				error_type=type(e).__name__,
				variables_used=variables or {},
			)

		try:
			rendered_prompt = substitute_variables(template.prompt_template, resolved_vars)
		except KeyError as e:
			return TaskTemplateExecutionResult(
				template_name=template.name,
				status=TaskTemplateStatus.FAILED,
				error=f'Prompt rendering failed: {e}',
				error_type=type(e).__name__,
				variables_used=resolved_vars,
			)

		effective_max_steps = max_steps if max_steps is not None else template.max_steps

		try:
			agent = await self._build_agent(
				template=template,
				rendered_prompt=rendered_prompt,
				resolved_vars=resolved_vars,
				overrides=overrides,
				browser_session=browser_session,
				llm=llm,
				tools=tools,
			)
		except Exception as e:
			logger.exception('Failed to build agent for template execution')
			return TaskTemplateExecutionResult(
				template_name=template.name,
				status=TaskTemplateStatus.FAILED,
				error=f'Failed to build agent: {e}',
				error_type=type(e).__name__,
				variables_used=resolved_vars,
			)

		history = None
		try:
			history = await agent.run(max_steps=effective_max_steps)
		except Exception as e:
			logger.exception(f'Template {template.name!r} execution raised exception')
			duration = time.time() - start
			found, missing = self.collect_output_files(template, resolved_vars)
			return TaskTemplateExecutionResult(
				template_name=template.name,
				status=TaskTemplateStatus.FAILED,
				error=f'Agent execution error: {e}',
				error_type=type(e).__name__,
				duration_seconds=duration,
				variables_used=resolved_vars,
				output_files=found,
				missing_output_files=missing,
			)
		finally:
			try:
				await agent.close()
			except Exception:
				pass

		duration = time.time() - start
		found, missing = self.collect_output_files(template, resolved_vars)

		result = TaskTemplateExecutionResult.from_agent_history(
			template_name=template.name,
			history=history,
			duration_seconds=duration,
			variables_used=resolved_vars,
			output_files=found,
			missing_output_files=missing,
		)

		# If required output files are missing, demote SUCCESS -> PARTIAL
		if result.status == TaskTemplateStatus.SUCCESS and missing:
			result.status = TaskTemplateStatus.PARTIAL
			result.error = f'Missing required output files: {missing}'
			result.success = False

		return result

	# ------------------------------------------------------------------
	# Internal: agent construction
	# ------------------------------------------------------------------

	async def _build_agent(
		self,
		*,
		template: TaskTemplate,
		rendered_prompt: str,
		resolved_vars: dict[str, Any],
		overrides: dict[str, Any] | None,
		browser_session: Any | None,
		llm: Any | None,
		tools: Any | None,
	):
		"""Build an Agent instance from the template + resolved state."""
		# Lazy imports - avoid heavy import cost when just listing templates
		from browser_use.agent.service import Agent
		from browser_use.browser.profile import BrowserProfile
		from browser_use.tools.service import Tools

		# ---- LLM instance ------------------------------------------------
		agent_llm = llm
		if agent_llm is None and template.llm is not None:
			agent_llm = self._instantiate_llm(template.llm)

		# ---- Tools / Tool registry --------------------------------------
		agent_tools = tools
		if agent_tools is None and (template.default_tools is not None or template.exclude_tools):
			exclude = list(template.exclude_tools)
			include = template.default_tools
			if include is not None:
				all_default = self._get_default_tool_names()
				# Compute the complement: keep only include + always preserve 'done'
				effective_include = set(include) | {'done'}
				exclude.extend([n for n in all_default if n not in effective_include])
			# Safety: never allow 'done' to be excluded, as agent cannot terminate without it
			if 'done' in exclude:
				logger.warning("Template requested to exclude tool 'done' — ignoring (agent requires 'done' to complete)")
				exclude = [x for x in exclude if x != 'done']
			agent_tools = Tools(exclude_actions=exclude) if exclude else Tools()

			# Debug: report which tools ended up available
			try:
				final_names = sorted(agent_tools.registry.registry.actions.keys())
				logger.debug(
					f'Template "{template.name}" tool filter: '
					f'whitelist={template.default_tools!r}  blacklist={template.exclude_tools!r}  '
					f'final_available({len(final_names)}): {final_names}'
				)
			except Exception:
				pass

		# ---- Browser profile --------------------------------------------
		agent_browser = browser_session
		if agent_browser is None and template.browser_profile is not None:
			profile_data = dict(template.browser_profile)
			# Substitute variables in string-valued browser profile entries
			for key, value in list(profile_data.items()):
				if isinstance(value, str):
					try:
						profile_data[key] = substitute_variables(value, resolved_vars)
					except KeyError:
						pass
			if overrides:
				profile_data.update(overrides.get('browser_profile', {}))
			agent_browser = BrowserProfile(**profile_data)

		# Apply top-level overrides last
		agent_kwargs: dict[str, Any] = {
			'task': rendered_prompt,
			'max_actions_per_step': overrides.get('max_actions_per_step', 3) if overrides else 3,
			'use_vision': template.use_vision,
		}
		if agent_llm is not None:
			agent_kwargs['llm'] = agent_llm
		if agent_tools is not None:
			agent_kwargs['tools'] = agent_tools
		if agent_browser is not None:
			if isinstance(agent_browser, BrowserProfile):
				agent_kwargs['browser_profile'] = agent_browser
			else:
				agent_kwargs['browser_session'] = agent_browser

		if overrides:
			for key in ('output_model_schema', 'save_conversation_path', 'calculate_cost'):
				if key in overrides:
					agent_kwargs[key] = overrides[key]

		return Agent(**agent_kwargs)

	@staticmethod
	def _get_default_tool_names() -> list[str]:
		"""Return the canonical list of default tool action names.

		This list is maintained alongside Tools._register_default_actions.
		For safety, we dynamically inspect a fresh Tools() instance when possible.
		"""
		try:
			from browser_use.tools.service import Tools

			t = Tools()
			return list(t.registry.registry.actions.keys())
		except Exception:
			return [
				'search',
				'navigate',
				'go_back',
				'wait',
				'click',
				'input',
				'upload_file',
				'scroll',
				'find_text',
				'send_keys',
				'evaluate',
				'switch',
				'close',
				'extract',
				'screenshot',
				'dropdown_options',
				'select_dropdown',
				'write_file',
				'read_file',
				'replace_file',
				'done',
			]

	@staticmethod
	def _instantiate_llm(cfg: LLMProviderConfig) -> Any:
		"""Create an LLM chat instance from a provider config.

		Uses lazy imports and falls back gracefully if a provider is unavailable.
		"""
		api_key = None
		if cfg.api_key_env:
			api_key = os.environ.get(cfg.api_key_env)

		provider = (cfg.provider or '').lower()

		# Provider detection by model name as a fallback
		model = cfg.model or ''
		model_lower = model.lower()
		if not provider:
			if model_lower.startswith('gpt') or model_lower.startswith('o'):
				provider = 'openai'
			elif 'claude' in model_lower:
				provider = 'anthropic'
			elif 'gemini' in model_lower:
				provider = 'google'
			elif 'groq' in model_lower:
				provider = 'groq'
			else:
				provider = 'browser_use'

		common_kwargs: dict[str, Any] = {}
		if model:
			common_kwargs['model'] = model
		if cfg.temperature is not None:
			common_kwargs['temperature'] = cfg.temperature
		if cfg.base_url:
			common_kwargs['base_url'] = cfg.base_url
		if api_key:
			common_kwargs['api_key'] = api_key

		try:
			if provider in ('browser_use', 'browseruse', 'bu'):
				from browser_use.llm.browser_use.chat import ChatBrowserUse

				return ChatBrowserUse(**common_kwargs)

			if provider == 'openai':
				from browser_use.llm.openai.chat import ChatOpenAI

				return ChatOpenAI(**common_kwargs)

			if provider == 'anthropic':
				from browser_use.llm.anthropic.chat import ChatAnthropic

				return ChatAnthropic(**common_kwargs)

			if provider == 'google':
				from browser_use.llm.google.chat import ChatGoogle

				return ChatGoogle(**common_kwargs)

			if provider == 'groq':
				from browser_use.llm.groq.chat import ChatGroq

				return ChatGroq(**common_kwargs)

			if provider == 'mistral':
				from browser_use.llm.mistral.chat import ChatMistral

				return ChatMistral(**common_kwargs)

			if provider in ('ollama', 'local'):
				from browser_use.llm.ollama.chat import ChatOllama

				return ChatOllama(**common_kwargs)

			if provider in ('azure', 'azure_openai'):
				from browser_use.llm.azure.chat import ChatAzureOpenAI

				return ChatAzureOpenAI(**common_kwargs)

			if provider in ('vercel', 'ai'):
				from browser_use.llm.vercel.chat import ChatVercel

				return ChatVercel(**common_kwargs)

			if provider in ('litellm', 'lite_llm'):
				from browser_use.llm.litellm.chat import ChatLiteLLM

				return ChatLiteLLM(**common_kwargs)

			# Final fallback: try ChatOpenAI with whatever config
			from browser_use.llm.openai.chat import ChatOpenAI

			return ChatOpenAI(**common_kwargs)
		except Exception as e:
			logger.warning(f'Failed to instantiate LLM provider={provider!r}: {e}. Agent will fall back to default.')
			return None


# ------------------------------------------------------------------
# Module-level convenience API
# ------------------------------------------------------------------

_default_manager: TemplateManager | None = None


def get_template_manager(templates_dir: Path | str | None = None) -> TemplateManager:
	"""Return the process-wide default TemplateManager, creating it on first use."""
	global _default_manager
	if _default_manager is None or templates_dir is not None:
		_default_manager = TemplateManager(templates_dir=templates_dir)
	return _default_manager


async def run_template(
	template_name: str,
	variables: dict[str, Any] | None = None,
	*,
	templates_dir: Path | str | None = None,
	**kwargs: Any,
) -> TaskTemplateExecutionResult:
	"""Convenience wrapper: load and run a template using the default manager."""
	mgr = get_template_manager(templates_dir=templates_dir)
	return await mgr.run(template_name, variables=variables, **kwargs)
