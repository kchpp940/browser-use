"""Tests for task templates module - views models and service helpers.

These tests do NOT require a browser or LLM — they validate the data model,
variable substitution, template storage, and other pure-logic components.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from browser_use.task_templates.service import (
	TemplateManager,
	_coerce_value,
	resolve_template_variables,
	substitute_variables,
)
from browser_use.task_templates.views import (
	LLMProviderConfig,
	OutputFileRule,
	TaskTemplate,
	TaskTemplateExecutionResult,
	TaskTemplateStatus,
	TemplateVariable,
	TemplateVariableType,
)


# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestTemplateVariable:
	def test_default_string_type(self):
		v = TemplateVariable()
		assert v.type == TemplateVariableType.STRING

	def test_coerce_string_type(self):
		v = TemplateVariable(type='string')
		assert v.type == TemplateVariableType.STRING

	def test_required_flag(self):
		v = TemplateVariable(required=True, description='must have')
		assert v.required is True
		assert v.description == 'must have'

	def test_choices_restriction(self):
		v = TemplateVariable(choices=['a', 'b'], default='a')
		assert v.choices == ['a', 'b']


class TestTaskTemplate:
	def test_minimal_template(self):
		tpl = TaskTemplate(name='simple', prompt_template='do something')
		assert tpl.name == 'simple'
		assert tpl.prompt_template == 'do something'
		assert tpl.variables == {}
		assert tpl.default_tools is None

	def test_full_template(self):
		tpl = TaskTemplate(
			name='hn-scraper',
			description='scrape hn',
			prompt_template='Get {{ n }} posts',
			variables={
				'n': TemplateVariable(type=TemplateVariableType.INTEGER, default=10, description='how many')
			},
			browser_profile={'headless': True},
			llm=LLMProviderConfig(provider='openai', model='gpt-4', temperature=0.0),
			default_tools=['navigate', 'extract'],
			exclude_tools=['write_file'],
			output_files=[OutputFileRule(pattern='./out.csv', required=True, description='results')],
			max_steps=25,
			use_vision=False,
			tags=['hn', 'scrape'],
			author='dev',
			version='0.2.0',
		)
		assert tpl.get_required_variables() == []
		assert 'n' in tpl.variables
		assert tpl.llm.model == 'gpt-4'
		assert tpl.tags == ['hn', 'scrape']

	def test_invalid_name_fails(self):
		with pytest.raises(Exception):
			TaskTemplate(name='bad name!', prompt_template='x')

	def test_get_required_variables(self):
		tpl = TaskTemplate(
			name='req',
			prompt_template='hello {{ user }}',
			variables={
				'user': TemplateVariable(required=True),
				'color': TemplateVariable(default='blue'),
			},
		)
		assert tpl.get_required_variables() == ['user']

	def test_template_model_json_roundtrip(self):
		tpl = TaskTemplate(
			name='roundtrip',
			description='test',
			prompt_template='Visit {{ url }}',
			variables={
				'url': TemplateVariable(
					type=TemplateVariableType.STRING,
					required=True,
					description='target URL',
				)
			},
			browser_profile={'headless': False},
			llm=LLMProviderConfig(provider='browser_use', temperature=0.5),
			output_files=[
				OutputFileRule(pattern='./result.txt', description='the output')
			],
			tags=['test'],
		)
		data = tpl.model_dump(mode='json')
		# serialize + deserialize via JSON to simulate on-disk roundtrip
		reloaded = TaskTemplate.model_validate_json(json.dumps(data))
		assert reloaded.name == tpl.name
		assert reloaded.prompt_template == tpl.prompt_template
		assert reloaded.variables['url'].required is True
		assert reloaded.browser_profile == {'headless': False}
		assert len(reloaded.output_files) == 1
		assert reloaded.output_files[0].pattern == './result.txt'


# ---------------------------------------------------------------------------
# Variable coercion / resolution tests
# ---------------------------------------------------------------------------


class TestCoerceValue:
	def test_string_passthrough(self):
		assert _coerce_value('hello', TemplateVariableType.STRING) == 'hello'

	def test_int_from_string(self):
		assert _coerce_value('42', TemplateVariableType.INTEGER) == 42

	def test_float_from_int(self):
		assert _coerce_value(3, TemplateVariableType.FLOAT) == 3.0

	def test_bool_truthy_strings(self):
		for s in ['true', '1', 'yes', 'y', 'on']:
			assert _coerce_value(s, TemplateVariableType.BOOLEAN) is True, s

	def test_bool_falsy_from_zero(self):
		assert _coerce_value(0, TemplateVariableType.BOOLEAN) is False

	def test_list_from_csv_string(self):
		assert _coerce_value('a,b,c', TemplateVariableType.LIST) == ['a', 'b', 'c']

	def test_list_from_json_string(self):
		assert _coerce_value('[1,2,3]', TemplateVariableType.LIST) == [1, 2, 3]

	def test_dict_from_json_string(self):
		assert _coerce_value('{"k": 1}', TemplateVariableType.DICT) == {'k': 1}

	def test_none_returns_none(self):
		assert _coerce_value(None, TemplateVariableType.STRING) is None


class TestResolveTemplateVariables:
	def test_applies_defaults(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='x={{ x }}',
			variables={'x': TemplateVariable(type=TemplateVariableType.INTEGER, default=5)},
		)
		out = resolve_template_variables(tpl, {})
		assert out == {'x': 5}

	def test_overrides_beat_defaults(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='x={{ x }}',
			variables={'x': TemplateVariable(type=TemplateVariableType.INTEGER, default=5)},
		)
		out = resolve_template_variables(tpl, {'x': '99'})
		assert out['x'] == 99

	def test_missing_required_raises(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='need {{ thing }}',
			variables={'thing': TemplateVariable(required=True)},
		)
		with pytest.raises(ValueError, match='thing'):
			resolve_template_variables(tpl, {})

	def test_invalid_choice_raises(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='color={{ color }}',
			variables={'color': TemplateVariable(choices=['red', 'blue'])},
		)
		with pytest.raises(ValueError, match='green'):
			resolve_template_variables(tpl, {'color': 'green'})

	def test_unknown_variable_warns_but_does_not_fail(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='nothing',
			variables={},
		)
		out = resolve_template_variables(tpl, {'extra': 1})
		assert out == {}


class TestSubstituteVariables:
	def test_simple_substitution(self):
		result = substitute_variables('Hello {{ name }}!', {'name': 'World'})
		assert result == 'Hello World!'

	def test_multiple_substitutions(self):
		result = substitute_variables(
			'{{ a }} + {{ b }} = {{ c }}',
			{'a': 1, 'b': 2, 'c': 'three'},
		)
		assert result == '1 + 2 = three'

	def test_missing_variable_raises_keyerror(self):
		with pytest.raises(KeyError):
			substitute_variables('Hi {{ missing }}', {})

	def test_whitespace_in_tags(self):
		result = substitute_variables('x={{  val  }}', {'val': 42})
		assert result == 'x=42'

	def test_substitution_in_paths(self):
		vars = {'user': 'alice', 'date': '2025-01-01'}
		result = substitute_variables('/tmp/{{ user }}/report-{{ date }}.csv', vars)
		assert result == '/tmp/alice/report-2025-01-01.csv'


# ---------------------------------------------------------------------------
# Template storage / manager tests
# ---------------------------------------------------------------------------


class TestTemplateManagerStorage:
	def test_save_and_load_roundtrip(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		tpl = TaskTemplate(
			name='my-scraper',
			description='scrape a page',
			prompt_template='open {{ url }} and scrape',
			variables={'url': TemplateVariable(required=True)},
			tags=['scrape'],
		)
		path = mgr.save(tpl)
		assert path.exists()
		assert path.name == 'my-scraper.json'

		loaded = mgr.load('my-scraper')
		assert loaded.name == 'my-scraper'
		assert loaded.tags == ['scrape']
		assert 'url' in loaded.variables

	def test_list_names(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		for n in ['zebra', 'alpha', 'middle']:
			mgr.save(TaskTemplate(name=n, prompt_template='x'))
		assert mgr.list_names() == ['alpha', 'middle', 'zebra']

	def test_exists_and_delete(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		assert mgr.exists('foo') is False
		mgr.save(TaskTemplate(name='foo', prompt_template='x'))
		assert mgr.exists('foo') is True
		assert mgr.delete('foo') is True
		assert mgr.exists('foo') is False
		assert mgr.delete('foo') is False

	def test_create_from_dict(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		raw = {
			'name': 'from-dict',
			'prompt_template': 'search for {{ query }}',
			'variables': {
				'query': {'type': 'string', 'description': 'search terms', 'required': True}
			},
			'max_steps': 50,
		}
		tpl = mgr.create_from_dict(raw)
		assert tpl.name == 'from-dict'
		assert tpl.max_steps == 50
		assert tpl.variables['query'].required is True

	def test_overwrite_flag(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		tpl1 = TaskTemplate(name='same', prompt_template='v1')
		tpl2 = TaskTemplate(name='same', prompt_template='v2')
		mgr.save(tpl1)
		with pytest.raises(FileExistsError):
			mgr.create_from_dict(tpl2.model_dump(mode='json'), overwrite=False)
		mgr.save(tpl2, overwrite=True)
		assert mgr.load('same').prompt_template == 'v2'

	def test_load_missing_raises(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		with pytest.raises(FileNotFoundError):
			mgr.load('does-not-exist')

	def test_load_corrupt_json_raises(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		bad = tmp_path / 'broken.json'
		bad.write_text('{not valid json')
		with pytest.raises(ValueError):
			mgr.load('broken')

	def test_list_templates_skips_corrupt(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		mgr.save(TaskTemplate(name='good', prompt_template='ok'))
		(tmp_path / 'bad.json').write_text('nope')
		templates = mgr.list_templates()
		assert [t.name for t in templates] == ['good']

	def test_default_dir_from_env(self, monkeypatch, tmp_path):
		monkeypatch.setenv('BROWSER_USE_TEMPLATES_DIR', str(tmp_path))
		mgr = TemplateManager()  # no explicit dir
		assert mgr.templates_dir == tmp_path


def _agent_tool_names_from_template(tpl: TaskTemplate) -> set[str]:
	"""Replicate the _build_agent tool filtering logic and return resulting tools."""
	from browser_use.tools.service import Tools

	exclude = list(tpl.exclude_tools)
	include = tpl.default_tools
	if include is not None:
		all_default = TemplateManager._get_default_tool_names()
		effective_include = set(include) | {'done'}
		exclude.extend([n for n in all_default if n not in effective_include])
	if 'done' in exclude:
		exclude = [x for x in exclude if x != 'done']
	t = Tools(exclude_actions=exclude) if exclude else Tools()
	return set(t.registry.registry.actions.keys())


class TestAgentToolFiltering:
	def test_whitelist_always_keeps_done(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			default_tools=['navigate', 'extract'],
		)
		tools = _agent_tool_names_from_template(tpl)
		assert tools == {'navigate', 'extract', 'done'}

	def test_whitelist_empty_still_keeps_done(self):
		tpl = TaskTemplate(name='t', prompt_template='...', default_tools=[])
		tools = _agent_tool_names_from_template(tpl)
		assert tools == {'done'}

	def test_blacklist_removes_only_named(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			exclude_tools=['write_file', 'read_file', 'replace_file'],
		)
		tools = _agent_tool_names_from_template(tpl)
		assert 'write_file' not in tools
		assert 'read_file' not in tools
		assert 'replace_file' not in tools
		assert 'done' in tools
		assert 'navigate' in tools
		# Default list has 24, minus 3 → 21 (including done)
		assert len(tools) == 21

	def test_explicit_done_in_exclude_is_ignored(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			exclude_tools=['done', 'search'],
		)
		tools = _agent_tool_names_from_template(tpl)
		assert 'done' in tools
		assert 'search' not in tools

	def test_whitelist_and_blacklist_combined(self):
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			default_tools=['search', 'navigate', 'click', 'input', 'extract', 'write_file'],
			exclude_tools=['write_file'],
		)
		tools = _agent_tool_names_from_template(tpl)
		assert tools == {'search', 'navigate', 'click', 'input', 'extract', 'done'}

	def test_no_filter_returns_all_tools(self):
		tpl = TaskTemplate(name='t', prompt_template='...')
		# When neither default_tools nor exclude_tools is set, _build_agent
		# passes no agent_tools to Agent (i.e. default Tools()) - simulate.
		from browser_use.tools.service import Tools

		default = set(Tools().registry.registry.actions.keys())
		assert len(default) == 24
		assert 'done' in default

	def test_get_default_tool_names_has_24_tools(self):
		names = TemplateManager._get_default_tool_names()
		assert len(names) == 24
		assert 'done' in names
		assert 'navigate' in names
		assert 'write_file' in names


class TestRenderPrompt:
	def test_render_applies_substitution(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		tpl = TaskTemplate(
			name='t',
			prompt_template='Search for {{ term }} on {{ site }}',
			variables={
				'term': TemplateVariable(default='weather'),
				'site': TemplateVariable(default='google.com'),
			},
		)
		# No overrides → defaults
		assert mgr.render_prompt(tpl, {}) == 'Search for weather on google.com'
		# Partial override
		assert mgr.render_prompt(tpl, {'term': 'cats'}) == 'Search for cats on google.com'
		# Full override
		assert (
			mgr.render_prompt(tpl, {'term': 'jobs', 'site': 'linkedin.com'})
			== 'Search for jobs on linkedin.com'
		)


class TestCollectOutputFiles:
	def test_finds_files_by_pattern(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		# create some fake outputs
		(tmp_path / 'report.csv').write_text('a,b')
		(tmp_path / 'other.txt').write_text('txt')
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			output_files=[OutputFileRule(pattern='*.csv', description='csv')],
		)
		found, missing = mgr.collect_output_files(tpl, {}, base_dir=tmp_path)
		assert found == [str((tmp_path / 'report.csv').resolve())]
		assert missing == []

	def test_required_missing_reported(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			output_files=[
				OutputFileRule(pattern='nope.csv', required=True, description='required')
			],
		)
		found, missing = mgr.collect_output_files(tpl, {}, base_dir=tmp_path)
		assert found == []
		assert missing == ['nope.csv']

	def test_substitutes_vars_in_pattern(self, tmp_path):
		mgr = TemplateManager(templates_dir=tmp_path)
		subdir = tmp_path / 'alice'
		subdir.mkdir()
		(subdir / 'result.json').write_text('{}')
		tpl = TaskTemplate(
			name='t',
			prompt_template='...',
			variables={'user': TemplateVariable(default='alice')},
			output_files=[OutputFileRule(pattern=str(tmp_path / '{{ user }}' / '*.json'))],
		)
		found, _ = mgr.collect_output_files(tpl, {})
		assert len(found) == 1
		assert found[0].endswith('result.json')


# ---------------------------------------------------------------------------
# Execution result tests
# ---------------------------------------------------------------------------


class FakeAgentHistory:
	"""Minimal stand-in for AgentHistoryList to test from_agent_history()."""

	def __init__(self, *, success: bool, steps: int, final: str | None, urls: list):
		self._success = success
		self._steps = steps
		self._final = final
		self._urls = urls
		self.history = [object()] * steps
		self.structured_output = {'parsed': True} if success else None

	def is_successful(self):
		return self._success

	def final_result(self):
		return self._final

	def urls(self):
		return self._urls


class TestExecutionResult:
	def test_from_successful_history(self):
		h = FakeAgentHistory(
			success=True, steps=12, final='Got: 42', urls=['https://a.com', 'https://b.com']
		)
		res = TaskTemplateExecutionResult.from_agent_history(
			template_name='tpl',
			history=h,
			duration_seconds=4.2,
			variables_used={'q': 'test'},
			output_files=['/tmp/a.csv'],
			missing_output_files=[],
		)
		assert res.status == TaskTemplateStatus.SUCCESS
		assert res.success is True
		assert res.num_steps == 12
		assert res.extracted_content == 'Got: 42'
		assert res.urls_visited == ['https://a.com', 'https://b.com']
		assert res.structured_output == {'parsed': True}
		assert res.duration_seconds == pytest.approx(4.2, 0.01)

	def test_from_failed_history(self):
		h = FakeAgentHistory(success=False, steps=3, final=None, urls=[])
		res = TaskTemplateExecutionResult.from_agent_history(
			template_name='tpl',
			history=h,
			duration_seconds=0.5,
			variables_used={},
			output_files=[],
			missing_output_files=['must-have.csv'],
		)
		assert res.status == TaskTemplateStatus.FAILED
		assert res.success is False
		assert 'Agent did not complete' in (res.error or '')

	def test_success_demoted_to_partial_when_missing_required_outputs(self):
		h = FakeAgentHistory(success=True, steps=5, final='ok', urls=[])
		res = TaskTemplateExecutionResult.from_agent_history(
			template_name='tpl',
			history=h,
			duration_seconds=1.0,
			variables_used={},
			output_files=[],
			missing_output_files=['required.csv'],
		)
		# Simulate the manager's post-processing step
		if res.status == TaskTemplateStatus.SUCCESS and res.missing_output_files:
			res.status = TaskTemplateStatus.PARTIAL
			res.error = f'Missing required output files: {res.missing_output_files}'
			res.success = False
		assert res.status == TaskTemplateStatus.PARTIAL
		assert res.success is False
		assert 'required.csv' in (res.error or '')

	def test_model_json_serialization(self):
		res = TaskTemplateExecutionResult(
			template_name='t',
			status=TaskTemplateStatus.FAILED,
			error_type='RuntimeError',
			error='boom',
			num_steps=1,
			duration_seconds=0.1,
			success=False,
		)
		payload = json.loads(res.model_dump_json())
		assert payload['template_name'] == 't'
		assert payload['status'] == 'failed'
		assert payload['error_type'] == 'RuntimeError'
		assert payload['success'] is False
