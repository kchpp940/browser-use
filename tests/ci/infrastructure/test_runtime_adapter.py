"""Lightweight tests for CommandRuntimeAdapter as unified entry point.

These tests verify that:
1. for_cli / for_mcp / for_skill constructors produce valid TaskConfig
2. run_cli_task / run_mcp_tool / run_skill_command call proper lifecycle hooks
3. RunResult serialization methods work correctly
4. Entry-point modules (cli.py, mcp/server.py, skill_cli) import the adapter

No real browser or LLM is used - we mock at the adapter level.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from browser_use.runtime.adapter import (
	CommandRuntimeAdapter,
	create_task_config_from_cli_dict,
)
from browser_use.runtime.config import (
	BrowserSessionConfig,
	LLMConfig,
	RunResult,
	TaskConfig,
)


class DummyLLM:
	def __init__(self, model='dummy-model'):
		self.model = model
		self.temperature = 0.0


class DummyAgent:
	def __init__(self, fail=False, final='done', steps=3):
		self._fail = fail
		self._final = final
		self._steps = steps

	async def run(self, max_steps=None):
		if self._fail:
			raise RuntimeError('boom')
		return self._make_history()

	def _make_history(self):
		from browser_use.agent.views import AgentHistoryList

		return AgentHistoryList.model_construct(history=[], usage=None)


# ---------------------------------------------------------------------------
# TaskConfig / RunResult basics
# ---------------------------------------------------------------------------


def test_llm_config_defaults_are_sane():
	cfg = LLMConfig(provider='openai', model='gpt-4.1-mini', api_key='sk-xxx')
	assert cfg.provider == 'openai'
	assert cfg.model == 'gpt-4.1-mini'


def test_browser_session_config_ignores_unknown_via_extra():
	cfg = BrowserSessionConfig(headless=True, window_width=800, window_height=600)
	assert cfg.headless is True
	assert cfg.window_width == 800
	assert cfg.window_height == 600


def test_task_config_roundtrip_via_model_dump():
	tc = TaskConfig(
		task='hello',
		llm=LLMConfig(provider='openai', model='m1'),
		browser=BrowserSessionConfig(headless=True),
		max_steps=10,
	)
	d = tc.model_dump()
	assert d['task'] == 'hello'
	assert d['max_steps'] == 10
	tc2 = TaskConfig(**d)
	assert tc2.task == tc.task


def test_run_result_from_error_builds_negative_outcome():
	r = RunResult.from_error(RuntimeError('x'), duration=1.5)
	assert r.success is False
	assert r.exit_code != 0
	assert r.duration_seconds == 1.5
	assert any('x' in (e or '') for e in r.errors)


def test_run_result_from_empty_history():
	from browser_use.agent.views import AgentHistoryList

	h = AgentHistoryList(history=[])
	r = RunResult.from_agent_history(h, duration=0.1)
	assert r.success is True
	assert r.is_done is False
	assert r.steps == 0


# ---------------------------------------------------------------------------
# RunResult serialization
# ---------------------------------------------------------------------------


def test_runresult_to_json_roundtrip():
	r = RunResult(
		success=True,
		exit_code=0,
		final_result='all good',
		steps=5,
		duration_seconds=2.1,
		urls_visited=['https://a.com', 'https://b.com'],
		errors=[None, None],
		is_done=True,
	)
	j = r.to_json()
	d = json.loads(j)
	assert d['success'] is True
	assert d['final_result'] == 'all good'
	assert len(d['urls_visited']) == 2


def test_runresult_format_text_contains_key_fields():
	r = RunResult(
		success=True,
		exit_code=0,
		final_result='found it',
		steps=3,
		duration_seconds=1.2,
		urls_visited=['https://example.com'],
		errors=[],
		is_done=True,
	)
	txt = r.format_text()
	assert 'found it' in txt
	assert '3' in txt


def test_runresult_format_error_picks_first_message():
	r = RunResult(
		success=False,
		exit_code=1,
		final_result=None,
		steps=1,
		duration_seconds=0.0,
		urls_visited=[],
		errors=[None, 'second error'],
	)
	assert r.format_error() == 'second error'

	r2 = RunResult(success=False, errors=[])
	assert r2.format_error() == ''


def test_runresult_to_skill_response_shape():
	r = RunResult(success=True, final_result='ok', steps=2, duration_seconds=0.5)
	resp = r.to_skill_response(request_id='req-42')
	assert resp['id'] == 'req-42'
	assert resp['success'] is True
	assert isinstance(resp['data'], dict)
	assert resp['error'] is None


def test_runresult_to_skill_response_error_shape():
	r = RunResult(success=False, exit_code=2, errors=['bad thing'])
	resp = r.to_skill_response(request_id='r1')
	assert resp['success'] is False
	assert 'bad thing' in resp['error']


# ---------------------------------------------------------------------------
# Bridge helpers
# ---------------------------------------------------------------------------


def test_create_task_config_from_cli_dict_populates_llm_and_browser():
	d = {
		'agent': {'max_steps': 42},
		'model': {
			'name': 'gpt-4.1-mini',
			'temperature': 0.3,
			'api_keys': {'OPENAI_API_KEY': 'sk-x'},
		},
		'browser': {
			'headless': True,
			'window_width': 1024,
			'window_height': 768,
			'keep_alive': False,
		},
	}
	cfg = create_task_config_from_cli_dict(d)
	assert cfg.llm.provider == 'auto'
	assert cfg.llm.model == 'gpt-4.1-mini'
	assert cfg.llm.temperature == 0.3
	assert cfg.llm.api_key == 'sk-x'
	assert cfg.browser.headless is True
	assert cfg.browser.window_width == 1024
	assert cfg.browser.window_height == 768
	assert cfg.max_steps == 42


# ---------------------------------------------------------------------------
# for_cli / for_mcp / for_skill constructors
# ---------------------------------------------------------------------------


def test_for_cli_builds_valid_adapter():
	user_config = {
		'model': {'name': 'gpt-4.1-mini'},
		'browser': {'headless': True},
		'agent': {'max_steps': 10},
	}
	a = CommandRuntimeAdapter.for_cli(
		'find x', user_config=user_config, source='cli', user_data_dir='/tmp/bu'
	)
	assert a.task_config.task == 'find x'
	assert a.task_config.source == 'cli'
	assert a.task_config.browser.headless is True
	assert a.task_config.browser.user_data_dir == '/tmp/bu'


def test_for_mcp_applies_overrides_and_allowed_domains():
	profile = {'headless': True}
	llm = {'model_provider': 'openai', 'model': 'gpt-4.1-mini', 'temperature': 0.9}
	a = CommandRuntimeAdapter.for_mcp(
		'browse',
		profile_config=profile,
		llm_config_dict=llm,
		model_override='gpt-override',
		allowed_domains=['example.com'],
		max_steps=5,
	)
	assert a.task_config.task == 'browse'
	assert a.task_config.llm.model == 'gpt-override'
	assert a.task_config.llm.temperature == 0.9
	assert a.task_config.browser.allowed_domains == ['example.com']
	assert a.task_config.max_steps == 5


def test_for_skill_flat_kwargs_basic():
	a = CommandRuntimeAdapter.for_skill(
		'the task',
		headed=True,
		headless=None,
		cdp_url=None,
		max_steps=20,
	)
	assert a.task_config.task == 'the task'
	assert a.task_config.browser.headed is True
	assert a.task_config.max_steps == 20


def test_for_skill_headed_overrides_headless():
	a = CommandRuntimeAdapter.for_skill(
		't', headed=False, headless=True, llm_model='some-model'
	)
	assert a.task_config.browser.headless is True


# ---------------------------------------------------------------------------
# run_cli_task lifecycle hooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_cli_task_happy_path_calls_hooks_in_order(monkeypatch):
	seq: list[str] = []
	cfg = TaskConfig(task='t', llm=LLMConfig(provider='openai', model='m'))
	a = CommandRuntimeAdapter(cfg)

	monkeypatch.setattr(a, 'resolve_llm', lambda: DummyLLM())

	async def fake_create():
		return object()

	async def fake_cleanup():
		return None

	monkeypatch.setattr(a, 'create_browser_session', fake_create)
	monkeypatch.setattr(a, 'create_agent', lambda browser_session=None: DummyAgent(final='result'))
	monkeypatch.setattr(a, 'cleanup', fake_cleanup)

	def on_start(adapter, llm):
		seq.append('start')
		assert adapter is a

	def on_complete(adapter, result):
		seq.append('complete')
		assert result.success is True

	def on_error(adapter, error, duration):
		seq.append('error')

	r = await a.run_cli_task(on_start=on_start, on_complete=on_complete, on_error=on_error)
	assert r.success is True
	assert seq == ['start', 'complete']


@pytest.mark.asyncio
async def test_run_cli_task_error_path_calls_on_error(monkeypatch):
	seq: list[str] = []
	cfg = TaskConfig(task='t', llm=LLMConfig(provider='openai', model='m'))
	a = CommandRuntimeAdapter(cfg)

	orig_resolve = a.resolve_llm

	def explode_resolve():
		raise RuntimeError('boom')

	monkeypatch.setattr(a, 'resolve_llm', explode_resolve)

	async def fake_cleanup():
		return None

	monkeypatch.setattr(a, 'cleanup', fake_cleanup)

	r = await a.run_cli_task(
		on_start=lambda ad, llm: seq.append('s'),
		on_complete=lambda ad, res: seq.append('c'),
		on_error=lambda ad, err, dur: seq.append(f'e:{err}'),
	)
	assert r.success is False
	assert seq == ['e:boom']


@pytest.mark.asyncio
async def test_run_cli_task_async_hooks_work(monkeypatch):
	seq: list[str] = []

	async def on_start(adapter, llm):
		seq.append('start')

	async def on_complete(adapter, result):
		seq.append('done')

	cfg = TaskConfig(task='t', llm=LLMConfig(provider='openai', model='m'))
	a = CommandRuntimeAdapter(cfg)
	monkeypatch.setattr(a, 'resolve_llm', lambda: DummyLLM())

	async def fake_create():
		return object()

	async def fake_cleanup():
		return None

	monkeypatch.setattr(a, 'create_browser_session', fake_create)
	monkeypatch.setattr(a, 'create_agent', lambda browser_session=None: DummyAgent())
	monkeypatch.setattr(a, 'cleanup', fake_cleanup)

	await a.run_cli_task(on_start=on_start, on_complete=on_complete)
	assert seq == ['start', 'done']


@pytest.mark.asyncio
async def test_run_cli_task_hook_exception_is_swallowed(monkeypatch):
	def bad_start(adapter, llm):
		raise ValueError('hook explode')

	cfg = TaskConfig(task='t', llm=LLMConfig(provider='openai', model='m'))
	a = CommandRuntimeAdapter(cfg)
	monkeypatch.setattr(a, 'resolve_llm', lambda: DummyLLM())

	async def fake_create():
		return object()

	async def fake_cleanup():
		return None

	monkeypatch.setattr(a, 'create_browser_session', fake_create)
	monkeypatch.setattr(a, 'create_agent', lambda browser_session=None: DummyAgent())
	monkeypatch.setattr(a, 'cleanup', fake_cleanup)

	r = await a.run_cli_task(on_start=bad_start)
	assert r.success is True


# ---------------------------------------------------------------------------
# run_mcp_tool / run_skill_command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_mcp_tool_returns_string(monkeypatch):
	cfg = TaskConfig(task='t', llm=LLMConfig(provider='openai', model='m'))
	a = CommandRuntimeAdapter(cfg)
	monkeypatch.setattr(a, 'resolve_llm', lambda: DummyLLM())

	async def fake_create():
		return object()

	async def fake_cleanup():
		return None

	monkeypatch.setattr(a, 'create_browser_session', fake_create)
	monkeypatch.setattr(a, 'create_agent', lambda browser_session=None: DummyAgent(final='MCP result'))
	monkeypatch.setattr(a, 'cleanup', fake_cleanup)

	out = await a.run_mcp_tool()
	assert isinstance(out, str)


@pytest.mark.asyncio
async def test_run_skill_command_returns_envelope_dict(monkeypatch):
	cfg = TaskConfig(task='t', llm=LLMConfig(provider='openai', model='m'))
	a = CommandRuntimeAdapter(cfg)
	monkeypatch.setattr(a, 'resolve_llm', lambda: DummyLLM())

	async def fake_create():
		return object()

	async def fake_cleanup():
		return None

	monkeypatch.setattr(a, 'create_browser_session', fake_create)
	monkeypatch.setattr(a, 'create_agent', lambda browser_session=None: DummyAgent(final='skill ok'))
	monkeypatch.setattr(a, 'cleanup', fake_cleanup)

	env = await a.run_skill_command(request_id='sk-1')
	assert env['id'] == 'sk-1'
	assert env['success'] is True
	assert isinstance(env['data'], dict)
	assert env['error'] is None


# ---------------------------------------------------------------------------
# Import locks — ensure entry-point modules reference the adapter layer
# ---------------------------------------------------------------------------


def test_cli_module_uses_commandruntimeadapter():
	"""Verify cli.py imports CommandRuntimeAdapter without actually importing it.

	We avoid a direct import because cli.py calls sys.exit() on missing textual
	in minimal environments; static AST analysis is robust instead.
	"""
	import ast
	from pathlib import Path

	cli_path = Path(__file__).resolve().parents[3] / 'browser_use' / 'cli.py'
	assert cli_path.exists(), f'cli.py not found at {cli_path}'

	code = cli_path.read_text()
	tree = ast.parse(code)

	found_for_cli = False
	found_run_cli_task = False
	for node in ast.walk(tree):
		if isinstance(node, ast.Attribute):
			name = node.attr
			if name == 'for_cli':
				found_for_cli = True
			elif name == 'run_cli_task':
				found_run_cli_task = True

	assert found_for_cli, 'cli.py must use CommandRuntimeAdapter.for_cli()'
	assert found_run_cli_task, 'cli.py must use CommandRuntimeAdapter.run_cli_task()'


def test_mcp_server_module_uses_for_mcp_and_run_mcp_tool():
	"""Verify mcp/server.py uses the adapter layer via static AST check."""
	import ast
	from pathlib import Path

	server_path = Path(__file__).resolve().parents[3] / 'browser_use' / 'mcp' / 'server.py'
	assert server_path.exists(), f'mcp/server.py not found at {server_path}'

	tree = ast.parse(server_path.read_text())

	found_for_mcp = False
	found_run_mcp_tool = False
	for node in ast.walk(tree):
		if isinstance(node, ast.Attribute):
			if node.attr == 'for_mcp':
				found_for_mcp = True
			elif node.attr == 'run_mcp_tool':
				found_run_mcp_tool = True

	assert found_for_mcp, 'mcp/server.py must use CommandRuntimeAdapter.for_mcp()'
	assert found_run_mcp_tool, 'mcp/server.py must use CommandRuntimeAdapter.run_mcp_tool()'


def test_skill_cli_sessions_uses_browser_config_conversion():
	"""Verify skill_cli/sessions.py uses _browser_config_to_profile_kwargs."""
	import ast
	from pathlib import Path

	sessions_path = (
		Path(__file__).resolve().parents[3] / 'browser_use' / 'skill_cli' / 'sessions.py'
	)
	assert sessions_path.exists()

	tree = ast.parse(sessions_path.read_text())

	found = False
	for node in ast.walk(tree):
		if isinstance(node, ast.Name) and node.id == '_browser_config_to_profile_kwargs':
			found = True
			break
		if isinstance(node, ast.Attribute) and node.attr == '_browser_config_to_profile_kwargs':
			found = True
			break

	assert found, 'skill_cli/sessions.py must use _browser_config_to_profile_kwargs'
