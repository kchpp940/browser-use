from datetime import datetime, timezone

from browser_use.observability_runtime import (
	ErrorInfo,
	EventSeverity,
	EventSource,
	EventType,
	OutputFile,
	RuntimeEvent,
	TokenUsage,
	validate_event_consistency,
)


def _make_full_event() -> RuntimeEvent:
	return RuntimeEvent(
		source=EventSource.AGENT,
		event_type=EventType.TASK_END,
		severity=EventSeverity.INFO,
		message='test task done',
		task_id='task-abc-123',
		session_id='session-def-456',
		step=5,
		error=ErrorInfo(
			error_type='TestError',
			error_message='something went wrong',
			error_stack='Traceback (most recent call last):\n  ...',
		),
		output_files=[
			OutputFile(path='/tmp/report.pdf', file_name='report.pdf', content_type='application/pdf', size_bytes=1024),
			OutputFile(path='/tmp/shot.png', file_name='screenshot.png', content_type='image/png', size_bytes=2048),
		],
		tokens=TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.001),
		duration_ms=1234.5,
		timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
		data={'custom_key': 'custom_value', 'numeric_field': 42},
	)


class TestConsoleAdapter:
	def test_console_extra_contains_identity_fields(self):
		event = _make_full_event()
		extra = event.to_console_extra_dict()

		assert extra['runtime_event_id'] == event.event_id
		assert extra['runtime_event_type'] == event.event_type.value
		assert extra['runtime_source'] == event.source.value
		assert extra['task_id'] == 'task-abc-123'
		assert extra['session_id'] == 'session-def-456'
		assert extra['step'] == 5

	def test_console_extra_omits_null_fields(self):
		event = RuntimeEvent(source=EventSource.MCP_SERVER, event_type=EventType.MCP_TOOL_CALL, message='hi')
		extra = event.to_console_extra_dict()

		assert 'task_id' not in extra
		assert 'session_id' not in extra
		assert 'step' not in extra
		assert extra['runtime_source'] == 'mcp_server'

	def test_console_line_is_nonempty_string(self):
		event = _make_full_event()
		line = event.to_console_line()
		assert isinstance(line, str)
		assert len(line) > 0
		assert 'test task done' in line


class TestTelemetryAdapter:
	def test_telemetry_properties_contain_canonical_fields(self):
		event = _make_full_event()
		props = event.to_telemetry_properties()

		assert props['source'] == 'agent'
		assert props['event_type'] == 'task_end'
		assert props['task_id'] == 'task-abc-123'
		assert props['session_id'] == 'session-def-456'
		assert props['step'] == 5
		assert props['error_type'] == 'TestError'
		assert props['error_message'] == 'something went wrong'
		assert props['output_file_count'] == 2

	def test_telemetry_properties_flatten_tokens(self):
		event = _make_full_event()
		props = event.to_telemetry_properties()

		assert props['tokens_input_tokens'] == 100
		assert props['tokens_output_tokens'] == 50
		assert props['tokens_total_tokens'] == 150
		assert props['tokens_cost_usd'] == 0.001

	def test_telemetry_properties_flatten_data(self):
		event = _make_full_event()
		props = event.to_telemetry_properties()

		assert props['data_custom_key'] == 'custom_value'
		assert props['data_numeric_field'] == 42

	def test_telemetry_event_name_mapping(self):
		assert (
			RuntimeEvent(source=EventSource.AGENT, event_type=EventType.TASK_START, message='').to_telemetry_event_name()
			== 'agent_task'
		)
		assert (
			RuntimeEvent(source=EventSource.AGENT, event_type=EventType.STEP_END, message='').to_telemetry_event_name()
			== 'agent_step'
		)
		assert (
			RuntimeEvent(source=EventSource.AGENT, event_type=EventType.AGENT_ACTION, message='').to_telemetry_event_name()
			== 'agent_action'
		)
		assert (
			RuntimeEvent(source=EventSource.AGENT, event_type=EventType.EXCEPTION, message='').to_telemetry_event_name()
			== 'agent_error'
		)
		assert (
			RuntimeEvent(source=EventSource.AGENT, event_type=EventType.LLM_CALL, message='').to_telemetry_event_name()
			== 'agent_llm'
		)
		assert (
			RuntimeEvent(source=EventSource.MCP_SERVER, event_type=EventType.MCP_TOOL_CALL, message='').to_telemetry_event_name()
			== 'mcp_server_mcp_tool'
		)
		assert (
			RuntimeEvent(source=EventSource.SANDBOX, event_type=EventType.SANDBOX_START, message='').to_telemetry_event_name()
			== 'sandbox_sandbox'
		)
		assert (
			RuntimeEvent(source=EventSource.CLI, event_type=EventType.CLI_START, message='').to_telemetry_event_name()
			== 'cli_event'
		)


class TestCloudAdapter:
	def test_cloud_dict_contains_all_canonical_fields(self):
		event = _make_full_event()
		d = event.to_cloud_event_dict()

		assert d['source'] == 'agent'
		assert d['event_type'] == 'task_end'
		assert d['task_id'] == 'task-abc-123'
		assert d['session_id'] == 'session-def-456'
		assert d['step'] == 5

		assert isinstance(d['error'], dict)
		assert d['error']['error_type'] == 'TestError'
		assert d['error']['error_message'] == 'something went wrong'
		assert d['error']['error_stack'] is not None

		assert isinstance(d['output_files'], list)
		assert len(d['output_files']) == 2
		assert d['output_files'][0]['file_name'] == 'report.pdf'
		assert d['output_files'][0]['content_type'] == 'application/pdf'

	def test_cloud_dict_includes_tokens_and_data(self):
		event = _make_full_event()
		d = event.to_cloud_event_dict()

		assert isinstance(d['tokens'], dict)
		assert d['tokens']['input_tokens'] == 100
		assert isinstance(d['data'], dict)
		assert d['data']['custom_key'] == 'custom_value'

	def test_cloud_dict_includes_environment_metadata(self):
		event = _make_full_event()
		d = event.to_cloud_event_dict()

		assert d['event_id'] == event.event_id
		assert d['timestamp'] == '2025-01-01T12:00:00+00:00'
		assert d['duration_ms'] == 1234.5


class TestValidateEventConsistency:
	def test_full_agent_event_passes(self):
		event = _make_full_event()
		violations = validate_event_consistency(event)
		assert violations == []

	def test_missing_required_fields_triggers_violations(self):
		event = RuntimeEvent(source=EventSource.AGENT, event_type=EventType.TASK_START, message='hi')
		violations = validate_event_consistency(event)
		assert any('task_id' in v for v in violations)
		assert any('session_id' in v for v in violations)

	def test_step_on_non_step_source_triggers_violation(self):
		event = RuntimeEvent(
			source=EventSource.CLI,
			event_type=EventType.CLI_START,
			message='hi',
			task_id='t1',
			step=0,
		)
		violations = validate_event_consistency(event)
		assert any('step' in v and 'cli' in v for v in violations)

	def test_error_on_non_error_source_triggers_violation(self):
		event = RuntimeEvent(
			source=EventSource.BROWSER,
			event_type=EventType.SESSION_START,
			message='hi',
			session_id='s1',
			error=ErrorInfo(error_type='Err', error_message='err'),
		)
		violations = validate_event_consistency(event)
		assert len(violations) == 0  # browser source allows errors

	def test_output_files_on_agent_source_allowed(self):
		event = _make_full_event()
		violations = validate_event_consistency(event)
		assert not any('output_files' in v for v in violations)
