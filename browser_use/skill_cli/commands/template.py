"""CLI command handlers for `browser-use template ...` subcommands.

Implements list/show/run/create/edit/delete/init operations for saved
TaskTemplate definitions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _parse_var_spec(var_list: list[str] | None) -> dict[str, Any]:
	"""Parse `--var key=value` CLI arguments into a dict.

	Values are JSON-parsed if possible (so `--var count=3` gives int, not str).
	"""
	result: dict[str, Any] = {}
	if not var_list:
		return result

	for spec in var_list:
		if '=' not in spec:
			print(f'Error: --var expects key=value, got {spec!r}', file=sys.stderr)
			sys.exit(1)
		key, _, raw = spec.partition('=')
		key = key.strip()
		if not key:
			print(f'Error: --var has empty key in {spec!r}', file=sys.stderr)
			sys.exit(1)
		# Try JSON parsing for non-string values
		try:
			value: Any = json.loads(raw)
		except (json.JSONDecodeError, TypeError):
			# Fall back to bare string
			value = raw
		result[key] = value
	return result


def _load_vars_json(arg: str | None) -> dict[str, Any]:
	"""Load variables from --vars-json (inline JSON object or file path)."""
	if not arg:
		return {}
	# Try as file path first
	p = Path(arg).expanduser()
	if p.exists() and p.is_file():
		try:
			return json.loads(p.read_text(encoding='utf-8'))
		except json.JSONDecodeError as e:
			print(f'Error: --vars-json file is not valid JSON: {e}', file=sys.stderr)
			sys.exit(1)
	# Try as inline JSON string
	try:
		data = json.loads(arg)
	except json.JSONDecodeError as e:
		print(f'Error: --vars-json is neither a file path nor valid JSON: {e}', file=sys.stderr)
		sys.exit(1)
	if not isinstance(data, dict):
		print(f'Error: --vars-json must resolve to a JSON object, got {type(data).__name__}', file=sys.stderr)
		sys.exit(1)
	return data


def _merge_vars(cli_vars: dict[str, Any], json_vars: dict[str, Any]) -> dict[str, Any]:
	"""Merge variable sources; --var takes precedence over --vars-json."""
	merged = dict(json_vars)
	merged.update(cli_vars)
	return merged


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
	"""Write a sample template JSON file to get users started."""
	from browser_use.task_templates.views import (
		LLMProviderConfig,
		OutputFileRule,
		TaskTemplate,
		TemplateVariable,
		TemplateVariableType,
	)

	sample = TaskTemplate(
		name='hn-top-post',
		description='Find a top-ranked Hacker News post and extract its metadata',
		prompt_template=(
			'Go to https://news.ycombinator.com and find the {{ position }} post on the front page. '
			'Extract its title, URL, author, and point count. '
			'Save results to {{ output_file }} using the write_file action.'
		),
		variables={
			'position': TemplateVariable(
				type=TemplateVariableType.STRING,
				default='number 1',
				description='Which post to find, e.g. "number 1" or "top 3"',
			),
			'output_file': TemplateVariable(
				type=TemplateVariableType.STRING,
				default='./hn-result.csv',
				description='Where to write extracted data on disk',
				required=True,
			),
		},
		browser_profile={'headless': True, 'window_size': {'width': 1280, 'height': 720}},
		llm=LLMProviderConfig(provider='browser_use', temperature=0.0),
		default_tools=['navigate', 'click', 'extract', 'write_file', 'done'],
		output_files=[
			OutputFileRule(
				pattern='{{ output_file }}',
				description='CSV containing extracted post metadata',
				required=True,
			)
		],
		max_steps=30,
		use_vision='auto',
		tags=['hacker-news', 'extraction'],
	)

	out_path = Path(args.output).expanduser() if args.output else Path.cwd() / 'sample-template.json'
	if out_path.exists() and not args.force:
		print(f'Error: {out_path} already exists. Pass --force to overwrite.', file=sys.stderr)
		return 1

	data = json.dumps(sample.model_dump(mode='json'), indent=2, ensure_ascii=False) + '\n'
	out_path.write_text(data, encoding='utf-8')
	print(f'Wrote sample template to: {out_path}')
	print('Edit it to suit your task, then create it with:')
	print(f'  browser-use template create {out_path}')
	return 0


def _cmd_list(args: argparse.Namespace) -> int:
	"""List saved templates with optional tag filtering."""
	from browser_use.task_templates.service import get_template_manager

	mgr = get_template_manager()
	templates = mgr.list_templates()

	if args.tags:
		tag_set = set(t.lower() for t in args.tags)
		templates = [
			t for t in templates
			if tag_set & {tg.lower() for tg in t.tags}
		]

	if args.json:
		entries = []
		for t in templates:
			entries.append({
				'name': t.name,
				'version': t.version,
				'description': t.description,
				'tags': t.tags,
				'variables': [
					{'name': k, 'type': v.type.value, 'required': v.required}
					for k, v in t.variables.items()
				],
				'max_steps': t.max_steps,
			})
		print(json.dumps({'templates': entries, 'count': len(entries)}, indent=2))
		return 0

	if not templates:
		print('No templates saved yet.')
		print('Create one from a JSON file with:  browser-use template create my-template.json')
		print('Or generate a sample with:         browser-use template init')
		return 0

	verbose = getattr(args, 'verbose', False)
	templates_dir = mgr.templates_dir
	print(f'Saved templates ({len(templates)}) — location: {templates_dir}')
	print()
	for t in templates:
		header = f'  {t.name} (v{t.version})'
		if t.tags:
			header += '  [' + ', '.join(t.tags) + ']'
		print(header)
		if t.description:
			print(f'      {t.description}')
		if verbose:
			if t.variables:
				print(f'      variables: {", ".join(t.variables.keys())}')
			if t.default_tools is not None:
				print(f'      tools: {len(t.default_tools)} allowed')
			if t.llm:
				llm_parts = []
				if t.llm.provider:
					llm_parts.append(t.llm.provider)
				if t.llm.model:
					llm_parts.append(t.llm.model)
				print(f'      llm: {"/".join(llm_parts) or "default"}')
			if t.output_files:
				print(f'      output_files: {len(t.output_files)} rule(s)')
		print()
	return 0


def _cmd_show(args: argparse.Namespace) -> int:
	"""Print a template definition."""
	from browser_use.task_templates.service import get_template_manager

	mgr = get_template_manager()
	try:
		tpl = mgr.load(args.name)
	except FileNotFoundError:
		print(f'Error: template {args.name!r} not found', file=sys.stderr)
		return 1
	except ValueError as e:
		print(f'Error: {e}', file=sys.stderr)
		return 1

	data = tpl.model_dump(mode='json')
	if args.raw or args.json:
		print(json.dumps(data, indent=2, ensure_ascii=False))
	else:
		print(f'Template: {tpl.name}  (v{tpl.version})')
		if tpl.author:
			print(f'Author:   {tpl.author}')
		if tpl.tags:
			print(f'Tags:     {", ".join(tpl.tags)}')
		if tpl.description:
			print(f'Desc:     {tpl.description}')
		print()
		print(f'Max steps: {tpl.max_steps}   Vision: {tpl.use_vision}')
		if tpl.llm:
			llm_parts = []
			if tpl.llm.provider:
				llm_parts.append(f'provider={tpl.llm.provider}')
			if tpl.llm.model:
				llm_parts.append(f'model={tpl.llm.model}')
			if tpl.llm.temperature is not None:
				llm_parts.append(f't={tpl.llm.temperature}')
			print(f'LLM:      {", ".join(llm_parts) or "default"}')
		if tpl.default_tools is not None:
			print(f'Tools:    allow={", ".join(tpl.default_tools)}')
		if tpl.exclude_tools:
			print(f'Tools:    exclude={", ".join(tpl.exclude_tools)}')
		if tpl.variables:
			print()
			print('Variables:')
			for name, var in tpl.variables.items():
				marker = ' (required)' if var.required else ''
				parts = [f'  {name}: {var.type.value}{marker}']
				if var.default is not None:
					parts.append(f'  default={var.default!r}')
				if var.description:
					parts.append(f'  — {var.description}')
				if var.choices:
					parts.append(f'  choices={var.choices}')
				print(''.join(parts))
		if tpl.output_files:
			print()
			print('Output file rules:')
			for rule in tpl.output_files:
				marker = ' (required)' if rule.required else ''
				print(f'  {rule.pattern}{marker}')
				if rule.description:
					print(f'      {rule.description}')
		print()
		print('Prompt template:')
		print('  ' + tpl.prompt_template.replace('\n', '\n  '))
	return 0


def _cmd_create(args: argparse.Namespace) -> int:
	"""Create/save a template from a JSON definition file."""
	from browser_use.task_templates.service import get_template_manager

	path = Path(args.file).expanduser()
	if not path.exists():
		print(f'Error: file not found: {path}', file=sys.stderr)
		return 1

	try:
		data = json.loads(path.read_text(encoding='utf-8'))
	except json.JSONDecodeError as e:
		print(f'Error: {path} is not valid JSON: {e}', file=sys.stderr)
		return 1

	mgr = get_template_manager()
	try:
		tpl = mgr.create_from_dict(data, overwrite=bool(args.force))
	except ValueError as e:
		print(f'Error: template validation failed: {e}', file=sys.stderr)
		return 1
	except FileExistsError as e:
		print(f'Error: {e}. Pass --force to overwrite.', file=sys.stderr)
		return 1

	print(f'Saved template {tpl.name!r} to {mgr.templates_dir / (tpl.name + ".json")}')
	return 0


def _cmd_edit(args: argparse.Namespace) -> int:
	"""Open a template JSON in $EDITOR and validate the result on save."""
	from browser_use.task_templates.service import get_template_manager
	from browser_use.task_templates.views import TaskTemplate

	mgr = get_template_manager()
	try:
		tpl = mgr.load(args.name)
	except FileNotFoundError:
		print(f'Error: template {args.name!r} not found', file=sys.stderr)
		return 1
	except ValueError as e:
		print(f'Error: {e}', file=sys.stderr)
		return 1

	editor = os.environ.get('EDITOR') or os.environ.get('VISUAL')
	if not editor:
		print('Error: $EDITOR is not set. Set it (e.g. export EDITOR=vim) or edit the JSON directly:', file=sys.stderr)
		print(f'  {mgr.templates_dir / (tpl.name + ".json")}')
		return 1

	path = mgr.templates_dir / (tpl.name + '.json')
	original_mtime = path.stat().st_mtime_ns

	try:
		subprocess.run([editor, str(path)], check=True)
	except (subprocess.CalledProcessError, FileNotFoundError) as e:
		print(f'Error: editor {editor!r} failed: {e}', file=sys.stderr)
		return 1

	if path.stat().st_mtime_ns == original_mtime:
		print('No changes detected.')
		return 0

	# Validate and re-load
	try:
		raw = json.loads(path.read_text(encoding='utf-8'))
		validated = TaskTemplate.model_validate(raw)
	except json.JSONDecodeError as e:
		print(f'Error: file is not valid JSON after edit: {e}', file=sys.stderr)
		print('File was NOT renamed or deleted — fix the JSON manually.')
		return 1
	except Exception as e:
		print(f'Error: template validation failed after edit: {e}', file=sys.stderr)
		print('File was NOT renamed or deleted — fix the JSON manually.')
		return 1

	mgr._cache.pop(validated.name, None)
	print(f'Saved and validated template {validated.name!r}')
	return 0


def _cmd_delete(args: argparse.Namespace) -> int:
	"""Delete a saved template."""
	from browser_use.task_templates.service import get_template_manager

	mgr = get_template_manager()
	if not mgr.exists(args.name):
		print(f'Error: template {args.name!r} not found', file=sys.stderr)
		return 1

	if not args.yes:
		ans = input(f'Delete template {args.name!r}? [y/N] ')
		if ans.lower() not in ('y', 'yes'):
			print('Aborted.')
			return 0

	ok = mgr.delete(args.name)
	if ok:
		print(f'Deleted template {args.name!r}')
		return 0
	print(f'Error: could not delete template {args.name!r}', file=sys.stderr)
	return 1


async def _cmd_run_async(args: argparse.Namespace) -> int:
	"""Run a saved template (async implementation)."""
	from browser_use.task_templates.service import get_template_manager
	from browser_use.task_templates.views import TaskTemplateStatus

	mgr = get_template_manager()

	# Resolve variables
	cli_vars = _parse_var_spec(getattr(args, 'vars', None))
	json_vars = _load_vars_json(getattr(args, 'vars_json', None))
	merged_vars = _merge_vars(cli_vars, json_vars)

	quiet = getattr(args, 'quiet', False)
	if not quiet:
		# Pre-load just to show prompt rendering errors early
		try:
			tpl = mgr.load(args.name)
		except FileNotFoundError:
			print(f'Error: template {args.name!r} not found', file=sys.stderr)
			return 1
		from browser_use.task_templates.service import resolve_template_variables, substitute_variables
		try:
			resolved = resolve_template_variables(tpl, merged_vars)
			rendered = substitute_variables(tpl.prompt_template, resolved)
		except (ValueError, KeyError) as e:
			print(f'Error: {e}', file=sys.stderr)
			return 1
		print(f'Running template: {tpl.name}')
		if tpl.description:
			print(f'  {tpl.description}')
		if merged_vars:
			import pprint
			print(f'  variables: {pprint.pformat(resolved, width=120)}')
		print(f'  max_steps: {args.max_steps or tpl.max_steps}')
		print()
		print('Prompt:')
		for line in rendered.splitlines():
			print(f'  {line}')
		print()
		print('Executing... (Ctrl+C to interrupt)')
		sys.stdout.flush()

	try:
		result = await mgr.run(
			args.name,
			variables=merged_vars,
			max_steps=args.max_steps,
		)
	except KeyboardInterrupt:
		print()
		print('Interrupted by user.')
		return 130

	# Output
	if getattr(args, 'output_json', False) or getattr(args, 'json', False):
		print(result.model_dump_json(indent=2))
	else:
		print()
		print('=' * 56)
		status_icon = {
			TaskTemplateStatus.SUCCESS: '✅',
			TaskTemplateStatus.PARTIAL: '⚠️',
			TaskTemplateStatus.FAILED: '❌',
			TaskTemplateStatus.RUNNING: '⏳',
		}.get(result.status, '?')
		print(f'{status_icon}  Status: {result.status.value}   (steps={result.num_steps}, {result.duration_seconds:.2f}s)')

		if result.extracted_content:
			print()
			print('Extracted content:')
			for line in str(result.extracted_content).splitlines():
				print(f'  {line}')

		if result.urls_visited:
			print()
			print(f'URLs visited ({len(result.urls_visited)}):')
			for u in result.urls_visited[:8]:
				print(f'  - {u}')
			if len(result.urls_visited) > 8:
				print(f'  ... and {len(result.urls_visited) - 8} more')

		if result.output_files:
			print()
			print(f'Output files ({len(result.output_files)}):')
			for f in result.output_files:
				print(f'  - {f}')

		if result.missing_output_files:
			print()
			print(f'Missing required output files ({len(result.missing_output_files)}):')
			for pat in result.missing_output_files:
				print(f'  - {pat}')

		if result.error:
			print()
			if result.error_type:
				print(f'Error [{result.error_type}]:')
			else:
				print('Error:')
			print(f'  {result.error}')

	return 0 if result.success else 1


def _cmd_run(args: argparse.Namespace) -> int:
	"""Run a saved template (sync entry point that wraps the async version)."""
	return asyncio.run(_cmd_run_async(args))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_CMD_TABLE: dict[str, Any] = {
	'list': _cmd_list,
	'show': _cmd_show,
	'run': _cmd_run,
	'create': _cmd_create,
	'edit': _cmd_edit,
	'delete': _cmd_delete,
	'init': _cmd_init,
}


def handle_template_command(args: argparse.Namespace) -> int:
	"""Main entry point for `browser-use template ...` called from skill_cli/main.py."""
	sub = getattr(args, 'template_command', None)

	if sub is None:
		print('Usage: browser-use template <list|show|run|create|edit|delete|init> --help')
		return 0

	handler = _CMD_TABLE.get(sub)
	if handler is None:
		print(f'Error: unknown template subcommand {sub!r}', file=sys.stderr)
		return 1

	return handler(args)
