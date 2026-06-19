"""
Standalone init command for browser-use template generation.

This module provides a minimal command-line interface for generating
browser-use templates without requiring heavy TUI dependencies.
"""

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError

import click
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.utils import InquirerPyStyle
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Rich console for styled output
console = Console()

# GitHub template repository URL (for runtime fetching)
TEMPLATE_REPO_URL = 'https://raw.githubusercontent.com/browser-use/template-library/main'

# Export for backward compatibility with cli.py
# Templates are fetched at runtime via _get_template_list()
INIT_TEMPLATES: dict[str, Any] = {}


def _fetch_template_list() -> dict[str, Any] | None:
	"""
	Fetch template list from GitHub templates.json.

	Returns template dict if successful, None if failed.
	"""
	try:
		url = f'{TEMPLATE_REPO_URL}/templates.json'
		with request.urlopen(url, timeout=5) as response:
			data = response.read().decode('utf-8')
			return json.loads(data)
	except (URLError, TimeoutError, json.JSONDecodeError, Exception):
		return None


def _get_template_list() -> dict[str, Any]:
	"""
	Get template list from GitHub.

	Raises FileNotFoundError if GitHub fetch fails.
	"""
	templates = _fetch_template_list()
	if templates is not None:
		return templates
	raise FileNotFoundError('Could not fetch templates from GitHub. Check your internet connection.')


def _fetch_from_github(file_path: str) -> str | None:
	"""
	Fetch template file from GitHub.

	Returns file content if successful, None if failed.
	"""
	try:
		url = f'{TEMPLATE_REPO_URL}/{file_path}'
		with request.urlopen(url, timeout=5) as response:
			return response.read().decode('utf-8')
	except (URLError, TimeoutError, Exception):
		return None


def _fetch_binary_from_github(file_path: str) -> bytes | None:
	"""
	Fetch binary file from GitHub.

	Returns file content if successful, None if failed.
	"""
	try:
		url = f'{TEMPLATE_REPO_URL}/{file_path}'
		with request.urlopen(url, timeout=5) as response:
			return response.read()
	except (URLError, TimeoutError, Exception):
		return None


def _get_template_content(file_path: str) -> str:
	"""
	Get template file content from GitHub.

	Raises exception if fetch fails.
	"""
	content = _fetch_from_github(file_path)

	if content is not None:
		return content

	raise FileNotFoundError(f'Could not fetch template from GitHub: {file_path}')


# InquirerPy style for template selection (browser-use orange theme)
inquirer_style = InquirerPyStyle(
	{
		'pointer': '#fe750e bold',
		'highlighted': '#fe750e bold',
		'question': 'bold',
		'answer': '#fe750e bold',
		'questionmark': '#fe750e bold',
	}
)


def _get_terminal_width() -> int:
	"""Get current terminal width in columns."""
	return shutil.get_terminal_size().columns


def _format_choice(name: str, metadata: dict[str, Any], width: int, is_default: bool = False) -> str:
	"""
	Format a template choice with responsive display based on terminal width.

	Styling:
	- Featured templates get [FEATURED] prefix
	- Author name included when width allows (except for default templates)
	- Everything turns orange when highlighted (InquirerPy's built-in behavior)

	Args:
		name: Template name
		metadata: Template metadata (description, featured, author)
		width: Terminal width in columns
		is_default: Whether this is a default template (default, advanced, tools)

	Returns:
		Formatted choice string
	"""
	is_featured = metadata.get('featured', False)
	description = metadata.get('description', '')
	author_name = metadata.get('author', {}).get('name', '') if isinstance(metadata.get('author'), dict) else ''

	# Build the choice string based on terminal width
	if width > 100:
		# Wide: show everything including author (except for default templates)
		if is_featured:
			if author_name:
				return f'[FEATURED] {name} by {author_name} - {description}'
			else:
				return f'[FEATURED] {name} - {description}'
		else:
			# Non-featured templates
			if author_name and not is_default:
				return f'{name} by {author_name} - {description}'
			else:
				return f'{name} - {description}'

	elif width > 60:
		# Medium: show name and description, no author
		if is_featured:
			return f'[FEATURED] {name} - {description}'
		else:
			return f'{name} - {description}'

	else:
		# Narrow: show name only
		return name


def _upgrade_template_content(content: str, file_name: str = 'main.py') -> str:
	"""Upgrade template content to use run_with_result() + ResultSerializer.

	This function ensures that any template downloaded from GitHub, even
	older ones that use agent.run() or agent.run_sync(), will be upgraded
	to the modern structured result API before being written to disk.

	Upgrade rules:
	1. If content already contains 'run_with_result', return as-is (already modern)
	2. Add 'ResultSerializer' to the browser_use import line
	3. For scripts using agent.run_sync() at top level, wrap in async main()
	4. Replace agent.run() / agent.run_sync() with run_with_result() + ResultSerializer
	5. Ensure ResultSerializer import is present
	"""
	# Already modern — skip
	if 'run_with_result' in content:
		return content

	lines = content.split('\n')
	out_lines: list[str] = []
	in_async_main = False
	has_async_main = False
	has_agent_run_sync = False
	seen_run_call = False

	# First pass: detect patterns
	for line in lines:
		if 'async def main()' in line:
			has_async_main = True
		if 'agent.run_sync()' in line or '.run_sync()' in line:
			has_agent_run_sync = True

	# Build ResultSerializer import line
	def _fix_import_line(line: str) -> str:
		"""Add ResultSerializer to the browser_use import if not present."""
		if 'from browser_use import' not in line:
			return line
		if 'ResultSerializer' in line:
			return line
		# Has a multi-line import — inject before the closing )
		if line.rstrip().endswith('('):
			return line
		# Single line: from browser_use import X, Y
		# Split and add ResultSerializer
		match = re.match(r'^(from browser_use import\s+)(.*?)(\s*#.*)?$', line)
		if match:
			prefix, imports, comment = match.groups()
			imports = imports.strip()
			if imports.endswith(','):
				new_imports = f'{imports} ResultSerializer'
			else:
				new_imports = f'{imports}, ResultSerializer'
			comment = comment or ''
			return f'{prefix}{new_imports}{comment}'
		return line

	def _replace_run_call(line: str) -> str:
		"""Replace agent.run() or agent.run_sync() with run_with_result() + ResultSerializer output."""
		# Top-level sync run_sync(): replace with asyncio.run(main()) and create async main
		run_pattern = re.compile(r'^(\s*)([\w\.]+)\.(run|run_sync)\(\s*\)\s*$')
		match = run_pattern.match(line.rstrip())
		if match:
			indent, agent_var, _run_type = match.groups()
			# Return the modern pattern: run_with_result() + ResultSerializer
			return (
				f'{indent}result = await {agent_var}.run_with_result()\n'
				f'{indent}serializer = ResultSerializer(result)\n'
				f'{indent}print(serializer.to_text())'
			)
		# run() with kwargs (e.g., agent.run(max_steps=100))
		run_kwargs_pattern = re.compile(r'^(\s*)([\w\.]+)\.(run|run_sync)\((.*)\)\s*$')
		match = run_kwargs_pattern.match(line.rstrip())
		if match:
			indent, agent_var, _run_type, kwargs = match.groups()
			return (
				f'{indent}result = await {agent_var}.run_with_result({kwargs})\n'
				f'{indent}serializer = ResultSerializer(result)\n'
				f'{indent}print(serializer.to_text())'
			)
		# await agent.run() pattern
		await_pattern = re.compile(r'^(\s*)await\s+([\w\.]+)\.(run|run_sync)\(\s*\)\s*$')
		match = await_pattern.match(line.rstrip())
		if match:
			indent, agent_var, _run_type = match.groups()
			return (
				f'{indent}result = await {agent_var}.run_with_result()\n'
				f'{indent}serializer = ResultSerializer(result)\n'
				f'{indent}print(serializer.to_text())'
			)
		# await agent.run(...) with kwargs
		await_kwargs_pattern = re.compile(r'^(\s*)await\s+([\w\.]+)\.(run|run_sync)\((.*)\)\s*$')
		match = await_kwargs_pattern.match(line.rstrip())
		if match:
			indent, agent_var, _run_type, kwargs = match.groups()
			return (
				f'{indent}result = await {agent_var}.run_with_result({kwargs})\n'
				f'{indent}serializer = ResultSerializer(result)\n'
				f'{indent}print(serializer.to_text())'
			)
		return line

	i = 0
	added_asyncio_import = False
	added_asyncio_run = False
	has_asyncio_import = 'import asyncio' in content or 'from asyncio import' in content
	result_lines: list[str] = []

	while i < len(lines):
		line = lines[i]

		# 1. Fix imports: add ResultSerializer, add asyncio import if needed
		fixed_line = _fix_import_line(line)
		if 'from browser_use import' in line and 'ResultSerializer' in fixed_line and 'ResultSerializer' not in line:
			line = fixed_line

		# Add asyncio import if script uses run_sync() and doesn't have one
		if not added_asyncio_import and has_agent_run_sync and not has_asyncio_import:
			if line.strip().startswith('from dotenv import load_dotenv') or line.strip().startswith('import dotenv'):
				result_lines.append(line)
				result_lines.append('import asyncio')
				i += 1
				added_asyncio_import = True
				continue

		# 2. For scripts with agent.run_sync() but no async main, wrap in async main
		if has_agent_run_sync and not has_async_main and not in_async_main:
			run_match = re.match(r'^(\s*)([\w\.]+)\.(run|run_sync)\(\s*\)\s*$', line.rstrip())
			run_kwargs_match = re.match(r'^(\s*)([\w\.]+)\.(run|run_sync)\((.*)\)\s*$', line.rstrip())
			if run_match or run_kwargs_match:
				# Create async main wrapper before this call
				result_lines.append('')
				result_lines.append('async def main():')
				in_async_main = True
				# Replace the run call inside async main
				replaced = _replace_run_call(f'    {line.lstrip()}')
				for rl in replaced.split('\n'):
					result_lines.append(rl)
				added_asyncio_run = True
				i += 1
				continue

		# 3. Replace run() calls with run_with_result() + ResultSerializer
		if not seen_run_call and ('agent.run' in line or '.run_sync()' in line or 'await agent.run' in line):
			replaced = _replace_run_call(line)
			if replaced != line:
				seen_run_call = True
				for rl in replaced.split('\n'):
					result_lines.append(rl)
				i += 1
				continue

		result_lines.append(line)
		i += 1

	# 4. Add asyncio.run(main()) at end if we wrapped in async main
	if added_asyncio_run:
		# Append before any final comments/empty lines at EOF
		# Remove trailing empty lines first
		while result_lines and result_lines[-1].strip() == '':
			result_lines.pop()
		# Check if __main__ block exists
		has_main_block = any("if __name__ == '__main__'" in line or 'if __name__ == "__main__"' in line for line in result_lines)
		if not has_main_block:
			result_lines.append('')
			result_lines.append("if __name__ == '__main__':")
			result_lines.append('    asyncio.run(main())')

	return '\n'.join(result_lines)


def _write_init_file(output_path: Path, content: str, force: bool = False) -> bool:
	"""Write content to a file, with safety checks."""
	# Check if file already exists
	if output_path.exists() and not force:
		console.print(f'[yellow]⚠[/yellow]  File already exists: [cyan]{output_path}[/cyan]')
		if not click.confirm('Overwrite?', default=False):
			console.print('[red]✗[/red] Cancelled')
			return False

	# Ensure parent directory exists
	output_path.parent.mkdir(parents=True, exist_ok=True)

	# Write file
	try:
		output_path.write_text(content, encoding='utf-8')
		return True
	except Exception as e:
		console.print(f'[red]✗[/red] Error writing file: {e}')
		return False


@click.command('browser-use-init')
@click.option(
	'--template',
	'-t',
	type=str,
	help='Template to use',
)
@click.option(
	'--output',
	'-o',
	type=click.Path(),
	help='Output file path (default: browser_use_<template>.py)',
)
@click.option(
	'--force',
	'-f',
	is_flag=True,
	help='Overwrite existing files without asking',
)
@click.option(
	'--list',
	'-l',
	'list_templates',
	is_flag=True,
	help='List available templates',
)
def main(
	template: str | None,
	output: str | None,
	force: bool,
	list_templates: bool,
):
	"""
	Generate a browser-use template file to get started quickly.

	Examples:

	\b
	# Interactive mode - prompts for template selection
	uvx browser-use init
	uvx browser-use init --template

	\b
	# Generate default template
	uvx browser-use init --template default

	\b
	# Generate advanced template with custom filename
	uvx browser-use init --template advanced --output my_script.py

	\b
	# List available templates
	uvx browser-use init --list
	"""

	# Fetch template list at runtime
	try:
		INIT_TEMPLATES = _get_template_list()
	except FileNotFoundError as e:
		console.print(f'[red]✗[/red] {e}')
		sys.exit(1)

	# Handle --list flag
	if list_templates:
		console.print('\n[bold]Available templates:[/bold]\n')
		for name, info in INIT_TEMPLATES.items():
			console.print(f'  [#fe750e]{name:12}[/#fe750e] - {info["description"]}')
		console.print()
		return

	# Interactive template selection if not provided
	if not template:
		# Get terminal width for responsive formatting
		width = _get_terminal_width()

		# Separate default and featured templates
		default_template_names = ['default', 'advanced', 'tools']
		featured_templates = [(name, info) for name, info in INIT_TEMPLATES.items() if info.get('featured', False)]
		other_templates = [
			(name, info)
			for name, info in INIT_TEMPLATES.items()
			if name not in default_template_names and not info.get('featured', False)
		]

		# Sort by last_modified_date (most recent first)
		def get_last_modified(item):
			name, info = item
			date_str = (
				info.get('author', {}).get('last_modified_date', '1970-01-01')
				if isinstance(info.get('author'), dict)
				else '1970-01-01'
			)
			return date_str

		# Sort default templates by last modified
		default_templates = [(name, INIT_TEMPLATES[name]) for name in default_template_names if name in INIT_TEMPLATES]
		default_templates.sort(key=get_last_modified, reverse=True)

		# Sort featured and other templates by last modified
		featured_templates.sort(key=get_last_modified, reverse=True)
		other_templates.sort(key=get_last_modified, reverse=True)

		# Build choices in order: defaults first, then featured, then others
		choices = []

		# Add default templates
		for i, (name, info) in enumerate(default_templates):
			formatted = _format_choice(name, info, width, is_default=True)
			choices.append(Choice(name=formatted, value=name))

		# Add featured templates
		for i, (name, info) in enumerate(featured_templates):
			formatted = _format_choice(name, info, width, is_default=False)
			choices.append(Choice(name=formatted, value=name))

		# Add other templates (if any)
		for name, info in other_templates:
			formatted = _format_choice(name, info, width, is_default=False)
			choices.append(Choice(name=formatted, value=name))

		# Use fuzzy prompt for search functionality
		# Use getattr to avoid static analysis complaining about non-exported names
		_fuzzy = getattr(inquirer, 'fuzzy')
		template = _fuzzy(
			message='Select a template (type to search):',
			choices=choices,
			style=inquirer_style,
			max_height='70%',
		).execute()

		# Handle user cancellation (Ctrl+C)
		if template is None:
			console.print('\n[red]✗[/red] Cancelled')
			sys.exit(1)

	# Template is guaranteed to be set at this point (either from option or prompt)
	assert template is not None

	# Create template directory
	template_dir = Path.cwd() / template
	if template_dir.exists() and not force:
		console.print(f'[yellow]⚠[/yellow]  Directory already exists: [cyan]{template_dir}[/cyan]')
		if not click.confirm('Continue and overwrite files?', default=False):
			console.print('[red]✗[/red] Cancelled')
			sys.exit(1)

	# Create directory
	template_dir.mkdir(parents=True, exist_ok=True)

	# Determine output path
	if output:
		output_path = template_dir / Path(output)
	else:
		output_path = template_dir / 'main.py'

	# Read template file from GitHub
	try:
		template_file = INIT_TEMPLATES[template]['file']
		content = _get_template_content(template_file)
		# Upgrade to modern structured result API before writing
		content = _upgrade_template_content(content, output_path.name)
	except Exception as e:
		console.print(f'[red]✗[/red] Error reading template: {e}')
		sys.exit(1)

	# Write file
	if _write_init_file(output_path, content, force):
		console.print(f'\n[green]✓[/green] Created [cyan]{output_path}[/cyan]')

		# Generate additional files if template has a manifest
		if 'files' in INIT_TEMPLATES[template]:
			import stat

			for file_spec in INIT_TEMPLATES[template]['files']:
				source_path = file_spec['source']
				dest_name = file_spec['dest']
				dest_path = output_path.parent / dest_name
				is_binary = file_spec.get('binary', False)
				is_executable = file_spec.get('executable', False)

				# Skip if we already wrote this file (main.py)
				if dest_path == output_path:
					continue

				# Fetch and write file
				try:
					if is_binary:
						file_content = _fetch_binary_from_github(source_path)
						if file_content:
							if not dest_path.exists() or force:
								dest_path.write_bytes(file_content)
								console.print(f'[green]✓[/green] Created [cyan]{dest_name}[/cyan]')
						else:
							console.print(f'[yellow]⚠[/yellow]  Could not fetch [cyan]{dest_name}[/cyan] from GitHub')
					else:
						file_content = _get_template_content(source_path)
						# Upgrade .py files to modern structured result API
						if dest_name.endswith('.py'):
							file_content = _upgrade_template_content(file_content, dest_name)
						if _write_init_file(dest_path, file_content, force):
							console.print(f'[green]✓[/green] Created [cyan]{dest_name}[/cyan]')
							# Make executable if needed
							if is_executable and sys.platform != 'win32':
								dest_path.chmod(dest_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
				except Exception as e:
					console.print(f'[yellow]⚠[/yellow]  Error generating [cyan]{dest_name}[/cyan]: {e}')

		# Create a nice panel for next steps
		next_steps = Text()

		# Display next steps from manifest if available
		if 'next_steps' in INIT_TEMPLATES[template]:
			steps = INIT_TEMPLATES[template]['next_steps']
			for i, step in enumerate(steps, 1):
				# Handle footer separately (no numbering)
				if 'footer' in step:
					next_steps.append(f'{step["footer"]}\n', style='dim italic')
					continue

				# Step title
				next_steps.append(f'\n{i}. {step["title"]}:\n', style='bold')

				# Step commands
				for cmd in step.get('commands', []):
					# Replace placeholders
					cmd = cmd.replace('{template}', template)
					cmd = cmd.replace('{output}', output_path.name)
					next_steps.append(f'   {cmd}\n', style='dim')

				# Optional note
				if 'note' in step:
					next_steps.append(f'   {step["note"]}\n', style='dim italic')

				next_steps.append('\n')
		else:
			# Default workflow for templates without custom next_steps
			next_steps.append('\n1. Navigate to project directory:\n', style='bold')
			next_steps.append(f'   cd {template}\n\n', style='dim')
			next_steps.append('2. Initialize uv project:\n', style='bold')
			next_steps.append('   uv init\n\n', style='dim')
			next_steps.append('3. Install browser-use:\n', style='bold')
			next_steps.append('   uv add browser-use\n\n', style='dim')
			next_steps.append('4. Set up your API key in .env file or environment:\n', style='bold')
			next_steps.append('   BROWSER_USE_API_KEY=your-key\n', style='dim')
			next_steps.append(
				'   (Get your key at https://cloud.browser-use.com/dashboard/settings?tab=api-keys&new&utm_source=oss&utm_medium=cli)\n\n',
				style='dim italic',
			)
			next_steps.append('5. Run your script:\n', style='bold')
			next_steps.append(f'   uv run {output_path.name}\n', style='dim')

		console.print(
			Panel(
				next_steps,
				title='[bold]Next steps[/bold]',
				border_style='#fe750e',
				padding=(1, 2),
			)
		)


if __name__ == '__main__':
	main()
