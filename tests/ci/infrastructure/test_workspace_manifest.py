"""Tests for the WorkspaceManifest functionality."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from browser_use.filesystem.file_system import FileSystem, FileSystemState
from browser_use.filesystem.workspace_manifest import (
	FileSource,
	ManifestEntry,
	PathType,
	WorkspaceManifest,
	WorkspaceManifestState,
)


class TestManifestEntry:
	def test_entry_creation(self):
		entry = ManifestEntry(
			display_name='test.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/data/test.md',
			size_bytes=100,
			readable=True,
			uploadable=True,
			last_action='write',
		)
		assert entry.display_name == 'test.md'
		assert entry.source == FileSource.VIRTUAL
		assert entry.path_type == PathType.VIRTUAL_NAME
		assert entry.is_virtual is True
		assert entry.lookup_key == 'test.md'

	def test_entry_local_absolute_lookup_key(self):
		entry = ManifestEntry(
			display_name='report.pdf',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/downloads/report.pdf',
			size_bytes=5000,
		)
		assert entry.is_virtual is False
		assert entry.lookup_key == '/tmp/downloads/report.pdf'

	def test_entry_defaults(self):
		entry = ManifestEntry(
			display_name='image.png',
			source=FileSource.SCREENSHOT,
			path_type=PathType.LOCAL_ABSOLUTE,
		)
		assert entry.size_bytes == 0
		assert entry.readable is True
		assert entry.uploadable is False
		assert entry.last_action is None
		assert entry.metadata == {}


class TestWorkspaceManifest:
	def test_register_and_get(self):
		manifest = WorkspaceManifest()
		entry = manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/fs/data.json',
			size_bytes=256,
			readable=True,
			uploadable=True,
			last_action='write',
		)
		assert entry.display_name == 'data.json'
		got = manifest.get('data.json')
		assert got is not None
		assert got.display_name == 'data.json'

	def test_register_update_existing(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			size_bytes=100,
			last_action='write',
		)
		manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			size_bytes=200,
			last_action='append',
		)
		entry = manifest.get('data.json')
		assert entry.size_bytes == 200
		assert entry.last_action == 'append'
		assert len(manifest.all_entries()) == 1

	def test_touch(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='test.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			last_action='write',
		)
		manifest.touch('test.md', 'read')
		entry = manifest.get('test.md')
		assert entry.last_action == 'read'

	def test_resolve_by_display_name(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='results.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/fs/results.md',
		)
		result = manifest.resolve('results.md')
		assert result is not None

	def test_resolve_by_real_path_suffix(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='report.pdf',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/downloads/report.pdf',
		)
		result = manifest.resolve('/tmp/downloads/report.pdf')
		assert result is not None
		assert result.display_name == 'report.pdf'

	def test_resolve_not_found(self):
		manifest = WorkspaceManifest()
		assert manifest.resolve('nonexistent.md') is None

	def test_remove(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='temp.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
		)
		assert manifest.get('temp.md') is not None
		manifest.remove('temp.md')
		assert manifest.get('temp.md') is None

	def test_summary_empty(self):
		manifest = WorkspaceManifest()
		assert manifest.summary() == ''

	def test_summary_with_entries(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			size_bytes=1024,
			readable=True,
			uploadable=True,
			last_action='write',
		)
		manifest.register(
			display_name='report.pdf',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/report.pdf',
			size_bytes=50000,
			readable=True,
			uploadable=True,
			last_action='download',
		)
		summary = manifest.summary()
		assert '<workspace_files>' in summary
		assert '</workspace_files>' in summary
		assert '[virtual]' in summary
		assert '[download]' in summary
		assert 'data.json' in summary
		assert 'report.pdf' in summary
		assert 'R/U' in summary
		assert '1.0KB' in summary
		assert '48.8KB' in summary

	def test_suggest_for_read_found_readable(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='test.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			readable=True,
		)
		assert manifest.suggest_for_read('test.md') is None

	def test_suggest_for_read_not_readable(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='secret.bin',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			readable=False,
		)
		hint = manifest.suggest_for_read('secret.bin')
		assert hint is not None
		assert 'not readable' in hint

	def test_suggest_for_read_fuzzy_match(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='results.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			readable=True,
		)
		hint = manifest.suggest_for_read('result.md')
		assert hint is not None
		assert 'results.md' in hint

	def test_suggest_for_read_no_match_lists_available(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			readable=True,
		)
		hint = manifest.suggest_for_read('missing.md')
		assert hint is not None
		assert 'data.json' in hint

	def test_suggest_for_upload_virtual_no_real_path(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='notes.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path=None,
			uploadable=False,
		)
		hint = manifest.suggest_for_upload('notes.md')
		assert hint is not None
		assert 'virtual file' in hint.lower()

	def test_suggest_for_upload_uploadable(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.csv',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/data.csv',
			uploadable=True,
		)
		assert manifest.suggest_for_upload('data.csv') is None

	def test_resolve_read_virtual_file(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='notes.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/fs/notes.md',
			readable=True,
		)
		entry, error = manifest.resolve_read('notes.md')
		assert entry is not None
		assert error is None
		assert entry.is_virtual is True

	def test_resolve_read_real_path(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='report.pdf',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/downloads/report.pdf',
			readable=True,
		)
		entry, error = manifest.resolve_read('/tmp/downloads/report.pdf')
		assert entry is not None
		assert error is None
		assert entry.real_path == '/tmp/downloads/report.pdf'

	def test_resolve_read_not_readable(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='secret.bin',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			readable=False,
		)
		entry, error = manifest.resolve_read('secret.bin')
		assert entry is not None
		assert error is not None
		assert 'not readable' in error

	def test_resolve_read_not_found(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			readable=True,
		)
		entry, error = manifest.resolve_read('missing.md')
		assert entry is None
		assert error is not None
		assert 'not found' in error
		assert 'data.json' in error

	def test_resolve_upload_virtual_with_path(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='output.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/fs/output.md',
			uploadable=True,
		)
		entry, error = manifest.resolve_upload('output.md')
		assert entry is not None
		assert error is None
		assert entry.real_path == '/tmp/fs/output.md'

	def test_resolve_upload_not_uploadable(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='step_1.png',
			source=FileSource.SCREENSHOT,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/screenshots/step_1.png',
			uploadable=False,
		)
		entry, error = manifest.resolve_upload('step_1.png')
		assert entry is not None
		assert error is not None
		assert 'not uploadable' in error

	def test_resolve_upload_not_found_with_suggestions(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.csv',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/data.csv',
			uploadable=True,
		)
		entry, error = manifest.resolve_upload('missing.csv')
		assert entry is None
		assert error is not None
		assert 'data.csv' in error

	def test_register_user_files(self):
		manifest = WorkspaceManifest()
		with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
			f.write(b'hello world')
			path = f.name
		try:
			manifest.register_user_files([path])
			entry = manifest.get(path)
			assert entry is not None
			assert entry.source == FileSource.USER_PROVIDED
			assert entry.uploadable is True
			assert entry.size_bytes > 0
		finally:
			os.unlink(path)

	def test_register_download(self):
		manifest = WorkspaceManifest()
		entry = manifest.register_download('/tmp/downloads/file.zip', size_bytes=1024)
		assert entry.source == FileSource.DOWNLOAD
		assert entry.display_name == 'file.zip'
		assert entry.uploadable is True
		assert entry.last_action == 'download'

	def test_register_screenshot(self):
		manifest = WorkspaceManifest()
		entry = manifest.register_screenshot('/tmp/screenshots/step_1.png', size_bytes=50000)
		assert entry.source == FileSource.SCREENSHOT
		assert entry.uploadable is False
		assert entry.last_action == 'screenshot'

	def test_register_pdf_save(self):
		manifest = WorkspaceManifest()
		entry = manifest.register_pdf_save('page.pdf', '/tmp/fs/page.pdf', size_bytes=30000)
		assert entry.source == FileSource.PDF_SAVE
		assert entry.uploadable is True
		assert entry.last_action == 'save_as_pdf'

	def test_state_roundtrip(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.json',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/fs/data.json',
			size_bytes=256,
			readable=True,
			uploadable=True,
			last_action='write',
		)
		manifest.register(
			display_name='report.pdf',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/downloads/report.pdf',
			size_bytes=50000,
			readable=True,
			uploadable=True,
			last_action='download',
		)
		state = manifest.get_state()
		assert isinstance(state, WorkspaceManifestState)
		assert len(state.entries) == 2

		restored = WorkspaceManifest.from_state(state)
		assert len(restored.all_entries()) == 2
		entry = restored.get('data.json')
		assert entry is not None
		assert entry.source == FileSource.VIRTUAL
		assert entry.size_bytes == 256

	def test_format_size(self):
		assert WorkspaceManifest._format_size(0) == '0B'
		assert WorkspaceManifest._format_size(500) == '500B'
		assert WorkspaceManifest._format_size(1024) == '1.0KB'
		assert WorkspaceManifest._format_size(1048576) == '1.0MB'

	def test_summary_enabled_flag(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='test.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/test.md',
			size_bytes=100,
			readable=True,
			uploadable=True,
		)
		assert manifest.summary(enabled=False) == ''
		assert '<workspace_files>' in manifest.summary(enabled=True)

	def test_summary_max_items(self):
		manifest = WorkspaceManifest()
		for i in range(10):
			manifest.register(
				display_name=f'file_{i}.md',
				source=FileSource.VIRTUAL,
				path_type=PathType.VIRTUAL_NAME,
				real_path=f'/tmp/file_{i}.md',
				size_bytes=100 * i,
				readable=True,
				uploadable=True,
			)
		summary = manifest.summary(max_items=5)
		assert '... and 5 more file(s)' in summary
		assert summary.count('[virtual]') == 5

	def test_summary_filter_by_source(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='virtual.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/virtual.md',
			size_bytes=100,
			readable=True,
			uploadable=True,
		)
		manifest.register(
			display_name='download.pdf',
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/download.pdf',
			size_bytes=5000,
			readable=True,
			uploadable=True,
		)
		summary_virtual = manifest.summary(include_sources=[FileSource.VIRTUAL])
		assert 'virtual.md' in summary_virtual
		assert 'download.pdf' not in summary_virtual

		summary_exclude_virtual = manifest.summary(exclude_sources=[FileSource.VIRTUAL])
		assert 'virtual.md' not in summary_exclude_virtual
		assert 'download.pdf' in summary_exclude_virtual

	def test_summary_show_full_paths(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='data.csv',
			source=FileSource.USER_PROVIDED,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/var/data/data.csv',
			size_bytes=200,
			readable=True,
			uploadable=True,
		)
		summary_short = manifest.summary(show_full_paths=False)
		assert 'data.csv' in summary_short
		assert '/var/data/data.csv' not in summary_short

		summary_full = manifest.summary(show_full_paths=True)
		assert '/var/data/data.csv' in summary_full

	def test_list_files_basic(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='test.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/test.md',
			size_bytes=150,
			readable=True,
			uploadable=True,
			last_action='write',
		)
		files = manifest.list_files()
		assert len(files) == 1
		assert files[0]['display_name'] == 'test.md'
		assert files[0]['source'] == 'virtual'
		assert files[0]['size_bytes'] == 150
		assert files[0]['readable'] is True
		assert files[0]['uploadable'] is True
		assert files[0]['last_action'] == 'write'

	def test_list_files_filters(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='read_only.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/ro.md',
			size_bytes=100,
			readable=True,
			uploadable=False,
		)
		manifest.register(
			display_name='upload_only.png',
			source=FileSource.SCREENSHOT,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/up.png',
			size_bytes=200,
			readable=False,
			uploadable=True,
		)
		manifest.register(
			display_name='internal.png',
			source=FileSource.SCREENSHOT,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path='/tmp/internal.png',
			size_bytes=300,
			readable=True,
			uploadable=False,
		)

		readable = manifest.list_files(readable_only=True)
		assert len(readable) == 2
		assert all(f['readable'] for f in readable)

		uploadable = manifest.list_files(uploadable_only=True)
		assert len(uploadable) == 1
		assert uploadable[0]['display_name'] == 'upload_only.png'

		virtual_only = manifest.list_files(include_sources=[FileSource.VIRTUAL])
		assert len(virtual_only) == 1
		assert virtual_only[0]['display_name'] == 'read_only.md'

		exclude_screenshot = manifest.list_files(exclude_sources=[FileSource.SCREENSHOT])
		assert len(exclude_screenshot) == 1
		assert exclude_screenshot[0]['display_name'] == 'read_only.md'

	def test_format_files_for_display(self):
		manifest = WorkspaceManifest()
		manifest.register(
			display_name='test.md',
			source=FileSource.VIRTUAL,
			path_type=PathType.VIRTUAL_NAME,
			real_path='/tmp/test.md',
			size_bytes=1024,
			readable=True,
			uploadable=True,
			last_action='write',
		)
		files = manifest.list_files()
		display = manifest.format_files_for_display(files)
		assert 'Workspace Files (1 total)' in display
		assert 'test.md' in display
		assert '1.0KB' in display
		assert 'R/U' in display
		assert 'last:write' in display

	def test_format_files_for_display_empty(self):
		manifest = WorkspaceManifest()
		display = manifest.format_files_for_display([])
		assert display == 'No files in workspace.'


class TestFileSystemManifestIntegration:
	@pytest.fixture
	def temp_filesystem(self):
		with tempfile.TemporaryDirectory() as tmp_dir:
			fs = FileSystem(base_dir=tmp_dir, create_default_files=True)
			yield fs
			try:
				fs.nuke()
			except Exception:
				pass

	def test_manifest_initialized(self, temp_filesystem):
		fs = temp_filesystem
		assert fs.manifest is not None
		todo_entry = fs.manifest.get('todo.md')
		assert todo_entry is not None
		assert todo_entry.source == FileSource.VIRTUAL

	async def test_write_file_registers_manifest(self, temp_filesystem):
		fs = temp_filesystem
		await fs.write_file('results.md', '# Test Results')
		entry = fs.manifest.get('results.md')
		assert entry is not None
		assert entry.source == FileSource.VIRTUAL
		assert entry.last_action == 'write'
		assert entry.uploadable is True

	async def test_append_file_updates_manifest(self, temp_filesystem):
		fs = temp_filesystem
		await fs.write_file('notes.txt', 'Hello')
		await fs.append_file('notes.txt', '\nWorld')
		entry = fs.manifest.get('notes.txt')
		assert entry is not None
		assert entry.last_action == 'append'

	async def test_replace_file_updates_manifest(self, temp_filesystem):
		fs = temp_filesystem
		await fs.write_file('test.md', 'old text')
		await fs.replace_file_str('test.md', 'old', 'new')
		entry = fs.manifest.get('test.md')
		assert entry is not None
		assert entry.last_action == 'replace'

	async def test_read_file_touches_manifest(self, temp_filesystem):
		fs = temp_filesystem
		await fs.write_file('data.json', '{"key": "value"}')
		await fs.read_file('data.json')
		entry = fs.manifest.get('data.json')
		assert entry is not None
		assert entry.last_action == 'read'

	async def test_save_extracted_content_registers_manifest(self, temp_filesystem):
		fs = temp_filesystem
		filename = await fs.save_extracted_content('Extracted data')
		entry = fs.manifest.get(filename)
		assert entry is not None
		assert entry.source == FileSource.EXTRACTED
		assert entry.last_action == 'extract'

	async def test_state_roundtrip_preserves_manifest(self, temp_filesystem):
		fs = temp_filesystem
		await fs.write_file('data.json', '{"test": true}')
		await fs.append_file('data.json', ', "more": false}')

		state = fs.get_state()
		assert state.manifest_state is not None
		assert 'data.json' in state.manifest_state

		fs2 = FileSystem.from_state(state)
		entry = fs2.manifest.get('data.json')
		assert entry is not None
		assert entry.source == FileSource.VIRTUAL
		assert entry.last_action == 'append'
		fs2.nuke()

	def test_manifest_summary_in_prompt(self, temp_filesystem):
		fs = temp_filesystem
		summary = fs.manifest.summary()
		assert 'todo.md' in summary
		assert '<workspace_files>' in summary
