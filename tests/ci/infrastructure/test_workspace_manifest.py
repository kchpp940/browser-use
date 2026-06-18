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
