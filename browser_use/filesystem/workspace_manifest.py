from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FileSource(str, Enum):
	VIRTUAL = 'virtual'
	DOWNLOAD = 'download'
	SCREENSHOT = 'screenshot'
	PDF_SAVE = 'pdf_save'
	USER_PROVIDED = 'user_provided'
	EXTRACTED = 'extracted'


class PathType(str, Enum):
	VIRTUAL_NAME = 'virtual_name'
	LOCAL_ABSOLUTE = 'local_absolute'
	REMOTE_ABSOLUTE = 'remote_absolute'


class ManifestEntry(BaseModel):
	display_name: str
	source: FileSource
	path_type: PathType
	real_path: str | None = None
	size_bytes: int = 0
	created_at: float = Field(default_factory=time.time)
	readable: bool = True
	uploadable: bool = False
	last_action: str | None = None
	last_action_at: float | None = None
	metadata: dict[str, Any] = Field(default_factory=dict)

	@property
	def is_virtual(self) -> bool:
		return self.path_type == PathType.VIRTUAL_NAME

	@property
	def lookup_key(self) -> str:
		if self.is_virtual:
			return self.display_name
		return self.real_path or self.display_name


class WorkspaceManifestState(BaseModel):
	entries: dict[str, dict[str, Any]] = Field(default_factory=dict)


class WorkspaceManifest:
	def __init__(self) -> None:
		self._entries: dict[str, ManifestEntry] = {}

	def register(
		self,
		display_name: str,
		source: FileSource,
		path_type: PathType,
		real_path: str | None = None,
		size_bytes: int = 0,
		readable: bool = True,
		uploadable: bool = False,
		last_action: str | None = None,
		metadata: dict[str, Any] | None = None,
	) -> ManifestEntry:
		key = real_path if (path_type != PathType.VIRTUAL_NAME and real_path) else display_name
		now = time.time()
		if key in self._entries:
			entry = self._entries[key]
			entry.size_bytes = size_bytes
			entry.last_action = last_action
			entry.last_action_at = now
			if metadata:
				entry.metadata.update(metadata)
			return entry

		entry = ManifestEntry(
			display_name=display_name,
			source=source,
			path_type=path_type,
			real_path=real_path,
			size_bytes=size_bytes,
			readable=readable,
			uploadable=uploadable,
			last_action=last_action,
			last_action_at=now,
			metadata=metadata or {},
		)
		self._entries[key] = entry
		return entry

	def touch(self, key: str, action: str) -> None:
		entry = self._entries.get(key)
		if entry:
			entry.last_action = action
			entry.last_action_at = time.time()

	def get(self, key: str) -> ManifestEntry | None:
		return self._entries.get(key)

	def resolve(self, name: str) -> ManifestEntry | None:
		if name in self._entries:
			return self._entries[name]
		for entry in self._entries.values():
			if entry.display_name == name:
				return entry
			if entry.real_path and (entry.real_path == name or entry.real_path.endswith(name)):
				return entry
		return None

	def resolve_read(self, name: str) -> tuple[ManifestEntry | None, str | None]:
		"""
		Resolve a file name for reading.
		Returns (entry, error_message).
		- If entry exists and is readable: (entry, None)
		- If entry exists but not readable: (entry, error_message)
		- If entry not found: (None, error_message with suggestions)
		"""
		entry = self.resolve(name)
		if entry:
			if entry.readable:
				return entry, None
			return entry, f"File '{entry.display_name}' is not readable (source: {entry.source.value})."

		hint = self._find_alternative(name, readable=True)
		if hint:
			return None, f"File '{name}' not found in workspace. {hint}"
		return None, f"File '{name}' not found and no readable files are available."

	def resolve_upload(self, name: str) -> tuple[ManifestEntry | None, str | None]:
		"""
		Resolve a file name for uploading.
		Returns (entry, error_message).
		- If entry exists and is uploadable: (entry, None)
		- If entry exists but not uploadable: (entry, error_message explaining why)
		- If entry not found: (None, error_message with suggestions)
		"""
		entry = self.resolve(name)
		if entry:
			if entry.uploadable:
				return entry, None
			if entry.is_virtual:
				if entry.real_path:
					return entry, None
				return (
					entry,
					f"File '{entry.display_name}' is a virtual file with no real disk path. "
					f'It can be read with read_file but cannot be uploaded directly.',
				)
			return entry, f"File '{entry.display_name}' is marked as not uploadable (source: {entry.source.value})."

		hint = self._find_alternative(name, uploadable=True)
		if hint:
			return None, f"File '{name}' not found in workspace. {hint}"
		return None, f"File '{name}' not found and no uploadable files are available."

	def _find_alternative(self, name: str, readable: bool = False, uploadable: bool = False) -> str | None:
		closest = self._fuzzy_match(name)
		if closest:
			return f"Did you mean '{closest.display_name}'?"
		available = [
			e.display_name
			for e in self.all_entries()
			if (not readable or e.readable) and (not uploadable or e.uploadable)
		]
		if available:
			return f'Available files: {", ".join(sorted(available))}'
		return None

	def remove(self, key: str) -> None:
		self._entries.pop(key, None)

	def all_entries(self) -> list[ManifestEntry]:
		return list(self._entries.values())

	def list_files(
		self,
		include_sources: list[FileSource] | None = None,
		exclude_sources: list[FileSource] | None = None,
		readable_only: bool = False,
		uploadable_only: bool = False,
	) -> list[dict[str, Any]]:
		"""
		List all files in the manifest as structured dictionaries, filtered by criteria.

		Args:
			include_sources: If set, only include files from these sources.
			exclude_sources: If set, exclude files from these sources.
			readable_only: If True, only include readable files.
			uploadable_only: If True, only include uploadable files.

		Returns:
			List of dicts with keys: display_name, source, path_type, real_path,
			size_bytes, size_human, readable, uploadable, last_action, created_at
		"""
		entries = self.all_entries()

		if include_sources:
			include_set = set(include_sources)
			entries = [e for e in entries if e.source in include_set]
		if exclude_sources:
			exclude_set = set(exclude_sources)
			entries = [e for e in entries if e.source not in exclude_set]
		if readable_only:
			entries = [e for e in entries if e.readable]
		if uploadable_only:
			entries = [e for e in entries if e.uploadable]

		return [
			{
				'display_name': e.display_name,
				'source': e.source.value,
				'path_type': e.path_type.value,
				'real_path': e.real_path,
				'size_bytes': e.size_bytes,
				'size_human': self._format_size(e.size_bytes),
				'readable': e.readable,
				'uploadable': e.uploadable,
				'last_action': e.last_action,
				'created_at': e.created_at,
			}
			for e in sorted(entries, key=lambda e: e.created_at, reverse=True)
		]

	def format_files_for_display(self, files: list[dict[str, Any]], show_full_paths: bool = True) -> str:
		"""
		Format a list of files from list_files() for human-readable display.
		"""
		if not files:
			return 'No files in workspace.'

		lines = []
		lines.append(f'Workspace Files ({len(files)} total):')
		lines.append('-' * 80)
		for i, f in enumerate(files, 1):
			flags = []
			if f['readable']:
				flags.append('R')
			if f['uploadable']:
				flags.append('U')
			flag_str = '/'.join(flags) if flags else '-'

			path = f['display_name']
			if show_full_paths and f['real_path'] and f['path_type'] != 'virtual_name':
				path = f['real_path']

			lines.append(
				f'{i:2d}. [{f["source"]:<10}] {path:<40} '
				f'{f["size_human"]:>8} {flag_str:>3} '
				f'last:{f["last_action"] or "-"}'
			)
		return '\n'.join(lines)

	def summary(
		self,
		enabled: bool = True,
		max_items: int = 20,
		include_sources: list[FileSource] | None = None,
		exclude_sources: list[FileSource] | None = None,
		show_full_paths: bool = False,
	) -> str:
		"""
		Generate a workspace file summary for display in the agent prompt.

		Args:
			enabled: Whether to generate the summary at all.
			max_items: Maximum number of files to show; newer files are prioritized.
			include_sources: If set, only include files from these sources.
			exclude_sources: If set, exclude files from these sources.
			show_full_paths: If True, show full real paths for local absolute files.
		"""
		if not enabled:
			return ''

		entries = self.all_entries()
		if not entries:
			return ''

		# Filter by sources
		if include_sources:
			include_set = set(include_sources)
			entries = [e for e in entries if e.source in include_set]
		if exclude_sources:
			exclude_set = set(exclude_sources)
			entries = [e for e in entries if e.source not in exclude_set]

		if not entries:
			return ''

		# Sort by creation time (newest first) and truncate
		entries = sorted(entries, key=lambda e: e.created_at, reverse=True)
		total_count = len(entries)
		if max_items > 0 and len(entries) > max_items:
			entries = entries[:max_items]
			truncated = True
		else:
			truncated = False

		lines = ['<workspace_files>']
		for entry in entries:
			src = entry.source.value
			flags = []
			if entry.readable:
				flags.append('R')
			if entry.uploadable:
				flags.append('U')
			flag_str = '/'.join(flags) if flags else '-'
			size_str = self._format_size(entry.size_bytes)
			display_path = entry.display_name
			if show_full_paths and entry.real_path and not entry.is_virtual:
				display_path = entry.real_path
			line = f'  [{src}] {display_path} ({size_str}, {flag_str})'
			if entry.last_action:
				line += f' last:{entry.last_action}'
			lines.append(line)

		if truncated:
			lines.append(f'  ... and {total_count - max_items} more file(s). Use list_files to see all.')

		lines.append('</workspace_files>')
		return '\n'.join(lines)

	@staticmethod
	def _format_size(n: int) -> str:
		if n < 1024:
			return f'{n}B'
		elif n < 1024 * 1024:
			return f'{n / 1024:.1f}KB'
		else:
			return f'{n / (1024 * 1024):.1f}MB'

	def suggest_for_read(self, name: str) -> str | None:
		entry = self.resolve(name)
		if entry:
			if entry.readable:
				return None
			return f"File '{entry.display_name}' exists but is not readable (source: {entry.source.value})."
		closest = self._fuzzy_match(name)
		if closest:
			return f"File '{name}' not found. Did you mean '{closest.display_name}'?"
		available = [e.display_name for e in self.all_entries() if e.readable]
		if available:
			return f"File '{name}' not found. Available readable files: {', '.join(available)}"
		return f"File '{name}' not found and no readable files are available in the workspace."

	def suggest_for_upload(self, name: str) -> str | None:
		entry = self.resolve(name)
		if entry:
			if entry.uploadable:
				return None
			if entry.is_virtual:
				real = entry.real_path
				if real:
					return None
				return (
					f"File '{entry.display_name}' is a virtual file with no real disk path. "
					f'It can be read with read_file but cannot be uploaded directly. '
					f'Write it to a local path first or use the virtual name with upload_file.'
				)
			return f"File '{entry.display_name}' is not uploadable (source: {entry.source.value})."
		uploadable = [e.display_name for e in self.all_entries() if e.uploadable]
		if uploadable:
			return f"File '{name}' not found. Uploadable files: {', '.join(uploadable)}"
		return f"File '{name}' not found and no uploadable files are available in the workspace."

	def _fuzzy_match(self, name: str) -> ManifestEntry | None:
		name_lower = name.lower()
		for entry in self._entries.values():
			if entry.display_name.lower() == name_lower:
				return entry
		for entry in self._entries.values():
			if name_lower in entry.display_name.lower() or entry.display_name.lower() in name_lower:
				return entry
		return None

	def get_state(self) -> WorkspaceManifestState:
		data = {}
		for key, entry in self._entries.items():
			data[key] = entry.model_dump()
		return WorkspaceManifestState(entries=data)

	@classmethod
	def from_state(cls, state: WorkspaceManifestState) -> WorkspaceManifest:
		manifest = cls()
		for key, entry_data in state.entries.items():
			manifest._entries[key] = ManifestEntry(**entry_data)
		return manifest

	def register_user_files(self, file_paths: list[str]) -> None:
		for path in file_paths:
			import os

			if os.path.exists(path):
				size = os.path.getsize(path)
			else:
				size = 0
			self.register(
				display_name=os.path.basename(path),
				source=FileSource.USER_PROVIDED,
				path_type=PathType.LOCAL_ABSOLUTE,
				real_path=path,
				size_bytes=size,
				readable=True,
				uploadable=True,
			)

	def register_download(self, file_path: str, size_bytes: int = 0) -> ManifestEntry:
		import os

		return self.register(
			display_name=os.path.basename(file_path),
			source=FileSource.DOWNLOAD,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path=file_path,
			size_bytes=size_bytes,
			readable=True,
			uploadable=True,
			last_action='download',
		)

	def register_screenshot(self, file_path: str, size_bytes: int = 0) -> ManifestEntry:
		import os

		return self.register(
			display_name=os.path.basename(file_path),
			source=FileSource.SCREENSHOT,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path=file_path,
			size_bytes=size_bytes,
			readable=True,
			uploadable=False,
			last_action='screenshot',
		)

	def register_pdf_save(self, display_name: str, real_path: str, size_bytes: int = 0) -> ManifestEntry:
		return self.register(
			display_name=display_name,
			source=FileSource.PDF_SAVE,
			path_type=PathType.LOCAL_ABSOLUTE,
			real_path=real_path,
			size_bytes=size_bytes,
			readable=True,
			uploadable=True,
			last_action='save_as_pdf',
		)
