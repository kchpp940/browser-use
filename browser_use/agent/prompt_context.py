from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from browser_use.browser.views import PLACEHOLDER_4PX_SCREENSHOT
from browser_use.dom.views import NodeType, SimplifiedNode
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL
from browser_use.utils import is_new_tab_page, sanitize_surrogates

if TYPE_CHECKING:
	from browser_use.agent.views import AgentStepInfo
	from browser_use.browser.views import BrowserStateSummary
	from browser_use.filesystem.file_system import FileSystem

logger = logging.getLogger(__name__)


@dataclass
class PromptSectionContext:
	browser_state_summary: 'BrowserStateSummary' | None = None
	file_system: 'FileSystem | None' = None
	agent_history_description: str | None = None
	read_state_description: str | None = None
	task: str | None = None
	include_attributes: list[str] | None = None
	step_info: 'AgentStepInfo | None' = None
	page_filtered_actions: str | None = None
	max_clickable_elements_length: int = 40000
	sensitive_data: str | None = None
	available_file_paths: list[str] | None = None
	screenshots: list[str] | None = None
	vision_detail_level: str = 'auto'
	include_recent_events: bool = False
	sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None
	read_state_images: list[dict] | None = None
	llm_screenshot_size: tuple[int, int] | None = None
	unavailable_skills_info: str | None = None
	plan_description: str | None = None


class PromptSection(ABC):
	name: str
	enabled: bool = True
	wrap_tag: str | None = None

	def __init__(self, name: str, enabled: bool = True, wrap_tag: str | None = None):
		self.name = name
		self.enabled = enabled
		self.wrap_tag = wrap_tag

	@abstractmethod
	def render(self, ctx: PromptSectionContext) -> str | None:
		"""Render this section as text, or None to skip."""
		...

	def render_wrapped(self, ctx: PromptSectionContext) -> str | None:
		"""Render with optional XML tag wrapping."""
		if not self.enabled:
			return None
		content = self.render(ctx)
		if content is None:
			return None
		content = content.strip('\n')
		if self.wrap_tag:
			return f'<{self.wrap_tag}>\n{content}\n</{self.wrap_tag}>'
		return content


class UserRequestSection(PromptSection):
	def __init__(self):
		super().__init__(name='user_request', wrap_tag='user_request')

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not ctx.task:
			return None
		return ctx.task


class AgentHistorySection(PromptSection):
	def __init__(self):
		super().__init__(name='agent_history', wrap_tag='agent_history')

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not ctx.agent_history_description:
			return ''
		return ctx.agent_history_description.strip('\n')


class FileSystemSection(PromptSection):
	def __init__(self):
		super().__init__(name='file_system')

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not ctx.file_system:
			return None
		_todo_contents = ctx.file_system.get_todo_contents()
		if not len(_todo_contents):
			_todo_contents = '[empty todo.md, fill it when applicable]'
		return f"""<file_system>
{ctx.file_system.describe()}
</file_system>
<todo_contents>
{_todo_contents}
</todo_contents>"""


class PlanSection(PromptSection):
	def __init__(self):
		super().__init__(name='plan', wrap_tag='plan')

	def render(self, ctx: PromptSectionContext) -> str | None:
		return ctx.plan_description


class SensitiveDataSection(PromptSection):
	def __init__(self):
		super().__init__(name='sensitive_data', wrap_tag='sensitive_data')

	def render(self, ctx: PromptSectionContext) -> str | None:
		return ctx.sensitive_data


class AvailableFilePathsSection(PromptSection):
	def __init__(self):
		super().__init__(name='available_file_paths', wrap_tag='available_file_paths')

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not ctx.available_file_paths:
			return None
		available_file_paths_text = '\n'.join(ctx.available_file_paths)
		return f'{available_file_paths_text}\nUse with absolute paths'


class AgentStateGroupSection(PromptSection):
	"""Groups FileSystemSection, PlanSection, SensitiveDataSection, AvailableFilePathsSection
	inside a single <agent_state> wrapper."""

	def __init__(self):
		super().__init__(name='agent_state_group', wrap_tag='agent_state')
		self.sub_sections: list[PromptSection] = [
			FileSystemSection(),
			PlanSection(),
			SensitiveDataSection(),
			AvailableFilePathsSection(),
		]

	def get_section(self, name: str) -> PromptSection | None:
		for s in self.sub_sections:
			if s.name == name:
				return s
		return None

	def render(self, ctx: PromptSectionContext) -> str | None:
		parts: list[str] = []
		for sub in self.sub_sections:
			rendered = sub.render_wrapped(ctx)
			if rendered is not None:
				parts.append(rendered)
		if not parts:
			return None
		return '\n'.join(parts)


class BrowserStateSection(PromptSection):
	def __init__(self):
		super().__init__(name='browser_state', wrap_tag='browser_state')

	def _extract_page_statistics(self, ctx: PromptSectionContext) -> dict[str, int]:
		stats = {
			'links': 0,
			'iframes': 0,
			'shadow_open': 0,
			'shadow_closed': 0,
			'scroll_containers': 0,
			'images': 0,
			'interactive_elements': 0,
			'total_elements': 0,
			'text_chars': 0,
		}
		if not ctx.browser_state_summary or not ctx.browser_state_summary.dom_state or not ctx.browser_state_summary.dom_state._root:
			return stats

		def traverse_node(node: SimplifiedNode) -> None:
			if not node or not node.original_node:
				return
			original = node.original_node
			stats['total_elements'] += 1
			if original.node_type == NodeType.ELEMENT_NODE:
				tag = original.tag_name.lower() if original.tag_name else ''
				if tag == 'a':
					stats['links'] += 1
				elif tag in ('iframe', 'frame'):
					stats['iframes'] += 1
				elif tag == 'img':
					stats['images'] += 1
				if original.is_actually_scrollable:
					stats['scroll_containers'] += 1
				if node.is_interactive:
					stats['interactive_elements'] += 1
				if node.is_shadow_host:
					has_closed_shadow = any(
						child.original_node.node_type == NodeType.DOCUMENT_FRAGMENT_NODE
						and child.original_node.shadow_root_type
						and child.original_node.shadow_root_type.lower() == 'closed'
						for child in node.children
					)
					if has_closed_shadow:
						stats['shadow_closed'] += 1
					else:
						stats['shadow_open'] += 1
			elif original.node_type == NodeType.TEXT_NODE:
				stats['text_chars'] += len(original.node_value.strip())
			for child in node.children:
				traverse_node(child)

		traverse_node(ctx.browser_state_summary.dom_state._root)
		return stats

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not ctx.browser_state_summary:
			return None
		bs = ctx.browser_state_summary
		page_stats = self._extract_page_statistics(ctx)

		stats_text = '<page_stats>'
		if page_stats['total_elements'] < 10:
			stats_text += 'Page appears empty (SPA not loaded?) - '
		elif page_stats['total_elements'] > 20 and page_stats['text_chars'] < page_stats['total_elements'] * 5:
			stats_text += 'Page appears to show skeleton/placeholder content (still loading?) - '
		stats_text += f'{page_stats["links"]} links, {page_stats["interactive_elements"]} interactive, '
		stats_text += f'{page_stats["iframes"]} iframes'
		if page_stats['shadow_open'] > 0 or page_stats['shadow_closed'] > 0:
			stats_text += f', {page_stats["shadow_open"]} shadow(open), {page_stats["shadow_closed"]} shadow(closed)'
		if page_stats['images'] > 0:
			stats_text += f', {page_stats["images"]} images'
		stats_text += f', {page_stats["total_elements"]} total elements'
		stats_text += '</page_stats>\n'

		elements_text = bs.dom_state.llm_representation(include_attributes=ctx.include_attributes)
		if len(elements_text) > ctx.max_clickable_elements_length:
			elements_text = elements_text[: ctx.max_clickable_elements_length]
			truncated_text = f' (truncated to {ctx.max_clickable_elements_length} characters)'
		else:
			truncated_text = ''

		has_content_above = False
		has_content_below = False
		page_info_text = ''
		if bs.page_info:
			pi = bs.page_info
			pages_above = pi.pixels_above / pi.viewport_height if pi.viewport_height > 0 else 0
			pages_below = pi.pixels_below / pi.viewport_height if pi.viewport_height > 0 else 0
			has_content_above = pages_above > 0
			has_content_below = pages_below > 0
			page_info_text = '<page_info>'
			page_info_text += f'{pages_above:.1f} pages above, {pages_below:.1f} pages below'
			if pages_below > 0.2:
				page_info_text += ' — scroll down to reveal more content'
			page_info_text += '</page_info>\n'

		if elements_text != '':
			if not has_content_above:
				elements_text = f'[Start of page]\n{elements_text}'
			if not has_content_below:
				elements_text = f'{elements_text}\n[End of page]'
		else:
			elements_text = 'empty page'

		tabs_text = ''
		current_tab_candidates = []
		for tab in bs.tabs:
			if tab.url == bs.url and tab.title == bs.title:
				current_tab_candidates.append(tab.target_id)
		current_target_id = current_tab_candidates[0] if len(current_tab_candidates) == 1 else None

		for tab in bs.tabs:
			tabs_text += f'Tab {tab.target_id[-4:]}: {tab.url} - {tab.title[:30]}\n'

		current_tab_text = f'Current tab: {current_target_id[-4:]}' if current_target_id is not None else ''

		pdf_message = ''
		if bs.is_pdf_viewer:
			pdf_message = (
				'PDF viewer cannot be rendered. In this page, DO NOT use the extract action as PDF content cannot be rendered. '
			)
			pdf_message += (
				'Use the read_file action on the downloaded PDF in available_file_paths to read the full text content.\n\n'
			)

		recent_events_text = ''
		if ctx.include_recent_events and bs.recent_events:
			recent_events_text = f'Recent browser events: {bs.recent_events}\n'

		closed_popups_text = ''
		if bs.closed_popup_messages:
			closed_popups_text = 'Auto-closed JavaScript dialogs:\n'
			for popup_msg in bs.closed_popup_messages:
				closed_popups_text += f'  - {popup_msg}\n'
			closed_popups_text += '\n'

		return f"""{stats_text}{current_tab_text}
Available tabs:
{tabs_text}
{page_info_text}
{recent_events_text}{closed_popups_text}{pdf_message}Interactive elements{truncated_text}:
{elements_text}
"""


class ReadStateSection(PromptSection):
	def __init__(self):
		super().__init__(name='read_state', wrap_tag='read_state')

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not ctx.read_state_description:
			return None
		stripped = ctx.read_state_description.strip('\n').strip()
		if not stripped:
			return None
		return stripped


class PageSpecificActionsSection(PromptSection):
	def __init__(self):
		super().__init__(name='page_specific_actions', wrap_tag='page_specific_actions')

	def render(self, ctx: PromptSectionContext) -> str | None:
		return ctx.page_filtered_actions


class UnavailableSkillsSection(PromptSection):
	def __init__(self):
		super().__init__(name='unavailable_skills')

	def render(self, ctx: PromptSectionContext) -> str | None:
		return ctx.unavailable_skills_info


class StepMetaSection(PromptSection):
	def __init__(self):
		super().__init__(name='step_meta', wrap_tag='step_info')

	def render(self, ctx: PromptSectionContext) -> str | None:
		step_info_description = ''
		if ctx.step_info:
			step_info_description = f'Step{ctx.step_info.step_number + 1} maximum:{ctx.step_info.max_steps}\n'
		step_info_description += f'Today:{datetime.now().strftime("%Y-%m-%d")}'
		return step_info_description


class ContextMessageSection(PromptSection):
	"""Section for free-form context messages (nudges, warnings, etc.)."""

	def __init__(self):
		super().__init__(name='context_messages')
		self._messages: list[str] = []

	def add_message(self, msg: str) -> None:
		self._messages.append(msg)

	def clear(self) -> None:
		self._messages.clear()

	def render(self, ctx: PromptSectionContext) -> str | None:
		if not self._messages:
			return None
		return '\n\n'.join(self._messages)


@dataclass
class PromptImage:
	label: str
	base64_data: str
	media_type: str = 'image/png'


class PromptContextBuilder:
	"""Assembles the agent prompt from ordered, toggleable sections."""

	sections: list[PromptSection]
	context: PromptSectionContext
	images: list[PromptImage] = field(default_factory=list)
	sample_images: list[ContentPartTextParam | ContentPartImageParam] = field(default_factory=list)

	def __init__(self):
		self.sections: list[PromptSection] = [
			UserRequestSection(),
			AgentHistorySection(),
			AgentStateGroupSection(),
			BrowserStateSection(),
			ReadStateSection(),
			PageSpecificActionsSection(),
			UnavailableSkillsSection(),
			ContextMessageSection(),
			StepMetaSection(),
		]
		self.context = PromptSectionContext()
		self.images = []
		self.sample_images = []

	def get_section(self, name: str) -> PromptSection | None:
		for s in self.sections:
			if s.name == name:
				return s
			if isinstance(s, AgentStateGroupSection):
				sub = s.get_section(name)
				if sub is not None:
					return sub
		return None

	def enable_section(self, name: str) -> None:
		s = self.get_section(name)
		if s is not None:
			s.enabled = True

	def disable_section(self, name: str) -> None:
		s = self.get_section(name)
		if s is not None:
			s.enabled = False

	def set_context(self, **kwargs: Any) -> None:
		for key, value in kwargs.items():
			if hasattr(self.context, key):
				setattr(self.context, key, value)

	def add_context_message(self, msg: str) -> None:
		section = self.get_section('context_messages')
		if isinstance(section, ContextMessageSection):
			section.add_message(msg)

	def clear_context_messages(self) -> None:
		section = self.get_section('context_messages')
		if isinstance(section, ContextMessageSection):
			section.clear()

	def build_text(self) -> str:
		"""Build the complete text prompt from all enabled sections."""
		parts: list[str] = []
		for section in self.sections:
			rendered = section.render_wrapped(self.context)
			if rendered is not None:
				parts.append(rendered)
		full_text = '\n\n'.join(parts)
		if full_text:
			full_text += '\n'
		return sanitize_surrogates(full_text)

	def add_screenshot(self, base64_data: str, label: str = 'Current screenshot:') -> None:
		self.images.append(PromptImage(label=label, base64_data=base64_data))

	def add_read_state_image(self, base64_data: str, name: str) -> None:
		media_type = 'image/png' if name.lower().endswith('.png') else 'image/jpeg'
		self.images.append(PromptImage(label=f'Image from file: {name}', base64_data=base64_data, media_type=media_type))

	def _resize_screenshot(self, screenshot_b64: str, llm_screenshot_size: tuple[int, int] | None) -> str:
		if not llm_screenshot_size:
			return screenshot_b64
		try:
			import base64
			from io import BytesIO

			from PIL import Image

			img = Image.open(BytesIO(base64.b64decode(screenshot_b64)))
			if img.size == llm_screenshot_size:
				return screenshot_b64
			logger.info(
				f'🔄 Resizing screenshot from {img.size[0]}x{img.size[1]} to {llm_screenshot_size[0]}x{llm_screenshot_size[1]} for LLM'
			)
			img_resized = img.resize(llm_screenshot_size, Image.Resampling.LANCZOS)
			buffer = BytesIO()
			img_resized.save(buffer, format='PNG')
			return base64.b64encode(buffer.getvalue()).decode('utf-8')
		except Exception as e:
			logger.warning(f'Failed to resize screenshot: {e}, using original')
			return screenshot_b64

	def build_content_parts(
		self,
		use_vision: bool,
		screenshots: list[str],
		read_state_images: list[dict[str, Any]],
		sample_images: list[ContentPartTextParam | ContentPartImageParam],
		vision_detail_level: str,
		llm_screenshot_size: tuple[int, int] | None,
	) -> str | list[ContentPartTextParam | ContentPartImageParam]:
		"""Build the full content: text + images (if vision)."""
		text = self.build_text()

		if self.context.browser_state_summary and is_new_tab_page(self.context.browser_state_summary.url):
			use_vision = False

		valid_screenshots = [s for s in screenshots if s != PLACEHOLDER_4PX_SCREENSHOT]
		has_images = bool(read_state_images)

		if (use_vision and valid_screenshots) or has_images:
			content_parts: list[ContentPartTextParam | ContentPartImageParam] = [ContentPartTextParam(text=text)]
			content_parts.extend(sample_images)

			for i, screenshot in enumerate(valid_screenshots):
				label = 'Current screenshot:' if i == len(valid_screenshots) - 1 else 'Previous screenshot:'
				content_parts.append(ContentPartTextParam(text=label))
				processed = self._resize_screenshot(screenshot, llm_screenshot_size)
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{processed}',
							media_type='image/png',
							detail=vision_detail_level,
						),
					)
				)

			for img_data in read_state_images:
				img_name = img_data.get('name', 'unknown')
				img_base64 = img_data.get('data', '')
				if not img_base64:
					continue
				media_type = 'image/png' if img_name.lower().endswith('.png') else 'image/jpeg'
				content_parts.append(ContentPartTextParam(text=f'Image from file: {img_name}'))
				content_parts.append(
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:{media_type};base64,{img_base64}',
							media_type=media_type,
							detail=vision_detail_level,
						),
					)
				)

			return content_parts

		return text
