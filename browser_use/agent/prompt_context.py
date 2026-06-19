from __future__ import annotations

import importlib.resources
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from browser_use.browser.views import PLACEHOLDER_4PX_SCREENSHOT
from browser_use.dom.views import NodeType, SimplifiedNode
from browser_use.llm.messages import (
	ContentPartImageParam,
	ContentPartTextParam,
	ImageURL,
	SystemMessage,
	UserMessage,
)
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


# =============================================================================
# Prompt Section Registry
# =============================================================================


class PromptSectionRegistry:
	"""Central registry for prompt sections — both user-state and system-prompt.

	Instead of hardcoding section lists inside builders, all sections are
	registered here with an explicit ordering.  Builders consume the registry
	to build their section lists.  Adding a new section only requires calling
	``registry.register_user_section(MyNewSection())`` (or the system
	equivalent); no builder code needs to change.
	"""

	def __init__(self) -> None:
		self._user_entries: list[tuple[int, PromptSection]] = []
		self._system_entries: list[tuple[int, SystemPromptSection]] = []

	def register_user_section(self, section: PromptSection, order: int = 0) -> None:
		self._user_entries.append((order, section))

	def register_system_section(self, section: SystemPromptSection, order: int = 0) -> None:
		self._system_entries.append((order, section))

	def build_user_sections(self) -> list[PromptSection]:
		self._user_entries.sort(key=lambda e: e[0])
		return [s for _, s in self._user_entries]

	def build_system_sections(self) -> list[SystemPromptSection]:
		self._system_entries.sort(key=lambda e: e[0])
		return [s for _, s in self._system_entries]

	def user_section_names(self) -> list[str]:
		return [s.name for _, s in self._user_entries]

	def system_section_names(self) -> list[str]:
		return [s.name for _, s in self._system_entries]

	def get_user_section(self, name: str) -> PromptSection | None:
		for _, s in self._user_entries:
			if s.name == name:
				return s
			if isinstance(s, AgentStateGroupSection):
				sub = s.get_section(name)
				if sub is not None:
					return sub
		return None

	def get_system_section(self, name: str) -> SystemPromptSection | None:
		for _, s in self._system_entries:
			if s.name == name:
				return s
		return None


def default_registry() -> PromptSectionRegistry:
	"""Build the default registry containing all built-in sections.

	Sections are ordered so that the assembled prompt matches the original
	hardcoded layout exactly.
	"""
	reg = PromptSectionRegistry()

	reg.register_user_section(UserRequestSection(), order=100)
	reg.register_user_section(AgentHistorySection(), order=200)
	reg.register_user_section(AgentStateGroupSection(), order=300)
	reg.register_user_section(BrowserStateSection(), order=400)
	reg.register_user_section(ReadStateSection(), order=500)
	reg.register_user_section(PageSpecificActionsSection(), order=600)
	reg.register_user_section(UnavailableSkillsSection(), order=700)
	reg.register_user_section(ContextMessageSection(), order=800)
	reg.register_user_section(StepMetaSection(), order=900)

	reg.register_system_section(SystemCoreTemplateSection(), order=100)
	reg.register_system_section(SystemExtendMessageSection(), order=200)

	return reg


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

	def __init__(self, registry: PromptSectionRegistry | None = None):
		if registry is not None:
			self.sections = registry.build_user_sections()
		else:
			self.sections = default_registry().build_user_sections()
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

	def build_state_and_context_messages(
		self,
		use_vision: bool,
		screenshots: list[str],
		read_state_images: list[dict[str, Any]],
		sample_images: list[ContentPartTextParam | ContentPartImageParam],
		vision_detail_level: str,
		llm_screenshot_size: tuple[int, int] | None,
	) -> tuple[UserMessage, list[UserMessage]]:
		"""Build the main state message and any follow-up context messages (nudges)."""
		state_content = self.build_content_parts(
			use_vision=use_vision,
			screenshots=screenshots,
			read_state_images=read_state_images,
			sample_images=sample_images,
			vision_detail_level=vision_detail_level,
			llm_screenshot_size=llm_screenshot_size,
		)
		state_message = UserMessage(content=state_content, cache=True)

		context_messages: list[UserMessage] = []
		ctx_msg_section = self.get_section('context_messages')
		if isinstance(ctx_msg_section, ContextMessageSection):
			for msg in ctx_msg_section._messages:
				context_messages.append(UserMessage(content=msg))

		return state_message, context_messages

	def nudge_budget_warning(self, steps_used: int, max_steps: int, steps_remaining: int) -> None:
		"""Inject budget warning nudge."""
		pct = int(steps_used / max_steps * 100)
		msg = (
			f'BUDGET WARNING: You have used {steps_used}/{max_steps} steps '
			f'({pct}%). {steps_remaining} steps remaining. '
			f'If the task cannot be completed in the remaining steps, prioritize: '
			f'(1) consolidate your results (save to files if the file system is in use), '
			f'(2) call done with what you have. '
			f'Partial results are far more valuable than exhausting all steps with nothing saved.'
		)
		self.add_context_message(msg)

	def nudge_replan(self, consecutive_failures: int) -> None:
		"""Inject replan nudge after consecutive failures."""
		msg = (
			f'REPLAN SUGGESTED: You have failed {consecutive_failures} consecutive times. '
			'Your current plan may need revision. '
			'Output a new `plan_update` with revised steps to recover.'
		)
		self.add_context_message(msg)

	def nudge_exploration(self, n_steps: int) -> None:
		"""Inject planning nudge after exploring without a plan."""
		msg = (
			f'PLANNING NUDGE: You have taken {n_steps} steps without creating a plan. '
			'If the task is complex, output a `plan_update` with clear todo items now. '
			'If the task is already done or nearly done, call `done` instead.'
		)
		self.add_context_message(msg)

	def nudge_loop_detection(self, message: str) -> None:
		"""Inject loop detection nudge."""
		self.add_context_message(message)

	def nudge_last_step(self, max_steps: int) -> None:
		"""Inject last-step warning."""
		msg = (
			f'You reached max_steps ({max_steps}) - this is your last step. '
			'Your only tool available is the "done" tool. No other tool is available. '
			'All other tools which you see in history or examples are not available.'
			'\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! '
			'E.g. if not all steps are fully completed. Else success to true.'
			'\nInclude everything you found out for the ultimate task in the done text.'
		)
		self.add_context_message(msg)

	def nudge_force_done(self, max_failures: int) -> None:
		"""Inject force-done after max failures."""
		msg = (
			f'You failed {max_failures} times. Therefore we terminate the agent.'
			'\nYour only tool available is the "done" tool. No other tool is available. '
			'All other tools which you see in history or examples are not available.'
			'\nIf the task is not yet fully finished as requested by the user, set success in "done" to false! '
			'E.g. if not all steps are fully completed. Else success to true.'
			'\nInclude everything you found out for the ultimate task in the done text.'
		)
		self.add_context_message(msg)


# =============================================================================
# System Prompt Section System
# =============================================================================


@dataclass
class SystemPromptSectionContext:
	"""Data available for system prompt section rendering."""

	max_actions_per_step: int = 3
	use_thinking: bool = True
	flash_mode: bool = False
	is_anthropic: bool = False
	is_browser_use_model: bool = False
	model_name: str | None = None
	is_anthropic_4_5: bool = False
	override_system_message: str | None = None
	extend_system_message: str | None = None


class SystemPromptSection(ABC):
	"""Base class for system prompt sections."""

	name: str
	enabled: bool = True

	def __init__(self, name: str, enabled: bool = True):
		self.name = name
		self.enabled = enabled

	@abstractmethod
	def render(self, ctx: SystemPromptSectionContext) -> str | None:
		...


class SystemCoreTemplateSection(SystemPromptSection):
	"""Loads the core system prompt template from markdown based on provider/mode."""

	def __init__(self):
		super().__init__(name='core_template')

	def _select_template_filename(self, ctx: SystemPromptSectionContext) -> str:
		if ctx.is_browser_use_model:
			if ctx.flash_mode:
				return 'system_prompt_browser_use_flash.md'
			elif ctx.use_thinking:
				return 'system_prompt_browser_use.md'
			else:
				return 'system_prompt_browser_use_no_thinking.md'
		elif ctx.is_anthropic_4_5 and ctx.flash_mode:
			return 'system_prompt_anthropic_flash.md'
		elif ctx.flash_mode and ctx.is_anthropic:
			return 'system_prompt_flash_anthropic.md'
		elif ctx.flash_mode:
			return 'system_prompt_flash.md'
		elif ctx.use_thinking:
			return 'system_prompt.md'
		else:
			return 'system_prompt_no_thinking.md'

	def render(self, ctx: SystemPromptSectionContext) -> str | None:
		if ctx.override_system_message is not None:
			return ctx.override_system_message

		filename = self._select_template_filename(ctx)
		try:
			with (
				importlib.resources.files('browser_use.agent.system_prompts')
				.joinpath(filename)
				.open('r', encoding='utf-8') as f
			):
				template = f.read()
			return template.format(max_actions=ctx.max_actions_per_step)
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template {filename}: {e}')


class SystemExtendMessageSection(SystemPromptSection):
	"""Appends user-provided extend_system_message."""

	def __init__(self):
		super().__init__(name='extend_message')

	def render(self, ctx: SystemPromptSectionContext) -> str | None:
		return ctx.extend_system_message


class SystemPromptBuilder:
	"""Assembles the system prompt from ordered, toggleable sections."""

	sections: list[SystemPromptSection]
	context: SystemPromptSectionContext

	def __init__(self, registry: PromptSectionRegistry | None = None):
		if registry is not None:
			self.sections = registry.build_system_sections()
		else:
			self.sections = default_registry().build_system_sections()
		self.context = SystemPromptSectionContext()

	def get_section(self, name: str) -> SystemPromptSection | None:
		for s in self.sections:
			if s.name == name:
				return s
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

	def build_text(self) -> str:
		parts: list[str] = []
		for section in self.sections:
			if not section.enabled:
				continue
			rendered = section.render(self.context)
			if rendered is not None and rendered.strip():
				parts.append(rendered.strip('\n'))
		return '\n'.join(parts)

	def build_system_message(self) -> SystemMessage:
		return SystemMessage(content=self.build_text(), cache=True)


def _is_anthropic_4_5_model(model_name: str | None) -> bool:
	"""Check if the model is Claude Opus 4.5 or Haiku 4.5 (requires 4096+ token prompts for caching)."""
	if not model_name:
		return False
	model_lower = model_name.lower()
	is_opus_4_5 = 'opus' in model_lower and ('4.5' in model_lower or '4-5' in model_lower)
	is_haiku_4_5 = 'haiku' in model_lower and ('4.5' in model_lower or '4-5' in model_lower)
	return is_opus_4_5 or is_haiku_4_5


class PromptSectionConfig:
	"""Configuration for which prompt sections to enable, based on provider and runtime mode.

	Instead of hardcoding boolean toggles that must be kept in sync with
	section registrations, this config tracks **disabled** section names only.
	By default every registered section is enabled.  ``apply_to_*_builder``
	iterates the builder's actual sections — no manual toggle dict needed.

	To add a new section: register it in ``default_registry()`` (or a custom
	registry).  The config will automatically discover it.
	"""

	def __init__(self) -> None:
		self.disabled_names: set[str] = set()

	def enable(self, name: str) -> None:
		self.disabled_names.discard(name)

	def disable(self, name: str) -> None:
		self.disabled_names.add(name)

	def is_enabled(self, name: str) -> bool:
		return name not in self.disabled_names

	@classmethod
	def for_provider(
		cls,
		provider: str = '',
		flash_mode: bool = False,
		use_thinking: bool = True,
		enable_planning: bool = True,
	) -> 'PromptSectionConfig':
		"""Factory: build a config based on provider name and runtime mode."""
		cfg = cls()

		if flash_mode:
			cfg.disable('plan')
			cfg.enable('step_meta')

		if not enable_planning:
			cfg.disable('plan')

		provider_lower = provider.lower() if provider else ''

		if 'anthropic' in provider_lower:
			pass

		return cfg

	def apply_to_user_builder(self, builder: PromptContextBuilder) -> None:
		"""Apply this config to a user-state PromptContextBuilder.

		Iterates the builder's actual sections (which come from the registry)
		so there is no hardcoded name list to maintain.
		"""
		for section in builder.sections:
			if self.is_enabled(section.name):
				section.enabled = True
			else:
				section.enabled = False
			if isinstance(section, AgentStateGroupSection):
				for sub in section.sub_sections:
					if self.is_enabled(sub.name):
						sub.enabled = True
					else:
						sub.enabled = False

	def apply_to_system_builder(self, builder: SystemPromptBuilder) -> None:
		"""Apply this config to a SystemPromptBuilder."""
		for section in builder.sections:
			if self.is_enabled(section.name):
				section.enabled = True
			else:
				section.enabled = False
