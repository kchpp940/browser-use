import importlib.resources
from typing import TYPE_CHECKING, Literal, Optional

from browser_use.agent.prompt_context import PromptContextBuilder, PromptSectionContext
from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, SystemMessage, UserMessage
from browser_use.observability import observe_debug

if TYPE_CHECKING:
	from browser_use.agent.views import AgentStepInfo
	from browser_use.browser.views import BrowserStateSummary
	from browser_use.filesystem.file_system import FileSystem


def _is_anthropic_4_5_model(model_name: str | None) -> bool:
	"""Check if the model is Claude Opus 4.5 or Haiku 4.5 (requires 4096+ token prompts for caching)."""
	if not model_name:
		return False
	model_lower = model_name.lower()
	# Check for Opus 4.5 or Haiku 4.5 variants
	is_opus_4_5 = 'opus' in model_lower and ('4.5' in model_lower or '4-5' in model_lower)
	is_haiku_4_5 = 'haiku' in model_lower and ('4.5' in model_lower or '4-5' in model_lower)
	return is_opus_4_5 or is_haiku_4_5


class SystemPrompt:
	def __init__(
		self,
		max_actions_per_step: int = 3,
		override_system_message: str | None = None,
		extend_system_message: str | None = None,
		use_thinking: bool = True,
		flash_mode: bool = False,
		is_anthropic: bool = False,
		is_browser_use_model: bool = False,
		model_name: str | None = None,
	):
		self.max_actions_per_step = max_actions_per_step
		self.use_thinking = use_thinking
		self.flash_mode = flash_mode
		self.is_anthropic = is_anthropic
		self.is_browser_use_model = is_browser_use_model
		self.model_name = model_name
		# Check if this is an Anthropic 4.5 model that needs longer prompts for caching
		self.is_anthropic_4_5 = _is_anthropic_4_5_model(model_name)
		prompt = ''
		if override_system_message is not None:
			prompt = override_system_message
		else:
			self._load_prompt_template()
			prompt = self.prompt_template.format(max_actions=self.max_actions_per_step)

		if extend_system_message:
			prompt += f'\n{extend_system_message}'

		self.system_message = SystemMessage(content=prompt, cache=True)

	def _load_prompt_template(self) -> None:
		"""Load the prompt template from the markdown file."""
		try:
			# Choose the appropriate template based on model type and mode
			# Browser-use models use simplified prompts optimized for fine-tuned models
			if self.is_browser_use_model:
				if self.flash_mode:
					template_filename = 'system_prompt_browser_use_flash.md'
				elif self.use_thinking:
					template_filename = 'system_prompt_browser_use.md'
				else:
					template_filename = 'system_prompt_browser_use_no_thinking.md'
			# Anthropic 4.5 models (Opus 4.5, Haiku 4.5) need 4096+ token prompts for caching
			elif self.is_anthropic_4_5 and self.flash_mode:
				template_filename = 'system_prompt_anthropic_flash.md'
			elif self.flash_mode and self.is_anthropic:
				template_filename = 'system_prompt_flash_anthropic.md'
			elif self.flash_mode:
				template_filename = 'system_prompt_flash.md'
			elif self.use_thinking:
				template_filename = 'system_prompt.md'
			else:
				template_filename = 'system_prompt_no_thinking.md'

			# This works both in development and when installed as a package
			with (
				importlib.resources.files('browser_use.agent.system_prompts')
				.joinpath(template_filename)
				.open('r', encoding='utf-8') as f
			):
				self.prompt_template = f.read()
		except Exception as e:
			raise RuntimeError(f'Failed to load system prompt template: {e}')

	def get_system_message(self) -> SystemMessage:
		"""
		Get the system prompt for the agent.

		Returns:
		    SystemMessage: Formatted system prompt
		"""
		return self.system_message


class AgentMessagePrompt:
	vision_detail_level: Literal['auto', 'low', 'high']

	def __init__(
		self,
		browser_state_summary: 'BrowserStateSummary',
		file_system: 'FileSystem',
		agent_history_description: str | None = None,
		read_state_description: str | None = None,
		task: str | None = None,
		include_attributes: list[str] | None = None,
		step_info: Optional['AgentStepInfo'] = None,
		page_filtered_actions: str | None = None,
		max_clickable_elements_length: int = 40000,
		sensitive_data: str | None = None,
		available_file_paths: list[str] | None = None,
		screenshots: list[str] | None = None,
		vision_detail_level: Literal['auto', 'low', 'high'] = 'auto',
		include_recent_events: bool = False,
		sample_images: list[ContentPartTextParam | ContentPartImageParam] | None = None,
		read_state_images: list[dict] | None = None,
		llm_screenshot_size: tuple[int, int] | None = None,
		unavailable_skills_info: str | None = None,
		plan_description: str | None = None,
	):
		self.builder = PromptContextBuilder()
		self.builder.set_context(
			browser_state_summary=browser_state_summary,
			file_system=file_system,
			agent_history_description=agent_history_description,
			read_state_description=read_state_description,
			task=task,
			include_attributes=include_attributes,
			step_info=step_info,
			page_filtered_actions=page_filtered_actions,
			max_clickable_elements_length=max_clickable_elements_length,
			sensitive_data=sensitive_data,
			available_file_paths=available_file_paths,
			screenshots=screenshots or [],
			vision_detail_level=vision_detail_level,
			include_recent_events=include_recent_events,
			sample_images=sample_images or [],
			read_state_images=read_state_images or [],
			llm_screenshot_size=llm_screenshot_size,
			unavailable_skills_info=unavailable_skills_info,
			plan_description=plan_description,
		)
		self.vision_detail_level = vision_detail_level
		self.screenshots = screenshots or []
		self.sample_images = sample_images or []
		self.read_state_images = read_state_images or []
		self.llm_screenshot_size = llm_screenshot_size
		assert browser_state_summary

	@property
	def context(self) -> PromptSectionContext:
		return self.builder.context

	@observe_debug(ignore_input=True, ignore_output=True, name='get_user_message')
	def get_user_message(self, use_vision: bool = True) -> UserMessage:
		"""Get complete state as a single cached message"""
		content = self.builder.build_content_parts(
			use_vision=use_vision,
			screenshots=self.screenshots,
			read_state_images=self.read_state_images,
			sample_images=self.sample_images,
			vision_detail_level=self.vision_detail_level,
			llm_screenshot_size=self.llm_screenshot_size,
		)
		return UserMessage(content=content, cache=True)


def get_rerun_summary_prompt(original_task: str, total_steps: int, success_count: int, error_count: int) -> str:
	return f'''You are analyzing the completion of a rerun task. Based on the screenshot and execution info, provide a summary.

Original task: {original_task}

Execution statistics:
- Total steps: {total_steps}
- Successful steps: {success_count}
- Failed steps: {error_count}

Analyze the screenshot to determine:
1. Whether the task completed successfully
2. What the final state shows
3. Overall completion status (complete/partial/failed)

Respond with:
- summary: A clear, concise summary of what happened during the rerun
- success: Whether the task completed successfully (true/false)
- completion_status: One of "complete", "partial", or "failed"'''


def get_rerun_summary_message(prompt: str, screenshot_b64: str | None = None) -> UserMessage:
	"""
	Build a UserMessage for rerun summary generation.

	Args:
		prompt: The prompt text
		screenshot_b64: Optional base64-encoded screenshot

	Returns:
		UserMessage with prompt and optional screenshot
	"""
	if screenshot_b64:
		# With screenshot: use multi-part content
		content_parts: list[ContentPartTextParam | ContentPartImageParam] = [
			ContentPartTextParam(type='text', text=prompt),
			ContentPartImageParam(
				type='image_url',
				image_url=ImageURL(url=f'data:image/png;base64,{screenshot_b64}'),
			),
		]
		return UserMessage(content=content_parts)
	else:
		# Without screenshot: use simple string content
		return UserMessage(content=prompt)


def get_ai_step_system_prompt() -> str:
	"""
	Get system prompt for AI step action used during rerun.

	Returns:
		System prompt string for AI step
	"""
	return """
You are an expert at extracting data from webpages.

<input>
You will be given:
1. A query describing what to extract
2. The markdown of the webpage (filtered to remove noise)
3. Optionally, a screenshot of the current page state
</input>

<instructions>
- Extract information from the webpage that is relevant to the query
- ONLY use the information available in the webpage - do not make up information
- If the information is not available, mention that clearly
- If the query asks for all items, list all of them
</instructions>

<output>
- Present ALL relevant information in a concise way
- Do not use conversational format - directly output the relevant information
- If information is unavailable, state that clearly
</output>
""".strip()


def get_ai_step_user_prompt(query: str, stats_summary: str, content: str) -> str:
	"""
	Build user prompt for AI step action.

	Args:
		query: What to extract or analyze
		stats_summary: Content statistics summary
		content: Page markdown content

	Returns:
		Formatted prompt string
	"""
	return f'<query>\n{query}\n</query>\n\n<content_stats>\n{stats_summary}\n</content_stats>\n\n<webpage_content>\n{content}\n</webpage_content>'
