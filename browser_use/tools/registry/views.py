from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from browser_use.browser import BrowserSession
from browser_use.filesystem.file_system import FileSystem
from browser_use.llm.base import BaseChatModel

if TYPE_CHECKING:
	pass


class RegisteredAction(BaseModel):
	"""Model for a registered action"""

	name: str
	description: str
	function: Callable
	param_model: type[BaseModel]

	# If True, this action is known to change the page (e.g. navigate, search, go_back, switch).
	# multi_act() will abort remaining queued actions after executing a terminates_sequence action.
	terminates_sequence: bool = False

	# filters: provide specific domains to determine whether the action should be available on the given URL or not
	domains: list[str] | None = None  # e.g. ['*.google.com', 'www.bing.com', 'yahoo.*]

	# Category of the action - used for filtering and organization
	category: Literal['navigation', 'interaction', 'extraction', 'tab_management', 'file', 'system', 'custom'] = 'custom'

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def prompt_description(self) -> str:
		"""Get a description of the action for the prompt in unstructured format"""
		schema = self.param_model.model_json_schema()
		params = []

		if 'properties' in schema:
			for param_name, param_info in schema['properties'].items():
				# Build parameter description
				param_desc = param_name

				# Add type information if available
				if 'type' in param_info:
					param_type = param_info['type']
					param_desc += f'={param_type}'

				# Add description as comment if available
				if 'description' in param_info:
					param_desc += f' ({param_info["description"]})'

				params.append(param_desc)

		# Format: action_name: Description. (param1=type, param2=type, ...)
		if params:
			return f'{self.name}: {self.description}. ({", ".join(params)})'
		else:
			return f'{self.name}: {self.description}'


class ActionModel(BaseModel):
	"""Base model for dynamically created action models"""

	# this will have all the registered actions, e.g.
	# click_element = param_model = ClickElementParams
	# done = param_model = None
	#
	model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

	def get_index(self) -> int | None:
		"""Get the index of the action"""
		# {'clicked_element': {'index':5}}
		params = self.model_dump(exclude_unset=True).values()
		if not params:
			return None
		for param in params:
			if param is not None and 'index' in param:
				return param['index']
		return None

	def set_index(self, index: int):
		"""Overwrite the index of the action"""
		# Get the action name and params
		action_data = self.model_dump(exclude_unset=True)
		action_name = next(iter(action_data.keys()))
		action_params = getattr(self, action_name)

		# Update the index directly on the model
		if hasattr(action_params, 'index'):
			action_params.index = index


class ActionRegistry(BaseModel):
	"""Model representing the action registry"""

	actions: dict[str, RegisteredAction] = {}

	@staticmethod
	def _match_domains(domains: list[str] | None, url: str) -> bool:
		"""
		Match a list of domain glob patterns against a URL.

		Args:
			domains: A list of domain patterns that can include glob patterns (* wildcard)
			url: The URL to match against

		Returns:
			True if the URL's domain matches the pattern, False otherwise
		"""

		if domains is None or not url:
			return True

		# Use the centralized URL matching logic from utils
		from browser_use.utils import match_url_with_domain_pattern

		for domain_pattern in domains:
			if match_url_with_domain_pattern(url, domain_pattern):
				return True
		return False

	def get_prompt_description(self, page_url: str | None = None) -> str:
		"""Get a description of all actions for the prompt

		Args:
			page_url: If provided, filter actions by URL using domain filters.

		Returns:
			A string description of available actions.
			- If page is None: return only actions with no page_filter and no domains (for system prompt)
			- If page is provided: return only filtered actions that match the current page (excluding unfiltered actions)
		"""
		if page_url is None:
			# For system prompt (no URL provided), include only actions with no filters
			return '\n'.join(action.prompt_description() for action in self.actions.values() if action.domains is None)

		# only include filtered actions for the current page URL
		filtered_actions = []
		for action in self.actions.values():
			if not action.domains:
				# skip actions with no filters, they are already included in the system prompt
				continue

			# Check domain filter
			if self._match_domains(action.domains, page_url):
				filtered_actions.append(action)

		return '\n'.join(action.prompt_description() for action in filtered_actions)


class SpecialActionParameters(BaseModel):
	"""Model defining all special parameters that can be injected into actions"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	# optional user-provided context object passed down from Agent(context=...)
	# e.g. can contain anything, external db connections, file handles, queues, runtime config objects, etc.
	# that you might want to be able to access quickly from within many of your actions
	# browser-use code doesn't use this at all, we just pass it down to your actions for convenience
	context: Any | None = None

	# browser-use session object, can be used to create new tabs, navigate, access CDP
	browser_session: BrowserSession | None = None

	# Current page URL for filtering and context
	page_url: str | None = None

	# CDP client for direct Chrome DevTools Protocol access
	cdp_client: Any | None = None  # CDPClient type from cdp_use

	# extra injected config if the action asks for these arg names
	page_extraction_llm: BaseChatModel | None = None
	file_system: FileSystem | None = None
	available_file_paths: list[str] | None = None
	has_sensitive_data: bool = False
	extraction_schema: dict | None = None

	@classmethod
	def get_browser_requiring_params(cls) -> set[str]:
		"""Get parameter names that require browser_session"""
		return {'browser_session', 'cdp_client', 'page_url'}


class ToolParameterSchema(BaseModel):
	"""Unified parameter schema for a tool."""

	name: str
	type: str
	description: str | None = None
	required: bool = False
	default: Any = None
	enum: list[str] | None = None


class ToolCapability(BaseModel):
	"""Unified tool capability description.

	This is the single source of truth for tool metadata across all entry points:
	- Agent tools registry
	- MCP list_tools
	- Task template allowed tools filtering
	- Result formatting
	"""

	name: str
	description: str
	category: Literal['navigation', 'interaction', 'extraction', 'tab_management', 'file', 'system', 'custom'] = 'custom'
	param_schema: type[BaseModel]
	domains: list[str] | None = None
	terminates_sequence: bool = False
	requires_browser: bool = False
	requires_llm: bool = False
	result_is_structured: bool = False

	model_config = ConfigDict(arbitrary_types_allowed=True)

	@classmethod
	def from_registered_action(cls, action: RegisteredAction) -> 'ToolCapability':
		"""Create a ToolCapability from a RegisteredAction."""
		requires_browser = False
		requires_llm = False

		func = action.function
		if hasattr(func, '__wrapped__'):
			sig_params = _get_func_params(func.__wrapped__)
		else:
			sig_params = _get_func_params(func)

		special_params = SpecialActionParameters.model_fields.keys()
		for param_name in sig_params:
			if param_name in {'browser_session', 'cdp_client', 'page_url'}:
				requires_browser = True
			if param_name == 'page_extraction_llm':
				requires_llm = True

		return cls(
			name=action.name,
			description=action.description,
			category=action.category,
			param_schema=action.param_model,
			domains=action.domains,
			terminates_sequence=action.terminates_sequence,
			requires_browser=requires_browser,
			requires_llm=requires_llm,
		)

	def to_json_schema(self) -> dict:
		"""Convert to JSON Schema (for MCP and other API consumers)."""
		schema = self.param_schema.model_json_schema()
		properties = schema.get('properties', {})
		required = schema.get('required', [])

		result = {
			'type': 'object',
			'properties': {},
			'required': list(required),
		}

		for name, prop in properties.items():
			result['properties'][name] = prop

		return result

	def to_mcp_tool(self) -> dict:
		"""Convert to MCP Tool format."""
		return {
			'name': self.name,
			'description': self.description,
			'inputSchema': self.to_json_schema(),
		}

	def is_available_for_url(self, url: str | None) -> bool:
		"""Check if this tool is available for the given URL."""
		if self.domains is None or not url:
			return True
		from browser_use.utils import match_url_with_domain_pattern

		for domain_pattern in self.domains:
			if match_url_with_domain_pattern(url, domain_pattern):
				return True
		return False


class ToolInvocationSpec(BaseModel):
	"""Specification for invoking a tool."""

	tool_name: str
	params: dict[str, Any] = Field(default_factory=dict)

	@classmethod
	def from_action_model(cls, action: ActionModel) -> 'ToolInvocationSpec':
		"""Create a ToolInvocationSpec from an ActionModel instance."""
		action_data = action.model_dump(exclude_unset=True)
		action_name = next(iter(action_data.keys()))
		action_params = action_data[action_name] or {}
		return cls(tool_name=action_name, params=action_params)


class ToolResult(BaseModel):
	"""Unified tool result wrapper.

	All tools should return results in this format for consistency across:
	- Agent execution chain
	- MCP tool responses
	- Template execution results
	"""

	extracted_content: str | None = None
	error: str | None = None
	is_done: bool = False
	success: bool | None = None
	metadata: dict[str, Any] | None = None
	long_term_memory: str | None = None
	attachments: list[str] | None = None
	images: list[dict[str, Any]] | None = None

	model_config = ConfigDict(extra='allow')

	@classmethod
	def from_action_result(cls, action_result: Any) -> 'ToolResult':
		"""Create a ToolResult from an ActionResult or raw value.

		Handles both ActionResult objects and plain strings/dicts.
		"""
		from browser_use.agent.views import ActionResult

		if isinstance(action_result, ToolResult):
			return action_result

		if isinstance(action_result, ActionResult):
			return cls(
				extracted_content=action_result.extracted_content,
				error=action_result.error,
				is_done=action_result.is_done or False,
				success=action_result.success,
				metadata=action_result.metadata,
				long_term_memory=action_result.long_term_memory,
				attachments=action_result.attachments,
				images=action_result.images,
			)

		if isinstance(action_result, str):
			return cls(extracted_content=action_result)

		if isinstance(action_result, dict):
			return cls(
				extracted_content=action_result.get('extracted_content'),
				error=action_result.get('error'),
				is_done=action_result.get('is_done', False),
				success=action_result.get('success'),
				metadata={
					k: v for k, v in action_result.items() if k not in {'extracted_content', 'error', 'is_done', 'success'}
				},
			)

		return cls(extracted_content=str(action_result) if action_result is not None else None)

	def to_action_result(self) -> Any:
		"""Convert to Agent's ActionResult format."""
		from browser_use.agent.views import ActionResult

		return ActionResult(
			extracted_content=self.extracted_content,
			error=self.error,
			is_done=self.is_done,
			success=self.success,
			metadata=self.metadata,
			long_term_memory=self.long_term_memory,
			attachments=self.attachments,
			images=self.images,
		)

	def to_mcp_content(self) -> list[dict[str, Any]]:
		"""Convert to MCP content items format."""
		content = []

		if self.error:
			content.append({'type': 'text', 'text': f'Error: {self.error}'})
		elif self.extracted_content:
			content.append({'type': 'text', 'text': self.extracted_content})

		if self.images:
			for img in self.images:
				content.append({'type': 'image', 'data': img.get('data', ''), 'mimeType': img.get('mime_type', 'image/png')})

		if not content:
			content.append({'type': 'text', 'text': 'Task completed' if self.success else 'No output'})

		return content

	def is_successful(self) -> bool:
		"""Check if the tool execution was successful."""
		return self.error is None and self.success is not False


def _get_func_params(func: Callable) -> list[str]:
	"""Get parameter names from a function, handling wrapped functions."""
	import inspect

	try:
		sig = inspect.signature(func)
		return list(sig.parameters.keys())
	except (ValueError, TypeError):
		return []


class ToolRegistryAdapter(BaseModel):
	"""Adapter that provides unified tool metadata from an action registry.

	This is the central adapter that ensures consistency across:
	1. Agent tools registry
	2. MCP list_tools
	3. Template allowed tools filtering
	4. Execution result packaging
	"""

	registry: 'ActionRegistry'

	model_config = ConfigDict(arbitrary_types_allowed=True)

	def get_tool_capability(self, tool_name: str) -> ToolCapability | None:
		"""Get a ToolCapability for a specific tool."""
		action = self.registry.actions.get(tool_name)
		if action is None:
			return None
		return ToolCapability.from_registered_action(action)

	def list_tool_capabilities(
		self, page_url: str | None = None, include_categories: list[str] | None = None
	) -> list[ToolCapability]:
		"""List all tool capabilities, optionally filtered by URL and category."""
		tools = []
		for action in self.registry.actions.values():
			capability = ToolCapability.from_registered_action(action)

			if page_url is not None and not capability.is_available_for_url(page_url):
				continue

			if include_categories is not None and capability.category not in include_categories:
				continue

			tools.append(capability)
		return tools

	def list_mcp_tools(self, page_url: str | None = None) -> list[dict]:
		"""List all tools in MCP format."""
		return [cap.to_mcp_tool() for cap in self.list_tool_capabilities(page_url=page_url)]

	def filter_allowed_tools(self, allowed_tool_names: list[str]) -> 'ToolRegistryAdapter':
		"""Create a new adapter with only allowed tools.

		Used for task template tool whitelisting.
		"""
		filtered_actions = {name: action for name, action in self.registry.actions.items() if name in allowed_tool_names}
		filtered_registry = ActionRegistry(actions=filtered_actions)
		return ToolRegistryAdapter(registry=filtered_registry)

	def get_prompt_description(self, page_url: str | None = None) -> str:
		"""Get a human-readable description of all tools for prompts."""
		return self.registry.get_prompt_description(page_url=page_url)

	def create_action_model(
		self,
		include_actions: list[str] | None = None,
		page_url: str | None = None,
		include_categories: list[str] | None = None,
	) -> Any:
		"""Create an action model from filtered tools.

		Used for task template tool whitelisting and LLM tool calling.

		Args:
			include_actions: List of action names to include (None for all)
			page_url: Filter by page URL (domain matching)
			include_categories: List of categories to include

		Returns:
			An ActionModel type with only the allowed actions
		"""
		from browser_use.tools.registry.service import Registry

		# Create a temporary registry with filtered actions
		filtered_actions: dict[str, RegisteredAction] = {}
		for name, action in self.registry.actions.items():
			if include_actions is not None and name not in include_actions:
				continue

			if include_categories is not None and action.category not in include_categories:
				continue

			if page_url is not None and not self._registry_match_domains(action.domains, page_url):
				continue

			filtered_actions[name] = action

		# Create a temporary registry and use its create_action_model
		temp_registry_model = ActionRegistry(actions=filtered_actions)
		temp_registry = Registry()
		temp_registry.registry = temp_registry_model
		return temp_registry.create_action_model(page_url=page_url)

	@staticmethod
	def _registry_match_domains(domains: list[str] | None, url: str) -> bool:
		"""Match domain patterns against a URL."""
		if domains is None or not url:
			return True
		from browser_use.utils import match_url_with_domain_pattern

		for domain_pattern in domains:
			if match_url_with_domain_pattern(url, domain_pattern):
				return True
		return False
