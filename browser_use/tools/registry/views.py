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


def create_action_model_instance_from_params(
	action_name: str,
	params: dict[str, Any],
	registry: 'ActionRegistry',
) -> Any:
	"""Create an action model instance from raw params dict.

	This is a convenience function used by unified tool execution entry points.
	(MCP, skill_cli, etc.) to build action instances from raw arguments.
	"""
	action = registry.actions.get(action_name)
	if action is None:
		raise ValueError(f'Action not found: {action_name}')

	# Create a dynamic action model with only this action
	from browser_use.tools.registry.service import Registry

	temp_registry = Registry()
	filtered_registry = ActionRegistry(actions={action_name: action})
	temp_registry.registry = filtered_registry
	ActionModel = temp_registry.create_action_model()

	# Build instance with the action params
	action_instance = ActionModel.model_validate({action_name: params})
	return action_instance


class ToolRegistryAdapter(BaseModel):
	"""Unified adapter for consistent tool metadata access across all entry points.

	This adapter provides a single interface for:
	- Agent tools registry
	- MCP list_tools and call_tool
	- Template allowed tools filtering
	- Tool result formatting
	- External tool registration (for MCP-specific and other tools)
	"""

	registry: 'ActionRegistry'

	model_config = ConfigDict(arbitrary_types_allowed=True)

	# External capabilities registered outside the core registry
	_external_capabilities: dict[str, ToolCapability] = {}

	# External executors (callable functions for each external tool)
	_external_executors: dict[str, Callable[..., Any]] = {}

	# Name mappings: alias_name -> canonical_name (used for MCP prefix mapping)
	_name_mappings: dict[str, str] = {}

	def register_external_capability(
		self,
		capability: ToolCapability,
		executor: Callable[..., Any] | None = None,
		canonical_alias: str | None = None,
	) -> None:
		"""Register an external tool capability and optional executor.

		This allows extending the unified registry with tools not in the core,
		such as MCP-specific tools or integration-specific tools.

		Args:
			capability: The ToolCapability describing the tool
			executor: Optional async callable to execute the tool
			canonical_alias: Optional alias for the canonical name (e.g., MCP name)
		"""
		self._external_capabilities[capability.name] = capability
		if executor is not None:
			self._external_executors[capability.name] = executor
		if canonical_alias is not None:
			self._name_mappings[canonical_alias] = capability.name

	def register_name_mapping(self, alias_name: str, canonical_name: str) -> None:
		"""Register a name mapping from alias to canonical name.

		This is used for MCP-style prefixed names like "browser_navigate" -> "navigate".
		"""
		self._name_mappings[alias_name] = canonical_name

	def resolve_name(self, name: str) -> str:
		"""Resolve an alias name to its canonical name.

		Returns the canonical name if a mapping exists, otherwise returns the input.
		"""
		return self._name_mappings.get(name, name)

	def _get_all_capabilities(self) -> dict[str, ToolCapability]:
		"""Get all capabilities including core registry and external."""
		all_caps: dict[str, ToolCapability] = {}

		# First add core capabilities from registry
		for name, action in self.registry.actions.items():
			all_caps[name] = ToolCapability.from_registered_action(action)

		# Then add external (override core if same name)
		for name, cap in self._external_capabilities.items():
			all_caps[name] = cap

		return all_caps

	async def execute_tool(
		self,
		tool_name: str,
		arguments: dict[str, Any],
		browser_session: Any | None = None,
		page_extraction_llm: Any | None = None,
		file_system: Any | None = None,
		sensitive_data: Any | None = None,
		available_file_paths: Any | None = None,
		extraction_schema: Any | None = None,
	) -> Any:
		"""Execute a tool by name with unified parameter handling.

		This handles both core registry tools and external tools through a single interface.

		Args:
			tool_name: The tool name (may be an alias that gets resolved)
			arguments: Dictionary of tool arguments
			browser_session: Optional browser session context
			page_extraction_llm: Optional LLM for extraction
			file_system: Optional file system context
			sensitive_data: Optional sensitive data wrapper
			available_file_paths: Optional file path allowlist
			extraction_schema: Optional extraction schema

		Returns:
			The tool execution result (ActionResult-compatible format)
		"""
		# Resolve alias name to canonical name
		canonical_name = self.resolve_name(tool_name)

		# First check if it's an external tool
		if canonical_name in self._external_executors:
			executor = self._external_executors[canonical_name]
			cap = self._external_capabilities.get(canonical_name)

			# Validate and parse arguments using param_schema if available
			params_kwargs = dict(arguments)
			if cap is not None:
				try:
					parsed = cap.param_schema(**arguments)
					params_kwargs = parsed.model_dump()
				except Exception:
					pass

			# Inject special parameters if executor accepts them
			import inspect

			sig = inspect.signature(executor)
			executor_kwargs = dict(params_kwargs)
			param_names = set(sig.parameters.keys())

			if 'browser_session' in param_names and browser_session is not None:
				executor_kwargs['browser_session'] = browser_session
			if 'page_extraction_llm' in param_names and page_extraction_llm is not None:
				executor_kwargs['page_extraction_llm'] = page_extraction_llm
			if 'file_system' in param_names and file_system is not None:
				executor_kwargs['file_system'] = file_system
			if 'self' in param_names:
				raise ValueError('Executor must be a static or module-level function, not a bound method')

			if inspect.iscoroutinefunction(executor):
				return await executor(**executor_kwargs)
			return executor(**executor_kwargs)

		# Otherwise it's a core tool - execute through the standard mechanism
		if canonical_name in self.registry.actions:
			from browser_use.tools.registry.service import Registry

			# Create a temp registry instance to use its execute_action method
			temp_registry = Registry()
			temp_registry.registry = self.registry
			result = await temp_registry.execute_action(
				action_name=canonical_name,
				params=arguments,
				browser_session=browser_session,
				page_extraction_llm=page_extraction_llm,
				file_system=file_system,
				sensitive_data=sensitive_data,
				available_file_paths=available_file_paths,
				extraction_schema=extraction_schema,
			)
			return result

		# Tool not found
		raise ValueError(f'Tool not found: {tool_name} (resolved: {canonical_name})')

	def get_tool_capability(self, tool_name: str) -> ToolCapability | None:
		"""Get a ToolCapability for a specific tool.

		Resolves name aliases automatically.
		"""
		canonical_name = self.resolve_name(tool_name)

		# Check external first
		if canonical_name in self._external_capabilities:
			return self._external_capabilities[canonical_name]

		# Check core registry
		action = self.registry.actions.get(canonical_name)
		if action is None:
			return None
		return ToolCapability.from_registered_action(action)

	def list_tool_capabilities(
		self, page_url: str | None = None, include_categories: list[str] | None = None
	) -> list[ToolCapability]:
		"""List all tool capabilities, optionally filtered by URL and category.

		Includes both core registry tools and externally registered tools.
		"""
		tools = []
		all_caps = self._get_all_capabilities()

		for capability in all_caps.values():
			if page_url is not None and not capability.is_available_for_url(page_url):
				continue

			if include_categories is not None and capability.category not in include_categories:
				continue

			tools.append(capability)
		return tools

	def list_mcp_tools(
		self,
		page_url: str | None = None,
		name_prefix: str = '',
		name_mappings: dict[str, str] | None = None,
	) -> list[dict]:
		"""List all tools in MCP format.

		Args:
			page_url: Filter by page URL availability
			name_prefix: Optional prefix to add to all tool names (e.g., 'browser_')
			name_mappings: Optional custom name mappings {canonical: mcp_name}
		"""
		result = []
		caps = self.list_tool_capabilities(page_url=page_url)

		# Build reverse mapping: canonical_name -> mcp_name
		reverse_map: dict[str, str] = {}
		if name_mappings:
			for mcp, canonical in name_mappings.items():
				reverse_map[canonical] = mcp

		for cap in caps:
			# Determine MCP tool name
			mcp_name = reverse_map.get(cap.name)
			if mcp_name is None:
				# Check if the capability's name itself is already an alias (external)
				# Otherwise add prefix
				if cap.name in reverse_map:
					mcp_name = reverse_map[cap.name]
				elif cap.name in self._name_mappings.values():
					# Find which alias maps to this canonical name
					for alias, canonical in self._name_mappings.items():
						if canonical == cap.name and alias.startswith(name_prefix):
							mcp_name = alias
							break
					if mcp_name is None:
						mcp_name = name_prefix + cap.name
				else:
					mcp_name = name_prefix + cap.name

			# Create MCP tool dict with adjusted name
			mcp_tool = cap.to_mcp_tool()
			mcp_tool['name'] = mcp_name
			result.append(mcp_tool)

		return result

	def filter_allowed_tools(
		self,
		allowed_tool_names: list[str] | None = None,
		allowed_categories: list[str] | None = None,
		resolve_aliases: bool = True,
	) -> 'ToolRegistryAdapter':
		"""Create a new adapter with only allowed tools.

		Used for task template tool whitelisting and skill_cli command filtering.

		Args:
			allowed_tool_names: List of allowed tool names (may contain aliases).
				If None, all names are allowed (subject to category filter).
			allowed_categories: List of allowed categories. If None, all categories
				are allowed (subject to name filter).
			resolve_aliases: If True, automatically resolves aliases before filtering
		"""
		# Resolve all aliases in allowed list
		allowed_canonical: set[str] | None = None
		if allowed_tool_names is not None:
			allowed_canonical = set()
			for name in allowed_tool_names:
				if resolve_aliases:
					canonical = self.resolve_name(name)
				else:
					canonical = name
				allowed_canonical.add(canonical)

		# Filter core actions
		# Logic: if both names and categories are specified, use OR — an action
		# passes if its name is in the allow-list OR its category is allowed.
		# This matches the common use case of "allow these specific tools PLUS
		# all tools in these categories".
		filtered_actions: dict[str, RegisteredAction] = {}
		for name, action in self.registry.actions.items():
			name_ok = allowed_canonical is None or name in allowed_canonical
			cat_ok = allowed_categories is None or action.category in allowed_categories

			if allowed_canonical is not None and allowed_categories is not None:
				passes = name_ok or cat_ok
			else:
				passes = name_ok and cat_ok

			if passes:
				filtered_actions[name] = action

		# Create adapter with filtered core registry
		filtered_registry = ActionRegistry(actions=filtered_actions)
		new_adapter = ToolRegistryAdapter(registry=filtered_registry)

		# Copy name mappings
		new_adapter._name_mappings = dict(self._name_mappings)

		# Filter external capabilities — same OR logic as core actions
		for ext_name, ext_cap in self._external_capabilities.items():
			name_ok = allowed_canonical is None or ext_name in allowed_canonical
			cat_ok = allowed_categories is None or ext_cap.category in allowed_categories

			if allowed_canonical is not None and allowed_categories is not None:
				passes = name_ok or cat_ok
			else:
				passes = name_ok and cat_ok

			if passes:
				new_adapter._external_capabilities[ext_name] = ext_cap
				if ext_name in self._external_executors:
					new_adapter._external_executors[ext_name] = self._external_executors[ext_name]

		return new_adapter

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
