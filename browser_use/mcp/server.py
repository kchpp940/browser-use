"""MCP Server for browser-use - exposes browser automation capabilities via Model Context Protocol.

This server provides tools for:
- Running autonomous browser tasks with an AI agent
- Direct browser control (navigation, clicking, typing, etc.)
- Content extraction from web pages
- File system operations

Usage:
    uvx browser-use --mcp

Or as an MCP server in Claude Desktop or other MCP clients:
    {
        "mcpServers": {
            "browser-use": {
                "command": "uvx",
                "args": ["browser-use[cli]", "--mcp"],
                "env": {
                    "OPENAI_API_KEY": "sk-proj-1234567890",
                }
            }
        }
    }
"""

import os
import sys

# Set environment variables BEFORE any browser_use imports to prevent early logging
os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'critical'
os.environ['BROWSER_USE_SETUP_LOGGING'] = 'false'

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from browser_use.llm import ChatAWSBedrock

# Configure logging for MCP mode - redirect to stderr but preserve critical diagnostics
logging.basicConfig(
	stream=sys.stderr, level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True
)

try:
	import psutil

	PSUTIL_AVAILABLE = True
except ImportError:
	PSUTIL_AVAILABLE = False

# Add browser-use to path if running from source
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import and configure logging to use stderr before other imports
from browser_use.logging_config import setup_logging


def _configure_mcp_server_logging():
	"""Configure logging for MCP server mode - redirect all logs to stderr to prevent JSON RPC interference."""
	# Set environment to suppress browser-use logging during server mode
	os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'warning'
	os.environ['BROWSER_USE_SETUP_LOGGING'] = 'false'  # Prevent automatic logging setup

	# Configure logging to stderr for MCP mode - preserve warnings and above for troubleshooting
	setup_logging(stream=sys.stderr, log_level='warning', force_setup=True)

	# Also configure the root logger and all existing loggers to use stderr
	logging.root.handlers = []
	stderr_handler = logging.StreamHandler(sys.stderr)
	stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logging.root.addHandler(stderr_handler)
	logging.root.setLevel(logging.CRITICAL)

	# Configure all existing loggers to use stderr and CRITICAL level
	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = []
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.addHandler(stderr_handler)
		logger_obj.propagate = False


# Configure MCP server logging before any browser_use imports to capture early log lines
_configure_mcp_server_logging()

# Additional suppression - disable all logging completely for MCP mode
logging.disable(logging.CRITICAL)

# Import browser_use modules
from browser_use import ActionModel, Agent
from browser_use.agent.views import ActionResult
from browser_use.browser import BrowserProfile, BrowserSession
from browser_use.config import get_default_llm, get_default_profile, load_browser_use_config
from browser_use.filesystem.file_system import FileSystem
from browser_use.llm.openai.chat import ChatOpenAI
from browser_use.tools.service import Tools

logger = logging.getLogger(__name__)


def _ensure_all_loggers_use_stderr():
	"""Ensure ALL loggers only output to stderr, not stdout."""
	# Get the stderr handler
	stderr_handler = None
	for handler in logging.root.handlers:
		if hasattr(handler, 'stream') and handler.stream == sys.stderr:  # type: ignore
			stderr_handler = handler
			break

	if not stderr_handler:
		stderr_handler = logging.StreamHandler(sys.stderr)
		stderr_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

	# Configure root logger
	logging.root.handlers = [stderr_handler]
	logging.root.setLevel(logging.CRITICAL)

	# Configure all existing loggers
	for name in list(logging.root.manager.loggerDict.keys()):
		logger_obj = logging.getLogger(name)
		logger_obj.handlers = [stderr_handler]
		logger_obj.setLevel(logging.CRITICAL)
		logger_obj.propagate = False


# Ensure stderr logging after all imports
_ensure_all_loggers_use_stderr()


# Try to import MCP SDK
try:
	import mcp.server.stdio
	import mcp.types as types
	from mcp.server import NotificationOptions, Server
	from mcp.server.models import InitializationOptions

	MCP_AVAILABLE = True

	# Configure MCP SDK logging to stderr as well
	mcp_logger = logging.getLogger('mcp')
	mcp_logger.handlers = []
	mcp_logger.addHandler(logging.root.handlers[0] if logging.root.handlers else logging.StreamHandler(sys.stderr))
	mcp_logger.setLevel(logging.ERROR)
	mcp_logger.propagate = False
except ImportError:
	MCP_AVAILABLE = False
	logger.error('MCP SDK not installed. Install with: pip install mcp')
	sys.exit(1)

from browser_use.telemetry import MCPServerTelemetryEvent, ProductTelemetry
from browser_use.utils import create_task_with_error_handling, get_browser_use_version


def get_parent_process_cmdline() -> str | None:
	"""Get the command line of all parent processes up the chain."""
	if not PSUTIL_AVAILABLE:
		return None

	try:
		cmdlines = []
		current_process = psutil.Process()
		parent = current_process.parent()

		while parent:
			try:
				cmdline = parent.cmdline()
				if cmdline:
					cmdlines.append(' '.join(cmdline))
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				# Skip processes we can't access (like system processes)
				pass

			try:
				parent = parent.parent()
			except (psutil.AccessDenied, psutil.NoSuchProcess):
				# Can't go further up the chain
				break

		return ';'.join(cmdlines) if cmdlines else None
	except Exception:
		# If we can't get parent process info, just return None
		return None


class BrowserUseServer:
	"""MCP Server for browser-use capabilities."""

	def __init__(self, session_timeout_minutes: int = 10):
		# Ensure all logging goes to stderr (in case new loggers were created)
		_ensure_all_loggers_use_stderr()

		self.server = Server('browser-use')
		self.config = load_browser_use_config()
		self.agent: Agent | None = None
		self.browser_session: BrowserSession | None = None
		self.tools: Tools | None = None
		self.llm: ChatOpenAI | None = None
		self.file_system: FileSystem | None = None
		self._telemetry = ProductTelemetry()
		self._start_time = time.time()

		# Session management
		self.active_sessions: dict[str, dict[str, Any]] = {}  # session_id -> session info
		self.session_timeout_minutes = session_timeout_minutes
		self._cleanup_task: Any = None

		# Unified tool registry adapter — single source of truth for all tool metadata
		# and execution. This replaces the old separate _tool_capabilities list.
		self.tool_registry_adapter: Any | None = None

		# Initialize tools and metadata early so list_tools works before browser session is created
		# This ensures tool names, descriptions, and schemas are always available
		self.tools = Tools()
		self._init_tool_registry()

		# Setup handlers
		self._setup_handlers()

	def _setup_handlers(self):
		"""Setup MCP server handlers."""

		@self.server.list_tools()
		async def handle_list_tools() -> list[types.Tool]:
			"""List all available browser-use tools.

			All tool metadata is sourced from the unified ToolRegistryAdapter.
			This ensures consistent tool names, descriptions, and parameter
			schemas across all entry points (Python API, MCP, skill_cli).
			"""
			assert self.tool_registry_adapter is not None, 'Tool registry adapter must be initialized'

			# Use the unified adapter to generate MCP format tools
			# list_mcp_tools already handles name mappings and external capabilities
			mcp_tool_dicts = self.tool_registry_adapter.list_mcp_tools(
				name_prefix='',  # External tools already have full names (browser_*)
			)

			# Convert dicts to MCP Tool objects
			mcp_tools = []
			for tool_dict in mcp_tool_dicts:
				mcp_tools.append(
					types.Tool(
						name=tool_dict['name'],
						description=tool_dict['description'],
						inputSchema=tool_dict['inputSchema'],
					)
				)

			return mcp_tools

		@self.server.list_resources()
		async def handle_list_resources() -> list[types.Resource]:
			"""List available resources (none for browser-use)."""
			return []

		@self.server.list_prompts()
		async def handle_list_prompts() -> list[types.Prompt]:
			"""List available prompts (none for browser-use)."""
			return []

		@self.server.call_tool()
		async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent | types.ImageContent]:
			"""Handle tool execution.

			All results are formatted using the unified ToolResult format
			to ensure consistent output structure across all entry points.
			"""
			start_time = time.time()
			error_msg = None
			try:
				result = await self._execute_tool(name, arguments or {})
				return self._format_result_from_action_result(result)
			except Exception as e:
				error_msg = str(e)
				logger.error(f'Tool execution failed: {e}', exc_info=True)
				return self._format_result_from_action_result({'error': str(e)})
			finally:
				# Capture telemetry for tool calls
				duration = time.time() - start_time
				self._telemetry.capture(
					MCPServerTelemetryEvent(
						version=get_browser_use_version(),
						action='tool_call',
						tool_name=name,
						duration_seconds=duration,
						error_message=error_msg,
					)
				)

	async def _execute_tool(
		self, tool_name: str, arguments: dict[str, Any]
	) -> str | list[types.TextContent | types.ImageContent]:
		"""Execute a browser-use tool via the unified ToolRegistryAdapter.

		All tool execution now goes through a single path:
		1. Resolve name aliases (MCP names -> canonical names)
		2. Check if browser session is required
		3. Execute via adapter.execute_tool()
		4. Result is formatted consistently via _format_result_from_action_result()

		There are NO separate if/elif branches for different tools anymore.
		"""
		assert self.tool_registry_adapter is not None, 'Tool registry adapter must be initialized'

		# Step 1: Get tool capability to check requirements
		cap = self.tool_registry_adapter.get_tool_capability(tool_name)

		# Step 2: Initialize browser session if the tool requires it
		if cap is not None and cap.requires_browser and not self.browser_session:
			await self._init_browser_session()

		# Step 3: Execute via unified adapter
		# This handles both core tools (via name mapping) and external tools (via registered executors)
		try:
			result = await self.tool_registry_adapter.execute_tool(
				tool_name=tool_name,
				arguments=arguments,
				browser_session=self.browser_session,
				page_extraction_llm=self.llm,
				file_system=self.file_system,
			)
		except ValueError as e:
			# Handle unknown tool
			if 'Tool not found' in str(e):
				return f'Unknown tool: {tool_name}'
			raise

		# Step 4: Handle special result with embedded image data (_state_json + _screenshot_b64 pattern)
		# This comes from browser_get_state and browser_screenshot executors
		if isinstance(result, dict) and ('_state_json' in result or '_screenshot_b64' in result):
			content_list: list[types.TextContent | types.ImageContent] = []
			if '_state_json' in result and result['_state_json'] is not None:
				content_list.append(types.TextContent(type='text', text=str(result['_state_json'])))
			if '_screenshot_b64' in result and result['_screenshot_b64'] is not None:
				content_list.append(types.ImageContent(type='image', data=str(result['_screenshot_b64']), mimeType='image/png'))
			return content_list

		# Step 5: Convert all other result types to string for MCP compatibility
		# MCP CallToolResult requires str or list[TextContent|ImageContent]
		if isinstance(result, ActionResult):
			return self._format_result_from_action_result(result)
		if result is None:
			return 'Success'
		if isinstance(result, str):
			return result
		# For dict/list/other Pydantic models, serialize to JSON string
		import json

		try:
			if isinstance(result, dict):
				result_dict: Any = result
			elif hasattr(result, 'model_dump'):
				result_dict = result.model_dump()
			elif hasattr(result, '__dict__'):
				result_dict = result.__dict__
			else:
				result_dict = result
			return json.dumps(result_dict, default=str, ensure_ascii=False)
		except Exception:
			return str(result)

	async def _init_browser_session(self, allowed_domains: list[str] | None = None, **kwargs):
		"""Initialize browser session using config"""
		if self.browser_session:
			return

		# Ensure all logging goes to stderr before browser initialization
		_ensure_all_loggers_use_stderr()

		logger.debug('Initializing browser session...')

		# Get profile config
		profile_config = get_default_profile(self.config)

		# Merge profile config with defaults and overrides
		profile_data = {
			'downloads_path': str(Path.home() / 'Downloads' / 'browser-use-mcp'),
			'wait_between_actions': 0.5,
			'keep_alive': True,
			'user_data_dir': '~/.config/browseruse/profiles/default',
			'device_scale_factor': 1.0,
			'disable_security': False,
			'headless': False,
			**profile_config,  # Config values override defaults
		}

		# Tool parameter overrides (highest priority)
		if allowed_domains is not None:
			profile_data['allowed_domains'] = allowed_domains

		# Merge any additional kwargs that are valid BrowserProfile fields
		for key, value in kwargs.items():
			profile_data[key] = value

		# Create browser profile
		profile = BrowserProfile(**profile_data)

		# Create browser session
		self.browser_session = BrowserSession(browser_profile=profile)
		await self.browser_session.start()

		# Track the session for management
		self._track_session(self.browser_session)

		# Initialize tool metadata from the unified ToolRegistryAdapter
		# (already initialized in __init__, but refresh in case tools changed)
		self._init_tool_registry()

		# Initialize LLM from config
		llm_config = get_default_llm(self.config)
		base_url = llm_config.get('base_url', None)
		kwargs = {}
		if base_url:
			kwargs['base_url'] = base_url
		if api_key := llm_config.get('api_key'):
			self.llm = ChatOpenAI(
				model=llm_config.get('model', 'gpt-o4-mini'),
				api_key=api_key,
				temperature=llm_config.get('temperature', 0.7),
				**kwargs,
			)

		# Initialize FileSystem for extraction actions
		file_system_path = profile_config.get('file_system_path', '~/.browser-use-mcp')
		self.file_system = FileSystem(base_dir=Path(file_system_path).expanduser())

		logger.debug('Browser session initialized')

	def _init_tool_registry(self) -> None:
		"""Initialize the unified ToolRegistryAdapter via the centralized MCP registry.

		All tool metadata (capabilities, param schemas, descriptions) and executor
		factories now live in browser_use/mcp/registry.py. This method simply
		delegates to register_all_mcp_tools() with the current adapter and server.
		"""
		from browser_use.mcp.registry import register_all_mcp_tools

		assert self.tools is not None, 'Tools must be initialized before _init_tool_registry'
		adapter = self.tools.get_tool_registry_adapter()

		# Single call wires in ALL core mappings, MCP-specific capabilities,
		# custom param overrides, and server-bound executors.
		register_all_mcp_tools(adapter=adapter, server=self)

		# Store the unified adapter
		self.tool_registry_adapter = adapter

	def _format_result_from_action_result(self, action_result: Any) -> list[types.TextContent | types.ImageContent]:
		"""Format any tool result using the unified ToolResult format for MCP responses.

		This is the single entry point for formatting all tool execution results.
		It handles:
		- ActionResult objects (from core tools)
		- Strings (plain text results)
		- Dicts (structured results)
		- Lists (already formatted content, e.g., with images)
		- Tuples (e.g., (metadata_json, screenshot_b64))

		All output is converted to MCP TextContent/ImageContent objects.
		"""
		from browser_use.tools.registry.views import ToolResult

		# Handle lists of already-formatted content (backward compatibility)
		if isinstance(action_result, list) and action_result:
			converted = []
			for item in action_result:
				if isinstance(item, dict):
					if item.get('type') == 'text':
						converted.append(types.TextContent(type='text', text=item.get('text', '')))
					elif item.get('type') == 'image':
						converted.append(
							types.ImageContent(
								type='image',
								data=item.get('data', ''),
								mimeType=item.get('mimeType', 'image/png'),
							)
						)
				else:
					converted.append(types.TextContent(type='text', text=str(item)))
			return converted

		# Handle tuples (e.g., (metadata_json, screenshot_b64) from screenshot tools)
		if isinstance(action_result, tuple) and len(action_result) >= 1:
			text_content = str(action_result[0]) if action_result[0] is not None else ''
			images = []
			if len(action_result) >= 2 and action_result[1] is not None:
				images.append({'data': action_result[1], 'mime_type': 'image/png'})
			action_result = {'extracted_content': text_content, 'images': images}

		# Convert to ToolResult and then to MCP format
		tool_result = ToolResult.from_action_result(action_result)
		mcp_content = tool_result.to_mcp_content()

		# Convert dict format to MCP content objects
		result = []
		for item in mcp_content:
			if item.get('type') == 'text':
				result.append(types.TextContent(type='text', text=item.get('text', '')))
			elif item.get('type') == 'image':
				result.append(
					types.ImageContent(
						type='image',
						data=item.get('data', ''),
						mimeType=item.get('mimeType', 'image/png'),
					)
				)
		return result

	async def _retry_with_browser_use_agent(
		self,
		task: str,
		max_steps: int = 100,
		model: str | None = None,
		allowed_domains: list[str] | None = None,
		use_vision: bool = True,
	) -> str:
		"""Run an autonomous agent task."""
		logger.debug(f'Running agent task: {task}')

		# Get LLM config
		llm_config = get_default_llm(self.config)

		# Get LLM provider
		model_provider = llm_config.get('model_provider') or os.getenv('MODEL_PROVIDER')

		# Get Bedrock-specific config
		if model_provider and model_provider.lower() == 'bedrock':
			llm_model = llm_config.get('model') or os.getenv('MODEL') or 'us.anthropic.claude-sonnet-4-20250514-v1:0'
			aws_region = llm_config.get('region') or os.getenv('REGION')
			if not aws_region:
				aws_region = 'us-east-1'
			aws_sso_auth = llm_config.get('aws_sso_auth', False)
			llm = ChatAWSBedrock(
				model=llm_model,  # or any Bedrock model
				aws_region=aws_region,
				aws_sso_auth=aws_sso_auth,
			)
		else:
			api_key = llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')
			if not api_key:
				return 'Error: OPENAI_API_KEY not set in config or environment'

			# Use explicit model from tool call, otherwise fall back to configured default
			llm_model = model or llm_config.get('model', 'gpt-4o')

			base_url = llm_config.get('base_url', None)
			kwargs = {}
			if base_url:
				kwargs['base_url'] = base_url
			llm = ChatOpenAI(
				model=llm_model,
				api_key=api_key,
				temperature=llm_config.get('temperature', 0.7),
				**kwargs,
			)

		# Get profile config and merge with tool parameters
		profile_config = get_default_profile(self.config)

		# Override allowed_domains only when the client supplied a non-empty list.
		# Treating an empty list as an override would silently disable any
		# admin-configured allowlist on the default profile, since
		# SecurityWatchdog interprets allowed_domains=[] as "no restrictions".
		if allowed_domains:
			profile_config['allowed_domains'] = allowed_domains

		# Create browser profile using config
		profile = BrowserProfile(**profile_config)

		# Create and run agent
		agent = Agent(
			task=task,
			llm=llm,
			browser_profile=profile,
			use_vision=use_vision,
		)

		try:
			history = await agent.run(max_steps=max_steps)

			# Format results
			results = []
			results.append(f'Task completed in {len(history.history)} steps')
			results.append(f'Success: {history.is_successful()}')

			# Get final result if available
			final_result = history.final_result()
			if final_result:
				results.append(f'\nFinal result:\n{final_result}')

			# Include any errors
			errors = history.errors()
			if errors:
				results.append(f'\nErrors encountered:\n{json.dumps(errors, indent=2)}')

			# Include URLs visited
			urls = history.urls()
			if urls:
				# Filter out None values and convert to strings
				valid_urls = [str(url) for url in urls if url is not None]
				if valid_urls:
					results.append(f'\nURLs visited: {", ".join(valid_urls)}')

			return '\n'.join(results)

		except Exception as e:
			logger.error(f'Agent task failed: {e}', exc_info=True)
			return f'Agent task failed: {str(e)}'
		finally:
			# Clean up
			await agent.close()

	async def _navigate(self, url: str, new_tab: bool = False) -> str:
		"""Navigate to a URL."""
		if not self.browser_session:
			return 'Error: No browser session active'

		# Update session activity
		self._update_session_activity(self.browser_session.id)

		from browser_use.browser.events import NavigateToUrlEvent

		if new_tab:
			event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=True))
			await event
			return f'Opened new tab with URL: {url}'
		else:
			event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url))
			await event
			return f'Navigated to: {url}'

	async def _click(
		self,
		index: int | None = None,
		coordinate_x: int | None = None,
		coordinate_y: int | None = None,
		new_tab: bool = False,
	) -> str:
		"""Click an element by index or at viewport coordinates."""
		if not self.browser_session:
			return 'Error: No browser session active'

		# Update session activity
		self._update_session_activity(self.browser_session.id)

		# Coordinate-based clicking
		if coordinate_x is not None and coordinate_y is not None:
			from browser_use.browser.events import ClickCoordinateEvent

			event = self.browser_session.event_bus.dispatch(
				ClickCoordinateEvent(coordinate_x=coordinate_x, coordinate_y=coordinate_y)
			)
			await event
			return f'Clicked at coordinates ({coordinate_x}, {coordinate_y})'

		# Index-based clicking
		if index is None:
			return 'Error: Provide either index or both coordinate_x and coordinate_y'

		# Get the element
		element = await self.browser_session.get_dom_element_by_index(index)
		if not element:
			return f'Element with index {index} not found'

		if new_tab:
			# For links, extract href and open in new tab
			href = element.attributes.get('href')
			if href:
				# Convert relative href to absolute URL
				state = await self.browser_session.get_browser_state_summary()
				current_url = state.url
				if href.startswith('/'):
					# Relative URL - construct full URL
					from urllib.parse import urlparse

					parsed = urlparse(current_url)
					full_url = f'{parsed.scheme}://{parsed.netloc}{href}'
				else:
					full_url = href

				# Open link in new tab
				from browser_use.browser.events import NavigateToUrlEvent

				event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=full_url, new_tab=True))
				await event
				return f'Clicked element {index} and opened in new tab {full_url[:20]}...'
			else:
				# For non-link elements, just do a normal click
				from browser_use.browser.events import ClickElementEvent

				event = self.browser_session.event_bus.dispatch(ClickElementEvent(node=element))
				await event
				return f'Clicked element {index} (new tab not supported for non-link elements)'
		else:
			# Normal click
			from browser_use.browser.events import ClickElementEvent

			event = self.browser_session.event_bus.dispatch(ClickElementEvent(node=element))
			await event
			return f'Clicked element {index}'

	async def _type_text(self, index: int, text: str) -> str:
		"""Type text into an element."""
		if not self.browser_session:
			return 'Error: No browser session active'

		element = await self.browser_session.get_dom_element_by_index(index)
		if not element:
			return f'Element with index {index} not found'

		from browser_use.browser.events import TypeTextEvent

		# Conservative heuristic to detect potentially sensitive data
		# Only flag very obvious patterns to minimize false positives
		is_potentially_sensitive = len(text) >= 6 and (
			# Email pattern: contains @ and a domain-like suffix
			('@' in text and '.' in text.split('@')[-1] if '@' in text else False)
			# Mixed alphanumeric with reasonable complexity (likely API keys/tokens)
			or (
				len(text) >= 16
				and any(char.isdigit() for char in text)
				and any(char.isalpha() for char in text)
				and any(char in '.-_' for char in text)
			)
		)

		# Use generic key names to avoid information leakage about detection patterns
		sensitive_key_name = None
		if is_potentially_sensitive:
			if '@' in text and '.' in text.split('@')[-1]:
				sensitive_key_name = 'email'
			else:
				sensitive_key_name = 'credential'

		event = self.browser_session.event_bus.dispatch(
			TypeTextEvent(node=element, text=text, is_sensitive=is_potentially_sensitive, sensitive_key_name=sensitive_key_name)
		)
		await event

		if is_potentially_sensitive:
			if sensitive_key_name:
				return f'Typed <{sensitive_key_name}> into element {index}'
			else:
				return f'Typed <sensitive> into element {index}'
		else:
			return f"Typed '{text}' into element {index}"

	async def _get_browser_state(self, include_screenshot: bool = False) -> tuple[str, str | None]:
		"""Get current browser state. Returns (state_json, screenshot_b64 | None)."""
		if not self.browser_session:
			return 'Error: No browser session active', None

		state = await self.browser_session.get_browser_state_summary()

		result: dict[str, Any] = {
			'url': state.url,
			'title': state.title,
			'tabs': [{'url': tab.url, 'title': tab.title} for tab in state.tabs],
			'interactive_elements': [],
		}

		# Add viewport info so the LLM knows the coordinate space
		if state.page_info:
			pi = state.page_info
			result['viewport'] = {
				'width': pi.viewport_width,
				'height': pi.viewport_height,
			}
			result['page'] = {
				'width': pi.page_width,
				'height': pi.page_height,
			}
			result['scroll'] = {
				'x': pi.scroll_x,
				'y': pi.scroll_y,
			}

		# Add interactive elements with their indices
		for index, element in state.dom_state.selector_map.items():
			elem_info: dict[str, Any] = {
				'index': index,
				'tag': element.tag_name,
				'text': element.get_all_children_text(max_depth=2)[:100],
			}
			if element.attributes.get('placeholder'):
				elem_info['placeholder'] = element.attributes['placeholder']
			if element.attributes.get('href'):
				elem_info['href'] = element.attributes['href']
			result['interactive_elements'].append(elem_info)

		# Return screenshot separately as ImageContent instead of embedding base64 in JSON
		screenshot_b64 = None
		if include_screenshot and state.screenshot:
			screenshot_b64 = state.screenshot
			# Include viewport dimensions in JSON so LLM can map pixels to coordinates
			if state.page_info:
				result['screenshot_dimensions'] = {
					'width': state.page_info.viewport_width,
					'height': state.page_info.viewport_height,
				}

		return json.dumps(result, indent=2), screenshot_b64

	async def _get_html(self, selector: str | None = None) -> str:
		"""Get raw HTML of the page or a specific element."""
		if not self.browser_session:
			return 'Error: No browser session active'

		self._update_session_activity(self.browser_session.id)

		cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=None, focus=False)
		if not cdp_session:
			return 'Error: No active CDP session'

		if selector:
			js = (
				f'(function(){{ const el = document.querySelector({json.dumps(selector)}); return el ? el.outerHTML : null; }})()'
			)
		else:
			js = 'document.documentElement.outerHTML'

		result = await cdp_session.cdp_client.send.Runtime.evaluate(
			params={'expression': js, 'returnByValue': True},
			session_id=cdp_session.session_id,
		)
		html = result.get('result', {}).get('value')
		if html is None:
			return f'No element found for selector: {selector}' if selector else 'Error: Could not get page HTML'
		return html

	async def _screenshot(self, full_page: bool = False) -> tuple[str, str | None]:
		"""Take a screenshot. Returns (metadata_json, screenshot_b64 | None)."""
		if not self.browser_session:
			return 'Error: No browser session active', None

		import base64

		self._update_session_activity(self.browser_session.id)

		data = await self.browser_session.take_screenshot(full_page=full_page)
		b64 = base64.b64encode(data).decode()

		# Return screenshot separately as ImageContent instead of embedding base64 in JSON
		state = await self.browser_session.get_browser_state_summary()
		result: dict[str, Any] = {
			'size_bytes': len(data),
		}
		if state.page_info:
			result['viewport'] = {
				'width': state.page_info.viewport_width,
				'height': state.page_info.viewport_height,
			}
		return json.dumps(result), b64

	async def _extract_content(self, query: str, extract_links: bool = False) -> str:
		"""Extract content from current page."""
		if not self.llm:
			return 'Error: LLM not initialized (set OPENAI_API_KEY)'

		if not self.file_system:
			return 'Error: FileSystem not initialized'

		if not self.browser_session:
			return 'Error: No browser session active'

		if not self.tools:
			return 'Error: Tools not initialized'

		state = await self.browser_session.get_browser_state_summary()

		# Use the extract action
		# Create a dynamic action model that matches the tools's expectations
		from pydantic import create_model

		# Create action model dynamically
		ExtractAction = create_model(
			'ExtractAction',
			__base__=ActionModel,
			extract=dict[str, Any],
		)

		# Use model_validate because Pyright does not understand the dynamic model
		action = ExtractAction.model_validate(
			{
				'extract': {'query': query, 'extract_links': extract_links},
			}
		)
		action_result = await self.tools.act(
			action=action,
			browser_session=self.browser_session,
			page_extraction_llm=self.llm,
			file_system=self.file_system,
		)

		return action_result.extracted_content or 'No content extracted'

	async def _scroll(self, direction: str = 'down') -> str:
		"""Scroll the page."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from browser_use.browser.events import ScrollEvent

		# Scroll by a standard amount (500 pixels)
		event = self.browser_session.event_bus.dispatch(
			ScrollEvent(
				direction=direction,  # type: ignore
				amount=500,
			)
		)
		await event
		return f'Scrolled {direction}'

	async def _go_back(self) -> str:
		"""Go back in browser history."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from browser_use.browser.events import GoBackEvent

		event = self.browser_session.event_bus.dispatch(GoBackEvent())
		await event
		return 'Navigated back'

	async def _close_browser(self) -> str:
		"""Close the browser session."""
		if self.browser_session:
			from browser_use.browser.events import BrowserStopEvent

			event = self.browser_session.event_bus.dispatch(BrowserStopEvent())
			await event
			self.browser_session = None
			self.tools = None
			return 'Browser closed'
		return 'No browser session to close'

	async def _list_tabs(self) -> str:
		"""List all open tabs."""
		if not self.browser_session:
			return 'Error: No browser session active'

		tabs_info = await self.browser_session.get_tabs()
		tabs = []
		for i, tab in enumerate(tabs_info):
			tabs.append({'tab_id': tab.target_id[-4:], 'url': tab.url, 'title': tab.title or ''})
		return json.dumps(tabs, indent=2)

	async def _switch_tab(self, tab_id: str) -> str:
		"""Switch to a different tab."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from browser_use.browser.events import SwitchTabEvent

		target_id = await self.browser_session.get_target_id_from_tab_id(tab_id)
		event = self.browser_session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
		await event
		state = await self.browser_session.get_browser_state_summary()
		return f'Switched to tab {tab_id}: {state.url}'

	async def _close_tab(self, tab_id: str) -> str:
		"""Close a specific tab."""
		if not self.browser_session:
			return 'Error: No browser session active'

		from browser_use.browser.events import CloseTabEvent

		target_id = await self.browser_session.get_target_id_from_tab_id(tab_id)
		event = self.browser_session.event_bus.dispatch(CloseTabEvent(target_id=target_id))
		await event
		current_url = await self.browser_session.get_current_page_url()
		return f'Closed tab # {tab_id}, now on {current_url}'

	def _track_session(self, session: BrowserSession) -> None:
		"""Track a browser session for management."""
		self.active_sessions[session.id] = {
			'session': session,
			'created_at': time.time(),
			'last_activity': time.time(),
			'url': getattr(session, 'current_url', None),
		}

	def _update_session_activity(self, session_id: str) -> None:
		"""Update the last activity time for a session."""
		if session_id in self.active_sessions:
			self.active_sessions[session_id]['last_activity'] = time.time()

	async def _list_sessions(self) -> str:
		"""List all active browser sessions."""
		if not self.active_sessions:
			return 'No active browser sessions'

		sessions_info = []
		for session_id, session_data in self.active_sessions.items():
			session = session_data['session']
			created_at = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_data['created_at']))
			last_activity = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session_data['last_activity']))

			# Check if session is still active
			is_active = hasattr(session, 'cdp_client') and session.cdp_client is not None

			sessions_info.append(
				{
					'session_id': session_id,
					'created_at': created_at,
					'last_activity': last_activity,
					'active': is_active,
					'current_url': session_data.get('url', 'Unknown'),
					'age_minutes': (time.time() - session_data['created_at']) / 60,
				}
			)

		return json.dumps(sessions_info, indent=2)

	async def _close_session(self, session_id: str) -> str:
		"""Close a specific browser session."""
		if session_id not in self.active_sessions:
			return f'Session {session_id} not found'

		session_data = self.active_sessions[session_id]
		session = session_data['session']

		try:
			# Close the session
			if hasattr(session, 'kill'):
				await session.kill()
			elif hasattr(session, 'close'):
				await session.close()

			# Remove from tracking
			del self.active_sessions[session_id]

			# If this was the current session, clear it
			if self.browser_session and self.browser_session.id == session_id:
				self.browser_session = None
				self.tools = None

			return f'Successfully closed session {session_id}'
		except Exception as e:
			return f'Error closing session {session_id}: {str(e)}'

	async def _close_all_sessions(self) -> str:
		"""Close all active browser sessions."""
		if not self.active_sessions:
			return 'No active sessions to close'

		closed_count = 0
		errors = []

		for session_id in list(self.active_sessions.keys()):
			try:
				result = await self._close_session(session_id)
				if 'Successfully closed' in result:
					closed_count += 1
				else:
					errors.append(f'{session_id}: {result}')
			except Exception as e:
				errors.append(f'{session_id}: {str(e)}')

		# Clear current session references
		self.browser_session = None
		self.tools = None

		result = f'Closed {closed_count} sessions'
		if errors:
			result += f'. Errors: {"; ".join(errors)}'

		return result

	async def _cleanup_expired_sessions(self) -> None:
		"""Background task to clean up expired sessions."""
		current_time = time.time()
		timeout_seconds = self.session_timeout_minutes * 60

		expired_sessions = []
		for session_id, session_data in self.active_sessions.items():
			last_activity = session_data['last_activity']
			if current_time - last_activity > timeout_seconds:
				expired_sessions.append(session_id)

		for session_id in expired_sessions:
			try:
				await self._close_session(session_id)
				logger.info(f'Auto-closed expired session {session_id}')
			except Exception as e:
				logger.error(f'Error auto-closing session {session_id}: {e}')

	async def _start_cleanup_task(self) -> None:
		"""Start the background cleanup task."""

		async def cleanup_loop():
			while True:
				try:
					await self._cleanup_expired_sessions()
					# Check every 2 minutes
					await asyncio.sleep(120)
				except Exception as e:
					logger.error(f'Error in cleanup task: {e}')
					await asyncio.sleep(120)

		self._cleanup_task = create_task_with_error_handling(cleanup_loop(), name='mcp_cleanup_loop', suppress_exceptions=True)

	async def run(self):
		"""Run the MCP server."""
		# Start the cleanup task
		await self._start_cleanup_task()

		if sys.stdin is None:
			raise RuntimeError('MCP stdio transport requires stdin, but this process was launched without one.')

		async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
			try:
				await self.server.run(
					read_stream,
					write_stream,
					InitializationOptions(
						server_name='browser-use',
						server_version='0.1.0',
						capabilities=self.server.get_capabilities(
							notification_options=NotificationOptions(),
							experimental_capabilities={},
						),
					),
				)
			except BrokenPipeError:
				logger.warning('MCP client disconnected while writing to stdio; shutting down server cleanly.')


async def main(session_timeout_minutes: int = 10):
	if not MCP_AVAILABLE:
		print('MCP SDK is required. Install with: pip install mcp', file=sys.stderr)
		sys.exit(1)

	server = BrowserUseServer(session_timeout_minutes=session_timeout_minutes)
	server._telemetry.capture(
		MCPServerTelemetryEvent(
			version=get_browser_use_version(),
			action='start',
			parent_process_cmdline=get_parent_process_cmdline(),
		)
	)

	try:
		await server.run()
	finally:
		duration = time.time() - server._start_time
		server._telemetry.capture(
			MCPServerTelemetryEvent(
				version=get_browser_use_version(),
				action='stop',
				duration_seconds=duration,
				parent_process_cmdline=get_parent_process_cmdline(),
			)
		)
		server._telemetry.flush()


if __name__ == '__main__':
	asyncio.run(main())
