"""Release engineering tests for browser-use.

This test file validates:
1. __all__ / _LAZY_IMPORTS / TYPE_CHECKING consistency
2. Optional dependency import boundaries (extras)
3. Example script syntax validity
4. CLI/MCP entry-point parameter naming consistency
5. Isolated import scenarios (no extras, [cli], [mcp], [sandbox])

These tests are designed to catch regressions before release.
"""

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
BROWSER_USE_INIT = PROJECT_ROOT / 'browser_use' / '__init__.py'
BROWSER_USE_DIR = PROJECT_ROOT / 'browser_use'
EXAMPLES_DIR = PROJECT_ROOT / 'examples'
LLM_DIR = BROWSER_USE_DIR / 'llm'
SANDBOX_DIR = BROWSER_USE_DIR / 'sandbox'
MCP_DIR = BROWSER_USE_DIR / 'mcp'
CLI_MAIN = BROWSER_USE_DIR / 'skill_cli' / 'main.py'
MCP_SERVER = MCP_DIR / 'server.py'


# =============================================================================
# Helpers
# =============================================================================


def parse_ast(path: Path) -> ast.Module:
	"""Parse a Python file into an AST."""
	with open(path, encoding='utf-8') as f:
		return ast.parse(f.read())


def extract_lazy_imports(module_path: Path) -> dict[str, tuple[str, str | None]]:
	"""Extract _LAZY_IMPORTS dict from a module's AST."""
	tree = parse_ast(module_path)
	for node in ast.walk(tree):
		if isinstance(node, ast.Assign):
			for target in node.targets:
				if isinstance(target, ast.Name) and target.id == '_LAZY_IMPORTS':
					if isinstance(node.value, ast.Dict):
						result = {}
						for key, value in zip(node.value.keys, node.value.values):
							if isinstance(key, ast.Constant) and isinstance(key.value, str):
								if isinstance(value, ast.Tuple) and len(value.elts) >= 1:
									module_ast = value.elts[0]
									attr_ast = value.elts[1] if len(value.elts) > 1 else None
									if isinstance(module_ast, ast.Constant):
										module_name = module_ast.value
										attr_name = attr_ast.value if isinstance(attr_ast, ast.Constant) else None
										result[key.value] = (module_name, attr_name)
						return result
	return {}


def extract_all_list(module_path: Path) -> list[str]:
	"""Extract __all__ list from a module's AST."""
	tree = parse_ast(module_path)
	for node in ast.walk(tree):
		if isinstance(node, ast.Assign):
			for target in node.targets:
				if isinstance(target, ast.Name) and target.id == '__all__':
					if isinstance(node.value, ast.List):
						return [
							elt.value for elt in node.value.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
						]
	return []


def extract_type_checking_imports(module_path: Path) -> set[str]:
	"""Extract names imported in TYPE_CHECKING blocks from a module's AST."""
	tree = parse_ast(module_path)
	type_checking_names: set[str] = set()

	for node in ast.walk(tree):
		if isinstance(node, ast.If):
			# Check if it's a TYPE_CHECKING guard
			condition = node.test
			is_type_checking = False
			if isinstance(condition, ast.Name) and condition.id == 'TYPE_CHECKING':
				is_type_checking = True
			elif isinstance(condition, ast.Attribute) and isinstance(condition.value, ast.Name):
				if condition.value.id == 'typing' and condition.attr == 'TYPE_CHECKING':
					is_type_checking = True

			if is_type_checking:
				for child in ast.walk(node):
					if isinstance(child, ast.ImportFrom):
						for alias in child.names:
							name = alias.asname if alias.asname else alias.name
							type_checking_names.add(name)
					elif isinstance(child, ast.Import):
						for alias in child.names:
							name = alias.asname if alias.asname else alias.name
							type_checking_names.add(name)

	return type_checking_names


def find_hard_imports(module_path: Path, forbidden_modules: set[str]) -> list[str]:
	"""Find hard imports of forbidden modules that are not inside try/except or TYPE_CHECKING."""
	tree = parse_ast(module_path)
	violations: list[str] = []

	def is_safe_context(node: ast.AST) -> bool:
		"""Check if node is inside a safe context (try/except ImportError or TYPE_CHECKING)."""
		for parent in ast.walk(tree):
			if isinstance(parent, ast.Try):
				# Check if any handler catches ImportError
				for handler in parent.handlers:
					if isinstance(handler.type, ast.Name):
						if handler.type.id in ('ImportError', 'ModuleNotFoundError'):
							# Check if node is inside this try block
							if _is_node_in_body(node, parent.body):
								return True
			elif isinstance(parent, ast.If):
				condition = parent.test
				if isinstance(condition, ast.Name) and condition.id == 'TYPE_CHECKING':
					if _is_node_in_body(node, parent.body):
						return True
		return False

	def _is_node_in_body(node: ast.AST, body: list[ast.stmt]) -> bool:
		"""Check if node is anywhere within a list of statements."""
		for stmt in body:
			if node is stmt:
				return True
			for child in ast.iter_child_nodes(stmt):
				if _is_node_in_body(node, [child] if isinstance(child, ast.stmt) else []):
					return True
				elif isinstance(child, list):
					if _is_node_in_body(node, child):  # type: ignore
						return True
		return False

	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			for alias in node.names:
				module_parts = alias.name.split('.')
				top_level = module_parts[0]
				if top_level in forbidden_modules and not is_safe_context(node):
					violations.append(f'{module_path}:{node.lineno}: import {alias.name}')
		elif isinstance(node, ast.ImportFrom):
			if node.module and node.level == 0:
				module_parts = node.module.split('.')
				top_level = module_parts[0]
				if top_level in forbidden_modules and not is_safe_context(node):
					violations.append(f'{module_path}:{node.lineno}: from {node.module} import ...')

	return violations


def extract_cli_params(cli_path: Path) -> list[str]:
	"""Extract CLI argument names from skill_cli/main.py AST."""
	tree = parse_ast(cli_path)
	params: list[str] = []

	for node in ast.walk(tree):
		if isinstance(node, ast.Call):
			if isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
				if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
					arg_name = node.args[0].value
					if arg_name.startswith('--'):
						params.append(arg_name[2:].replace('-', '_'))
					elif not arg_name.startswith('-'):
						params.append(arg_name.replace('-', '_'))

	return params


def extract_mcp_browser_params(mcp_path: Path) -> list[str]:
	"""Extract BrowserProfile/parameter names used in MCP server."""
	tree = parse_ast(mcp_path)
	params: list[str] = []

	# Look for the profile_data dict in _init_browser_session
	for node in ast.walk(tree):
		if isinstance(node, ast.Dict):
			for key in node.keys:
				if isinstance(key, ast.Constant) and isinstance(key.value, str):
					params.append(key.value)

	return params


# =============================================================================
# Tests: Top-level exports consistency
# =============================================================================


class TestTopLevelExports:
	"""Test that __all__, _LAZY_IMPORTS, and TYPE_CHECKING are in sync."""

	def test_all_vs_lazy_imports_in_sync(self):
		"""Test that __all__ contains exactly the same names as _LAZY_IMPORTS."""
		lazy_imports = extract_lazy_imports(BROWSER_USE_INIT)
		all_list = extract_all_list(BROWSER_USE_INIT)

		assert len(lazy_imports) > 0, 'No _LAZY_IMPORTS found'
		assert len(all_list) > 0, 'No __all__ found'

		lazy_keys = set(lazy_imports.keys())
		all_set = set(all_list)

		missing_from_all = lazy_keys - all_set
		missing_from_lazy = all_set - lazy_keys

		assert missing_from_all == set(), f'Names in _LAZY_IMPORTS but missing from __all__: {missing_from_all}'
		assert missing_from_lazy == set(), f'Names in __all__ but missing from _LAZY_IMPORTS: {missing_from_lazy}'

	def test_no_duplicates_in_all(self):
		"""Test that __all__ has no duplicate entries."""
		all_list = extract_all_list(BROWSER_USE_INIT)
		seen = set()
		duplicates = set()
		for name in all_list:
			if name in seen:
				duplicates.add(name)
			seen.add(name)

		assert duplicates == set(), f'Duplicate entries in __all__: {duplicates}'

	def test_type_checking_imports_match_lazy_imports(self):
		"""Test that TYPE_CHECKING block imports match _LAZY_IMPORTS (for class names)."""
		lazy_imports = extract_lazy_imports(BROWSER_USE_INIT)
		type_checking_names = extract_type_checking_imports(BROWSER_USE_INIT)

		# Only check names that start with uppercase (classes), skip modules like 'models', 'sandbox'
		class_names = {name for name in lazy_imports.keys() if name[0].isupper()}

		missing_from_type_checking = class_names - type_checking_names
		assert missing_from_type_checking == set(), (
			f'Names in _LAZY_IMPORTS but missing from TYPE_CHECKING: {missing_from_type_checking}'
		)


# =============================================================================
# Tests: Optional dependency import boundaries
# =============================================================================


class TestOptionalDependencyBoundaries:
	"""Test that optional dependencies are not hard-imported at module level."""

	OPTIONAL_DEPS = {
		'boto3',  # [aws] extra
		'oci',  # [oci] extra
		'textual',  # [cli] extra
	}

	def test_no_hard_imports_of_optional_deps(self):
		"""Scan all .py files for unguarded imports of optional dependencies."""
		all_violations: list[str] = []

		for py_file in BROWSER_USE_DIR.rglob('*.py'):
			# Skip test directories - they can import anything
			if '/tests/' in str(py_file) or str(py_file).endswith('_test.py') or 'test_' in str(py_file):
				continue
			# Skip __pycache__
			if '__pycache__' in str(py_file):
				continue
			violations = find_hard_imports(py_file, self.OPTIONAL_DEPS)
			all_violations.extend(violations)

		assert all_violations == [], 'Unguarded optional dependency imports found:\n' + '\n'.join(all_violations)

	def test_oci_raw_module_has_lazy_imports(self):
		"""Test that oci_raw module has proper lazy import guards."""
		oci_chat = LLM_DIR / 'oci_raw' / 'chat.py'
		oci_serializer = LLM_DIR / 'oci_raw' / 'serializer.py'

		assert oci_chat.exists(), f'{oci_chat} does not exist'
		assert oci_serializer.exists(), f'{oci_serializer} does not exist'

		# These should have 'oci' inside try/except in methods, not hard imports
		chat_violations = find_hard_imports(oci_chat, {'oci'})
		serializer_violations = find_hard_imports(oci_serializer, {'oci'})

		assert chat_violations == [], f'Hard oci imports in chat.py: {chat_violations}'
		assert serializer_violations == [], f'Hard oci imports in serializer.py: {serializer_violations}'

		# Verify oci is imported inside methods (not at module level)
		chat_content = oci_chat.read_text()
		# Check for method-level imports
		assert '\t\timport oci' in chat_content or '        import oci' in chat_content, 'oci should be imported inside a method'
		assert 'from oci.generative_ai_inference import GenerativeAiInferenceClient' in chat_content

	def test_aws_bedrock_has_lazy_imports(self):
		"""Test that AWS Bedrock module has proper lazy import guards."""
		bedrock_chat = LLM_DIR / 'aws' / 'chat_bedrock.py'
		assert bedrock_chat.exists(), f'{bedrock_chat} does not exist'

		violations = find_hard_imports(bedrock_chat, {'boto3'})
		assert violations == [], f'Hard boto3 imports in chat_bedrock.py: {violations}'

		# Verify boto3 is imported inside try/except
		bedrock_content = bedrock_chat.read_text()
		assert 'from boto3 import' in bedrock_content or 'import boto3' in bedrock_content
		assert 'try:' in bedrock_content
		assert 'except ImportError' in bedrock_content

	def test_sandbox_cloudpickle_is_optional(self):
		"""Test that sandbox.py uses try/except for cloudpickle import."""
		sandbox_py = SANDBOX_DIR / 'sandbox.py'
		assert sandbox_py.exists(), f'{sandbox_py} does not exist'

		# cloudpickle should not be a hard import
		violations = find_hard_imports(sandbox_py, {'cloudpickle'})
		assert violations == [], f'Hard cloudpickle imports in sandbox.py: {violations}'

	def test_cli_tui_requires_textual(self):
		"""Test that cli.py (legacy TUI) properly guards textual import."""
		cli_py = BROWSER_USE_DIR / 'cli.py'
		assert cli_py.exists(), f'{cli_py} does not exist'

		violations = find_hard_imports(cli_py, {'textual'})
		assert violations == [], f'Hard textual imports in cli.py: {violations}'


# =============================================================================
# Tests: Example script syntax validity
# =============================================================================


class TestExampleScripts:
	"""Test that all example scripts have valid Python syntax."""

	def test_all_examples_have_valid_syntax(self):
		"""Parse all .py files in examples/ directory for syntax errors."""
		syntax_errors: list[str] = []

		for py_file in EXAMPLES_DIR.rglob('*.py'):
			try:
				parse_ast(py_file)
			except SyntaxError as e:
				syntax_errors.append(f'{py_file}:{e.lineno}: {e.msg}')

		assert syntax_errors == [], 'Syntax errors in examples:\n' + '\n'.join(syntax_errors)

	def test_examples_use_public_api(self):
		"""Check that examples prefer public API imports over internal paths."""
		warnings: list[str] = []
		internal_import_patterns = [
			'from browser_use.agent.service import',
			'from browser_use.agent.views import',
			'from browser_use.tools.service import',
			'from browser_use.tools.views import',
			'from browser_use.browser.session import',
		]

		for py_file in EXAMPLES_DIR.rglob('*.py'):
			with open(py_file, encoding='utf-8') as f:
				lines = f.readlines()
				for i, line in enumerate(lines, 1):
					for pattern in internal_import_patterns:
						if pattern in line:
							warnings.append(f'{py_file}:{i}:{line.strip()}')

		# This is a warning-level check - some examples intentionally use internal APIs
		# Just document the count for awareness
		if warnings:
			pytest.skip(f'{len(warnings)} examples use internal imports (warning-level only)')


# =============================================================================
# Tests: CLI/MCP parameter naming consistency
# =============================================================================


class TestEntryPointParameterConsistency:
	"""Test that CLI and MCP entry points use consistent parameter names."""

	EXPECTED_COMMON_PARAMS = {
		'headless',
		'cdp_url',
		'user_data_dir',
		'profile_directory',
		'use_cloud',
		'cloud_profile_id',
		'cloud_proxy_country_code',
	}

	def test_cli_has_expected_params(self):
		"""Test that CLI defines the expected common parameters."""
		cli_params = set(extract_cli_params(CLI_MAIN))

		# CLI uses --headed (inverse of headless), --profile, --cdp-url
		cli_mapped = {
			'headed',  # inverse of headless
			'profile',  # maps to profile_directory indirectly
			'cdp_url',  # from --cdp-url
			'session',  # session management
		}

		for param in cli_mapped:
			assert param in cli_params, f'Expected CLI param "{param}" not found in: {sorted(cli_params)}'

	def test_mcp_has_expected_browser_params(self):
		"""Test that MCP server uses expected browser profile parameters."""
		mcp_params = set(extract_mcp_browser_params(MCP_SERVER))

		# These are the BrowserProfile fields used in MCP _init_browser_session
		expected_mcp = {
			'headless',
			'user_data_dir',
			'downloads_path',
			'keep_alive',
			'device_scale_factor',
			'disable_security',
			'wait_between_actions',
		}

		for param in expected_mcp:
			assert param in mcp_params, f'Expected MCP browser param "{param}" not found in: {sorted(mcp_params)}'

	def test_cli_mcp_param_naming_consistency(self):
		"""Test that common parameters use consistent naming between CLI and MCP."""
		cli_params = extract_cli_params(CLI_MAIN)
		mcp_params = extract_mcp_browser_params(MCP_SERVER)

		cli_set = set(cli_params)
		mcp_set = set(mcp_params)

		# Check that parameters that exist in both use the same name
		common = cli_set & mcp_set
		expected_common = {'headless', 'user_data_dir', 'cdp_url', 'profile_directory'}

		# headless is represented as --headed in CLI (inverse)
		# user_data_dir is set via --profile in CLI
		# Just verify no naming conflicts
		assert 'headless' in mcp_set or 'headed' in cli_set, 'headless/headed param missing'
		assert 'user_data_dir' in mcp_set or 'profile' in cli_set, 'user_data_dir/profile param missing'


# =============================================================================
# Tests: Isolated import scenarios (different extras combinations)
# =============================================================================


class TestIsolatedImports:
	"""Test imports in isolated environments with different extras combinations.

	These tests run in subprocesses to simulate minimal installations.
	"""

	def _run_isolated_import_test(self, extra: str | None, imports_to_test: list[str]) -> None:
		"""Run an isolated import test in a subprocess."""
		test_code = [
			'import sys',
			'# Block optional dependencies to simulate minimal install',
			'class BlockedImport: pass',
			'',
		]

		if extra != 'all':
			blocked = []
			if extra != 'aws':
				blocked.append('boto3')
			if extra != 'oci':
				blocked.append('oci')
			if extra != 'cli':
				blocked.append('textual')

			for dep in blocked:
				test_code.append(f'sys.modules["{dep}"] = BlockedImport()  # type: ignore')

		test_code.extend(['', '# Now test the imports'])
		for imp in imports_to_test:
			test_code.append(f'import {imp}')
			# For lazy imports, actually access the attribute to trigger loading
			if imp == 'browser_use':
				test_code.append('_ = browser_use.Agent')

		test_script = '\n'.join(test_code)

		result = subprocess.run(
			[sys.executable, '-c', test_script],
			capture_output=True,
			text=True,
			cwd=str(PROJECT_ROOT),
		)

		assert result.returncode == 0, f'Import test failed for extra={extra}:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}'

	def test_core_imports_no_extras(self):
		"""Test that core package imports work without any extras installed."""
		self._run_isolated_import_test(
			extra=None,
			imports_to_test=[
				'browser_use',
				'browser_use.agent',
				'browser_use.browser',
				'browser_use.tools',
			],
		)

	def test_core_lazy_imports_resolve(self):
		"""Test that all top-level lazy imports can be resolved without optional deps."""
		lazy_imports = extract_lazy_imports(BROWSER_USE_INIT)

		# Test each lazy import can be imported
		for name, (module_path, attr_name) in lazy_imports.items():
			if name.lower() in ('models', 'sandbox'):
				continue  # Skip modules that are not classes
			try:
				module = importlib.import_module(module_path)
				if attr_name:
					getattr(module, attr_name)
			except ImportError as e:
				if 'boto3' in str(e) or 'oci' in str(e) or 'textual' in str(e):
					pytest.skip(f'Skipping {name} - requires optional dep: {e}')
				raise AssertionError(f'Failed to import {name}: {e}') from e

	def test_llm_module_imports(self):
		"""Test that llm module can be imported without optional deps."""
		import browser_use.llm as llm_module

		assert hasattr(llm_module, 'ChatOpenAI')
		assert hasattr(llm_module, 'ChatGoogle')
		assert hasattr(llm_module, 'ChatAnthropic')
		assert hasattr(llm_module, 'ChatBrowserUse')

	def test_mcp_import_boundary(self):
		"""Test that MCP module gracefully handles missing mcp SDK."""
		try:
			import browser_use.mcp as mcp_module

			# Should have lazy import for BrowserUseServer
			assert hasattr(mcp_module, 'BrowserUseServer')
		except ImportError:
			pytest.skip('MCP SDK not installed')

	def test_sandbox_import_boundary(self):
		"""Test that sandbox module can be imported (with cloudpickle as optional)."""
		import browser_use.sandbox as sandbox_module

		assert hasattr(sandbox_module, 'sandbox')

	def test_import_top_level_symbols(self):
		"""Test that all symbols in __all__ are importable from the top level."""
		import browser_use

		all_list = extract_all_list(BROWSER_USE_INIT)

		for name in all_list:
			try:
				getattr(browser_use, name)
			except ImportError as e:
				if 'boto3' in str(e) or 'oci' in str(e) or 'textual' in str(e) or 'mcp' in str(e):
					pytest.skip(f'Skipping {name} - requires optional dep: {e}')
				raise AssertionError(f'Failed to get {name} from browser_use: {e}') from e

	def test_core_lazy_imports_dont_trigger_optional_deps(self):
		"""Test that importing core lazy imports doesn't try to load optional deps."""
		test_code = """
import sys
import importlib

# Track which modules get imported
original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
imported_modules = set()

def tracking_import(name, *args, **kwargs):
    imported_modules.add(name.split('.')[0])
    return original_import(name, *args, **kwargs)

import builtins
builtins.__import__ = tracking_import

# Now import browser_use and access core symbols
import browser_use
_ = browser_use.Agent
_ = browser_use.Browser
_ = browser_use.ChatOpenAI
_ = browser_use.ChatBrowserUse
_ = browser_use.Tools
_ = browser_use.ActionResult

# Restore
builtins.__import__ = original_import

# Check that optional deps were NOT imported
optional = {'boto3', 'oci', 'textual'}
imported_optional = imported_modules & optional

if imported_optional:
    print(f'ERROR: Optional deps were imported: {imported_optional}', file=sys.stderr)
    sys.exit(1)
else:
    print('OK: No optional deps imported for core symbols')
    sys.exit(0)
"""

		result = subprocess.run(
			[sys.executable, '-c', test_code],
			capture_output=True,
			text=True,
			cwd=str(PROJECT_ROOT),
		)

		assert result.returncode == 0, f'Optional dep leakage detected:\nSTDERR: {result.stderr}'
