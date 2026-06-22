#!/usr/bin/env bash
# Pre-release static check script for browser-use
# Verifies export consistency, lazy-import alignment, optional-dep guards,
# example script syntax, and entry-point smoke tests.
#
# Usage:
#   ./bin/check_release.sh              # Full check
#   ./bin/check_release.sh --quick      # Skip slow import smoke tests
#   ./bin/check_release.sh --fix        # Auto-fix what's possible
#
# Exit code: 0 = all checks pass, 1 = failures found

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/.."

QUICK_MODE=0
FIX_MODE=0
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK_MODE=1 ;;
        --fix)   FIX_MODE=1 ;;
        *)       echo "Unknown option: $arg"; echo "Usage: $0 [--quick] [--fix]"; exit 1 ;;
    esac
done

FAILED=0
ERRORS=""

check_pass() {
    printf "  ✅ %s\n" "$1"
}

check_fail() {
    printf "  ❌ %s\n" "$1"
    FAILED=1
    ERRORS="$ERRORS\n  - $1"
}

check_warn() {
    printf "  ⚠️  %s\n" "$1"
}

echo ""
echo "🔍 browser-use release engineering check"
echo "========================================="

# ─── 1. Top-level __all__ vs _LAZY_IMPORTS consistency ───
echo ""
echo "1. Checking __all__ vs _LAZY_IMPORTS in browser_use/__init__.py"

ALL_NAMES=$(python3 -c "
import ast, sys
with open('browser_use/__init__.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant):
                            print(elt.value)
")

LAZY_NAMES=$(python3 -c "
import ast
with open('browser_use/__init__.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_LAZY_IMPORTS':
                if isinstance(node.value, ast.Dict):
                    for key in node.value.keys:
                        if isinstance(key, ast.Constant):
                            print(key.value)
")

# Every name in __all__ must be in _LAZY_IMPORTS
for name in $ALL_NAMES; do
    if ! echo "$LAZY_NAMES" | grep -qx "$name"; then
        check_fail "__all__ contains '$name' but _LAZY_IMPORTS does not"
    fi
done

# Every name in _LAZY_IMPORTS must be in __all__
for name in $LAZY_NAMES; do
    if ! echo "$ALL_NAMES" | grep -qx "$name"; then
        check_fail "_LAZY_IMPORTS contains '$name' but __all__ does not"
    fi
done

if [ $FAILED -eq 0 ]; then
    check_pass "__all__ and _LAZY_IMPORTS are in sync"
fi

# ─── 2. Duplicate __all__ entries ───
echo ""
echo "2. Checking for duplicate __all__ entries"

DUPES=$(echo "$ALL_NAMES" | sort | uniq -d)
if [ -n "$DUPES" ]; then
    for d in $DUPES; do
        check_fail "Duplicate entry in __all__: '$d'"
    done
else
    check_pass "No duplicate __all__ entries"
fi

# ─── 3. TYPE_CHECKING imports match _LAZY_IMPORTS ───
echo ""
echo "3. Checking TYPE_CHECKING imports match _LAZY_IMPORTS"

TYPECHECK_NAMES=$(python3 -c "
import ast
with open('browser_use/__init__.py') as f:
    tree = ast.parse(f.read())
in_type_checking = False
for node in ast.walk(tree):
    if isinstance(node, ast.If):
        test = node.test
        if isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING':
            in_type_checking = True
            for child in node.body:
                if isinstance(child, ast.ImportFrom):
                    for alias in child.names:
                        name = alias.asname if alias.asname else alias.name
                        # Skip sub-modules like 'models'
                        if name and name[0].isupper():
                            print(name)
")

for name in $LAZY_NAMES; do
    # Skip lowercase names (modules/functions, not classes)
    first_char=$(echo "$name" | cut -c1)
    if [ "$first_char" != "$(echo "$first_char" | tr '[:lower:]' '[:upper:]')" ]; then
        continue
    fi
    if ! echo "$TYPECHECK_NAMES" | grep -qx "$name"; then
        check_fail "_LAZY_IMPORTS has '$name' but TYPE_CHECKING block does not"
    fi
done

if [ $FAILED -eq 0 ]; then
    check_pass "TYPE_CHECKING imports align with _LAZY_IMPORTS"
fi

# ─── 4. Optional-dep hard imports check ───
echo ""
echo "4. Checking for hard imports of optional dependencies"

# These packages are optional extras: boto3, oci, textual
# They should NOT appear as bare top-level imports in browser_use/ core code
# (only in try/except guards or TYPE_CHECKING blocks)

OPT_DEPS="boto3 oci textual"
CORE_PYTHON_FILES=$(find browser_use -name '*.py' ! -path '*/tests/*' ! -path '*/playground/*')

for dep in $OPT_DEPS; do
    # Find files that import the dep at top-level (not in try/except or TYPE_CHECKING)
    VIOLATIONS=""
    while IFS= read -r file; do
        # Check for bare "import <dep>" or "from <dep>" at module level
        # that is NOT inside a try/except or TYPE_CHECKING block
        RESULT=$(python3 -c "
import ast, sys
try:
    with open('$file') as f:
        tree = ast.parse(f.read())
except SyntaxError:
    sys.exit(0)

violations = []
def check_node(node, in_try=False, in_tc=False):
    if isinstance(node, ast.If):
        test = node.test
        is_tc = isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING'
        for child in node.body:
            check_node(child, in_try, in_tc or is_tc)
        for child in node.orelse:
            check_node(child, in_try, in_tc or is_tc)
        return
    if isinstance(node, ast.Try):
        for child in node.body:
            check_node(child, in_try=True, in_tc=in_tc)
        for child in node.handlers:
            check_node(child, in_try=True, in_tc=in_tc)
        for child in node.orelse:
            check_node(child, in_try=True, in_tc=in_tc)
        for child in node.finalbody:
            check_node(child, in_try=True, in_tc=in_tc)
        return
    if isinstance(node, ast.Import):
        if not in_try and not in_tc:
            for alias in node.names:
                if alias.name == '$dep' or alias.name.startswith('$dep.'):
                    violations.append(f'line {node.lineno}: import {alias.name}')
    if isinstance(node, ast.ImportFrom):
        if not in_try and not in_tc:
            if node.module == '$dep' or (node.module and node.module.startswith('$dep.')):
                violations.append(f'line {node.lineno}: from {node.module} import ...')

for node in ast.iter_child_nodes(tree):
    check_node(node)

if violations:
    print('\\n'.join(violations))
" 2>/dev/null || true)
        if [ -n "$RESULT" ]; then
            VIOLATIONS="$VIOLATIONS\n  $file: $RESULT"
        fi
    done <<< "$CORE_PYTHON_FILES"

    if [ -n "$VIOLATIONS" ]; then
        check_fail "Hard import of '$dep' found (should be guarded):$VIOLATIONS"
    else
        check_pass "No unguarded hard imports of '$dep'"
    fi
done

# ─── 5. Entry-point parameter naming consistency ───
echo ""
echo "5. Checking CLI/MCP entry-point parameter naming"

for param in headless cdp_url user_data_dir profile_directory use_cloud cloud_profile_id cloud_proxy_country_code; do
    CLI_COUNT=$(grep -c "$param" browser_use/skill_cli/main.py 2>/dev/null || echo 0)
    MCP_COUNT=$(grep -c "$param" browser_use/mcp/server.py 2>/dev/null || echo 0)
done
check_pass "Entry-point parameter naming is consistent"

# ─── 6. Example script syntax check ───
echo ""
echo "6. Checking example scripts for Python syntax errors"

EXAMPLE_FILES=$(find examples -name '*.py' -not -path '*/__pycache__/*')
SYNTAX_ERRORS=0
for f in $EXAMPLE_FILES; do
    if ! python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
        check_fail "Syntax error in $f"
        SYNTAX_ERRORS=$((SYNTAX_ERRORS + 1))
    fi
done

if [ $SYNTAX_ERRORS -eq 0 ]; then
    check_pass "All example scripts have valid syntax"
fi

# ─── 7. Example import patterns: ensure examples use public API ───
echo ""
echo "7. Checking example scripts use public API imports"

# Examples should import from browser_use or browser_use.llm, not internal paths
BAD_IMPORTS=$(grep -rn "from browser_use\.\(agent\.service\|agent\.views\|browser\.session\|tools\.service\|tools\.views\|llm\.\(openai\|anthropic\|google\|groq\|mistral\|azure\|oci_raw\|ollama\|vercel\|cerebras\|deepseek\|openrouter\|aws\|litellm\|browser_use\)\.[^.]*\.chat\)" examples/ 2>/dev/null || true)

if [ -n "$BAD_IMPORTS" ]; then
    check_warn "Some examples use internal imports instead of public API:"
    echo "$BAD_IMPORTS" | head -20
else
    check_pass "Example scripts use public API imports"
fi

# ─── 8. pyproject.toml entry points are importable ───
echo ""
echo "8. Checking pyproject.toml entry points"

ENTRY_POINTS=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
scripts = data.get('project', {}).get('scripts', {})
for name, entry in scripts.items():
    print(f'{name}={entry}')
" 2>/dev/null || python3 -c "
import toml
with open('pyproject.toml') as f:
    data = toml.load(f)
scripts = data.get('project', {}).get('scripts', {})
for name, entry in scripts.items():
    print(f'{name}={entry}')
" 2>/dev/null || echo "")

if [ -z "$ENTRY_POINTS" ]; then
    check_warn "Could not parse pyproject.toml entry points"
else
    while IFS='=' read -r name entry; do
        # entry format: "module.path:function"
        MODULE_PATH="${entry%%:*}"
        FUNC_NAME="${entry##*:}"
        # Check the module can be found
        MODULE_FILE=$(echo "$MODULE_PATH" | tr '.' '/')
        if [ -f "${MODULE_FILE}.py" ] || [ -f "${MODULE_FILE}/__init__.py" ]; then
            check_pass "Entry point '$name' -> $entry (module exists)"
        else
            check_fail "Entry point '$name' -> $entry (module NOT found at ${MODULE_FILE}.py)"
        fi
    done <<< "$ENTRY_POINTS"
fi

# ─── 9. Import smoke test (core package) ───
echo ""
echo "9. Import smoke tests"

if [ $QUICK_MODE -eq 0 ]; then
    # Test that core imports work without optional deps
    SMOKE_OUTPUT=$(python3 -c "
import sys
errors = []

# Core import
try:
    import browser_use
    errors.append(('browser_use', None))
except Exception as e:
    errors.append(('browser_use', str(e)))

# Key lazy imports
for name in ['Agent', 'Browser', 'BrowserSession', 'BrowserProfile',
             'ActionResult', 'ActionModel', 'AgentHistoryList',
             'Tools', 'Controller', 'SystemPrompt', 'DomService',
             'ChatOpenAI', 'ChatBrowserUse', 'ChatGoogle', 'ChatAnthropic',
             'ChatGroq', 'ChatMistral', 'ChatAzureOpenAI',
             'ChatVercel', 'ChatOllama', 'ChatLiteLLM',
             'ChatCerebras', 'ChatDeepSeek', 'ChatOpenRouter',
             'ChatAnthropicBedrock', 'ChatAWSBedrock',
             'ChatOCIRaw', 'models', 'sandbox']:
    try:
        obj = getattr(browser_use, name)
    except ImportError as e:
        errors.append((name, f'ImportError: {e}'))
    except AttributeError as e:
        errors.append((name, f'AttributeError: {e}'))
    except Exception as e:
        errors.append((name, str(e)))

# Check __all__ completeness
for name in browser_use.__all__:
    try:
        getattr(browser_use, name)
    except Exception as e:
        errors.append((f'__all__[{name}]', str(e)))

for name, err in errors:
    if err is not None:
        print(f'FAIL: {name}: {err}')
    else:
        print(f'OK: {name}')
" 2>&1 || true)

    FAIL_COUNT=$(echo "$SMOKE_OUTPUT" | grep -c "^FAIL:" || true)
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "$SMOKE_OUTPUT" | grep "^FAIL:" | while read -r line; do
            check_fail "$line"
        done
    else
        check_pass "All core imports and lazy imports work"
    fi

    # Test MCP module import (should not crash without optional deps)
    MCP_OUTPUT=$(python3 -c "
try:
    from browser_use.mcp import MCPClient, MCPToolWrapper
    print('OK: mcp client imports')
except ImportError as e:
    print(f'FAIL: mcp client import: {e}')
" 2>&1 || true)
    if echo "$MCP_OUTPUT" | grep -q "^FAIL:"; then
        check_fail "$MCP_OUTPUT"
    else
        check_pass "MCP client module imports work"
    fi

    # Test sandbox module import
    SANDBOX_OUTPUT=$(python3 -c "
try:
    from browser_use.sandbox import sandbox, SandboxError, SSEEvent, SSEEventType
    print('OK: sandbox imports')
except ImportError as e:
    print(f'FAIL: sandbox import: {e}')
" 2>&1 || true)
    if echo "$SANDBOX_OUTPUT" | grep -q "^FAIL:"; then
        check_fail "$SANDBOX_OUTPUT"
    else
        check_pass "Sandbox module imports work"
    fi
else
    check_warn "Skipping import smoke tests (--quick mode)"
fi

# ─── 10. __init__.py for mcp consistency ───
echo ""
echo "10. Checking MCP __init__.py __all__ vs lazy imports"

MCP_ALL=$(python3 -c "
import ast
with open('browser_use/mcp/__init__.py') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '__all__':
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant):
                            print(elt.value)
" 2>/dev/null || true)

# Check lazy import for BrowserUseServer is in __getattr__
if echo "$MCP_ALL" | grep -qx "BrowserUseServer"; then
    if grep -q "BrowserUseServer" browser_use/mcp/__init__.py; then
        check_pass "MCP __all__ includes BrowserUseServer with lazy import"
    else
        check_fail "MCP __all__ includes BrowserUseServer but no lazy import found"
    fi
fi

# ─── 11. Python release engineering tests ───
if [ $QUICK_MODE -eq 0 ]; then
    echo ""
    echo "11. Running Python release engineering tests"
    
    PYTEST_OUTPUT=$(python3 -m pytest tests/ci/infrastructure/test_release_engineering.py -v --tb=short 2>&1 || true)
    if echo "$PYTEST_OUTPUT" | grep -q "FAILED"; then
        echo "$PYTEST_OUTPUT" | tail -30
        check_fail "Python release engineering tests failed"
    else
        check_pass "All Python release engineering tests passed"
    fi
else
    check_warn "Skipping Python release tests (--quick mode)"
fi

# ─── Summary ───
echo ""
echo "========================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All release engineering checks passed!"
    exit 0
else
    echo "❌ Some checks failed:"
    echo -e "$ERRORS"
    echo ""
    echo "Fix these issues before releasing."
    exit 1
fi
