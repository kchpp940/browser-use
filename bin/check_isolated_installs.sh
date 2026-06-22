#!/usr/bin/env bash
# Isolated install verification script for browser-use release engineering.
# Tests the package installs cleanly under different extras combinations.
#
# Usage:
#   ./bin/check_isolated_installs.sh              # Test core + quick combos
#   ./bin/check_isolated_installs.sh --all        # Test all combos (slow)
#   ./bin/check_isolated_installs.sh --keep       # Keep temp venvs for debugging
#   ./bin/check_isolated_installs.sh --clean      # Clean up existing temp venvs
#
# Install combos tested:
#   core (no extras)
#   browser-use[cli]
#   browser-use[aws]
#   browser-use[oci]
#   browser-use[all]
#
# For each combo, verifies:
#   - Top-level imports (core symbols from __all__)
#   - CLI entry point (browser-use --help)
#   - MCP server import
#   - Sandbox import
#   - Example script syntax
#   - No optional dependency leakage

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/.."
PROJECT_ROOT="$(pwd)"

ALL_MODE=0
KEEP_VENVS=0
CLEAN_MODE=0

for arg in "$@"; do
    case "$arg" in
        --all)     ALL_MODE=1 ;;
        --quick)   ALL_MODE=0 ;;  # default
        --keep)    KEEP_VENVS=1 ;;
        --clean)   CLEAN_MODE=1 ;;
        *)         echo "Unknown option: $arg"; echo "Usage: $0 [--all|--quick] [--keep] [--clean]"; exit 1 ;;
    esac
done

TMP_BASE="${TMPDIR:-/tmp}/browser-use-isolated-test"
FAILED=0
ERRORS=""
PASS_COUNT=0
FAIL_COUNT=0

# ─── Clean mode ───
if [ "$CLEAN_MODE" -eq 1 ]; then
    echo "🧹 Cleaning up temp venvs at $TMP_BASE"
    rm -rf "$TMP_BASE"
    echo "Done."
    exit 0
fi

# ─── Helpers ───

log_pass() {
    printf "  ✅ %s\n" "$1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

log_fail() {
    printf "  ❌ %s\n" "$1"
    FAILED=1
    FAIL_COUNT=$((FAIL_COUNT + 1))
    ERRORS="$ERRORS\n  - $1"
}

log_warn() {
    printf "  ⚠️  %s\n" "$1"
}

log_section() {
    echo ""
    echo "📦 $1"
}

create_venv() {
    local name="$1"
    local venv_dir="$TMP_BASE/$name"

    if [ -d "$venv_dir" ] && [ -f "$venv_dir/bin/python" ]; then
        echo "  Reusing existing venv: $venv_dir"
        return 0
    fi

    echo "  Creating venv: $venv_dir"
    uv venv --python 3.12 "$venv_dir" >/dev/null 2>&1
}

install_combo() {
    local name="$1"
    local extra_spec="${2:-}"

    local venv_dir="$TMP_BASE/$name"

    if [ -n "$extra_spec" ]; then
        echo "  Installing browser-use[$extra_spec] from local source..."
        uv pip install --python "$venv_dir/bin/python" -e ".[$extra_spec]" >/dev/null 2>&1
    else
        echo "  Installing browser-use (core) from local source..."
        uv pip install --python "$venv_dir/bin/python" -e . >/dev/null 2>&1
    fi
}

run_in_venv() {
    local name="$1"
    shift
    local venv_dir="$TMP_BASE/$name"
    VIRTUAL_ENV="$venv_dir" "$venv_dir/bin/python" "$@"
}

cli_in_venv() {
    local name="$1"
    shift
    local venv_dir="$TMP_BASE/$name"
    VIRTUAL_ENV="$venv_dir" PATH="$venv_dir/bin:$PATH" "$venv_dir/bin/browser-use" "$@"
}

# ─── Test: top-level imports ───

test_top_level_imports() {
    local combo_name="$1"

    echo "  Testing top-level imports..."

    local test_code='
import sys
errors = []
try:
    import browser_use
    for name in ["Agent", "Browser", "BrowserSession", "ChatBrowserUse",
                 "ChatOpenAI", "ChatAnthropic", "ChatGoogle",
                 "Tools", "ActionResult", "AgentHistoryList"]:
        try:
            getattr(browser_use, name)
        except Exception as e:
            errors.append(f"{name}: {e}")
    if errors:
        print("FAIL: import failures:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    else:
        print("OK: all core imports work")
except Exception as e:
    print(f"FAIL: top-level import failed: {e}")
    sys.exit(1)
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: all core imports work"; then
        log_pass "Top-level imports work"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Top-level imports: $(echo "$output" | tr '\n' ' ' | head -c 200)"
        return 1
    fi
}

# ─── Test: CLI entry point ───

test_cli_entry_point() {
    local combo_name="$1"

    echo "  Testing CLI entry point..."

    if cli_in_venv "$combo_name" --help >/dev/null 2>&1; then
        log_pass "CLI entry point works (browser-use --help)"
    else
        log_fail "CLI entry point failed: browser-use --help returned non-zero"
        return 1
    fi

    # Test install subcommand works
    if cli_in_venv "$combo_name" install --help >/dev/null 2>&1; then
        log_pass "CLI 'install' subcommand works"
    else
        log_fail "CLI 'install' subcommand failed"
    fi
}

# ─── Test: MCP import ───

test_mcp_import() {
    local combo_name="$1"

    echo "  Testing MCP import..."

    local test_code='
import sys
try:
    from browser_use.mcp import BrowserUseServer
    print("OK: MCP import works")
except ImportError as e:
    print(f"SKIP: MCP import failed (expected if mcp SDK not installed): {e}")
    sys.exit(2)
except Exception as e:
    print(f"FAIL: unexpected error: {e}")
    sys.exit(1)
'

    local output
    output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)

    if echo "$output" | grep -q "^OK: MCP import works"; then
        log_pass "MCP import works"
    elif echo "$output" | grep -q "^SKIP:"; then
        log_warn "MCP import skipped (mcp SDK not available)"
    else
        log_fail "MCP import: $(echo "$output" | tr '\n' ' ' | head -c 200)"
        return 1
    fi
}

# ─── Test: sandbox import ───

test_sandbox_import() {
    local combo_name="$1"

    echo "  Testing sandbox import..."

    local test_code='
import sys
try:
    from browser_use import sandbox
    from browser_use.sandbox.sandbox import sandbox as sandbox_fn
    print("OK: sandbox import works")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: sandbox import works"; then
        log_pass "Sandbox import works"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Sandbox import: $(echo "$output" | tr '\n' ' ' | head -c 200)"
        return 1
    fi
}

# ─── Test: examples syntax ───

test_examples_syntax() {
    local combo_name="$1"

    echo "  Testing example script syntax..."

    local test_code="
import ast
import sys
from pathlib import Path

examples_dir = Path('$PROJECT_ROOT') / 'examples'
errors = []

for py_file in examples_dir.rglob('*.py'):
    try:
        with open(py_file, 'r') as f:
            ast.parse(f.read())
    except SyntaxError as e:
        errors.append(f'{py_file}:{e.lineno}: {e.msg}')

if errors:
    print('FAIL: syntax errors:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    count = len(list(examples_dir.rglob('*.py')))
    print(f'OK: all {count} examples parse successfully')
"

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: all .* examples parse successfully"; then
        log_pass "All example scripts have valid syntax"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Example syntax: $(echo "$output" | tr '\n' ' ' | head -c 200)"
        return 1
    fi
}

# ─── Test: no optional dep leakage ───

test_no_optional_leakage() {
    local combo_name="$1"
    local expected_blocked=("$@")
    # Remove first arg (combo_name)
    shift

    echo "  Testing no optional dep leakage..."

    local test_code='
import sys
import builtins

# Track top-level module imports
original_import = builtins.__import__
imported_top_level = set()

def tracking_import(name, *args, **kwargs):
    imported_top_level.add(name.split(".")[0])
    return original_import(name, *args, **kwargs)

builtins.__import__ = tracking_import

# Import core symbols
import browser_use
_ = browser_use.Agent
_ = browser_use.Browser
_ = browser_use.ChatOpenAI
_ = browser_use.Tools
_ = browser_use.ActionResult

# Restore
builtins.__import__ = original_import

# Check optional deps
optional = {"boto3", "oci", "textual"}
leaked = imported_top_level & optional

if leaked:
    print(f"FAIL: Optional deps leaked: {leaked}")
    sys.exit(1)
else:
    print("OK: No optional deps imported for core symbols")
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: No optional deps imported"; then
        log_pass "No optional dependency leakage"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Optional dep leakage: $(echo "$output" | tr '\n' ' ' | head -c 200)"
        return 1
    fi
}

# ─── Run all tests for a combo ───

run_combo_test() {
    local combo_name="$1"
    local extra_spec="$2"

    log_section "Testing: $combo_name"

    create_venv "$combo_name"
    install_combo "$combo_name" "$extra_spec"

    echo ""

    test_top_level_imports "$combo_name" || true
    test_cli_entry_point "$combo_name" || true
    test_mcp_import "$combo_name" || true
    test_sandbox_import "$combo_name" || true
    test_examples_syntax "$combo_name" || true
    test_no_optional_leakage "$combo_name" || true
}

# ─── Main ───

echo "🔬 browser-use isolated install verification"
echo "============================================"
echo "Temp directory: $TMP_BASE"
echo "Mode: $([ "$ALL_MODE" -eq 1 ] && echo "all combos" || echo "quick (core only)")"

# Always test core
run_combo_test "core" ""

if [ "$ALL_MODE" -eq 1 ]; then
    run_combo_test "cli" "cli"
    run_combo_test "aws" "aws"
    run_combo_test "oci" "oci"
    run_combo_test "all-extras" "all"
fi

# ─── Summary ───
echo ""
echo "============================================"
echo "Results: $PASS_COUNT passed, $FAIL_COUNT failed"

if [ "$FAILED" -eq 0 ]; then
    echo "✅ All isolated install checks passed!"
    if [ "$KEEP_VENVS" -eq 0 ]; then
        echo "Cleaning up temp venvs..."
        rm -rf "$TMP_BASE"
    else
        echo "Temp venvs kept at: $TMP_BASE"
    fi
    exit 0
else
    echo "❌ Some isolated install checks failed:"
    echo -e "$ERRORS"
    echo ""
    echo "Temp venvs kept at: $TMP_BASE (for debugging)"
    exit 1
fi
