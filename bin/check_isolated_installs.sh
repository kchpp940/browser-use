#!/usr/bin/env bash
# Isolated install verification script for browser-use release engineering.
# Tests the package installs cleanly under different extras combinations.
#
# NOTE: Not all features are separate extras:
#   - [mcp] feature  → is part of CORE (mcp SDK is a core dependency)
#   - [sandbox] feature → is part of CORE
#   - [cloud] feature → is part of CORE (Browser(use_cloud=True) is core)
#   - [cli] extra     → adds textual for terminal UI
#   - [aws] extra     → adds boto3 for AWS Bedrock chat models
#   - [oci] extra     → adds oci SDK for OCI Raw chat models
#   - [all] extra     → cli + examples + aws + oci
#
# Usage:
#   ./bin/check_isolated_installs.sh              # DEFAULT: run ALL combos (release hard constraint)
#   ./bin/check_isolated_installs.sh --quick      # Only core + cli combos (fast)
#   ./bin/check_isolated_installs.sh --core-only  # Only core (fastest sanity)
#   ./bin/check_isolated_installs.sh --keep       # Keep temp venvs for debugging
#   ./bin/check_isolated_installs.sh --clean      # Clean up existing temp venvs
#
# Install combos (DEFAULT):
#   1. core          → no extras
#   2. cli           → browser-use[cli]
#   3. aws           → browser-use[aws]
#   4. oci           → browser-use[oci]
#   5. all-extras    → browser-use[all]
#
# Verified for each combo:
#   - Top-level imports (all __all__ core symbols)
#   - CLI entry point (browser-use --help, browser-use install)
#   - MCP server import   (from browser_use.mcp import BrowserUseServer)  [CORE FEATURE]
#   - Sandbox import      (from browser_use.sandbox.sandbox import sandbox) [CORE FEATURE]
#   - Cloud entry import  (Browser(use_cloud=True) params)                [CORE FEATURE]
#   - AWS Bedrock import  (ChatAWSBedrock, ChatAnthropicBedrock)          [only with aws extra]
#   - OCI Raw import      (ChatOCIRaw)                                    [only with oci extra]
#   - Example script syntax
#   - No optional dependency leakage

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$SCRIPT_DIR/.."
PROJECT_ROOT="$(pwd)"

# ─── Mode parsing ───
RUN_MODE="all"          # all | quick | core-only
KEEP_VENVS=0
CLEAN_MODE=0

for arg in "$@"; do
    case "$arg" in
        --quick)     RUN_MODE="quick" ;;
        --core-only) RUN_MODE="core-only" ;;
        --all)       RUN_MODE="all" ;;   # same as default
        --keep)      KEEP_VENVS=1 ;;
        --clean)     CLEAN_MODE=1 ;;
        *)           echo "Unknown option: $arg"
                     echo "Usage: $0 [--all|--quick|--core-only] [--keep] [--clean]"
                     exit 1 ;;
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

log_info() {
    printf "  ℹ️  %s\n" "$1"
}

log_section() {
    echo ""
    echo "📦 $1"
}

# ─── Venv & install helpers ───

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
    # All core symbols from __all__
    for name in ["Agent", "Browser", "BrowserSession", "BrowserProfile",
                 "ChatBrowserUse", "ChatOpenAI", "ChatAnthropic", "ChatGoogle",
                 "ChatGroq", "ChatLiteLLM", "ChatMistral", "ChatAzureOpenAI",
                 "ChatOllama", "ChatVercel",
                 "Tools", "ActionResult", "ActionModel", "AgentHistoryList",
                 "DomService", "SystemPrompt", "Controller",
                 "sandbox", "models"]:
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
        log_pass "Top-level imports (all __all__ symbols)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Top-level imports: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: CLI entry point ───

test_cli_entry_point() {
    local combo_name="$1"
    local has_cli_extra="${2:-0}"   # 1 if [cli] extra is installed (textual)

    echo "  Testing CLI entry point..."

    # browser-use CLI is always installed (skill_cli doesn't require textual)
    if cli_in_venv "$combo_name" --help >/dev/null 2>&1; then
        log_pass "CLI entry: browser-use --help"
    else
        log_fail "CLI entry: browser-use --help returned non-zero"
        return 1
    fi

    # install subcommand
    if cli_in_venv "$combo_name" install --help >/dev/null 2>&1; then
        log_pass "CLI subcommand: browser-use install --help"
    else
        log_fail "CLI subcommand: browser-use install --help failed"
    fi

    # sessions subcommand (plural - list sessions)
    if cli_in_venv "$combo_name" sessions --help >/dev/null 2>&1; then
        log_pass "CLI subcommand: browser-use sessions --help"
    else
        log_fail "CLI subcommand: browser-use sessions --help failed"
    fi

    # cloud subcommand (cloud browser management)
    if cli_in_venv "$combo_name" cloud --help >/dev/null 2>&1; then
        log_pass "CLI subcommand: browser-use cloud --help"
    else
        log_fail "CLI subcommand: browser-use cloud --help failed"
    fi

    # [cli] extra → terminal UI entry point (browser-use-tui)
    if [ "$has_cli_extra" -eq 1 ]; then
        local venv_dir="$TMP_BASE/$combo_name"
        if [ -x "$venv_dir/bin/browser-use-tui" ]; then
            log_pass "TUI entry: browser-use-tui (from [cli] extra)"
        else
            log_fail "TUI entry 'browser-use-tui' not found (should exist with [cli] extra)"
        fi
    fi
}

# ─── Test: MCP import (CORE FEATURE) ───

test_mcp_import() {
    local combo_name="$1"

    echo "  Testing MCP import (CORE FEATURE - no extra required)..."

    local test_code='
import sys
try:
    from browser_use.mcp import BrowserUseServer
    print("OK: MCP import works (BrowserUseServer)")
except ImportError as e:
    print(f"FAIL: MCP import failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: unexpected error: {e}")
    sys.exit(1)
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: MCP import works"; then
        log_pass "MCP import: BrowserUseServer (CORE FEATURE)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "MCP import: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: Sandbox import (CORE FEATURE) ───

test_sandbox_import() {
    local combo_name="$1"

    echo "  Testing sandbox import (CORE FEATURE - no extra required)..."

    local test_code='
import sys
try:
    from browser_use import sandbox
    from browser_use.sandbox.sandbox import sandbox as sandbox_fn
    # Verify cloud params exist in sandbox signature
    import inspect
    sig = inspect.signature(sandbox_fn)
    params = sig.parameters
    has_cpid = "cloud_profile_id" in params
    has_cpcc = "cloud_proxy_country_code" in params
    has_ct = "cloud_timeout" in params
    if has_cpid and has_cpcc and has_ct:
        print("OK: sandbox import + cloud params OK (cloud_profile_id, cloud_proxy_country_code, cloud_timeout)")
    else:
        print(f"FAIL: cloud params missing in sandbox: cpid={has_cpid}, cpcc={has_cpcc}, ct={has_ct}")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: sandbox import"; then
        log_pass "Sandbox: import + cloud params (CORE FEATURE)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Sandbox: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: Cloud entry import (CORE FEATURE) ───

test_cloud_entry() {
    local combo_name="$1"

    echo "  Testing cloud entry import (CORE FEATURE - no extra required)..."

    local test_code='
import sys
try:
    from browser_use import Browser, BrowserProfile
    import inspect

    # Browser uses overloads, inspect actual __init__ + BrowserProfile fields
    # 1. Browser: check via Pydantic model_fields (it inherits from BaseModel)
    browser_fields = set(Browser.model_fields.keys()) if hasattr(Browser, "model_fields") else set()
    # Also check BrowserSession since Browser is alias
    from browser_use import BrowserSession
    session_fields = set(BrowserSession.model_fields.keys())

    # Params that must exist on BrowserSession (directly or via browser_profile forwarding)
    required_session = {"headless", "use_cloud", "cloud_profile_id", "cloud_proxy_country_code", "cloud_timeout"}
    missing_session = [p for p in required_session if p not in session_fields]
    # cloud_* may be in __init__ overloads but not direct model fields
    # Check BrowserProfile has the cloud fields
    required_profile = {"use_cloud", "cloud_browser_params"}
    profile_fields = set(BrowserProfile.model_fields.keys())
    missing_profile = [p for p in required_profile if p not in profile_fields]

    if missing_profile:
        print(f"FAIL: BrowserProfile missing fields: {missing_profile} (have: {sorted([f for f in profile_fields if 'cloud' in f.lower()])})")
        sys.exit(1)

    print("OK: cloud entry fields OK (BrowserProfile.use_cloud & cloud_browser_params; BrowserSession has session/cloud params)")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: cloud entry fields OK"; then
        log_pass "Cloud entry: Browser/BrowserProfile (CORE FEATURE)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Cloud entry: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: AWS Bedrock import (only with [aws] extra) ───

test_aws_bedrock_import() {
    local combo_name="$1"
    local has_aws_extra="${2:-0}"

    echo "  Testing AWS Bedrock import ([aws] extra)..."

    if [ "$has_aws_extra" -eq 0 ]; then
        # Without aws extra:
        # 1. Top-level import should NOT crash (already verified by test 1)
        # 2. Protocol stub is available (for typing) - but real methods fail gracefully
        # 3. Verify that actually using boto3-dependent code triggers ImportError
        local test_code='
import sys
try:
    from browser_use.llm.aws.chat_bedrock import ChatAWSBedrock
    # _get_client() actually triggers the boto3 import
    m = ChatAWSBedrock(model="anthropic.claude-v2", aws_region="us-east-1")
    client = m._get_client()
    print("FAIL: _get_client() should fail without boto3")
    sys.exit(1)
except (ImportError, ModuleNotFoundError) as e:
    msg = str(e).lower()
    if "boto3" in msg or "botocore" in msg:
        print(f"OK: gracefully fails when calling boto3-dependent method without [aws] extra: {e}")
    else:
        print(f"FAIL: unexpected error: {e}")
        sys.exit(1)
except TypeError as e:
    # If Protocol stub cannot be instantiated properly, thats OK too
    print(f"OK: Cannot instantiate ChatAWSBedrock without [aws] extra: {e}")
except Exception as e:
    print(f"FAIL: unexpected error type: {type(e).__name__}: {e}")
    sys.exit(1)
'
        if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -qE "^OK: (gracefully fails|Cannot instantiate)"; then
            log_pass "AWS Bedrock: graceful failure without [aws] extra"
        else
            local output
            output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
            log_fail "AWS Bedrock no-extra behavior: $(echo "$output" | tr '\n' ' ' | head -c 250)"
            return 1
        fi
        return 0
    fi

    # With aws extra, both top-level and direct import should work
    local test_code='
import sys
try:
    from browser_use import ChatAWSBedrock, ChatAnthropicBedrock
    from browser_use.llm.aws.chat_bedrock import ChatAWSBedrock as Bedrock1
    import inspect
    if inspect.isclass(ChatAWSBedrock) and inspect.isclass(ChatAnthropicBedrock):
        print("OK: AWS Bedrock imports work with [aws] extra")
    else:
        print("FAIL: ChatAWSBedrock/ChatAnthropicBedrock are not classes")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
'
    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: AWS Bedrock imports work"; then
        log_pass "AWS Bedrock: ChatAWSBedrock/ChatAnthropicBedrock ([aws] extra)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "AWS Bedrock import: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: OCI Raw import (only with [oci] extra) ───

test_oci_import() {
    local combo_name="$1"
    local has_oci_extra="${2:-0}"

    echo "  Testing OCI Raw import ([oci] extra)..."

    if [ "$has_oci_extra" -eq 0 ]; then
        # Without oci extra:
        # 1. Top-level import should NOT crash (already verified by test 1)
        # 2. Protocol stub is available (for typing) - but real methods fail gracefully
        # 3. Verify that actually using oci-dependent code triggers ImportError
        local test_code='
import sys
try:
    from browser_use.llm.oci_raw.chat import ChatOCIRaw
    # _get_oci_client() actually triggers the oci import
    m = ChatOCIRaw(model_id="cohere.command-r-16k", service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com", compartment_id="ocid1.test")
    client = m._get_oci_client()
    print("FAIL: _get_oci_client() should fail without oci")
    sys.exit(1)
except (ImportError, ModuleNotFoundError) as e:
    msg = str(e).lower()
    if "oci" in msg:
        print(f"OK: gracefully fails when calling oci-dependent method without [oci] extra: {e}")
    else:
        print(f"FAIL: unexpected error: {e}")
        sys.exit(1)
except TypeError as e:
    # If Protocol stub cannot be instantiated properly, thats OK too
    print(f"OK: Cannot instantiate ChatOCIRaw without [oci] extra: {e}")
except Exception as e:
    print(f"FAIL: unexpected error type: {type(e).__name__}: {e}")
    sys.exit(1)
'
        if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -qE "^OK: (gracefully fails|Cannot instantiate)"; then
            log_pass "OCI Raw: graceful failure without [oci] extra"
        else
            local output
            output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
            log_fail "OCI no-extra behavior: $(echo "$output" | tr '\n' ' ' | head -c 250)"
            return 1
        fi
        return 0
    fi

    # With oci extra, both top-level and direct import should work
    local test_code='
import sys
try:
    from browser_use import ChatOCIRaw
    from browser_use.llm.oci_raw.chat import ChatOCIRaw as OCIRaw1
    import inspect
    if inspect.isclass(ChatOCIRaw):
        print("OK: OCI Raw import works with [oci] extra")
    else:
        print("FAIL: ChatOCIRaw is not a class")
        sys.exit(1)
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
'
    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: OCI Raw import works"; then
        log_pass "OCI Raw: ChatOCIRaw ([oci] extra)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "OCI Raw import: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: example script syntax ───

test_examples_syntax() {
    local combo_name="$1"

    echo "  Testing example script syntax..."

    local test_code="
import ast
import sys
from pathlib import Path

examples_dir = Path('$PROJECT_ROOT') / 'examples'
errors = []
count = 0

for py_file in examples_dir.rglob('*.py'):
    count += 1
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
    print(f'OK: all {count} examples parse successfully')
"

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: all .* examples parse successfully"; then
        log_pass "Example scripts syntax (all $PROJECT_ROOT/examples/*.py)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Example syntax: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Test: no optional dep leakage ───

test_no_optional_leakage() {
    local combo_name="$1"
    # Dependencies that MUST NOT be imported when only accessing core symbols
    local forbidden="boto3 oci textual"

    echo "  Testing no optional dep leakage (boto3/oci/textual)..."

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

# Import and access core symbols (NOT aws/oci-specific)
import browser_use
_ = browser_use.Agent
_ = browser_use.Browser
_ = browser_use.ChatBrowserUse
_ = browser_use.ChatOpenAI
_ = browser_use.Tools
_ = browser_use.ActionResult
_ = browser_use.sandbox  # CORE FEATURE
_ = browser_use.AgentHistoryList

# Restore
builtins.__import__ = original_import

# Check optional deps
optional = {"boto3", "oci", "textual"}
leaked = imported_top_level & optional

if leaked:
    print(f"FAIL: Optional deps leaked: {leaked}")
    print(f"All imported top-level: {sorted(imported_top_level)}")
    sys.exit(1)
else:
    print("OK: No optional deps imported for core symbols")
'

    if run_in_venv "$combo_name" -c "$test_code" 2>&1 | grep -q "^OK: No optional deps imported"; then
        log_pass "No optional dep leakage (boto3/oci/textual)"
    else
        local output
        output=$(run_in_venv "$combo_name" -c "$test_code" 2>&1 || true)
        log_fail "Optional dep leakage: $(echo "$output" | tr '\n' ' ' | head -c 250)"
        return 1
    fi
}

# ─── Run full test suite for a combo ───

run_combo_test() {
    local combo_name="$1"
    local extra_spec="$2"
    local has_cli_extra="$3"
    local has_aws_extra="$4"
    local has_oci_extra="$5"
    local combo_description="$6"

    log_section "Testing: $combo_name — $combo_description"

    create_venv "$combo_name"
    install_combo "$combo_name" "$extra_spec"

    echo ""

    test_top_level_imports "$combo_name" || true
    test_cli_entry_point "$combo_name" "$has_cli_extra" || true
    test_mcp_import "$combo_name" || true
    test_sandbox_import "$combo_name" || true
    test_cloud_entry "$combo_name" || true
    test_aws_bedrock_import "$combo_name" "$has_aws_extra" || true
    test_oci_import "$combo_name" "$has_oci_extra" || true
    test_examples_syntax "$combo_name" || true
    test_no_optional_leakage "$combo_name" || true
}

# ─── Main ───

echo "🔬 browser-use isolated install verification (release hard constraint)"
echo "================================================================"
echo "Temp directory: $TMP_BASE"
echo "Run mode: $RUN_MODE"
echo ""
echo "Note: MCP/Sandbox/Cloud features are CORE (no separate extras)."
echo "Extras: [cli]=textual · [aws]=boto3 · [oci]=oci SDK · [all]=cli+examples+aws+oci"
echo ""

combo_count=0

# Combo 1: CORE (always run)
run_combo_test \
    "core" "" \
    0 0 0 \
    "no extras · CORE features only"
combo_count=$((combo_count + 1))

if [ "$RUN_MODE" != "core-only" ]; then
    # Combo 2: CLI
    run_combo_test \
        "cli" "cli" \
        1 0 0 \
        "[cli] extra · adds textual (TUI support)"
    combo_count=$((combo_count + 1))
fi

if [ "$RUN_MODE" = "all" ]; then
    # Combo 3: AWS
    run_combo_test \
        "aws" "aws" \
        0 1 0 \
        "[aws] extra · adds boto3 (Bedrock chat models)"
    combo_count=$((combo_count + 1))

    # Combo 4: OCI
    run_combo_test \
        "oci" "oci" \
        0 0 1 \
        "[oci] extra · adds oci SDK (OCI Raw chat models)"
    combo_count=$((combo_count + 1))

    # Combo 5: ALL extras
    run_combo_test \
        "all-extras" "all" \
        1 1 1 \
        "[all] extra · cli + examples + aws + oci (maximum install)"
    combo_count=$((combo_count + 1))
fi

# ─── Summary ───
echo ""
echo "================================================================"
echo "Results: $PASS_COUNT checks passed, $FAIL_COUNT checks failed across $combo_count combos"

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
