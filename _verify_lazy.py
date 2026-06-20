"""
Forcefully verify each optional submodule against missing extras.

Method: 
1. Remove target extras packages from sys.modules
2. Block re-import by injecting ImportError via sys.modules override
3. Re-import the optional submodule under test
4. Check it does NOT crash at module level
"""
import sys
import os
import builtins

# Submodules to test - each entry: (human_name, submodule_path, list_of_packages_to_block)
TEST_CASES = [
    # LLM providers
    ("llm.models (all providers)", "browser_use.llm.models", 
     ["openai", "anthropic", "google", "google.genai", "groq", "ollama", "boto3", "oci"]),
    ("llm.openai.chat", "browser_use.llm.openai.chat", ["openai"]),
    ("llm.anthropic.chat", "browser_use.llm.anthropic.chat", ["anthropic"]),
    ("llm.google.chat", "browser_use.llm.google.chat", ["google", "google.genai", "google.auth"]),
    ("llm.groq.chat", "browser_use.llm.groq.chat", ["groq"]),
    ("llm.ollama.chat", "browser_use.llm.ollama.chat", ["ollama"]),
    ("llm.aws.chat_bedrock", "browser_use.llm.aws.chat_bedrock", ["boto3"]),
    
    # Sandbox / cloud
    ("sandbox.sandbox", "browser_use.sandbox.sandbox", ["cloudpickle"]),
    
    # Skills
    ("skills.service", "browser_use.skills.service", ["browser_use_sdk"]),
    
    # MCP
    ("mcp.client", "browser_use.mcp.client", ["mcp", "mcp.client", "mcp.server"]),
]

_original_import = builtins.__import__
_blocked: set[str] = set()

def _blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
    # For absolute imports only (level=0)
    if level == 0:
        short = name.split('.')[0]
        if short in _blocked:
            raise ImportError(f"[SIMULATED]")
        # Also check full name
        for b in _blocked:
            if name == b or name.startswith(b + "."):
                raise ImportError(f"[SIMULATED]")
    return _original_import(name, globals, locals, fromlist, level)

# --- Pre-import browser_use core first (Agent/Browser are safe)
builtins.__import__ = _original_import
print("Loading browser_use core first (baseline)...")
from browser_use import Agent, Browser, ChatBrowserUse
print("✓ Baseline loaded.\n")

# Now replace importer with blocker
builtins.__import__ = _blocking_import

passed_all = True

for name, mod_path, block_pkgs in TEST_CASES:
    # Install blocker
    _blocked = set(block_pkgs)
    
    # Remove any already-loaded submodule
    for pkg in block_pkgs:
        for k in list(sys.modules):
            if k == pkg or k.startswith(pkg + "."):
                del sys.modules[k]
    # Also remove the test module itself
    for k in list(sys.modules):
        if k == mod_path or k.startswith(mod_path + "."):
            del sys.modules[k]
    
    try:
        # Force fresh import
        import importlib
        mod = importlib.import_module(mod_path)
        print(f"✓ PASS: {name}")
        print(f"     → blocked: {block_pkgs}")
        print(f"     → imported without crash")
    except ImportError as e:
        if "[SIMULATED]" in str(e):
            # Get full traceback to locate the exact line
            passed_all = False
            import traceback
            tb = traceback.format_exc()
            # Find the relevant frame inside our package
            lines = tb.split("\n")
            site_line = None
            for line in lines:
                if mod_path.replace(".", "/") in line or "browser_use" in line:
                    if line.strip().startswith("File"):
                        site_line = line.strip()
                        break
            print(f"✗ FAIL: {name}")
            print(f"     → blocked: {block_pkgs}")
            print(f"     → ImportError at: {site_line or '(see above)'}")
            print(f"     → error: {e}")
            print(f"     (move this import to be lazy/inside try/except with runtime check)")
        else:
            # Some other ImportError - not our simulated one
            print(f"? WARN: {name}")
            print(f"     → blocked: {block_pkgs}")
            print(f"     → non-simulated ImportError: {e}")
    except Exception as e:
        passed_all = False
        print(f"✗ FAIL [{type(e).__name__}: {name}")
        print(f"     → blocked: {block_pkgs}")
        print(f"     → {e}")

# Restore original import
builtins.__import__ = _original_import
_blocked = set()

print()
if passed_all:
    print("=" * 60)
    print("✅ ALL OPTIONAL SUBMODULES PASS LAZY-LOADING CHECKS!")
    print("=" * 60)
    sys.exit(0)
else:
    print("=" * 60)
    print("❌ Some submodules FAILED — need lazy-import fixes")
    print("=" * 60)
    sys.exit(1)
