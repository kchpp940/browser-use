"""Trace exact import chain for core Agent API in a minimal environment."""
import sys
import builtins
import os

# Capture ALL imports before touching browser_use
_original_import = builtins.__import__
_import_stack = []
_import_trail = []  # (depth, module_name, importing_file)
_max_depth = 0

def _tracing_import(name, globals=None, locals=None, fromlist=(), level=0):
    global _max_depth
    depth = len(_import_stack)
    _max_depth = max(_max_depth, depth)
    
    # Get caller info
    frame = sys._getframe(1)
    caller_file = frame.f_code.co_filename if frame else '?'
    
    # Only track browser_use modules
    short_name = name.split('.')[0]
    is_browser_use = (short_name == 'browser_use')
    
    if is_browser_use:
        _import_trail.append((depth, name, caller_file))
        print(f"{'  ' * depth}↳ {name}  (from {os.path.basename(caller_file)})")
    
    _import_stack.append(name)
    try:
        return _original_import(name, globals, locals, fromlist, level)
    finally:
        _import_stack.pop()

builtins.__import__ = _tracing_import

# Now do the core imports
print("=" * 70)
print("CORE IMPORT CHAIN: from browser_use import Agent, Browser, ChatBrowserUse")
print("=" * 70)

try:
    from browser_use import Agent, Browser, ChatBrowserUse
    print()
    print("✅ Core import successful")
except Exception as e:
    print()
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Now check which specific submodules were actually loaded
print()
print("=" * 70)
print("LOADED browser_use SUBMODULES:")
print("=" * 70)
bu_mods = sorted([k for k in sys.modules.keys() if k.startswith('browser_use')])
for m in bu_mods:
    # Flag potentially problematic ones
    flag = ""
    if any(w in m for w in ['filesystem', 'file_system']):
        flag = "  ⚠️  FILESYSTEM (files extra)"
    elif any(w in m for w in ['sandbox', 'cloud']):
        flag = "  ☁️  SANDBOX/CLOUD (cloud extra)"
    elif any(w in m for w in ['mcp']):
        flag = "  📡 MCP (mcp extra)"
    elif any(w in m for w in ['skills']):
        flag = "  🤖 SKILLS (cloud extra)"
    elif any(w in m for w in ['llm.openai', 'llm.anthropic', 'llm.google', 'llm.groq', 'llm.ollama', 'llm.aws', 'llm.oci']):
        flag = "  🧠 SPECIFIC LLM (llm-* extras)"
    elif any(w in m for w in ['cli']):
        flag = "  💻 CLI (cli extra)"
    print(f"  {m}{flag}")

# Check for third-party imports that are extras-only
print()
print("=" * 70)
print("CRITICAL CHECK: Extras-only third-party packages LOADED during core import")
print("=" * 70)
EXTRAS_ONLY = {
    # LLM providers
    'openai': 'llm-openai',
    'anthropic': 'llm-anthropic',
    'google': 'llm-google',
    'google.genai': 'llm-google',
    'groq': 'llm-groq',
    'ollama': 'llm-ollama',
    'boto3': 'llm-aws',
    'oci': 'llm-oci',
    # Cloud/Sandbox
    'cloudpickle': 'cloud',
    'browser_use_sdk': 'cloud',
    # MCP
    'mcp': 'mcp',
    # CLI
    'click': 'cli',
    'InquirerPy': 'cli',
    'textual': 'cli',
    'rich': 'cli',
    # Files (filesystem)
    'pypdf': 'files',
    'reportlab': 'files',
    'docx': 'files',
    'python_docx': 'files',
    # Video
    'imageio': 'video',
    'numpy': 'video',
}

violations = []
for pkg, extra in EXTRAS_ONLY.items():
    if pkg in sys.modules:
        violations.append((pkg, extra))

if violations:
    print("❌ VIOLATIONS FOUND - these extras were loaded during core import:")
    for pkg, extra in violations:
        print(f"  - {pkg} ({extra} extra)")
    sys.exit(1)
else:
    print("✅ NO violations - all extras-only packages are NOT loaded")
