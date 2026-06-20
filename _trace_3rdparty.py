"""Trace exactly which browser_use module imports each extras-only third-party pkg."""
import sys
import builtins
import os
import traceback

PACKAGES_TO_TRACK = {
    'click': 'cli',
    'rich': 'cli',
    'google': 'llm-google',
    'pypdf': 'files',
    'reportlab': 'files',
    'docx': 'files',
    'cloudpickle': 'cloud',
    'browser_use_sdk': 'cloud',
    'mcp': 'mcp',
    'openai': 'llm-openai',
    'anthropic': 'llm-anthropic',
    'groq': 'llm-groq',
    'ollama': 'llm-ollama',
    'boto3': 'llm-aws',
    'oci': 'llm-oci',
}

_original_import = builtins.__import__
hit_points = {}  # pkg -> [(caller_file, caller_line, stack_snippet)]

def _tracing_import(name, globals=None, locals=None, fromlist=(), level=0):
    # Check if this import triggers any of our tracked packages (directly or transitively)
    short_pkg = name.split('.')[0]
    
    # For tracking: check if any tracked pkg is the one being imported, or a submodule
    for tracked_pkg in PACKAGES_TO_TRACK:
        if name == tracked_pkg or name.startswith(tracked_pkg + '.'):
            # Get call stack from outside browser_use's own import machinery
            stack = traceback.extract_stack()
            # Find first caller outside this module and outside importlib
            caller_info = None
            for frame in reversed(stack[:-1]):  # skip last (this function)
                fname = frame.filename
                # Ignore this tracing script and Python internals
                if '_trace_' in fname or fname.endswith('.py') is False:
                    continue
                if 'importlib' in fname:
                    continue
                # Find a file inside browser_use source
                if 'browser_use' in fname and os.path.isfile(fname):
                    caller_info = (fname, frame.lineno, frame.name)
                    break
                elif not fname.startswith('<') and os.path.isfile(fname):
                    # Outside browser_use but a real file - also capture
                    caller_info = (fname, frame.lineno, frame.name)
                    break
            
            if caller_info:
                fname, lineno, func = caller_info
                rel = fname.replace('/Users/pkcha/browser-use/', '')
                entry = (rel, lineno, func, name)
                if tracked_pkg not in hit_points:
                    hit_points[tracked_pkg] = []
                # Only record unique call sites
                if entry not in [x[:4] for x in hit_points[tracked_pkg]]:
                    # Grab 1 line of context from source
                    ctx = ""
                    try:
                        with open(fname, errors='ignore') as f:
                            lines = f.readlines()
                            if 0 <= lineno-1 < len(lines):
                                ctx = lines[lineno-1].strip()[:100]
                    except:
                        pass
                    hit_points[tracked_pkg].append(entry + (ctx,))
            break
    
    return _original_import(name, globals, locals, fromlist, level)

builtins.__import__ = _tracing_import

# Now run the core import
print("Running core import (Agent, Browser, ChatBrowserUse)...")
from browser_use import Agent, Browser, ChatBrowserUse
print("Core import done.\n")

# Report
print("=" * 70)
print("EXTRAS-ONLY THIRD-PARTY PACKAGE IMPORT SOURCES:")
print("=" * 70)

found_any = False
for pkg, extra in sorted(PACKAGES_TO_TRACK.items()):
    if pkg in hit_points and hit_points[pkg]:
        found_any = True
        print(f"\n❌ [{pkg}]  ({extra} extra) — imported from {len(hit_points[pkg])} site(s):")
        for (fname, lineno, func, full_name, ctx) in hit_points[pkg]:
            print(f"   {fname}:{lineno}  ({func})")
            if ctx:
                print(f"     → {ctx}")

if not found_any:
    print("\n✅ NO extras-only third-party packages imported during core import!")
else:
    print("\n" + "=" * 70)
    print("ACTION REQUIRED: The above imports must be deferred (lazy loading or runtime check).")
    sys.exit(1)
