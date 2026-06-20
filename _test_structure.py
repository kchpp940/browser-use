"""Validate friendly error messages for optional dependencies"""
import sys

passed = 0
failed = 0

def check(name, condition, detail=''):
    global passed, failed
    if condition:
        passed += 1
        print(f'✓ {name}')
    else:
        failed += 1
        print(f'✗ {name}  {detail}')

# --- 1. Check llm.models has all provider AVAILABLE flags ---
print('--- LLM provider flags ---')
from browser_use.llm import models as llm_models

flags = ['OPENAI_AVAILABLE', 'ANTHROPIC_AVAILABLE', 'GOOGLE_AVAILABLE', 'GROQ_AVAILABLE', 'OLLAMA_AVAILABLE']
for f in flags:
    val = getattr(llm_models, f, None)
    check(f, val is not None, f'missing attribute {f}')
    if val is not None:
        check(f'  is bool ({val})', isinstance(val, bool))

# --- 2. Check sandbox/cloud optional import flags ---
print()
print('--- Sandbox/Cloud flags ---')
from browser_use.sandbox import sandbox as sb_mod
check('CLOUDPICKLE_AVAILABLE', hasattr(sb_mod, 'CLOUDPICKLE_AVAILABLE'))
check('_require_cloudpickle callable', callable(getattr(sb_mod, '_require_cloudpickle', None)))

# --- 3. Check skills service optional import flags ---
print()
print('--- Skills SDK flags ---')
from browser_use.skills import service as skills_mod
check('SKILLS_SDK_AVAILABLE', hasattr(skills_mod, 'SKILLS_SDK_AVAILABLE'))
check('_require_skills_sdk callable', callable(getattr(skills_mod, '_require_skills_sdk', None)))

# --- 4. Check MCP client optional import flags ---
print()
print('--- MCP client flags ---')
from browser_use.mcp import client as mcp_client_mod
check('MCP_AVAILABLE', hasattr(mcp_client_mod, 'MCP_AVAILABLE'))
check('_require_mcp callable', callable(getattr(mcp_client_mod, '_require_mcp', None)))

# --- 5. Check __init__.py _EXTRA_HINTS map ---
print()
print('--- _EXTRA_HINTS map (friendly install suggestions) ---')
import browser_use
hints = getattr(browser_use, '_EXTRA_HINTS', None)
check('_EXTRA_HINTS exists', hints is not None, 'missing _EXTRA_HINTS map')
if hints:
    expected_keys = ['ChatOpenAI', 'ChatAnthropic', 'ChatGoogle', 'sandbox', 'SkillService']
    for k in expected_keys:
        check(f'  hint for {k}', k in hints)

# --- 6. Check pyproject.toml extras structure ---
print()
print('--- pyproject.toml extras ---')
import tomllib
with open('/Users/pkcha/browser-use/pyproject.toml', 'rb') as f:
    pyproject = tomllib.load(f)

optional_deps = pyproject['project']['optional-dependencies']
expected_extras = ['cli', 'mcp', 'cloud', 'sandbox', 'llm-openai', 'llm-anthropic', 'llm-google',
                   'llm-groq', 'llm-ollama', 'llm-aws', 'llm-oci', 'llms', 'files', 'video', 'all']
for e in expected_extras:
    check(f'  extra [{e}]', e in optional_deps)

# --- 7. Verify core deps don't include CLI/LLM/MCP/Cloud packages ---
print()
print('--- Core dependencies boundary check ---')
core_deps = pyproject['project']['dependencies']
forbidden_in_core = ['click', 'InquirerPy', 'rich', 'textual', 'mcp', 'openai', 'anthropic',
                     'cloudpickle', 'browser-use-sdk', 'boto3', 'oci', 'pypdf', 'reportlab']
for dep in forbidden_in_core:
    found = any(dep in d for d in core_deps)
    check(f'  {dep} NOT in core', not found, f'found in core deps!')

# --- 8. Verify pydantic-settings is NOW in core (was a bug) ---
check('  pydantic-settings in core', any('pydantic-settings' in d for d in core_deps))

# --- 9. Verify no duplicate deps between core and eval ---
print()
print('--- Duplicate deps check ---')
eval_deps = optional_deps.get('eval', [])
core_dep_names = [d.split('==')[0].split(';')[0].strip() for d in core_deps]
eval_dep_names = [d.split('==')[0].split(';')[0].strip() for d in eval_deps]
duplicates = set(core_dep_names) & set(eval_dep_names)
check('  no core/eval duplicates', len(duplicates) == 0, f'duplicates: {duplicates}')

# --- 10. Verify dev-deps no longer has lmnr duplicate ---
print()
dev_deps = pyproject['tool']['uv'].get('dev-dependencies', [])
dev_dep_names = [d.split('==')[0] for d in dev_deps]
check('  lmnr NOT in dev-deps', 'lmnr[all]' not in dev_dep_names and 'lmnr' not in dev_dep_names)

print()
print(f'=== Results: {passed} passed, {failed} failed ===')
sys.exit(0 if failed == 0 else 1)
