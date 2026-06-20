import sys

print('=== Simulating missing optional dependencies ===')
print()

# --- Test 1: Simulate missing OpenAI SDK ---
print('Test 1: Simulate missing openai SDK')
if 'openai' in sys.modules:
    del sys.modules['openai']
if 'browser_use.llm.openai.chat' in sys.modules:
    del sys.modules['browser_use.llm.openai.chat']

import importlib
import browser_use.llm.models as llm_models
importlib.reload(llm_models)

if not llm_models.OPENAI_AVAILABLE:
    print('✓ OPENAI_AVAILABLE correctly set to False')
else:
    print('✗ OPENAI_AVAILABLE should be False after simulation')

try:
    _ = llm_models.ChatOpenAI
except (ImportError, AttributeError) as e:
    if 'pip install' in str(e):
        print(f'✓ ChatOpenAI access triggers friendly error')
    else:
        print(f'⊘ ChatOpenAI error type: {type(e).__name__}: {e}')

print()

# --- Test 2: Check _require_provider helper ---
print('Test 2: llm.models _require_provider with missing provider')
try:
    from browser_use.llm.models import _require_provider, _OPENAI_MISSING
    _require_provider('OpenAI', 'llm-openai', _OPENAI_MISSING)
    print('✗ Should have raised ImportError')
except ImportError as e:
    if 'pip install "browser-use[llm-openai]"' in str(e):
        print('✓ _require_provider returns correct install command')
    else:
        print(f'⊘ Error message: {str(e)[:100]}')

print()

# --- Test 3: Simulate missing cloudpickle ---
print('Test 3: Simulate missing cloudpickle (sandbox/cloud dep)')
if 'cloudpickle' in sys.modules:
    del sys.modules['cloudpickle']
if 'browser_use.sandbox.sandbox' in sys.modules:
    del sys.modules['browser_use.sandbox.sandbox']

import browser_use.sandbox.sandbox as sb_module
importlib.reload(sb_module)

if not sb_module.CLOUDPICKLE_AVAILABLE:
    print('✓ CLOUDPICKLE_AVAILABLE correctly set to False')
else:
    print('✗ CLOUDPICKLE_AVAILABLE should be False')

try:
    sb_module._require_cloudpickle()
    print('✗ Should have raised ImportError')
except ImportError as e:
    if 'pip install "browser-use[cloud]"' in str(e):
        print('✓ _require_cloudpickle returns correct install command')
    else:
        print(f'⊘ Error message: {str(e)[:100]}')

print()

# --- Test 4: Simulate missing mcp SDK ---
print('Test 4: Simulate missing mcp SDK')
for k in list(sys.modules.keys()):
    if k.startswith('mcp'):
        del sys.modules[k]
if 'browser_use.mcp.client' in sys.modules:
    del sys.modules['browser_use.mcp.client']

import browser_use.mcp.client as mcp_client
importlib.reload(mcp_client)

if not mcp_client.MCP_AVAILABLE:
    print('✓ MCP_AVAILABLE correctly set to False')
else:
    print('✗ MCP_AVAILABLE should be False')

try:
    mcp_client._require_mcp()
    print('✗ Should have raised ImportError')
except ImportError as e:
    if 'pip install "browser-use[mcp]"' in str(e):
        print('✓ _require_mcp returns correct install command')
    else:
        print(f'⊘ Error message: {str(e)[:100]}')

print()

# --- Test 5: Simulate missing browser-use-sdk ---
print('Test 5: Simulate missing browser-use-sdk (skills/cloud dep)')
for k in list(sys.modules.keys()):
    if k.startswith('browser_use_sdk'):
        del sys.modules[k]
if 'browser_use.skills.service' in sys.modules:
    del sys.modules['browser_use.skills.service']

import browser_use.skills.service as skills_service
importlib.reload(skills_service)

if not skills_service.SKILLS_SDK_AVAILABLE:
    print('✓ SKILLS_SDK_AVAILABLE correctly set to False')
else:
    print('✗ SKILLS_SDK_AVAILABLE should be False')

try:
    skills_service._require_skills_sdk()
    print('✗ Should have raised ImportError')
except ImportError as e:
    if 'pip install "browser-use[cloud]"' in str(e):
        print('✓ _require_skills_sdk returns correct install command')
    else:
        print(f'⊘ Error message: {str(e)[:100]}')

print()
print('=== All missing-dependency simulation tests completed ===')
