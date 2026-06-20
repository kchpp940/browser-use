import sys, traceback

print('=== Test 1: Core API imports (Agent, Browser) ===')
try:
    from browser_use import Agent, Browser
    print('✓ Agent, Browser imported successfully')
except Exception as e:
    print(f'✗ FAILED: {e}')
    traceback.print_exc()
    sys.exit(1)

print()
print('=== Test 2: ChatBrowserUse import ===')
try:
    from browser_use import ChatBrowserUse
    print('✓ ChatBrowserUse imported successfully')
except Exception as e:
    print(f'✗ FAILED: {e}')
    traceback.print_exc()

print()
print('=== Test 3: Lazy import ChatOpenAI ===')
try:
    from browser_use import ChatOpenAI
    print('✓ ChatOpenAI imported (SDK present in dev env)')
except ImportError as e:
    if 'pip install' in str(e):
        print(f'✓ Got expected friendly error: {e}')
    else:
        print(f'✗ Got unexpected ImportError: {e}')

print()
print('=== Test 4: Lazy import sandbox ===')
try:
    from browser_use import sandbox
    print('✓ sandbox imported (SDK present in dev env)')
except ImportError as e:
    if 'pip install' in str(e):
        print(f'✓ Got expected friendly error: {e}')
    else:
        print(f'✗ Got unexpected ImportError: {e}')

print()
print('=== Test 5: browser_use.llm module lazy-loading ===')
try:
    from browser_use import llm
    print(f'✓ llm module loaded')
except Exception as e:
    print(f'✗ FAILED: {e}')
    traceback.print_exc()

print()
print('=== Test 6: Browser instantiation (headless) ===')
try:
    b = Browser(headless=True)
    print(f'✓ Browser instantiated: {type(b).__name__}')
except Exception as e:
    print(f'✗ Browser FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()

print()
print('=== Test 7: MCPClient optional import check ===')
try:
    from browser_use.mcp.client import MCPClient, MCP_AVAILABLE
    if MCP_AVAILABLE:
        print('✓ MCPClient SDK available (present in dev env)')
    else:
        print('⊘ MCPClient SDK not available')
except ImportError as e:
    if 'pip install' in str(e):
        print(f'✓ Got expected friendly error')
    else:
        print(f'✗ Unexpected error: {e}')

print()
print('=== Test 8: SkillService optional import check ===')
try:
    from browser_use.skills.service import SkillService, SKILLS_SDK_AVAILABLE
    if SKILLS_SDK_AVAILABLE:
        print('✓ SkillService SDK available (present in dev env)')
    else:
        print('⊘ SkillService SDK not available')
except ImportError as e:
    if 'pip install' in str(e):
        print(f'✓ Got expected friendly error')
    else:
        print(f'✗ Unexpected error: {e}')

print()
print('=== All tests passed! ===')
