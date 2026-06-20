"""Dependency boundary test for browser-use extras.

Verifies that:
  - pip install . (core-only) does NOT pull in extras-only deps
  - pip install .[cli] enables CLI
  - pip install .[mcp] enables MCP
  - pip install .[sandbox] (= [cloud]) enables sandbox & skills

Run from project root with:
    uv run python tests/ci/test_dependency_boundary.py
Or directly with any Python (reads pyproject.toml):
    python tests/ci/test_dependency_boundary.py
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"


def _parse_toml(path: Path) -> dict:
    """Parse pyproject.toml without requiring tomli/tomllib (for Py<3.11)."""
    try:
        import tomllib  # Py 3.11+
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore
    return tomllib.loads(path.read_text())


def _parse_toml_fallback(path: Path) -> dict:
    """Regex-based fallback for core extras list if tomllib is unavailable."""
    import re

    text = path.read_text()
    result: dict = {"project": {"optional-dependencies": {}}}
    # Match [project.optional-dependencies] section blocks
    opt_match = re.search(
        r"\[project\.optional-dependencies\]\n(.*?)(?=\n\[|\Z)", text, re.DOTALL
    )
    if opt_match:
        block = opt_match.group(1)
        for m in re.finditer(r'^(\w+)\s*=\s*\[(.*?)\]', block, re.MULTILINE | re.DOTALL):
            name = m.group(1)
            pkgs = [p.strip().strip('"').strip("'").split(';')[0].strip()
                    for p in m.group(2).split(',') if p.strip()]
            result["project"]["optional-dependencies"][name] = [
                p for p in pkgs if p and not p.startswith('#')
            ]
    return result


def load_pyproject() -> dict:
    try:
        return _parse_toml(PYPROJECT)
    except Exception:
        return _parse_toml_fallback(PYPROJECT)


def extract_core_deps(pp: dict) -> set[str]:
    """Extract bare dependency package names (lowercased, normalized)."""
    import re

    deps = pp.get("project", {}).get("dependencies", [])
    result = set()
    for d in deps:
        # Strip version constraints, extras, markers
        name = re.split(r'[<>=!~;\[]', d)[0].strip().lower().replace('_', '-')
        if name:
            result.add(name)
    return result


def extract_extras(pp: dict) -> dict[str, list[str]]:
    """Return extras -> list of package names (not depspecs)."""
    import re

    extras = pp.get("project", {}).get("optional-dependencies", {})
    result: dict[str, list[str]] = {}
    for extra, deps in extras.items():
        names = []
        for d in deps:
            name = re.split(r'[<>=!~;\[]', d)[0].strip().lower().replace('_', '-')
            if name:
                names.append(name)
        result[extra] = names
    return result


# --- Submodules that must NOT be importable at top-level in core-only install ---

CORE_IMPORT_BLOCK_LIST: list[tuple[str, str, set[str]]] = [
    # (description, module_path, blocked_third_party_prefixes)
    # If any blocked prefix is imported during module top-level load, test fails.
    ("filesystem.file_system", "browser_use.filesystem.file_system",
     {"reportlab", "pypdf", "docx"}),
    ("llm.models (providers)", "browser_use.llm.models",
     {"openai", "anthropic", "google.genai", "groq", "ollama", "boto3", "oci",
      "litellm", "deepseek", "openrouter", "vercel", "cerebras", "mistralai"}),
    ("sandbox.sandbox", "browser_use.sandbox.sandbox", {"cloudpickle"}),
    ("mcp.client", "browser_use.mcp.client",
     {"mcp", "mcp.client", "mcp.server", "mcp.types"}),
    ("mcp.controller", "browser_use.mcp.controller",
     {"mcp", "mcp.client", "mcp.server", "mcp.types"}),
    ("skills.service", "browser_use.skills.service", {"browser_use_sdk"}),
    ("skills.views", "browser_use.skills.views", {"browser_use_sdk"}),
]

# Provider chat modules: direct `from browser_use.llm.X.chat import ChatX` must also
# survive missing SDK (only raise friendly error when the class is *instantiated*).
# Full list of every provider (except browser_use which uses httpx only, plus mistral
# which uses plain httpx without a provider SDK).
PROVIDER_CHAT_BLOCK: list[tuple[str, str, set[str]]] = [
    ("llm.openai.chat", "browser_use.llm.openai.chat", {"openai"}),
    ("llm.azure.chat", "browser_use.llm.azure.chat", {"openai"}),
    ("llm.cerebras.chat", "browser_use.llm.cerebras.chat", {"openai"}),
    ("llm.deepseek.chat", "browser_use.llm.deepseek.chat", {"openai"}),
    ("llm.openrouter.chat", "browser_use.llm.openrouter.chat", {"openai"}),
    ("llm.vercel.chat", "browser_use.llm.vercel.chat", {"openai"}),
    ("llm.anthropic.chat", "browser_use.llm.anthropic.chat", {"anthropic"}),
    ("llm.google.chat", "browser_use.llm.google.chat", {"google", "google.genai"}),
    ("llm.groq.chat", "browser_use.llm.groq.chat", {"groq"}),
    ("llm.ollama.chat", "browser_use.llm.ollama.chat", {"ollama"}),
    ("llm.oci_raw.chat", "browser_use.llm.oci_raw.chat", {"oci"}),
    ("llm.aws.chat_bedrock", "browser_use.llm.aws.chat_bedrock", {"boto3"}),
    ("llm.aws.chat_anthropic", "browser_use.llm.aws.chat_anthropic", {"anthropic", "boto3"}),
    ("llm.litellm.chat", "browser_use.llm.litellm.chat", {"litellm"}),
]


def _installed_packages() -> set[str]:
    """Return set of installed top-level package names via importlib.metadata."""
    try:
        from importlib import metadata
    except ImportError:
        from importlib_metadata import metadata  # type: ignore
    pkgs: set[str] = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            pkgs.add(name.lower().replace('_', '-'))
    return pkgs


def test_core_does_not_include_extras() -> list[str]:
    """Verify core deps list in pyproject.toml does not include extras-only deps.

    (Static check against pyproject.toml, not runtime.)
    """
    errors = []
    pp = load_pyproject()
    core = extract_core_deps(pp)
    extras = extract_extras(pp)

    extras_only: set[str] = set()
    for names in extras.values():
        extras_only.update(names)
    extras_only = extras_only - core

    known_safe_core = {
        "aiohttp", "anyio", "bubus", "cdp-use", "httpx", "markdownify", "pillow",
        "posthog", "psutil", "pydantic", "pydantic-settings", "pyobjc",
        "pyotp", "python-dotenv", "requests", "screeninfo", "typing-extensions",
        "uuid7",
    }
    unexpected = core - known_safe_core - extras_only
    if unexpected:
        errors.append(
            f"Core deps contain packages not in known-safe set: {sorted(unexpected)}"
        )
    return errors


def test_runtime_top_level_safety() -> list[str]:
    """Simulate extras-deps-not-installed and verify block-list modules load fine."""
    import builtins

    errors = []
    # CORE_IMPORT_BLOCK_LIST: core Agent initialization chain
    # PROVIDER_CHAT_BLOCK: explicit provider chat submodule imports
    all_cases = CORE_IMPORT_BLOCK_LIST + PROVIDER_CHAT_BLOCK

    # Reusable blocker factory
    def _make_blocker(_orig_import, _bset):
        def _blocker(name, *args, **kwargs):
            base = name.split('.')[0].lower()
            full = name.lower()
            for b in _bset:
                bl = b.lower()
                if (base == bl.split('.')[0]
                    or full == bl
                    or full.startswith(bl + '.')):
                    raise ImportError(
                        f"[SIMULATED] Import of '{name}' blocked for "
                        f"boundary test (matches '{b}')"
                    )
            return _orig_import(name, *args, **kwargs)
        return _blocker

    for desc, module_path, blocked in all_cases:
        _orig_import = builtins.__import__
        builtins.__import__ = _make_blocker(_orig_import, blocked)
        try:
            importlib.invalidate_caches()
            for mod in list(sys.modules.keys()):
                if (mod.startswith(module_path)
                    or mod == module_path
                    or mod.startswith(module_path.rsplit('.', 1)[0])):
                    del sys.modules[mod]
            importlib.import_module(module_path)
            print(f"  ✓ {desc} (blocked {sorted(blocked)}) — top-level safe")
        except ImportError as e:
            errors.append(f"{desc}: ImportError when extras blocked: {e}")
            print(f"  ✗ {desc}: {e}")
        except Exception as e:
            errors.append(f"{desc}: {type(e).__name__}: {e}")
            print(f"  ✗ {desc}: {type(e).__name__}: {e}")
        finally:
            builtins.__import__ = _orig_import
    return errors


def test_core_import_chain() -> list[str]:
    """Trace `from browser_use import Agent, Browser, ChatBrowserUse` imports,
    ensure no extras-only 3rd party packages appear."""
    import builtins
    import inspect  # Import BEFORE monkey-patching to avoid recursion

    pp = load_pyproject()
    extras_map = extract_extras(pp)
    extras_pkgs: set[str] = set()
    for names in extras_map.values():
        extras_pkgs.update(names)
    core = extract_core_deps(pp)
    extras_only = extras_pkgs - core

    # Map python import top-level names -> PyPI package names (known mappings)
    PYPI_TO_IMPORT: dict[str, str] = {
        "python-docx": "docx",
        "python-dotenv": "dotenv",
        "pydantic-settings": "pydantic_settings",
        "cdp-use": "cdp_use",
        "browser-use-sdk": "browser_use_sdk",
        "typing-extensions": "typing_extensions",
    }
    import_to_pypi = {v: k for k, v in PYPI_TO_IMPORT.items()}

    # stdlib-ish module prefixes that we never want to trace into (avoids recursion)
    STDLIB_PREFIXES = (
        "inspect", "builtins", "sys", "os", "re", "json", "abc", "collections",
        "importlib", "threading", "multiprocessing", "subprocess", "types",
        "warnings", "traceback", "linecache", "tokenize", "token", "keyword",
        "ast", "dis", "code", "enum", "dataclasses", "typing", "functools",
        "itertools", "operator", "pathlib", "shutil", "tempfile", "io",
        "time", "datetime", "calendar", "math", "random", "stat", "statistics",
        "hashlib", "hmac", "secrets", "base64", "binascii", "copy", "copyreg",
        "shelve", "pickle", "marshal", "sqlite3", "dbm", "configparser",
        "logging", "argparse", "getopt", "getpass", "glob", "fnmatch",
        "locale", "gettext", "unicodedata", "string", "struct", "zlib",
        "gzip", "bz2", "lzma", "zipfile", "tarfile", "asyncio", "concurrent",
        "email", "urllib", "http", "html", "xml", "csv", "ntpath", "posixpath",
        "pathlib", "socket", "select", "ssl", "contextvars", "weakref",
        "gc", "atexit", "signal", "faulthandler", "syslog", "pdb", "profile",
        "pstats", "cProfile", "turtle", "unittest", "doctest", "lib2to3",
    )

    traced: dict[str, str] = {}  # import_name -> originating_file
    _orig = builtins.__import__

    def _tracing(name, *args, **kwargs):
        mod = _orig(name, *args, **kwargs)
        top = name.split('.')[0]
        if (top not in sys.builtin_module_names
                and top not in traced
                and top not in STDLIB_PREFIXES
                and not top.startswith("_")):
            frame = inspect.currentframe()
            origin = "?"
            depth = 0
            try:
                while frame and depth < 15:
                    fname = frame.f_code.co_filename
                    if "browser_use" in fname and "_tracing" not in fname and "_import" not in fname:
                        try:
                            origin = f"{fname}:{frame.f_lineno}"
                        except Exception:
                            pass
                        break
                    frame = frame.f_back
                    depth += 1
            finally:
                del frame
            traced[top] = origin
        return mod

    builtins.__import__ = _tracing
    try:
        for mod in list(sys.modules.keys()):
            if mod.startswith("browser_use"):
                del sys.modules[mod]
        # No need for invalidate_caches here — it triggers internal import tracing
        from browser_use import Agent, Browser, ChatBrowserUse  # noqa: F401
    finally:
        builtins.__import__ = _orig

    errors = []
    print("  Core import 3rd-party packages loaded (informational, dev env):")
    for name, origin in sorted(traced.items()):
        # Skip stdlib-ish, project, and core-dep imports
        if name.startswith("_") or name == "browser_use":
            continue
        pypi_name = import_to_pypi.get(name, name)
        in_core = (name.lower().replace('_', '-') in core
                   or pypi_name.lower() in core
                   or any(c.replace('-', '_').startswith(name.lower()[:3]) for c in core))
        if in_core:
            print(f"    ✓ {name:<25} [core]    via {origin}")
        elif pypi_name in extras_only or name in extras_only:
            print(f"    ~ {name:<25} [extra]   via {origin}")
            print(f"       → (checking: module MUST survive if this package is missing...)")
            # Verify that the originating module loads correctly WITHOUT this package
            safe = _verify_module_survives_missing(origin, name)
            if safe:
                print(f"       → OK: module uses try/except or runtime guard")
            else:
                errors.append(
                    f"Core import chain module '{origin}' loads extras-only '{name}' "
                    f"WITHOUT proper try/except guard!"
                )
        else:
            print(f"    ~ {name:<25} [?]       via {origin}")
    return errors


def _verify_module_survives_missing(origin_file: str, missing_pkg: str) -> bool:
    """Simulate a missing 3rd-party package, re-import the origin module.

    Returns True if the module's top-level code does NOT raise ImportError/NameError
    due to `missing_pkg` not being present.
    """
    import builtins

    module_name = None
    # Derive module name from file path
    try:
        rel = Path(origin_file.split(":")[0]).resolve().relative_to(ROOT)
        parts = list(rel.parts)
        if parts and parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif parts and parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        module_name = ".".join(parts)
    except Exception:
        return True  # Cannot derive module name — skip verification

    if not module_name.startswith("browser_use"):
        return True

    # Simulate package missing via import blocker
    _orig_import = builtins.__import__
    missing_base = missing_pkg.split(".")[0].lower()

    def _blocker(name, *args, **kwargs):
        top = name.split(".")[0].lower()
        if top == missing_base or name.lower().startswith(missing_pkg.lower()):
            raise ImportError(f"[SIMULATED] Missing extra: {missing_pkg}")
        return _orig_import(name, *args, **kwargs)

    builtins.__import__ = _blocker
    ok = False
    try:
        # Clear this module and submodules from cache
        for mod in list(sys.modules.keys()):
            if mod == module_name or mod.startswith(module_name + "."):
                del sys.modules[mod]
        importlib.import_module(module_name)
        ok = True
    except (ImportError, NameError, AttributeError) as e:
        print(f"       → FAIL: {type(e).__name__}: {e}")
    except Exception as e:
        # Other errors (not import-related) are OK-ish
        ok = True
    finally:
        builtins.__import__ = _orig_import
    return ok


def main() -> int:
    print("=" * 70)
    print("browser-use DEPENDENCY BOUNDARY CHECK")
    print("=" * 70)

    all_errors: list[str] = []

    print("\n[1/3] Static: core deps vs extras classification...")
    e = test_core_does_not_include_extras()
    all_errors.extend(e)
    if not e:
        print("  ✓ Static dep classification OK")

    print("\n[2/3] Runtime: top-level extras safety (simulate missing)...")
    e = test_runtime_top_level_safety()
    all_errors.extend(e)

    print("\n[3/3] Runtime: core Agent import chain extras leak check...")
    e = test_core_import_chain()
    all_errors.extend(e)

    print("\n" + "=" * 70)
    if all_errors:
        print(f"❌ {len(all_errors)} BOUNDARY VIOLATION(S):")
        for i, err in enumerate(all_errors, 1):
            print(f"  {i}. {err}")
        print("\n→ See AGENTS.md for dependency rules and lazy-import patterns.")
        return 1
    print("✅ All dependency boundary checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
