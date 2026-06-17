"""Pure-function helpers for browser storage state handling.

This module is the single source of truth for:

* Normalizing cookie dicts (session cookie expires, sameSite defaults)
* Merging two storage states (cookies by key, origins by deep-merge)
* Resolving whether a ``storage_state`` value represents a writable file path
  or a read-only in-memory dict
* Atomically writing storage state JSON to disk

No code in here imports from ``StorageStateWatchdog`` or ``BrowserSession`` —
this layer is pure and can be imported from either side without creating a
dependency cycle.

Type boundary
-------------
``storage_state`` as found on ``BrowserProfile`` can be:

* **File path** (``str | Path``) → both initial load *and* auto-save apply.
  The file is the source of truth and watchdog writes back to it.
* **In-memory dict** → only used for initial load into the browser session;
  the watchdog must **never** write back to it (it's seed data only).
* ``None`` → nothing to load or save.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

_SESSION_COOKIE_EXPIRES_VALUES: frozenset[float | int] = frozenset({0, 0.0, -1, -1.0})


# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


class StorageStateSource(NamedTuple):
    """A resolved storage state with both its value and origin path.

    Attributes:
        value: The parsed storage state dict (``cookies`` + ``origins``).
        path: Absolute Path to the source file, or ``None`` if the state
            was provided as an in-memory dict / doesn't have a file backing.
            When ``path`` is not ``None``, watchdogs may write state changes
            back to this file.
    """

    value: dict[str, Any]
    path: Path | None

    @property
    def is_persistent(self) -> bool:
        """True if this state is backed by a file we can save to."""
        return self.path is not None

    @classmethod
    def from_any(cls, source: str | Path | dict[str, Any] | None) -> "StorageStateSource | None":
        """Resolve *source* into a ``StorageStateSource``.

        * ``str | Path`` → read file as JSON, ``path`` set to the resolved path.
        * ``dict`` → use as-is, ``path`` is ``None`` (in-memory, not writable).
        * ``None`` → return ``None``.

        If a file path is given but the file doesn't exist or can't be parsed,
        we still return a ``StorageStateSource`` with an empty ``value`` dict
        and the ``path`` set — because the file is still the save target
        (it will be created on first save).
        """
        if source is None:
            return None

        if isinstance(source, dict):
            return cls(value=dict(source), path=None)

        # It's a path — resolve it and try to read it.
        file_path = Path(str(source)).expanduser().resolve()
        value: dict[str, Any] = {"cookies": [], "origins": []}
        if file_path.exists():
            try:
                loaded = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    value = loaded
            except (OSError, json.JSONDecodeError):
                # Corrupt file → start fresh, but path is still valid for saving.
                pass

        return cls(value=value, path=file_path)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_cookie(cookie: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized copy of a single cookie dict.

    * Session cookies (``expires`` in ``{0, -1}``) have the ``expires`` key
      removed so that CDP does not treat them as already-expired when
      re-loaded.
    * ``sameSite`` is defaulted to ``'Lax'`` when absent or ``'None'``
      (which is not a valid CDP value).
    """
    c = dict(cookie)
    expires = c.get("expires")
    if expires in _SESSION_COOKIE_EXPIRES_VALUES:
        c.pop("expires", None)
    same_site = c.get("sameSite")
    if not same_site or same_site == "None":
        c["sameSite"] = "Lax"
    return c


def normalize_storage_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-ish copy of *state* with every cookie normalized."""
    normalized = dict(state)
    if "cookies" in normalized:
        normalized["cookies"] = [
            normalize_cookie(c) if isinstance(c, dict) else c for c in normalized["cookies"]
        ]
    return normalized


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_storage_states(
    existing: dict[str, Any], new: dict[str, Any]
) -> dict[str, Any]:
    """Merge two storage states with *new* taking precedence.

    Cookies are keyed by ``(name, domain, path)``; *new* overwrites
    *existing* on collision.  Origins are keyed by their ``origin`` string
    and ``localStorage`` / ``sessionStorage`` entries are **deep-merged**
    so that items present in *existing* but absent in *new* are not lost.
    """
    merged: dict[str, Any] = {}

    # --- cookies ---
    existing_cookies: dict[tuple[str, str, str], dict[str, Any]] = {
        (c["name"], c["domain"], c["path"]): c
        for c in existing.get("cookies", [])
        if isinstance(c, dict)
    }
    for cookie in new.get("cookies", []):
        if not isinstance(cookie, dict):
            continue
        key = (cookie["name"], cookie["domain"], cookie["path"])
        existing_cookies[key] = cookie
    merged["cookies"] = list(existing_cookies.values())

    # --- origins (deep-merge localStorage / sessionStorage items) ---
    def _merge_storage_items(
        existing_items: list[dict[str, str]] | None,
        new_items: list[dict[str, str]] | None,
    ) -> list[dict[str, str]] | None:
        if not existing_items and not new_items:
            return None
        items_by_name: dict[str, dict[str, str]] = {}
        for item in existing_items or []:
            items_by_name[item["name"]] = item
        for item in new_items or []:
            items_by_name[item["name"]] = item
        return list(items_by_name.values())

    existing_origins: dict[str, dict[str, Any]] = {
        o["origin"]: o for o in existing.get("origins", [])
    }
    for origin in new.get("origins", []):
        origin_name = origin.get("origin")
        if not origin_name:
            continue
        if origin_name not in existing_origins:
            existing_origins[origin_name] = origin
        else:
            merged_origin: dict[str, Any] = {"origin": origin_name}
            merged_origin["localStorage"] = _merge_storage_items(
                existing_origins[origin_name].get("localStorage"),
                origin.get("localStorage"),
            )
            merged_origin["sessionStorage"] = _merge_storage_items(
                existing_origins[origin_name].get("sessionStorage"),
                origin.get("sessionStorage"),
            )
            if merged_origin["localStorage"] is None:
                merged_origin.pop("localStorage")
            if merged_origin["sessionStorage"] is None:
                merged_origin.pop("sessionStorage")
            existing_origins[origin_name] = merged_origin

    merged["origins"] = list(existing_origins.values())

    return merged


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_storage_state_path(
    storage_state: str | Path | dict[str, Any] | None,
) -> Path | None:
    """Return the writable file path for *storage_state*, or ``None``.

    * ``str | Path`` → resolved ``Path`` (may or may not exist yet).
    * ``dict`` → ``None`` (in-memory only, not a valid save target).
    * ``None`` → ``None``.
    """
    if storage_state is None:
        return None
    if isinstance(storage_state, dict):
        return None
    return Path(str(storage_state)).expanduser().resolve()


# ---------------------------------------------------------------------------
# Atomic file write
# ---------------------------------------------------------------------------


def write_storage_state_atomically(json_path: Path, state: dict[str, Any]) -> None:
    """Write *state* to *json_path* atomically.

    Uses ``tempfile.mkstemp`` in the same directory followed by
    ``os.replace`` so that a crash never leaves the destination file
    in a partial or missing state.
    """
    json_path = Path(json_path).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(json_path.parent),
        prefix=json_path.stem + "_",
        suffix=".json.tmp",
    )
    tmp_path_str = tmp_path
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(state, indent=4, ensure_ascii=False))
        os.replace(tmp_path_str, str(json_path))
        tmp_path_str = None  # type: ignore[assignment]
    finally:
        if tmp_path_str and os.path.exists(tmp_path_str):  # type: ignore[arg-type]
            os.unlink(tmp_path_str)  # type: ignore[arg-type]
