"""Persistent handle table for cursor sessions — explicit handles, no heuristics.

EXPLICIT session handles, no auto-resume heuristics: ``cursor_create_session``
mints a session NAME and every other tool takes it back (the cursor ACP
UUID resolves as an alias). This module is the tiny persistence layer under
that model — a JSON file mapping
``session_name -> {repo, status, task, model, cursor_session_id, ...}`` so a
handle minted on turn T is still resolvable on turn T+1 even across a process
restart (the live job table in ``jobs.py`` is process-local).

What this is NOT: there is no ``get_recent()``, no repo matching, no age
window, no auto-resume. Lookup is by exact handle only. If the caller lost
the handle, the run's completion delivery re-states it; nothing is guessed.

Session scoping: each entry records the dispatching Hermes ``session_key``
(empty string = CLI). LISTING (``known_handles``) is scoped to the caller's
session_key by default (``scope="session"``); ``scope="all"`` opts into the
global view. Direct ``get(session_id)`` lookups are deliberately UNSCOPED —
an explicit handle is explicit intent, and cross-session continuation must
keep working.

Storage: ``<HERMES_HOME>/state/ghost_cursor_handles.json``.

Bounded growth: terminal-state entries older than
:data:`PRUNE_TERMINAL_AFTER_S` are dropped on every write, and the table is
capped at :data:`MAX_ENTRIES` — evicting oldest TERMINAL entries first so a
crowd of finished runs can never push out another session's live handle.

Contract: never raises. A missing/corrupt/unwritable file degrades to the
in-memory (process-local) view.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Retain this many handles in the file. Sized for many concurrent Hermes
# sessions sharing one table (the entries are small); terminal entries are
# evicted before running ones, so the cap degrades gracefully under load.
MAX_ENTRIES = 500
# Terminal-state entries older than this are pruned on every write.
PRUNE_TERMINAL_AFTER_S = 7 * 24 * 3600

# Mirrors jobs.TERMINAL_STATUSES (duplicated to avoid a circular import —
# jobs.py imports this module).
_TERMINAL_STATUSES = ("completed", "failed", "cancelled", "timeout")

VALID_SCOPES = ("session", "all")

_lock = threading.Lock()
_table: Dict[str, Dict[str, Any]] = {}
_loaded = False


def _state_file() -> Optional[Path]:
    """Path to the JSON backing file (profile-aware). None if unresolvable."""
    try:
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:
            home = Path.home() / ".hermes"
        return home / "state" / "ghost_cursor_handles.json"
    except Exception:
        return None


def _load_locked() -> None:
    """Merge the backing file into the in-memory table (once per process)."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    path = _state_file()
    try:
        if path is not None and path.is_file():
            data = json.loads(path.read_text("utf-8"))
            if isinstance(data, dict):
                for key, entry in data.items():
                    if isinstance(key, str) and isinstance(entry, dict):
                        # In-memory (this process) entries are fresher.
                        _table.setdefault(key, entry)
    except Exception:
        logger.debug("ghost_cursor handle table load failed", exc_info=True)


def _save_locked() -> None:
    path = _state_file()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(_table, ensure_ascii=False), "utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("ghost_cursor handle table save failed", exc_info=True)


def _is_terminal(entry: Dict[str, Any]) -> bool:
    return str(entry.get("status") or "") in _TERMINAL_STATUSES


def _prune_locked(now: Optional[float] = None) -> None:
    """Age out old terminal entries, then enforce the size cap.

    Eviction order under the cap: oldest TERMINAL entries first; only when
    the table is over cap with no terminal entries left do the oldest
    non-terminal (running/unknown) entries go — a burst of finished runs
    can't push out live handles.
    """
    now = time.time() if now is None else now
    for key in [
        k for k, e in _table.items()
        if _is_terminal(e)
        and now - float(e.get("updated_at") or 0.0) > PRUNE_TERMINAL_AFTER_S
    ]:
        _table.pop(key, None)

    if len(_table) <= MAX_ENTRIES:
        return
    by_age = sorted(
        _table.items(),
        key=lambda kv: (not _is_terminal(kv[1]), float(kv[1].get("updated_at") or 0.0)),
    )
    for key, _ in by_age[: len(_table) - MAX_ENTRIES]:
        _table.pop(key, None)


def record(session_id: Optional[str], **fields: Any) -> None:
    """Merge ``fields`` into the entry for ``session_id``. Never raises.

    No-ops on a missing session_id (e.g. a run that failed before the ACP
    session was ever established has no handle to record).
    """
    try:
        if not session_id:
            return
        with _lock:
            _load_locked()
            entry = _table.setdefault(str(session_id), {})
            entry.update({k: v for k, v in fields.items() if v is not None})
            entry["updated_at"] = time.time()
            _prune_locked()
            _save_locked()
    except Exception:
        logger.debug("ghost_cursor handle record failed", exc_info=True)


def _resolve_locked(identifier: str) -> Optional[str]:
    """Canonical handle name for ``identifier`` (name or UUID alias)."""
    if identifier in _table:
        return identifier
    for name, entry in _table.items():
        if (
            isinstance(entry, dict)
            and str(entry.get("cursor_session_id") or "") == identifier
        ):
            return name
    return None


def resolve(identifier: Optional[str]) -> Optional[str]:
    """The canonical session NAME for a name-or-UUID identifier, or None.

    v0.4 keys the table by human slug (``playful-space-bunny``); the cursor
    ACP UUID is recorded on the entry as ``cursor_session_id`` and stays a
    working alias — this is the lookup that makes UUIDs resolve everywhere
    a name is accepted. Never raises.
    """
    try:
        ident = str(identifier or "").strip()
        if not ident:
            return None
        with _lock:
            _load_locked()
            return _resolve_locked(ident)
    except Exception:
        logger.debug("ghost_cursor handle resolve failed", exc_info=True)
        return None


def get(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The persisted entry for a name or UUID alias, or None. Never raises.

    Deliberately UNSCOPED: an explicit handle is explicit intent, so direct
    lookups resolve across Hermes sessions (see module docstring).
    """
    try:
        if not session_id:
            return None
        with _lock:
            _load_locked()
            name = _resolve_locked(str(session_id).strip())
            entry = _table.get(name) if name else None
        return dict(entry) if isinstance(entry, dict) else None
    except Exception:
        logger.debug("ghost_cursor handle lookup failed", exc_info=True)
        return None


def _in_scope(entry: Dict[str, Any], scope: str, session_key: str) -> bool:
    if scope == "all":
        return True
    # Session scope: entries recorded by the same Hermes session. Entries
    # written before session_key existed (legacy) count as CLI (key "").
    return str(entry.get("session_key") or "") == session_key


def known_handles(
    limit: int = 10,
    scope: str = "session",
    session_key: str = "",
) -> List[str]:
    """The most recently updated handles (for actionable error messages).

    ``scope="session"`` (default) lists only handles recorded by
    ``session_key``; ``scope="all"`` lists everything. An unknown scope
    value falls back to "session" — the conservative view.
    """
    try:
        scope = scope if scope in VALID_SCOPES else "session"
        with _lock:
            _load_locked()
            items = sorted(
                (
                    kv for kv in _table.items()
                    if _in_scope(kv[1], scope, str(session_key or ""))
                ),
                key=lambda kv: float(kv[1].get("updated_at") or 0.0),
                reverse=True,
            )
        return [k for k, _ in items[: max(int(limit), 0)]]
    except Exception:
        return []


def entries(
    scope: str = "session",
    session_key: str = "",
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Scoped handle entries, most recently updated first (for cursor_list).

    Each item is a copy of the persisted entry with the handle name added
    under ``"session"``. Never raises.
    """
    try:
        scope = scope if scope in VALID_SCOPES else "session"
        with _lock:
            _load_locked()
            items = sorted(
                (
                    kv for kv in _table.items()
                    if isinstance(kv[1], dict)
                    and _in_scope(kv[1], scope, str(session_key or ""))
                ),
                key=lambda kv: float(kv[1].get("updated_at") or 0.0),
                reverse=True,
            )
            return [
                {"session": name, **entry}
                for name, entry in items[: max(int(limit), 0)]
            ]
    except Exception:
        return []
