"""Persistent handle table for cursor sessions — explicit handles, no heuristics.

v0.3 replaces the old ``session_registry.py`` repo+timestamp auto-resume
heuristic with EXPLICIT session handles: ``cursor_start`` returns the cursor
``session_id`` and every other tool takes it back. This module is the tiny
persistence layer under that model — a JSON file mapping
``cursor_session_id -> {repo, status, task, model, updated_at, ...}`` so a
handle minted on turn T is still resolvable on turn T+1 even across a process
restart (the live job table in ``jobs.py`` is process-local).

What this is NOT: there is no ``get_recent()``, no repo matching, no age
window, no auto-resume. Lookup is by exact handle only. If the caller lost
the handle, the run's completion delivery re-states it; nothing is guessed.

Storage: ``<HERMES_HOME>/state/ghost_cursor_handles.json``.

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

# Retain this many handles in the file (pruned oldest-first by updated_at).
MAX_ENTRIES = 50

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


def _prune_locked() -> None:
    if len(_table) <= MAX_ENTRIES:
        return
    by_age = sorted(
        _table.items(), key=lambda kv: float(kv[1].get("updated_at") or 0.0)
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


def get(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """The persisted entry for ``session_id``, or None. Never raises."""
    try:
        if not session_id:
            return None
        with _lock:
            _load_locked()
            entry = _table.get(str(session_id))
        return dict(entry) if isinstance(entry, dict) else None
    except Exception:
        logger.debug("ghost_cursor handle lookup failed", exc_info=True)
        return None


def known_handles(limit: int = 10) -> List[str]:
    """The most recently updated handles (for actionable error messages)."""
    try:
        with _lock:
            _load_locked()
            items = sorted(
                _table.items(),
                key=lambda kv: float(kv[1].get("updated_at") or 0.0),
                reverse=True,
            )
        return [k for k, _ in items[: max(int(limit), 0)]]
    except Exception:
        return []
