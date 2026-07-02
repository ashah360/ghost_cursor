"""Eager cursor-session persistence for interject-via-resume.

When a Hermes turn is interrupted mid-``cursor_edit`` (stop + nudge), the
in-flight tool result — which carries the cursor ``session_id`` — may never
be appended to the transcript, so the next turn has nothing to resume with.
This registry closes that gap: ``cursor_edit`` records the active cursor
session id the instant ACP establishes it (before/independent of the tool
returning), keyed by the calling Hermes session id, so it survives
cancellation and the next ``cursor_edit`` can auto-resume it.

Storage: a process-global dict backed by a tiny JSON file at
``<HERMES_HOME>/state/ghost_cursor_sessions.json`` mapping
``hermes_session_id -> {cursor_session_id, repo, updated_at, status}``
where ``status`` is ``"running" | "cancelled" | "completed"``.

Contract: never raises. A missing/corrupt/unwritable file degrades to
no-auto-resume (the pre-registry behavior).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Only an interrupted run ("running" that never settled, or "cancelled") is
# continuable — a cleanly completed run must NOT be auto-resumed by the next
# unrelated task.
CONTINUABLE_STATUSES = ("running", "cancelled")

# Auto-resume window: a prior entry older than this is stale (the nudge
# use case is seconds-to-minutes after the stop, not hours).
DEFAULT_MAX_AGE_S = 600.0

_lock = threading.Lock()
_registry: Dict[str, Dict[str, Any]] = {}
_loaded = False


def _state_file() -> Optional[Path]:
    """Path to the JSON backing file (profile-aware). None if unresolvable."""
    try:
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:
            home = Path.home() / ".hermes"
        return home / "state" / "ghost_cursor_sessions.json"
    except Exception:
        return None


def _load_locked() -> None:
    """Merge the backing file into the in-memory dict (once per process)."""
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
                        _registry.setdefault(key, entry)
    except Exception:
        logger.debug("ghost_cursor session registry load failed", exc_info=True)


def _save_locked() -> None:
    path = _state_file()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(_registry, ensure_ascii=False), "utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("ghost_cursor session registry save failed", exc_info=True)


def record(
    hermes_sid: Optional[str],
    cursor_sid: Optional[str],
    repo: Optional[str],
    status: str,
) -> None:
    """Persist the active cursor session for ``hermes_sid``. Never raises.

    No-ops when either id is missing (e.g. the calling agent was not
    resolvable) — that degrades to the pre-registry behavior.
    """
    try:
        if not hermes_sid or not cursor_sid:
            return
        with _lock:
            _load_locked()
            _registry[str(hermes_sid)] = {
                "cursor_session_id": str(cursor_sid),
                "repo": str(repo or ""),
                "updated_at": time.time(),
                "status": str(status),
            }
            _save_locked()
    except Exception:
        logger.debug("ghost_cursor session registry record failed", exc_info=True)


def get_recent(
    hermes_sid: Optional[str],
    repo: Optional[str],
    max_age_s: float = DEFAULT_MAX_AGE_S,
) -> Optional[str]:
    """Cursor session id continuable for ``hermes_sid`` + ``repo``, else None.

    Returns the id ONLY when a recent (<= ``max_age_s``) entry exists for
    that Hermes session AND repo AND its status is continuable
    ("running"/"cancelled" — i.e. an interrupted run, not a completed one).
    Never raises.
    """
    try:
        if not hermes_sid:
            return None
        with _lock:
            _load_locked()
            entry = _registry.get(str(hermes_sid))
        if not isinstance(entry, dict):
            return None
        if str(entry.get("repo") or "") != str(repo or ""):
            return None
        if str(entry.get("status") or "") not in CONTINUABLE_STATUSES:
            return None
        updated_at = float(entry.get("updated_at") or 0.0)
        if time.time() - updated_at > float(max_age_s):
            return None
        cursor_sid = str(entry.get("cursor_session_id") or "")
        return cursor_sid or None
    except Exception:
        logger.debug("ghost_cursor session registry lookup failed", exc_info=True)
        return None
