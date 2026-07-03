"""Per-session JSONL spill log for cursor progress events.

Full-fidelity companion to the in-memory rolling buffer in ``jobs.py``:
every canonical envelope a run produces is appended as one JSON line to
``<HERMES_HOME>/state/ghost_cursor/logs/<session_id>.jsonl``, so history
that the compact in-memory tails evict stays recoverable and pageable
(``cursor_status(offset=..., limit=...)``).

Record shape::

    {"seq": N, "ts": <epoch seconds>, ...envelope}

``seq`` is a monotonically increasing per-session event index. It continues
across runs that resume the same cursor session and across process restarts
(the writer re-derives the next seq from the file tail), so ``seq`` doubles
as the pagination cursor and ``last_seq + 1`` as the total event count.

Bounding: a log that grows past :data:`MAX_LOG_BYTES` is compacted in place
to head (oldest events) + tail (newest events) retention, with one marker
line ``{"kind": "log_compaction", "first_dropped_seq": ..., ...}`` in the
gap. Seq numbers are preserved, so pagination stays truthful about which
events were dropped instead of silently renumbering.

Contract: mirrors ``handles.py`` — never raises. Disk problems degrade to a
missing/partial log, never a failed run.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hard cap per session log; past it the file is compacted to head + tail.
MAX_LOG_BYTES = 10 * 1024 * 1024
# Retention split on compaction: oldest events (head) + newest (tail). The
# tail gets the larger share — recent history is what callers page for.
HEAD_RETAIN_BYTES = MAX_LOG_BYTES // 4
TAIL_RETAIN_BYTES = MAX_LOG_BYTES // 2

# Inline per-field clip applied to paged events returned by cursor_status.
# The JSONL line itself keeps the full value.
PAGE_FIELD_CHARS = 4096
_PAGE_CLIP_FIELDS = ("output", "diff", "before", "after", "delta", "text")

DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 500

_lock = threading.Lock()
# Writer state keyed by str(path) (not session_id) so a relocated
# HERMES_HOME — e.g. per-test temp homes — never reuses stale counters.
_state: Dict[str, Dict[str, int]] = {}

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def logs_dir() -> Optional[Path]:
    """The per-session log directory (profile-aware). None if unresolvable."""
    try:
        try:
            from hermes_constants import get_hermes_home

            home = Path(get_hermes_home())
        except Exception:
            home = Path.home() / ".hermes"
        return home / "state" / "ghost_cursor" / "logs"
    except Exception:
        return None


def log_path(session_id: Optional[str]) -> Optional[Path]:
    """The JSONL path for a session handle. None on a missing handle/home."""
    sid = str(session_id or "").strip()
    if not sid:
        return None
    base = logs_dir()
    if base is None:
        return None
    return base / f"{_UNSAFE_NAME.sub('_', sid)}.jsonl"


def _load_state_locked(path: Path) -> Dict[str, int]:
    """(Re)derive {next_seq, size} for ``path``, reading only the file tail."""
    key = str(path)
    st = _state.get(key)
    if st is not None:
        return st
    next_seq = 0
    size = 0
    try:
        if path.is_file():
            size = path.stat().st_size
            with path.open("rb") as fh:
                fh.seek(max(0, size - 65536))
                tail = fh.read().decode("utf-8", errors="replace")
            for line in reversed(tail.splitlines()):
                try:
                    seq = json.loads(line).get("seq")
                except Exception:
                    continue
                if isinstance(seq, int):
                    next_seq = seq + 1
                    break
    except Exception:
        logger.debug("ghost_cursor event log state load failed", exc_info=True)
    st = {"next_seq": next_seq, "size": size}
    _state[key] = st
    return st


def append(session_id: Optional[str], envelope: Dict[str, Any]) -> None:
    """Append one full-fidelity envelope line. Never raises."""
    try:
        path = log_path(session_id)
        if path is None:
            return
        with _lock:
            st = _load_state_locked(path)
            record = {"seq": st["next_seq"], "ts": round(time.time(), 3), **envelope}
            try:
                line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            except Exception:
                line = json.dumps({"seq": st["next_seq"], "ts": record["ts"],
                                   "kind": "unserializable_event"}) + "\n"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            st["next_seq"] += 1
            st["size"] += len(line.encode("utf-8"))
            if st["size"] > MAX_LOG_BYTES:
                _compact_locked(path, st)
    except Exception:
        logger.debug("ghost_cursor event log append failed", exc_info=True)


def _line_seq(line: str) -> Optional[int]:
    try:
        seq = json.loads(line).get("seq")
    except Exception:
        return None
    return seq if isinstance(seq, int) else None


def _compact_locked(path: Path, st: Dict[str, int]) -> None:
    """Rewrite the log as head + compaction marker + tail (atomic replace)."""
    lines = path.read_text("utf-8").splitlines(keepends=True)

    head: List[str] = []
    head_bytes = 0
    for line in lines:
        head_bytes += len(line.encode("utf-8"))
        if head_bytes > HEAD_RETAIN_BYTES:
            break
        head.append(line)

    tail: List[str] = []
    tail_bytes = 0
    for line in reversed(lines[len(head):]):
        tail_bytes += len(line.encode("utf-8"))
        if tail_bytes > TAIL_RETAIN_BYTES:
            break
        tail.append(line)
    tail.reverse()

    dropped = lines[len(head): len(lines) - len(tail)]
    if not dropped:
        return
    dropped_seqs = [s for s in (_line_seq(l) for l in dropped) if s is not None]
    marker = json.dumps({
        "kind": "log_compaction",
        "ts": round(time.time(), 3),
        "dropped_events": len(dropped),
        "first_dropped_seq": min(dropped_seqs) if dropped_seqs else None,
        "last_dropped_seq": max(dropped_seqs) if dropped_seqs else None,
    }, ensure_ascii=False) + "\n"

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(head) + marker + "".join(tail), "utf-8")
    tmp.replace(path)
    st["size"] = path.stat().st_size


def _clip_for_page(record: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a record with oversized string fields clipped inline."""
    out = dict(record)
    for field in _PAGE_CLIP_FIELDS:
        val = out.get(field)
        if isinstance(val, str) and len(val) > PAGE_FIELD_CHARS:
            out[field] = val[:PAGE_FIELD_CHARS] + (
                f"… [truncated {len(val) - PAGE_FIELD_CHARS} chars — "
                "full event preserved in the JSONL log]"
            )
    return out


def stats(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """{"path", "total_events"} for a session's log, or None if it has none.

    ``total_events`` counts every event ever appended (``last seq + 1``),
    including events later dropped by compaction. Never raises.
    """
    try:
        path = log_path(session_id)
        if path is None or not path.is_file():
            return None
        with _lock:
            st = _load_state_locked(path)
            return {"path": str(path), "total_events": st["next_seq"]}
    except Exception:
        logger.debug("ghost_cursor event log stats failed", exc_info=True)
        return None


def read_page(
    session_id: Optional[str],
    offset: Any = 0,
    limit: Any = DEFAULT_PAGE_LIMIT,
) -> Optional[Dict[str, Any]]:
    """One page of persisted events, by seq: ``offset <= seq < offset+limit``.

    Returns ``{"events", "offset", "limit", "total_events", "log_path"}``
    plus ``"gaps"`` when compaction dropped events inside the requested
    range (so a pager can tell "dropped" from "never existed"). Oversized
    per-event string fields are clipped inline (:data:`PAGE_FIELD_CHARS`);
    the JSONL file keeps the full values. Returns None when the session has
    no log. Never raises.
    """
    try:
        path = log_path(session_id)
        if path is None or not path.is_file():
            return None
        try:
            off = max(int(offset), 0)
        except (TypeError, ValueError):
            off = 0
        try:
            lim = min(max(int(limit), 1), MAX_PAGE_LIMIT)
        except (TypeError, ValueError):
            lim = DEFAULT_PAGE_LIMIT
        end = off + lim

        events: List[Dict[str, Any]] = []
        gaps: List[Dict[str, Any]] = []
        total = 0
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                seq = record.get("seq")
                if isinstance(seq, int):
                    total = max(total, seq + 1)
                    if off <= seq < end:
                        events.append(_clip_for_page(record))
                elif record.get("kind") == "log_compaction":
                    first = record.get("first_dropped_seq")
                    last = record.get("last_dropped_seq")
                    if (isinstance(first, int) and isinstance(last, int)
                            and first < end and last >= off):
                        gaps.append(record)

        page: Dict[str, Any] = {
            "events": events,
            "offset": off,
            "limit": lim,
            "total_events": total,
            "log_path": str(path),
        }
        if gaps:
            page["gaps"] = gaps
            page["note"] = (
                "Some events in this range were dropped by log compaction "
                "(the log exceeded its size cap); see `gaps`."
            )
        return page
    except Exception:
        logger.debug("ghost_cursor event log read failed", exc_info=True)
        return None


def _reset_for_tests() -> None:
    """Drop cached writer state (test isolation only)."""
    with _lock:
        _state.clear()
