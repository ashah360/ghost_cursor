"""Task-backed session supervisor (RFC: docs/rfcs/session-supervisor.md).

One supervisor loop per live cursor session, spawned by a reconciler that
runs at plugin init and every :data:`RECONCILE_INTERVAL_S`. The supervisor
owns everything that previously lived across three modules with implicit
coupling: SSE stream consumption, per-session seq assignment, jsonl
append, digest ticks, completion delivery, and the poll watchdog.

Durable state lives on the session's handle entry as the ``supervision``
record (``handles.supervision_of`` / ``handles.record_supervision``)::

    {phase, current_attempt_id, attempt_n,
     last_seq_delivered: {subscriber: seq},
     watchdog: {last_poll_ts, last_remote_status}}

so a gateway restart is a non-event: the reconciler sees handles in a
non-terminal phase with no live supervisor task and re-attaches (k8s
controller pattern — RFC §1).

Core invariants (enforced here, tested in the unit suite):

* **Single-writer settlement (RFC §3):** only the supervisor settles a
  session (terminal phase + completion fan-out). Tool calls request
  transitions; the supervisor applies them.
* **Terminal precedence (RFC §2):** a remote ``GET`` run status that is
  terminal ALWAYS wins over a replayed stream's terminal status — a
  cancelled run's replay emits ``status: FINISHED`` while the GET says
  CANCELLED (verified live). Settlement never reads terminal state from
  replay.
* **Ingest boundary (RFC §4):** dedupe ``interaction_update`` twins by
  provider event id BEFORE seq assignment; assign monotonic per-session
  seq post-dedupe; stamp ``attemptId`` on every event;
  ``lifecycle.durable_progress`` is derived supervisor-side from observed
  ``file_diff`` / irreversible completed ``tool_use`` events — never
  trusted from agent self-report.
* **Retry policy (RFC §5):** zero durable progress → auto-retry allowed
  (cap :data:`MAX_AUTO_RETRIES`), ``lifecycle.retry_started``; any
  durable progress → ``lifecycle.retry_suppressed``, explicit resume
  required. Terminal events carry a death-shape tag (``fast_fail`` vs
  ``mid_flight``).
* **Delivery cursors:** ``last_seq_delivered`` advances only after a
  successful enqueue onto the completion queue; duplicate digests are
  acceptable (consumers dedupe on delegation id), completion delivery is
  exactly-once per subscriber.

Threading model: consistent with the rest of the plugin — one daemon
thread per supervised session, a daemon reconciler thread, all state
handed off through ``handles.py`` / ``eventlog.py`` (both never-raise).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from . import eventlog as _eventlog
from . import handles as _handles

logger = logging.getLogger(__name__)

# Reconciler cadence (RFC §1: plugin init + every 60s).
RECONCILE_INTERVAL_S = 60.0
# Watchdog: poll GET /v1/agents/{id}/runs when the stream has been silent
# AND unreconnectable this long (RFC §2).
WATCHDOG_INTERVAL_S = 60.0
# Auto-retry cap for zero-progress attempts (RFC §5).
MAX_AUTO_RETRIES = 3
# Death-shape boundary: a terminal attempt under this age with
# lifecycle-only events is a fast_fail; anything with real events is
# mid_flight (RFC §5).
FAST_FAIL_WINDOW_S = 120.0

# Supervision phases (mirrors handles.SUPERVISION_LIVE_PHASES /
# SUPERVISION_TERMINAL_PHASES).
PHASE_SPAWNING = "spawning"
PHASE_STREAMING = "streaming"
PHASE_RETRYING = "retrying"
TERMINAL_PHASES = _handles.SUPERVISION_TERMINAL_PHASES

# tool_use names whose COMPLETED calls count as durable progress (RFC §4:
# irreversible side effects observed by the deriver, never self-reported).
DURABLE_TOOL_NAMES = ("shell", "write", "edit_file", "str_replace", "delete")


def mint_attempt_id() -> str:
    """A fresh stable attempt id, stamped on every event as ``attemptId``."""
    return f"att-{uuid.uuid4().hex[:12]}"


def is_durable_progress(envelope: Dict[str, Any]) -> bool:
    """Whether one canonical envelope is durable-progress evidence.

    Controller-derived (RFC §4): any ``file_diff``, or a COMPLETED
    ``tool_use`` whose tool is on the irreversible list. Lifecycle
    chatter, reasoning, and assistant prose never count.
    """
    kind = str(envelope.get("kind") or "")
    if kind == "file_diff":
        return True
    if kind == "tool_use":
        status = str(envelope.get("status") or "")
        name = str(envelope.get("tool") or envelope.get("name") or "")
        return status == "completed" and name in DURABLE_TOOL_NAMES
    return False


class SessionSupervisor:
    """One session's supervisor loop (RFC §1) — skeleton.

    Owns the session's supervision record, ingest boundary, watchdog and
    settlement. The loop body lands incrementally; the record lifecycle
    and ingest helpers are functional now so dispatch can already write
    durable supervision state.
    """

    def __init__(self, session_name: str) -> None:
        self.session_name = str(session_name)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run,
            name=f"ghost-cursor-supervisor-{self.session_name}",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    # -- the loop ----------------------------------------------------------

    def _run(self) -> None:
        """Supervisor loop: stream → ingest → digest → settle.

        Skeleton: full stream consumption / watchdog / settlement land in
        follow-up commits on this branch.
        """
        logger.info(
            "supervisor attached to session %s (phase=%s)",
            self.session_name,
            _handles.supervision_of(_handles.get(self.session_name))["phase"],
        )


# ---------------------------------------------------------------------------
# Registry of live supervisor tasks + reconciler (RFC §1)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_supervisors: Dict[str, SessionSupervisor] = {}
_reconciler_timer: Optional[threading.Timer] = None
_reconciler_started = False


def ensure_supervisor(session_name: str) -> Optional[SessionSupervisor]:
    """The live supervisor for a session, spawning one if needed.

    No-op (returns None) for handles that are not in a live supervision
    phase — legacy entries and settled sessions never get a task.
    """
    name = str(session_name or "").strip()
    if not name:
        return None
    if not _handles.supervision_is_live(_handles.get(name)):
        return None
    with _lock:
        existing = _supervisors.get(name)
        if existing is not None and existing.is_alive():
            return existing
        sup = SessionSupervisor(name)
        _supervisors[name] = sup
    sup.start()
    return sup


def reconcile_once() -> List[str]:
    """One reconciler pass: spawn a supervisor for every handle in a
    non-terminal supervision phase with no live task. Returns the session
    names attached this pass (for logs/tests). Never raises."""
    attached: List[str] = []
    try:
        for entry in _handles.entries(scope="all", limit=_handles.MAX_ENTRIES):
            name = str(entry.get("session") or "")
            if not name or not _handles.supervision_is_live(entry):
                continue
            with _lock:
                existing = _supervisors.get(name)
                if existing is not None and existing.is_alive():
                    continue
            if ensure_supervisor(name) is not None:
                attached.append(name)
    except Exception:
        logger.exception("ghost_cursor reconciler pass failed")
    return attached


def _reconcile_tick() -> None:
    try:
        reconcile_once()
    finally:
        _arm_reconciler()


def _arm_reconciler() -> None:
    global _reconciler_timer
    with _lock:
        if not _reconciler_started:
            return
        timer = threading.Timer(RECONCILE_INTERVAL_S, _reconcile_tick)
        timer.daemon = True
        timer.name = "ghost-cursor-reconciler"
        _reconciler_timer = timer
    timer.start()


def start_reconciler() -> None:
    """Start the periodic reconciler (idempotent; called at plugin init).

    Runs one pass immediately — re-attaching supervisors to any handle
    left in a live phase by a previous process — then every
    :data:`RECONCILE_INTERVAL_S`.
    """
    global _reconciler_started
    with _lock:
        if _reconciler_started:
            return
        _reconciler_started = True
    reconcile_once()
    _arm_reconciler()


def _reset_for_tests() -> None:
    """Stop the reconciler and every supervisor (test isolation only)."""
    global _reconciler_timer, _reconciler_started
    with _lock:
        timer = _reconciler_timer
        _reconciler_timer = None
        _reconciler_started = False
        sups = list(_supervisors.values())
        _supervisors.clear()
    if timer is not None:
        timer.cancel()
    for sup in sups:
        sup.stop()
