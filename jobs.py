"""Job table for cursor runs — one entry per dispatched run, keyed by handle.

v0.3 session-handle model
-------------------------
Every cursor run executes as a background :class:`CursorJob` on a worker
thread. The SINGLE public handle for a run is the cursor ``session_id``
(minted by ACP ``session/new``, surfaced at the ``acp.session`` event):
``cursor_start`` returns it, and ``cursor_send`` / ``cursor_status`` /
``cursor_stop`` take it back. This registry is the in-process job table
behind that handle — rolling progress buffer, status, files-changed
aggregation, and deliver-on-complete. A tiny JSON persistence layer
(``handles.py``) mirrors handle → status/repo across process restarts;
the live job state itself is process-local (like async delegation).

There is deliberately NO auto-resume heuristic here (the v0.2
``session_registry`` repo+timestamp guesswork is gone). Lookup is by
explicit handle only: :meth:`CursorJobRegistry.get_by_session`.

Completion delivery (reuses core infra — nothing reinvented)
------------------------------------------------------------
On EVERY terminal state (completed / failed / cancelled / timeout — never
silent), a job whose delivery flag is still set pushes an event onto the
shared ``tools.process_registry.process_registry.completion_queue`` with
``type="async_delegation"`` — the exact rail ``delegate_task(background=true)``
uses (see ``tools/async_delegation.py``). The CLI ``process_loop`` drain and
the gateway's ``_async_delegation_watcher`` already consume that queue while
the agent is idle and inject each event as a fresh turn, which keeps strict
message-role alternation legal and the prompt cache intact. The event's
``session_key`` (captured on the dispatching thread) routes it back to the
originating gateway session; an empty key means CLI.

Delivery is ARMED, not assumed: a job is dispatched with ``deliver=False``
and the dispatching tool arms it (:meth:`CursorJob.arm_delivery`) only
after it has returned the running-handle shape to the caller. A run that
reaches a terminal state BEFORE that (handshake failure, ultra-fast
completion) is reported in-turn by the dispatching tool itself — arming
races finalize under the job lock, so the outcome lands exactly once:
either in the tool result or as a delivered message, never both, never
neither. Symmetrically, when ``cursor_stop`` / ``cursor_send`` settle a
run in-turn they disarm delivery first (:meth:`CursorJob.mark_handled`).

The completion payload carries the FULL final result dict (success / status /
summary / files_changed / session_id / resumed) both as a structured
``result`` field and rendered into the human-readable ``summary`` block, so
continuation (pass ``session_id`` to ``cursor_send``) survives the async
boundary.

Concurrency guard (same repo)
-----------------------------
Two agents editing one working tree corrupts it. ``dispatch()`` atomically
rejects a new job when an active job already holds the same resolved repo,
returning the existing job so the caller can surface its handle. Parallel
runs on DIFFERENT repos are fine — different handles.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import eventlog as _eventlog
from . import handles as _handles
from . import render as _render

logger = logging.getLogger(__name__)

# Rolling progress buffer cap — mirrors process_registry.MAX_OUTPUT_CHARS.
MAX_PROGRESS_CHARS = 200_000
# How many finished jobs to retain for cursor_status queries.
MAX_FINISHED_JOBS = 20
# Envelopes produced before the ACP session (the spill-log key) exists are
# held here, then flushed the moment the handle arrives. Bounded so a run
# that never gets a session can't grow it forever.
MAX_PENDING_SPILL = 1_000

# Per-envelope trims for the rolling buffer (full-fidelity diffs live in the
# job's ``files`` aggregation, same caps as the final result).
_BUFFER_DIFF_CHARS = 2_000
_BUFFER_OUTPUT_CHARS = 1_000
# Trims for the completion payload / status snapshots.
_PAYLOAD_DIFF_CHARS = 2_000
_PAYLOAD_SUMMARY_CHARS = 4_000

TERMINAL_STATUSES = ("completed", "failed", "cancelled", "timeout")


def _clip(text: Any, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [truncated {len(s) - limit} chars]"


def trim_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of a final result dict safe to embed in a delivery payload.

    Per-file diffs and the prose summary are clipped; everything else —
    notably ``session_id`` / ``resumed`` / ``files_changed`` structure —
    passes through untouched so continuation keeps working.
    """
    out = dict(result)
    if isinstance(out.get("summary"), str):
        out["summary"] = _clip(out["summary"], _PAYLOAD_SUMMARY_CHARS)
    files = out.get("files_changed")
    if isinstance(files, list):
        trimmed = []
        for f in files:
            if isinstance(f, dict) and isinstance(f.get("diff"), str):
                f = {**f, "diff": _clip(f["diff"], _PAYLOAD_DIFF_CHARS)}
            trimmed.append(f)
        out["files_changed"] = trimmed
    return out


@dataclass
class CursorJob:
    """One dispatched cursor run (always a background worker thread)."""

    job_id: str  # internal dispatch id; the PUBLIC handle is session_name
    task: str
    repo: str
    # Watchdog knobs (see acp_runner): abort after this much SILENCE (no ACP
    # events) — activity resets the clock — plus an optional hard ceiling on
    # total run time (0 = disabled).
    inactivity_timeout_s: float
    max_wall_s: float = 0.0
    # v0.4 handle: the human slug minted by cursor_create_session (e.g.
    # "playful-space-bunny"). The cursor ACP UUID (cursor_session_id below)
    # stays a resolvable alias. Empty only for direct registry use in tests.
    session_name: str = ""
    session_key: str = ""
    requested_session_id: Optional[str] = None
    requested_model: Optional[str] = None
    deliver: bool = False             # armed by the dispatching tool (see module doc)
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cursor_session_id: str = ""       # THE handle, set at acp.session
    resumed: bool = False
    model: str = ""                   # actual model reported by acp.session
    # Wall-clock time of the last ACP event received for this run (None
    # until the first event). Advisory only — feeds the last_activity_s
    # field of status snapshots so callers can flag silent runs; the
    # actual inactivity watchdog lives in acp_runner.
    last_event_at: Optional[float] = None
    # --- aggregation state (guarded by _lock) ---
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assistant_parts: List[str] = field(default_factory=list)
    reasoning_tail: str = ""
    progress_buffer: str = ""
    progress_events: int = 0
    run_error: Optional[str] = None
    timed_out: bool = False
    completed: bool = False
    cancelled: bool = False
    result: Optional[Dict[str, Any]] = None
    # --- control ---
    _pending_spill: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    # Set the instant the ACP session is established (handle available).
    session_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    # -- control -------------------------------------------------------------

    def request_cancel(self) -> None:
        """Ask the run to stop via cursor's native session/cancel path."""
        self.cancel_event.set()

    def arm_delivery(self) -> bool:
        """Turn on completion delivery for this run.

        Called by the dispatching tool AFTER it has handed the running
        handle back to the caller — from then on the outcome must arrive as
        a delivered message. Returns False when the job already reached a
        terminal state (finalize won the race with ``deliver`` still False,
        so nothing was enqueued and the caller must report the final result
        in-turn instead). Atomic with respect to finalize via the job lock.
        """
        with self._lock:
            if self.status in TERMINAL_STATUSES:
                return False
            self.deliver = True
            return True

    def mark_handled(self) -> bool:
        """Suppress completion delivery — the caller reports the outcome in-turn.

        Used by ``cursor_stop`` / ``cursor_send`` before they cancel: the
        terminal state reaches the conversation through the tool result, so
        the async delivery would be a duplicate message. Returns False when
        the job already reached a terminal state (finalize won the race —
        delivery may already have fired; the caller just reads the result).
        """
        with self._lock:
            if self.status in TERMINAL_STATUSES:
                return False
            self.deliver = False
            return True

    # -- progress buffer -------------------------------------------------------

    def append_progress(self, envelope: Dict[str, Any]) -> None:
        """Append one canonical envelope to the rolling progress buffer.

        Mirrors the process-registry rolling-output pattern: bounded size,
        trimmed from the front at a line boundary. Full before/after file
        content is dropped (the diff carries the signal); diffs and shell
        output are clipped so one giant edit can't evict all history.

        The UNCLIPPED envelope is also spilled to the per-session JSONL
        event log (``eventlog.py``) so everything the compact buffer evicts
        or trims stays recoverable and pageable via ``cursor_status``.
        Envelopes that arrive before the session handle exists are held in
        a bounded pending list and flushed the moment it does.
        """
        compact = {k: v for k, v in envelope.items() if k not in ("before", "after")}
        if isinstance(compact.get("diff"), str):
            compact["diff"] = _clip(compact["diff"], _BUFFER_DIFF_CHARS)
        if isinstance(compact.get("output"), str):
            compact["output"] = _clip(compact["output"], _BUFFER_OUTPUT_CHARS)
        try:
            line = json.dumps(compact, ensure_ascii=False, default=str)
        except Exception:
            line = str(compact)
        with self._lock:
            self.progress_buffer += line + "\n"
            if len(self.progress_buffer) > MAX_PROGRESS_CHARS:
                cut = self.progress_buffer[-MAX_PROGRESS_CHARS:]
                nl = cut.find("\n")
                self.progress_buffer = cut[nl + 1:] if nl >= 0 else cut
            self.progress_events += 1
            # Spill key: the session NAME (stable across runs and ACP
            # session ids, so one named session = one log). Fallbacks keep
            # name-less jobs (direct registry use) spilling by cursor sid.
            spill_sid = (
                self.session_name
                or self.cursor_session_id
                or self.requested_session_id
                or ""
            )
            if spill_sid:
                to_spill = self._pending_spill + [envelope]
                self._pending_spill = []
            else:
                if len(self._pending_spill) < MAX_PENDING_SPILL:
                    self._pending_spill.append(dict(envelope))
                to_spill = []
        # File I/O outside the job lock (eventlog serializes internally);
        # only the worker thread appends progress, so order is preserved.
        for env in to_spill:
            _eventlog.append(spill_sid, env)

    # -- read-only view ---------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """A read-only status snapshot (what ``cursor_status`` returns).

        STRICTLY read-only: takes the lock only to copy state. Never touches
        the cancel event, the ACP session, or the worker thread.
        """
        now = time.time()
        with self._lock:
            files = []
            for f in sorted(self.files.values(), key=lambda x: x.get("path", "")):
                entry = dict(f)
                if isinstance(entry.get("diff"), str):
                    entry["diff"] = _clip(entry["diff"], _PAYLOAD_DIFF_CHARS)
                files.append(entry)
            snap: Dict[str, Any] = {
                "session": self.session_name,
                "session_id": self.cursor_session_id,
                "status": self.status,
                "repo": self.repo,
                "task": _clip(self.task, 400),
                "elapsed_s": round((self.finished_at or now) - self.created_at, 1),
                # Seconds since the last ACP event (advisory: lets callers
                # flag a silent run without touching it). Frozen at
                # finished_at for terminal runs; falls back to created_at
                # before the first event arrives.
                "last_activity_s": round(
                    (self.finished_at or now) - (self.last_event_at or self.created_at), 1
                ),
                "cursor_session_id": self.cursor_session_id,
                "resumed": self.resumed,
                "model": self.model or (self.requested_model or ""),
                "summary_so_far": _clip(
                    "".join(self.assistant_parts).strip(), _PAYLOAD_SUMMARY_CHARS
                ),
                "files_changed_so_far": files,
                "files_changed_count": len(files),
                "latest_reasoning": self.reasoning_tail[-1500:],
                "progress_tail": self.progress_buffer[-4000:],
                "progress_events": self.progress_events,
            }
            if self.result is not None:
                snap["result"] = trim_result(self.result)
        return snap


class CursorJobRegistry:
    """Process-global tracker of cursor jobs (thread-safe)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, CursorJob] = {}
        self._lock = threading.Lock()

    # -- dispatch ---------------------------------------------------------------

    def dispatch(
        self,
        *,
        runner: Callable[[CursorJob], Dict[str, Any]],
        task: str,
        repo: str,
        inactivity_timeout_s: float,
        max_wall_s: float = 0.0,
        session_name: str = "",
        session_key: str = "",
        requested_session_id: Optional[str] = None,
        requested_model: Optional[str] = None,
    ) -> Tuple[Optional[CursorJob], Optional[CursorJob]]:
        """Start a cursor run on a worker thread.

        Returns ``(job, None)`` on success or ``(None, existing_job)`` when
        an active job already holds ``repo`` (same-repo concurrency guard —
        two cursor agents on one working tree corrupt it). The check and the
        insert happen under one lock hold, so two concurrent dispatches can't
        both pass.
        """
        job = CursorJob(
            job_id=f"cursor_{uuid.uuid4().hex[:8]}",
            task=task,
            repo=repo,
            inactivity_timeout_s=inactivity_timeout_s,
            max_wall_s=max_wall_s,
            session_name=session_name or "",
            session_key=session_key or "",
            requested_session_id=requested_session_id,
            requested_model=requested_model,
        )
        with self._lock:
            existing = self._find_active_for_repo_locked(repo)
            if existing is not None:
                return None, existing
            self._jobs[job.job_id] = job
            self._prune_locked()

        thread = threading.Thread(
            target=self._worker,
            args=(job, runner),
            name=f"ghost-cursor-job-{job.job_id}",
            daemon=True,
        )
        job._thread = thread
        thread.start()
        logger.info(
            "Dispatched cursor run %s (repo=%s, resume=%s): %.80s",
            job.job_id, repo, requested_session_id or "-", task,
        )
        return job, None

    def _worker(self, job: CursorJob, runner: Callable[[CursorJob], Dict[str, Any]]) -> None:
        result: Dict[str, Any] = {}
        try:
            result = runner(job) or {}
        except Exception as exc:  # noqa: BLE001 — must never strand the job
            logger.exception("cursor job %s crashed", job.job_id)
            result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if not isinstance(result, dict):
                result = {"success": False, "error": "cursor run produced no result"}
            self._finalize(job, result)

    # -- lifecycle ----------------------------------------------------------------

    @staticmethod
    def _terminal_status(job: CursorJob, result: Dict[str, Any]) -> str:
        """Map a final result dict onto the job's terminal status.

        The result dict itself keeps the run's exact status vocabulary (a
        native cancel is result status "failed" — unchanged); the JOB status
        distinguishes "cancelled" so status queries and the completion
        message name the real terminal state.
        """
        st = str(result.get("status") or "")
        if st == "timeout":
            return "timeout"
        if st == "completed" and result.get("success"):
            return "completed"
        if job.cancelled or job.cancel_event.is_set():
            return "cancelled"
        return "failed"

    def _finalize(self, job: CursorJob, result: Dict[str, Any]) -> None:
        """Settle the job and, when delivery is still on, push the completion.

        The status write and the delivery-flag read share one lock hold so a
        concurrent ``mark_handled()`` either lands before (delivery is
        suppressed, the caller reports in-turn) or loses (delivery fires) —
        never neither.
        """
        status = self._terminal_status(job, result)
        with job._lock:
            if job.status in TERMINAL_STATUSES:
                return  # defensive: never double-finalize / double-deliver
            job.result = result
            job.status = status
            job.finished_at = time.time()
            deliver = job.deliver
        logger.info("Cursor job %s finished: %s (deliver=%s)", job.job_id, status, deliver)
        # Settle the persistent handle table so the handle stays resolvable
        # (and correctly non-running) across process restarts. Keyed by the
        # session NAME (v0.4 handle); name-less jobs fall back to the sid.
        handle_key = job.session_name or job.cursor_session_id
        if handle_key:
            _handles.record(
                handle_key,
                status=status,
                cursor_session_id=job.cursor_session_id or None,
                files_changed_count=result.get("files_changed_count"),
                duration_s=round(
                    (job.finished_at or time.time()) - job.created_at, 1
                ),
            )
        # Enqueue BEFORE signalling done: anyone who observes the job as
        # finished (waiters, tests, drains) must also find the completion
        # event already on the queue — no observe-then-miss window.
        if deliver:
            self._push_completion_event(job, result)
        job.done_event.set()
        # Unblock anyone still waiting for a session that will never come
        # (e.g. handshake failure before acp.session).
        job.session_event.set()

    def _push_completion_event(self, job: CursorJob, result: Dict[str, Any]) -> None:
        """Deliver the terminal result into the originating session.

        Rides the shared ``process_registry.completion_queue`` as a
        ``type="async_delegation"`` event — the same rail
        ``delegate_task(background=true)`` uses, already drained by the CLI
        process_loop and the gateway's async-delegation watcher and injected
        as a fresh turn. Fires for EVERY terminal state (unless the outcome
        was already reported in-turn via mark_handled); a failure to enqueue
        is logged loudly because it would mean a silently-lost result.
        """
        try:
            from tools.process_registry import process_registry
        except Exception as exc:  # pragma: no cover — core import failure
            logger.error(
                "Cursor job %s finished but process_registry import failed; "
                "completion delivery lost: %s", job.job_id, exc,
            )
            return

        payload_result = trim_result(result)
        evt = {
            "type": "async_delegation",
            "delegation_id": (
                job.session_name or job.cursor_session_id or job.job_id
            ),
            "session_key": job.session_key,
            "goal": f"cursor: {_clip(job.task, 200)}",
            "context": None,
            "toolsets": None,
            "role": "cursor",
            "model": job.model or job.requested_model or "cursor-agent",
            "status": job.status,
            "summary": self._completion_summary(job, result),
            "error": result.get("error"),
            "api_calls": 0,
            "duration_seconds": round((job.finished_at or time.time()) - job.created_at, 2),
            "dispatched_at": job.created_at,
            "completed_at": job.finished_at or time.time(),
            # Structured extras for programmatic consumers (formatters
            # ignore unknown keys). ``result`` carries the FULL final dict —
            # success/status/summary/files_changed/session_id/resumed — so
            # continuation survives the async boundary.
            "result": payload_result,
            "cursor_job_id": job.job_id,
            "cursor_session_id": job.cursor_session_id,
        }
        try:
            process_registry.completion_queue.put(evt)
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Cursor job %s: failed to enqueue completion event; "
                "result lost: %s", job.job_id, exc,
            )

    @staticmethod
    def _completion_summary(job: CursorJob, result: Dict[str, Any]) -> str:
        """The delivered completion message — the v0.4 plain-text format
        (labeled headers, prose, raw fenced diffs; never a JSON blob)."""
        name = job.session_name or job.cursor_session_id or job.job_id
        text = _render.completion_text(
            name=name,
            status=job.status,
            elapsed_s=(job.finished_at or time.time()) - job.created_at,
            repo=job.repo,
            summary=str(result.get("summary") or ""),
            files=result.get("files_changed") or [],
            error=str(result.get("error") or ""),
        )
        return (
            f"{text}\n\nfollow up in this session: "
            f"cursor_send_message('{name}', ...)"
        )

    # -- queries ---------------------------------------------------------------

    def get(self, job_id: str) -> Optional[CursorJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_by_name(self, session_name: str) -> Optional[CursorJob]:
        """The newest job dispatched under a session name (v0.4 handle)."""
        name = str(session_name or "").strip()
        if not name:
            return None
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        for job in jobs:
            if job.session_name == name:
                return job
        return None

    def get_by_session(self, session_id: str) -> Optional[CursorJob]:
        """The newest job for a cursor session handle (explicit lookup only).

        Matches the ESTABLISHED handle first; falls back to the REQUESTED
        resume handle so a just-dispatched continuation (``cursor_send`` /
        resume) is addressable in the window before its ``acp.session``
        event fires.
        """
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        for job in jobs:
            if job.cursor_session_id == sid:
                return job
        for job in jobs:
            if not job.cursor_session_id and job.requested_session_id == sid:
                return job
        return None

    def list_jobs(self) -> List[CursorJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at)

    def find_active_for_repo(self, repo: str) -> Optional[CursorJob]:
        with self._lock:
            return self._find_active_for_repo_locked(repo)

    def _find_active_for_repo_locked(self, repo: str) -> Optional[CursorJob]:
        for job in self._jobs.values():
            if job.status != "running" or job.repo != repo:
                continue
            # Safety valve: a worker that died without finalizing (should be
            # impossible — the worker finalizes in a finally) must not hold
            # the repo hostage forever.
            thread = job._thread
            if thread is not None and not thread.is_alive():
                job.status = "failed"
                job.run_error = job.run_error or "worker thread died without finalizing"
                job.finished_at = job.finished_at or time.time()
                job.done_event.set()
                job.session_event.set()
                continue
            return job
        return None

    # -- housekeeping ------------------------------------------------------------

    def _prune_locked(self) -> None:
        finished = [j for j in self._jobs.values() if j.status in TERMINAL_STATUSES]
        if len(finished) <= MAX_FINISHED_JOBS:
            return
        finished.sort(key=lambda j: j.finished_at or j.created_at)
        for job in finished[: len(finished) - MAX_FINISHED_JOBS]:
            self._jobs.pop(job.job_id, None)

    def _reset_for_tests(self) -> None:
        """Cancel + drop everything (test isolation only)."""
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            job.request_cancel()
        for job in jobs:
            thread = job._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)


# Module-level singleton (process-local, like the async-delegation records).
registry = CursorJobRegistry()
