"""Background job registry for ``cursor_edit`` — decouple cursor runs from turns.

Why this exists
---------------
The synchronous ``cursor_edit`` holds the Hermes turn open for the whole
cursor run (minutes). While the turn is open the only way the user can talk
to the agent is an out-of-band interrupt — which cancels the turn and kills
the running cursor session. This module lets a cursor run execute as a
tracked BACKGROUND JOB decoupled from the turn: the tool returns a job handle
immediately, chat stays free, progress accumulates in a readable rolling
buffer (mirroring ``tools/process_registry``'s output-buffer pattern), and
completion is delivered into the session as a NEW message when the job ends.

Every ``cursor_edit`` run — synchronous or background — executes on a worker
thread as a :class:`CursorJob`. The synchronous path simply *waits* on the
job (proxying the caller's interrupt flag to the job's cancel event), which
is what makes auto-promote-on-overrun a clean detach: promotion just stops
waiting; the run genuinely continues on its worker thread.

Completion delivery (reuses core infra — nothing reinvented)
------------------------------------------------------------
On EVERY terminal state (completed / failed / cancelled / timeout — never
silent), a job whose delivery flag is set pushes an event onto the shared
``tools.process_registry.process_registry.completion_queue`` with
``type="async_delegation"`` — the exact rail ``delegate_task(background=true)``
uses (see ``tools/async_delegation.py``). The CLI ``process_loop`` drain and
the gateway's ``_async_delegation_watcher`` already consume that queue while
the agent is idle and inject each event as a fresh turn, which keeps strict
message-role alternation legal and the prompt cache intact. The event's
``session_key`` (captured on the dispatching thread) routes it back to the
originating gateway session; an empty key means CLI.

The completion payload carries the FULL final result dict (success / status /
summary / files_changed / session_id / resumed) both as a structured
``result`` field and rendered into the human-readable ``summary`` block, so
multi-turn continuation (pass ``session_id`` back to ``cursor_edit``)
survives the async boundary.

Concurrency guard (G1)
----------------------
Two agents editing one working tree corrupts it. ``dispatch()`` atomically
rejects a new job when an active job already holds the same resolved repo,
returning the existing job so the caller can point at it.

State is process-local (like async delegation): jobs do not survive a
restart. The cursor session id is still persisted eagerly through
``session_registry`` so interject/auto-resume works across processes.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Rolling progress buffer cap — mirrors process_registry.MAX_OUTPUT_CHARS.
MAX_PROGRESS_CHARS = 200_000
# How many finished jobs to retain for cursor_status queries.
MAX_FINISHED_JOBS = 20

# Per-envelope trims for the rolling buffer (full-fidelity diffs live in the
# job's ``files`` aggregation, same caps as the sync result).
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
    """One cursor run — sync-waited, background, or promoted mid-run."""

    job_id: str
    task: str
    repo: str
    timeout: float
    hermes_session_id: str = ""
    session_key: str = ""
    requested_session_id: Optional[str] = None
    auto_resumed: bool = False
    background: bool = False          # dispatched with background=True
    detached: bool = False            # promoted from a sync wait (overrun)
    deliver: bool = False             # push a completion event at finalize
    live_progress: bool = False       # sync runs stream via the agent pcb
    status: str = "running"
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    cursor_session_id: str = ""
    resumed: bool = False
    model: str = ""
    # --- aggregation state (guarded by _lock) ---
    files: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    assistant_parts: List[str] = field(default_factory=list)
    reasoning_tail: str = ""
    progress_buffer: str = ""
    progress_events: int = 0
    emitted: int = 0
    run_error: Optional[str] = None
    timed_out: bool = False
    completed: bool = False
    cancelled: bool = False
    result: Optional[Dict[str, Any]] = None
    # --- control ---
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    done_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)

    # -- control -------------------------------------------------------------

    def emit_live(self) -> bool:
        """Whether progress should still stream through the caller's pcb.

        Background jobs never stream (their turn already ended); a promoted
        job stops streaming the moment it detaches from the waiting turn.
        """
        return not (self.background or self.detached)

    def request_cancel(self) -> None:
        """Ask the run to stop via cursor's native session/cancel path."""
        self.cancel_event.set()

    def detach(self) -> bool:
        """Promote a sync-waited job to background delivery.

        Returns False when the job already reached a terminal state (the
        caller should return the final result instead). Atomic with respect
        to finalize: whichever wins the lock decides who reports the result.
        """
        with self._lock:
            if self.status in TERMINAL_STATUSES:
                return False
            self.detached = True
            self.deliver = True
            return True

    # -- progress buffer -------------------------------------------------------

    def append_progress(self, envelope: Dict[str, Any]) -> None:
        """Append one canonical envelope to the rolling progress buffer.

        Mirrors the process-registry rolling-output pattern: bounded size,
        trimmed from the front at a line boundary. Full before/after file
        content is dropped (the diff carries the signal); diffs and shell
        output are clipped so one giant edit can't evict all history.
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
                "job_id": self.job_id,
                "status": self.status,
                "repo": self.repo,
                "task": _clip(self.task, 400),
                "background": bool(self.background or self.detached),
                "promoted": self.detached,
                "elapsed_s": round((self.finished_at or now) - self.created_at, 1),
                "cursor_session_id": self.cursor_session_id,
                "resumed": self.resumed,
                "auto_resumed": self.auto_resumed,
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
        timeout: float,
        hermes_session_id: str = "",
        session_key: str = "",
        requested_session_id: Optional[str] = None,
        auto_resumed: bool = False,
        background: bool = False,
        live_progress: bool = False,
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
            timeout=timeout,
            hermes_session_id=hermes_session_id or "",
            session_key=session_key or "",
            requested_session_id=requested_session_id,
            auto_resumed=auto_resumed,
            background=background,
            deliver=background,
            live_progress=live_progress,
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
            "Dispatched cursor job %s (background=%s, repo=%s): %.80s",
            job.job_id, background, repo, task,
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

        The result dict itself keeps the synchronous tool's exact status
        vocabulary (a native cancel is result status "failed" — unchanged);
        the JOB status distinguishes "cancelled" so status queries and the
        completion message name the real terminal state.
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
        """Settle the job and, when delivery is on, push the completion event.

        The status write and the delivery-flag read share one lock hold so a
        concurrent ``detach()`` either lands before (delivery fires) or loses
        (the sync waiter returns the final result itself) — never neither.
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
        # Enqueue BEFORE signalling done: anyone who observes the job as
        # finished (sync waiter, tests, drains) must also find the completion
        # event already on the queue — no observe-then-miss window.
        if deliver:
            self._push_completion_event(job, result)
        job.done_event.set()

    def _push_completion_event(self, job: CursorJob, result: Dict[str, Any]) -> None:
        """Deliver the terminal result into the originating session.

        Rides the shared ``process_registry.completion_queue`` as a
        ``type="async_delegation"`` event — the same rail
        ``delegate_task(background=true)`` uses, already drained by the CLI
        process_loop and the gateway's async-delegation watcher and injected
        as a fresh turn. Fires for EVERY terminal state; a failure to enqueue
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
            "delegation_id": job.job_id,
            "session_key": job.session_key,
            "goal": f"cursor_edit (background): {_clip(job.task, 200)}",
            "context": None,
            "toolsets": None,
            "role": "cursor",
            "model": job.model or "cursor-agent",
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
        """Human-readable completion block, distinct per terminal state."""
        lines = [
            f"Background cursor_edit job {job.job_id} finished — status: {job.status}.",
            f"Repo: {job.repo}",
        ]
        files = result.get("files_changed") or []
        if files:
            lines.append(f"Files changed ({len(files)}):")
            for f in files[:20]:
                if isinstance(f, dict):
                    lines.append(
                        f"  {f.get('status', 'M')} {f.get('path', '?')} "
                        f"(+{f.get('added', 0)}/-{f.get('removed', 0)})"
                    )
            if len(files) > 20:
                lines.append(f"  … and {len(files) - 20} more")
        sid = result.get("session_id") or job.cursor_session_id
        if sid:
            lines.append(
                f"Cursor session_id: {sid} — pass it back as `session_id` to "
                "cursor_edit to continue this cursor session with full prior context."
            )
        if result.get("error"):
            lines.append(f"Error: {result['error']}")
        lines.append("Final result:")
        lines.append(json.dumps(trim_result(result), ensure_ascii=False, default=str))
        return "\n".join(lines)

    # -- queries ---------------------------------------------------------------

    def get(self, job_id: str) -> Optional[CursorJob]:
        with self._lock:
            return self._jobs.get(job_id)

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
                continue
            return job
        return None

    def most_recent(self, hermes_session_id: Optional[str] = None) -> Optional[CursorJob]:
        """The job ``cursor_status`` should report when no job_id was given.

        Preference order: a running job for this Hermes session, any running
        job, the most recent job for this session, the most recent job.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        if not jobs:
            return None
        sid = str(hermes_session_id or "")
        newest = lambda seq: max(seq, key=lambda j: j.created_at)  # noqa: E731
        running = [j for j in jobs if j.status == "running"]
        if sid:
            mine_running = [j for j in running if j.hermes_session_id == sid]
            if mine_running:
                return newest(mine_running)
        if running:
            return newest(running)
        if sid:
            mine = [j for j in jobs if j.hermes_session_id == sid]
            if mine:
                return newest(mine)
        return newest(jobs)

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
