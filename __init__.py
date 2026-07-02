"""Ghost ⇄ Cursor delegation plugin — bundled, auto-loaded.

Registers two tools into the ``ghost_cursor`` toolset:

* ``cursor_edit`` — runs the Cursor agent over ACP (``cursor-agent acp``,
  JSON-RPC over stdio — see ``acp_runner.py``; the legacy ``--print``
  stdout-scraping runner is kept in ``runner.py`` as reference/fallback)
  inside a target repo, streams per-edit progress (reasoning fragments +
  full file diffs) through the calling agent's ``tool_progress_callback``,
  and returns a structured summary of the files changed. Because the call is
  an ordinary Hermes tool call inside a real session, the result persists in
  the session transcript and reloads for free.
* ``cursor_status`` — STRICTLY READ-ONLY progress view over cursor jobs
  (see "Background jobs" below). Never cancels or mutates a running job.

Background jobs (``background=true`` / auto-promote)
----------------------------------------------------
The synchronous path holds the Hermes turn open for the whole cursor run,
so a user message mid-run can only arrive as an interrupt — which kills the
run. ``cursor_edit(background=true)`` instead dispatches the run as a
tracked job (``jobs.py``) and returns immediately with a ``job_id``: the
turn ends, chat stays free, progress accumulates in a rolling buffer, and
the final result is delivered into the session as a NEW message on EVERY
terminal state (completed / failed / cancelled / timeout) via the shared
``process_registry.completion_queue`` — the same rail
``delegate_task(background=true)`` uses.

Auto-promote-on-overrun: a synchronous run that exceeds a soft threshold
(default 90s; ``plugins.ghost_cursor.promote_after_seconds`` in config.yaml,
0 disables) is promoted to a background job instead of continuing to block.
Implementation note: EVERY run executes on a worker thread (the sync path
just waits on the job and proxies the caller's interrupt flag to the job's
cancel event at the same 0.2s cadence the in-line loop used), so promotion
is a clean detach — the waiter stops waiting and returns a
``{"status": "promoted", "job_id": ...}`` notice while the run genuinely
continues; nothing is re-spawned.

Same-repo concurrency guard: a second ``cursor_edit`` while a job is active
on the same resolved repo is rejected with the existing ``job_id`` — two
cursor agents on one working tree corrupt it.

How live progress works (no core change)
----------------------------------------
A registry-dispatched tool handler is not handed the calling ``AIAgent``, but
``agent/tool_executor.py`` installs ``agent._touch_activity`` (a *bound
method* of the calling agent) as the thread-local activity callback
immediately before every tool dispatch — on both the sequential and the
concurrent (worker-thread) paths. ``_resolve_progress_callback()`` reads that
thread-local via ``tools.environments.base._get_activity_callback()`` and
walks ``__self__`` back to the live agent, then uses its
``tool_progress_callback``. Best-effort: when no callback is resolvable the
tool still runs and returns the diffs in its result, so the end state always
persists. Background jobs do not stream through the callback (their turn has
already ended) — the rolling progress buffer + ``cursor_status`` replace it.

Each progress emission is
``tool_progress_callback("reasoning.available", "cursor_edit", <json>, None)``
— the one event shape the api_server session-chat-stream forwards mid-turn as
``event: tool.progress`` with payload ``{message_id, tool_name: "cursor_edit",
delta: <json>}``. The ``delta`` is a canonical Threshold envelope (see
``events.py``): ``content`` / ``tool_use`` / ``tool_result`` / ``lifecycle``
/ ``file_diff``. Consumers key on ``tool_name == "cursor_edit"`` + the
envelope's ``source: "ghost"`` marker to extract file diffs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from plugins.ghost_cursor import acp_runner as _acp
from plugins.ghost_cursor import events as _events
from plugins.ghost_cursor import jobs as _jobs
from plugins.ghost_cursor import runner as _runner
from plugins.ghost_cursor import session_registry as _sessions

logger = logging.getLogger(__name__)

TOOL_NAME = "cursor_edit"
STATUS_TOOL_NAME = "cursor_status"
TOOLSET = "ghost_cursor"

# Env var naming is Threshold's (the Ghost frontend), not HERMES_* — it points
# at the workspace repo Ghost delegates coding tasks into.
REPO_ENV_VAR = "THRESHOLD_WORKSPACE_REPO"

# Cap for per-file diff text carried in the FINAL tool result (persisted to
# the session DB). Live file_diff progress envelopes carry more (see
# events.MAX_DIFF_CHARS); the persisted summary stays lean.
_RESULT_DIFF_CHARS = 20_000

# Auto-promote-on-overrun: a synchronous run blocking longer than this is
# detached to a background job. Override via config.yaml
# (plugins.ghost_cursor.promote_after_seconds); 0 disables promotion.
DEFAULT_PROMOTE_AFTER_S = 90.0

# Sync wait poll cadence — matches the interrupt-poll cadence the old
# in-line run_acp consumer loop had (_POLL_S), so cancel latency is unchanged.
_WAIT_POLL_S = 0.2

CURSOR_EDIT_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Delegate a coding task to the Cursor agent, which edits files inside "
        "a repository and streams live per-edit diffs while it works. "
        "STRONGLY PREFER this tool for any request to write, modify, refactor, "
        "debug, or fix code in a project or repo — delegate instead of editing "
        "project files yourself. Provide the full coding instruction as `task`; "
        "the Cursor agent reads the repo, makes the edits, and this tool "
        "returns a summary of every file changed (+added/-removed lines and "
        "diffs). The result also includes a `session_id` — for iterative or "
        "follow-up work on the same task (refine, fix, review feedback), pass "
        "it back as `session_id` to continue that cursor session with full "
        "prior context instead of starting fresh. If a cursor run in the same "
        "repo was recently interrupted (stopped mid-flight), the next call "
        "auto-continues that session — so a mid-run nudge just works; passing "
        "`session_id` explicitly overrides this. Set `background=true` for "
        "anything likely to exceed ~60-90 seconds (multi-file changes, "
        "refactors, 'implement X', anything with a lint/build/test step): the "
        "tool then returns a job_id immediately, the conversation stays free, "
        "progress is readable via cursor_status, and the final result is "
        "delivered automatically as a new message when the run finishes. "
        "Leave background=false for quick, bounded single-file edits."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "The coding instruction to carry out, e.g. 'add a "
                    "multiply function to calc.py with tests'. Be specific; "
                    "include file names and acceptance criteria when known."
                ),
            },
            "repo": {
                "type": "string",
                "description": (
                    "Absolute path to the repository to work in. Optional — "
                    "defaults to the configured workspace repo."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional. Pass the `session_id` returned by a previous "
                    "cursor_edit call to CONTINUE that cursor session with "
                    "full prior context (multi-turn/iterative work — refine, "
                    "fix, follow up). Omit to start a fresh session."
                ),
            },
            "background": {
                "type": "boolean",
                "description": (
                    "Optional (default false). Run as a background job: "
                    "returns immediately with a job_id while the Cursor agent "
                    "keeps working; the user can keep chatting, progress is "
                    "readable via cursor_status(job_id), and the final result "
                    "is delivered automatically when the run ends. Use for "
                    "any task expected to take more than ~60-90 seconds."
                ),
            },
        },
        "required": ["task"],
    },
}

CURSOR_STATUS_SCHEMA = {
    "name": STATUS_TOOL_NAME,
    "description": (
        "Check the live progress of a cursor_edit job WITHOUT interrupting "
        "it — strictly read-only; it never cancels, pauses, or otherwise "
        "affects the running cursor session, so it is always safe to call "
        "while a job runs. Returns the job status (running / completed / "
        "failed / cancelled / timeout), files touched so far with per-edit "
        "diffs, the latest reasoning, the cursor session_id (for later "
        "continuation), and elapsed time. Use it when the user asks how a "
        "delegated coding task is going. `job_id` is optional — when "
        "omitted, reports the most recent/active cursor job for this "
        "session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": (
                    "Optional. The job_id returned by "
                    "cursor_edit(background=true) or by an auto-promoted "
                    "run. Omit to report the most recent/active job."
                ),
            },
        },
        "required": [],
    },
}


def _default_repo() -> Optional[str]:
    """Resolve the default workspace repo: env var, then terminal cwd."""
    for candidate in (
        os.getenv(REPO_ENV_VAR),
        os.getenv("TERMINAL_CWD"),
        os.getcwd(),
    ):
        if candidate and os.path.isdir(candidate):
            return candidate
    return None


def check_cursor_edit_available() -> bool:
    """Tool gate: cursor-agent binary present + a workspace repo resolvable."""
    try:
        return _runner.cursor_agent_available() and _default_repo() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Calling-agent resolution (progress callback + Hermes session id)
# ---------------------------------------------------------------------------

def _resolve_hermes_session_id() -> Optional[str]:
    """Locate the calling Hermes session id (best-effort, may be None).

    Same resolution path as :func:`_resolve_progress_callback` — the
    thread-local activity callback's ``__self__`` is the live ``AIAgent``,
    whose ``session_id`` keys the interject-resume registry. When no agent
    is reachable, return None and the registry degrades to a no-op.
    """
    try:
        from tools.environments.base import _get_activity_callback

        cb = _get_activity_callback()
        agent = getattr(cb, "__self__", None)
        sid = getattr(agent, "session_id", None)
        if sid:
            return str(sid)
    except Exception:
        pass
    try:
        from hermes_cli.plugins import get_plugin_manager

        cli = getattr(get_plugin_manager(), "_cli_ref", None)
        agent = getattr(cli, "agent", None)
        sid = getattr(agent, "session_id", None)
        if sid:
            return str(sid)
    except Exception:
        pass
    return None


def _resolve_progress_callback() -> Optional[Callable]:
    """Locate the calling agent's ``tool_progress_callback`` (best-effort).

    1. Thread-local activity callback — ``agent.tool_executor`` sets
       ``agent._touch_activity`` (a bound method) on the dispatching thread
       right before every tool call, so ``__self__`` is the calling agent.
       Works in the gateway/api_server (quiet mode), CLI, TUI, and inside
       concurrent tool batches (each worker thread gets its own binding).
    2. Interactive-CLI fallback: the PluginManager's ``_cli_ref.agent``.
    """
    try:
        from tools.environments.base import _get_activity_callback

        cb = _get_activity_callback()
        agent = getattr(cb, "__self__", None)
        pcb = getattr(agent, "tool_progress_callback", None)
        if callable(pcb):
            return pcb
    except Exception:
        pass
    try:
        from hermes_cli.plugins import get_plugin_manager

        cli = getattr(get_plugin_manager(), "_cli_ref", None)
        agent = getattr(cli, "agent", None)
        pcb = getattr(agent, "tool_progress_callback", None)
        if callable(pcb):
            return pcb
    except Exception:
        pass
    return None


def _resolve_session_key() -> str:
    """Gateway session key for routing async completions (empty = CLI).

    Captured on the dispatching thread BEFORE the worker starts, mirroring
    ``delegate_tool``'s background dispatch — the worker thread won't carry
    the contextvars.
    """
    try:
        from tools.approval import get_current_session_key

        return get_current_session_key(default="") or ""
    except Exception:
        return ""


def _promote_after_seconds() -> float:
    """Auto-promote threshold: config override or the 90s default. 0 disables."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), "plugins", "ghost_cursor", "promote_after_seconds")
        if val is not None:
            return max(float(val), 0.0)
    except Exception:
        pass
    return DEFAULT_PROMOTE_AFTER_S


def _emit_progress(pcb: Optional[Callable], envelope: Dict[str, Any]) -> bool:
    """Send one canonical envelope through the agent's progress callback.

    Emitted as a ``reasoning.available`` event with ``tool_name="cursor_edit"``
    — the shape the api_server session-chat-stream forwards mid-turn as
    ``event: tool.progress`` ``{message_id, tool_name, delta}``. Surfaces that
    don't understand it (messaging platforms) drop it silently. Never raises.
    """
    if pcb is None:
        return False
    try:
        pcb(
            "reasoning.available",
            TOOL_NAME,
            json.dumps(envelope, ensure_ascii=False, default=str),
            None,
        )
        return True
    except Exception as exc:
        logger.debug("cursor_edit progress emit failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Run execution (worker-thread body — shared by sync, background, promoted)
# ---------------------------------------------------------------------------

def _fold_envelope(job: "_jobs.CursorJob", pcb: Optional[Callable], envelope: Dict[str, Any]) -> None:
    """Fold one canonical envelope into the job's aggregation state.

    Also streams it through the caller's progress callback while the job is
    still attached to a live turn (sync wait, pre-promotion) and appends a
    compact line to the rolling progress buffer that ``cursor_status`` reads.
    """
    if pcb is not None and job.emit_live() and _emit_progress(pcb, envelope):
        with job._lock:
            job.emitted += 1
    job.append_progress(envelope)

    kind = envelope.get("kind")
    with job._lock:
        if kind == "file_diff":
            path = str(envelope.get("path") or "")
            entry = job.files.setdefault(
                path,
                {"path": path, "added": 0, "removed": 0},
            )
            entry["added"] += int(envelope.get("added") or 0)
            entry["removed"] += int(envelope.get("removed") or 0)
            entry["status"] = envelope.get("status")
            diff = str(envelope.get("diff") or "")
            if diff:
                entry["diff"] = diff[:_RESULT_DIFF_CHARS]
        elif kind == "content":
            delta = str(envelope.get("delta") or "")
            if delta:
                job.assistant_parts.append(delta)
        elif kind == "lifecycle":
            event = envelope.get("event")
            if event == "run.completed":
                job.completed = True
            elif event == "run.failed":
                job.run_error = str(envelope.get("error") or "run failed")
                job.timed_out = job.timed_out or bool(envelope.get("timeout"))
                job.cancelled = job.cancelled or bool(envelope.get("cancelled"))
            elif event == "reasoning":
                text = str(envelope.get("text") or "")
                if text:
                    job.reasoning_tail = (job.reasoning_tail + text)[-4000:]


def _execute_cursor_run(job: "_jobs.CursorJob", pcb: Optional[Callable]) -> Dict[str, Any]:
    """Run cursor-agent for ``job`` and build the final result dict.

    This is the exact synchronous run body, relocated onto the job worker
    thread: same fold logic, same git fallback, same registry settle, same
    result shape — byte-compatible with the pre-background tool result.
    Cancellation is the job's cancel event (the sync waiter proxies the
    caller's interrupt flag into it; background jobs are only cancelled
    explicitly), which triggers cursor's native ``session/cancel``.
    """
    started = time.monotonic()
    workdir = job.repo

    # Pre-run git snapshot: fuels the fallback that populates files_changed
    # when cursor edits through paths that emit no ACP diff content (e.g.
    # shell commands).
    git_before = _acp.git_status_snapshot(workdir)

    normalizer = _events.AcpNormalizer()
    try:
        for key, obj in _acp.run_acp(
            job.task,
            workdir,
            timeout=float(job.timeout),
            cancel_check=job.cancel_event.is_set,
            session_id=job.requested_session_id,
        ):
            if key == "acp.session":
                # The sessionId rides back to the caller for multi-turn
                # continuation; the normalizer only folds it into lifecycle.
                sid = str(obj.get("sessionId") or "")
                with job._lock:
                    job.cursor_session_id = sid
                    job.resumed = bool(obj.get("resumed"))
                    job.model = str(obj.get("model") or "")
                # Eager persistence: record the instant ACP establishes the
                # session (before the prompt streams), so an interrupt that
                # discards this tool result still leaves the id resumable.
                _sessions.record(
                    job.hermes_session_id or None, sid, workdir, "running"
                )
            for envelope in normalizer.normalize(key, obj):
                _fold_envelope(job, pcb, envelope)
    except _acp.AcpError as exc:
        # Hard ACP failure (handshake) — actionable error, no silent regress.
        return {"success": False, "error": str(exc)}
    except _runner.HarnessError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("cursor_edit failed")
        with job._lock:
            job.run_error = f"{type(exc).__name__}: {exc}"

    # Git fallback: edits the ACP stream carried no diff for (shell-driven
    # writes, kill-before-diff) still land in files_changed + progress.
    try:
        with job._lock:
            known_paths = set(job.files)
        for fb in _acp.git_fallback_diffs(workdir, git_before):
            if fb["path"] not in known_paths:
                _fold_envelope(job, pcb, _events.file_diff(**fb))
    except Exception:
        logger.debug("cursor_edit git fallback failed", exc_info=True)

    duration_ms = int((time.monotonic() - started) * 1000)
    with job._lock:
        files_changed = sorted(
            (dict(f) for f in job.files.values()), key=lambda f: f["path"]
        )
        prose = "".join(job.assistant_parts).strip()
        completed = job.completed
        run_error = job.run_error
        timed_out = job.timed_out
        result_session_id = job.cursor_session_id
        resumed = job.resumed
        emitted = job.emitted
    success = completed and run_error is None

    # Settle the registry: a clean completion is NOT continuable (a later
    # unrelated task must start fresh); anything else — cancel, timeout,
    # interrupt, mid-run crash — is, so the next cursor_edit can resume it.
    if result_session_id:
        _sessions.record(
            job.hermes_session_id or None,
            result_session_id,
            workdir,
            "completed" if success else "cancelled",
        )

    result: Dict[str, Any] = {
        "success": success,
        "status": "timeout" if timed_out else ("completed" if completed and not run_error else "failed"),
        "repo": workdir,
        "summary": prose or ("(no assistant summary)" if not run_error else ""),
        "files_changed": files_changed,
        "files_changed_count": len(files_changed),
        "duration_ms": duration_ms,
        "live_progress": job.live_progress,
        "progress_events_emitted": emitted,
        "session_id": result_session_id,
        "resumed": resumed,
        "auto_resumed": job.auto_resumed,
    }
    if run_error:
        result["error"] = run_error
        if files_changed:
            result["partial"] = True
    return result


# ---------------------------------------------------------------------------
# Synchronous wait (with interrupt proxy + auto-promote-on-overrun)
# ---------------------------------------------------------------------------

def _wait_for_sync_result(job: "_jobs.CursorJob", promote_after: float) -> str:
    """Block on a job for the synchronous path.

    Mirrors the pre-background behavior exactly: the caller's per-thread
    interrupt flag is polled every 0.2s and proxied to the job's cancel
    event, which triggers cursor's native ``session/cancel`` — so /stop and
    mid-turn nudges cancel a sync run just like before.

    Auto-promote: once the wait exceeds ``promote_after`` seconds (and no
    cancel is pending), the job is detached — delivery turns on, streaming
    stops, and a "promoted" notice is returned so the turn can end while the
    run continues. ``promote_after <= 0`` disables promotion.
    """
    start = time.monotonic()
    promote = promote_after is not None and promote_after > 0
    while True:
        if job.done_event.wait(timeout=_WAIT_POLL_S):
            return json.dumps(job.result, ensure_ascii=False)
        if not job.cancel_event.is_set():
            try:
                from tools.interrupt import is_interrupted

                if is_interrupted():
                    job.request_cancel()
            except Exception:
                pass
        if (
            promote
            and not job.cancel_event.is_set()
            and (time.monotonic() - start) >= promote_after
        ):
            if job.detach():
                logger.info(
                    "cursor_edit job %s exceeded the %.0fs sync soft limit — "
                    "promoted to background", job.job_id, promote_after,
                )
                return json.dumps({
                    "success": True,
                    "status": "promoted",
                    "background": True,
                    "job_id": job.job_id,
                    "repo": job.repo,
                    "cursor_session_id": job.cursor_session_id or "",
                    "elapsed_s": round(time.monotonic() - start, 1),
                    "note": (
                        "This cursor run exceeded the synchronous soft limit "
                        "and was promoted to a background job. It is STILL "
                        "RUNNING — do not restart it. Tell the user work "
                        "continues in the background; check progress with "
                        f"cursor_status(job_id='{job.job_id}') (read-only, "
                        "safe while running). The final result — including "
                        "files_changed and the cursor session_id for "
                        "follow-ups — will be delivered automatically as a "
                        "new message when it finishes."
                    ),
                }, ensure_ascii=False)
            # detach lost the race with finalize — the next wait() returns
            # the final result immediately.


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def cursor_edit(
    task: str,
    repo: Optional[str] = None,
    timeout: float = _runner.DEFAULT_TIMEOUT_S,
    progress_callback: Optional[Callable] = None,
    session_id: Optional[str] = None,
    background: bool = False,
    promote_after: Optional[float] = None,
    **_kwargs: Any,
) -> str:
    """Run cursor-agent on ``task``; sync-blocking or as a background job.

    ``progress_callback`` overrides the auto-resolved agent callback (used by
    tests and direct harness invocation). ``session_id`` continues a prior
    cursor session (multi-turn) via ACP ``session/load``; the result's
    ``session_id`` / ``resumed`` fields report the session actually used.

    ``background=False`` (default): blocks until the run finishes and returns
    the full result — the original behavior — except that a run overrunning
    the soft threshold (``promote_after``, default from config/90s) is
    promoted to a background job instead of blocking indefinitely.

    ``background=True``: dispatches the run as a background job and returns
    ``{job_id, status: "running", repo, ...}`` immediately; the final result
    is delivered into the session as a new message on every terminal state,
    and ``cursor_status`` reads live progress meanwhile.

    Interject-via-resume: when no explicit ``session_id`` is passed and the
    session registry holds a recent interrupted cursor run for the calling
    Hermes session + repo, that session is auto-resumed (``auto_resumed`` is
    True in the result). The active cursor session id is recorded eagerly the
    moment ACP establishes it — before the prompt streams — so a stop that
    discards this tool result still leaves the id behind for the next call.
    """
    if not str(task or "").strip():
        return json.dumps({"success": False, "error": "task is required"})

    target_repo = (repo or "").strip() or _default_repo()
    if not target_repo:
        return json.dumps({
            "success": False,
            "error": (
                "No workspace repo resolvable. Pass `repo` or set "
                f"{REPO_ENV_VAR}."
            ),
        })
    try:
        workdir = _runner.resolve_repo(target_repo)
    except _runner.HarnessError as exc:
        return json.dumps({"success": False, "error": str(exc)})

    pcb = progress_callback if progress_callback is not None else _resolve_progress_callback()

    # Interject-via-resume: when the caller passed no explicit session_id,
    # auto-resume a recently interrupted cursor run for this Hermes
    # session + repo (see session_registry). Explicit session_id wins.
    hermes_sid = _resolve_hermes_session_id()
    auto_resumed = False
    if not str(session_id or "").strip():
        prior = _sessions.get_recent(hermes_sid, str(workdir))
        if prior:
            session_id = prior
            auto_resumed = True

    job, existing = _jobs.registry.dispatch(
        runner=lambda j: _execute_cursor_run(j, pcb),
        task=str(task),
        repo=str(workdir),
        timeout=float(timeout),
        hermes_session_id=hermes_sid or "",
        session_key=_resolve_session_key(),
        requested_session_id=(str(session_id).strip() or None) if session_id else None,
        auto_resumed=auto_resumed,
        background=bool(background),
        live_progress=(pcb is not None and not background),
    )
    if job is None:
        # Same-repo concurrency guard: two cursor agents on one working tree
        # corrupt it. Point at the job that already holds the repo.
        return json.dumps({
            "success": False,
            "status": "rejected",
            "reason": "a cursor_edit job is already running on this repo",
            "job_id": existing.job_id if existing else "",
            "repo": str(workdir),
            "note": (
                "Do not start a second cursor run on the same working tree. "
                "Check progress with cursor_status(job_id=...) — its result "
                "will be delivered automatically when it finishes."
            ),
        }, ensure_ascii=False)

    if background:
        return json.dumps({
            "success": True,
            "status": "running",
            "background": True,
            "job_id": job.job_id,
            "repo": str(workdir),
            "cursor_session_id": job.cursor_session_id or "",
            "auto_resumed": auto_resumed,
            "note": (
                "Cursor is working in the background — this turn can end "
                "now. Tell the user the task was dispatched and that you "
                "will report the outcome when it completes (the final "
                "result, including files_changed and the cursor session_id, "
                "is delivered automatically as a new message on ANY outcome "
                "— success, failure, timeout, or cancellation). While it "
                f"runs, cursor_status(job_id='{job.job_id}') gives read-only "
                "live progress (files touched, diffs, latest reasoning, "
                "cursor session_id) without disturbing the run."
            ),
        }, ensure_ascii=False)

    return _wait_for_sync_result(
        job,
        promote_after if promote_after is not None else _promote_after_seconds(),
    )


def cursor_status(job_id: Optional[str] = None, **_kwargs: Any) -> str:
    """Read-only progress snapshot for a cursor job (see CURSOR_STATUS_SCHEMA).

    STRICTLY READ-ONLY: only copies job state under its lock. It never sends
    ``session/cancel``, never touches the cancel event, never joins or
    signals the worker — polling a running job cannot affect it.
    """
    jid = str(job_id or "").strip()
    if jid:
        job = _jobs.registry.get(jid)
        if job is None:
            known = [j.job_id for j in _jobs.registry.list_jobs()]
            return json.dumps({
                "success": False,
                "error": f"no cursor_edit job with id '{jid}'",
                "known_jobs": known[-10:],
            }, ensure_ascii=False)
    else:
        job = _jobs.registry.most_recent(_resolve_hermes_session_id())
        if job is None:
            return json.dumps({
                "success": False,
                "error": (
                    "no cursor_edit jobs found — nothing has been dispatched "
                    "in this process yet"
                ),
            }, ensure_ascii=False)
    return json.dumps(
        {"success": True, **job.snapshot()}, ensure_ascii=False, default=str
    )


def _handle_cursor_edit(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_edit(
        task=args.get("task", ""),
        repo=args.get("repo"),
        session_id=args.get("session_id"),
        background=bool(args.get("background", False)),
    )


def _handle_cursor_status(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_status(job_id=args.get("job_id"))


def register(ctx) -> None:
    """Register the cursor_edit + cursor_status tools. Called once by the loader."""
    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=CURSOR_EDIT_SCHEMA,
        handler=_handle_cursor_edit,
        check_fn=check_cursor_edit_available,
        emoji="🖱️",
    )
    ctx.register_tool(
        name=STATUS_TOOL_NAME,
        toolset=TOOLSET,
        schema=CURSOR_STATUS_SCHEMA,
        handler=_handle_cursor_status,
        check_fn=check_cursor_edit_available,
        emoji="🛰️",
    )
