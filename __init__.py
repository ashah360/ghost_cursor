"""Ghost ⇄ Cursor delegation plugin — bundled, auto-loaded.

v0.3: explicit session-handle model. Four tools in the ``ghost_cursor``
toolset, mirroring the terminal/process split (start / send / status / stop):

* ``cursor_start(task, repo?, model?, session_id?)`` — dispatch a cursor run
  (``cursor-agent acp``, JSON-RPC over stdio — see ``acp_runner.py``) and
  return the cursor ``session_id`` (THE handle) + ``status: "running"`` as
  soon as the ACP session is established; the run continues in the
  background, progress accumulating in a rolling buffer. Passing
  ``session_id`` continues that prior cursor session (ACP ``session/load``)
  instead of starting fresh. ``model`` overrides the cursor-agent model.
* ``cursor_send(session_id, message)`` — steer / follow up on a run. Honest
  semantics: cursor's ACP has no queue, so send CANCELS the running prompt
  (native ``session/cancel``) and re-prompts the same session with
  ``message`` — full prior context, but "interrupt + re-prompt", not
  "append". Works mid-flight and after the run settled.
* ``cursor_status(session_id)`` — STRICTLY READ-ONLY snapshot: status,
  summary/transcript so far, files changed with diffs, elapsed. Never
  cancels or mutates the run (tested property).
* ``cursor_stop(session_id)`` — graceful native ``session/cancel`` (SIGKILL
  only on hang — the existing acp_runner cancel path); returns the final
  status + partial files_changed.

Handle model (what replaced the v0.2 auto-resume heuristic)
-----------------------------------------------------------
The single handle is the cursor ``session_id`` from ACP ``session/new``,
surfaced at the ``acp.session`` event. ``cursor_start`` returns it; every
other tool takes it back. The in-process job table (``jobs.py``) keys live
state on it; a tiny JSON file (``handles.py``, ``<HERMES_HOME>/state/``)
persists handle → repo/status/model so a handle minted on turn T still
resolves on turn T+1 and across restarts. The old ``session_registry.py``
repo+10-minute-timestamp auto-resume heuristic is DELETED — interrupted-run
recovery is explicit: the caller passes the handle it was given.

``cursor_edit`` is REMOVED, not aliased. A blocking convenience wrapper
would have needed the v0.2 sync-wait + interrupt-proxy + auto-promote
machinery back, which is most of the complexity this rewrite deletes; the
explicit start → (deliver-on-complete) flow covers the quick-edit case with
one extra message and zero hidden state. Breaking change, greenlit.

Completion delivery
-------------------
Background runs deliver a completion message into the session on ALL
terminal states (completed / failed / cancelled / timeout) via the shared
``process_registry.completion_queue`` — the same rail
``delegate_task(background=true)`` uses. The payload carries the final
result (files_changed, session_id) so continuation works. The one
exception: when ``cursor_stop`` / ``cursor_send`` settle the run in-turn,
the outcome is in that tool result and the duplicate delivery is
suppressed (``CursorJob.mark_handled``).

Model override
--------------
``cursor_start(model=...)`` > ``plugins.ghost_cursor.model`` in config.yaml
> cursor-agent's own default (unchanged from v0.2, which never passed a
model over ACP). Instrumented 2026-07-02 (cursor-agent 2026.07.01-777f564):
ACP ``session/new`` ignores model params and ``session/set_model`` rejects
plain ids, so the override is threaded as ``--model`` on the
``cursor-agent acp`` invocation — see ``acp_runner._AcpClient._spawn``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from . import acp_runner as _acp
from . import eventlog as _eventlog
from . import events as _events
from . import handles as _handles
from . import jobs as _jobs
from . import runner as _runner

logger = logging.getLogger(__name__)

START_TOOL_NAME = "cursor_start"
SEND_TOOL_NAME = "cursor_send"
STATUS_TOOL_NAME = "cursor_status"
STOP_TOOL_NAME = "cursor_stop"
TOOLSET = "ghost_cursor"

# Env var naming is Threshold's (the Ghost frontend), not HERMES_* — it points
# at the workspace repo Ghost delegates coding tasks into.
REPO_ENV_VAR = "THRESHOLD_WORKSPACE_REPO"

# Cap for per-file diff text carried in the FINAL result dict (persisted to
# the session DB via the completion message). Live file_diff progress
# envelopes carry more (see events.MAX_DIFF_CHARS).
_RESULT_DIFF_CHARS = 20_000

# How long cursor_start/cursor_send block waiting for the ACP session to be
# established (the handle). Bounded by the acp_runner handshake timeouts
# (initialize 30s + session/load 60s) plus slack; a healthy run yields the
# handle in a few seconds.
_HANDLE_WAIT_S = 150.0

# How long cursor_send/cursor_stop wait for a cancelled run to settle.
# Native session/cancel resolves ~immediately; the ceiling covers the
# acp_runner CANCEL_GRACE_S (15s) + TERM_GRACE_S (10s) SIGKILL escalation.
_INTERRUPT_WAIT_S = 40.0

_HANDLE_DOC = (
    "The `session_id` handle returned by cursor_start (also included in "
    "every cursor completion message)."
)

CURSOR_START_SCHEMA = {
    "name": START_TOOL_NAME,
    "description": (
        "Dispatch a coding task to the Cursor agent, which edits files "
        "inside a repository. STRONGLY PREFER this tool for any request to "
        "write, modify, refactor, debug, or fix code in a project or repo — "
        "delegate instead of editing project files yourself. Returns "
        "immediately with a `session_id` handle and status \"running\" "
        "while the Cursor agent keeps working in the background; the "
        "conversation stays free. Track progress with "
        "cursor_status(session_id) (read-only), steer or follow up with "
        "cursor_send(session_id, message), stop with "
        "cursor_stop(session_id). The final result — files changed with "
        "diffs and the session_id — is delivered automatically as a new "
        "message on ANY outcome (success, failure, timeout, cancellation). "
        "Pass a previous run's `session_id` to start a new task that "
        "continues that cursor session with its full prior context. Only "
        "one run may be active per repo at a time (runs on different repos "
        "proceed in parallel)."
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
            "model": {
                "type": "string",
                "description": (
                    "Optional cursor-agent model override for this run "
                    "(e.g. 'composer-2.5', 'gpt-5.3-codex'). Omit to use "
                    "the configured/default model."
                ),
            },
            "session_id": {
                "type": "string",
                "description": (
                    "Optional. A prior run's session_id handle — the new "
                    "task CONTINUES that cursor session with full prior "
                    "context instead of starting fresh. Omit for a new "
                    "session."
                ),
            },
        },
        "required": ["task"],
    },
}

CURSOR_SEND_SCHEMA = {
    "name": SEND_TOOL_NAME,
    "description": (
        "Send a follow-up or steering message to a cursor run. HONEST "
        "SEMANTICS: cursor has no message queue — if the run is still "
        "working, this INTERRUPTS its current prompt (native cancel) and "
        "re-prompts the same session with your message, so the agent "
        "continues with full prior context but abandons whatever step it "
        "was mid-way through; it is \"interrupt + re-prompt with context\", "
        "not \"append to a running turn\". On an already-finished run it "
        "simply continues the session (classic follow-up: refine, fix, "
        "review feedback). Returns the (possibly new) session_id and "
        "status \"running\"; the result is delivered automatically when "
        "the re-prompted run finishes."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": _HANDLE_DOC},
            "message": {
                "type": "string",
                "description": (
                    "The steering instruction or follow-up task, e.g. "
                    "'also add tests for the divide function'."
                ),
            },
        },
        "required": ["session_id", "message"],
    },
}

CURSOR_STATUS_SCHEMA = {
    "name": STATUS_TOOL_NAME,
    "description": (
        "Check a cursor run WITHOUT affecting it — strictly read-only; it "
        "never cancels, pauses, or otherwise touches the running session, "
        "so it is always safe to call mid-run. Returns the run status "
        "(running / completed / failed / cancelled / timeout), the "
        "assistant summary and files changed so far (with per-edit diffs), "
        "the latest reasoning, the cursor session_id, and elapsed time — "
        "plus the path to the full persisted event log (JSONL) and its "
        "total event count. Pass offset/limit to page through the "
        "persisted event history (the default response only carries "
        "compact tails). Use it when the user asks how a delegated coding "
        "task is going."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": _HANDLE_DOC},
            "offset": {
                "type": "integer",
                "description": (
                    "Optional. Page through the persisted event log: return "
                    "events with seq >= offset (0-based). Providing offset "
                    "and/or limit adds an `events_page` field to the result."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Optional. Max events per page (default 50, max 500). "
                    "Used with `offset` to page through the event log."
                ),
            },
            "scope": {
                "type": "string",
                "enum": ["session", "all"],
                "description": (
                    "Optional. Which handles the `known_sessions` hint in "
                    "an unknown-handle error may list: 'session' (default) "
                    "= only runs dispatched from this Hermes session, "
                    "'all' = every recorded run. Explicit session_id "
                    "lookups always resolve regardless of scope."
                ),
            },
        },
        "required": ["session_id"],
    },
}

CURSOR_STOP_SCHEMA = {
    "name": STOP_TOOL_NAME,
    "description": (
        "Stop a running cursor run gracefully (cursor's native cancel; "
        "the process is only force-killed if it hangs). Returns the final "
        "status and any partial files_changed. Idempotent: calling it on "
        "an already-finished run just reports that run's final state."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": _HANDLE_DOC},
        },
        "required": ["session_id"],
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


def check_cursor_available() -> bool:
    """Tool gate: cursor-agent binary present + a workspace repo resolvable."""
    try:
        return _runner.cursor_agent_available() and _default_repo() is not None
    except Exception:
        return False


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


def _known_sessions(scope: str = "session") -> list:
    """Scoped handle listing for actionable error messages.

    Default scope "session": only handles dispatched from the CURRENT
    Hermes session — with many concurrent sessions sharing the table, one
    session's error hint must not leak (or suggest steering) another
    session's runs. ``scope="all"`` opts into the global view.
    """
    return _handles.known_handles(
        scope=scope, session_key=_resolve_session_key()
    )


def _configured_model() -> Optional[str]:
    """The ``plugins.ghost_cursor.model`` config.yaml value, if set."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), "plugins", "ghost_cursor", "model")
        if val:
            return str(val).strip() or None
    except Exception:
        pass
    return None


def _resolve_model(explicit: Optional[str]) -> Optional[str]:
    """Model precedence: explicit param > config > None (cursor's default)."""
    val = (str(explicit).strip() or None) if explicit else None
    return val or _configured_model()


# ---------------------------------------------------------------------------
# Run execution (worker-thread body)
# ---------------------------------------------------------------------------

def _fold_envelope(job: "_jobs.CursorJob", envelope: Dict[str, Any]) -> None:
    """Fold one canonical envelope into the job's aggregation state and
    append a compact line to the rolling buffer ``cursor_status`` reads."""
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


def _execute_cursor_run(job: "_jobs.CursorJob") -> Dict[str, Any]:
    """Run cursor-agent for ``job`` and build the final result dict.

    Runs on the job worker thread. Cancellation is the job's cancel event
    (set by cursor_stop / cursor_send), which triggers cursor's native
    ``session/cancel``. The cursor session id (the handle) is persisted to
    the handle table the instant ACP establishes it, so it is resolvable
    on later turns and across restarts.
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
            model=job.requested_model,
        ):
            if key == "acp.session":
                sid = str(obj.get("sessionId") or "")
                with job._lock:
                    job.cursor_session_id = sid
                    job.resumed = bool(obj.get("resumed"))
                    job.model = str(obj.get("model") or "")
                # Persist the handle the moment it exists — the caller's
                # turn (blocked in _await_handle) and any later turn key
                # everything on it.
                _handles.record(
                    sid,
                    repo=workdir,
                    status="running",
                    task=job.task[:200],
                    model=job.model or job.requested_model,
                    # Scope tag: which Hermes session dispatched this run
                    # ("" = CLI). Captured on the dispatching thread at
                    # dispatch time (worker threads don't carry the
                    # contextvar) and threaded through the job.
                    session_key=job.session_key,
                )
                job.session_event.set()
            for envelope in normalizer.normalize(key, obj):
                _fold_envelope(job, envelope)
    except _acp.AcpError as exc:
        # Hard ACP failure (handshake) — actionable error, no silent regress.
        return {"success": False, "error": str(exc)}
    except _runner.HarnessError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("cursor run failed")
        with job._lock:
            job.run_error = f"{type(exc).__name__}: {exc}"

    # Git fallback: edits the ACP stream carried no diff for (shell-driven
    # writes, kill-before-diff) still land in files_changed + progress.
    try:
        with job._lock:
            known_paths = set(job.files)
        for fb in _acp.git_fallback_diffs(workdir, git_before):
            if fb["path"] not in known_paths:
                _fold_envelope(job, _events.file_diff(**fb))
    except Exception:
        logger.debug("cursor run git fallback failed", exc_info=True)

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
        model = job.model

    success = completed and run_error is None
    result: Dict[str, Any] = {
        "success": success,
        "status": "timeout" if timed_out else ("completed" if completed and not run_error else "failed"),
        "repo": workdir,
        "summary": prose or ("(no assistant summary)" if not run_error else ""),
        "files_changed": files_changed,
        "files_changed_count": len(files_changed),
        "duration_ms": duration_ms,
        "session_id": result_session_id,
        "resumed": resumed,
        "model": model,
    }
    if run_error:
        result["error"] = run_error
        if files_changed:
            result["partial"] = True
    return result


# ---------------------------------------------------------------------------
# Dispatch plumbing shared by cursor_start and cursor_send
# ---------------------------------------------------------------------------

def _dispatch_run(
    task: str,
    workdir: str,
    session_id: Optional[str],
    model: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    """Dispatch a run and block only until the handle exists.

    Returns the tool-result dict: the running-handle shape once the ACP
    session is established, the final result if the run terminated before
    a session existed (handshake failure) or terminated very fast, or the
    same-repo rejection.
    """
    job, existing = _jobs.registry.dispatch(
        runner=_execute_cursor_run,
        task=str(task),
        repo=str(workdir),
        timeout=float(timeout),
        session_key=_resolve_session_key(),
        requested_session_id=(str(session_id).strip() or None) if session_id else None,
        requested_model=model,
    )
    if job is None:
        # Same-repo concurrency guard: two cursor agents on one working tree
        # corrupt it. Surface the existing run's handle so the caller can
        # steer/inspect it instead. (Different repos run in parallel.)
        existing.session_event.wait(timeout=10)  # give a usable handle
        return {
            "success": False,
            "status": "rejected",
            "reason": "a cursor run is already active on this repo",
            "session_id": existing.cursor_session_id,
            "repo": str(workdir),
            "note": (
                "Do not start a second cursor run on the same working tree. "
                "Use cursor_status(session_id) to check the active run, "
                "cursor_send(session_id, message) to steer it, or "
                "cursor_stop(session_id) to stop it first."
            ),
        }
    return _await_handle(job)


def _await_handle(job: "_jobs.CursorJob") -> Dict[str, Any]:
    """Block until the run has a session handle (or died trying).

    Exactly-once outcome reporting: the job is dispatched with delivery
    DISARMED. If the run reaches a terminal state before this function
    returns the running shape (handshake failure, ultra-fast completion),
    the final result is returned right here, in-turn, and nothing is ever
    enqueued. Only once we are about to hand the running handle back do we
    arm delivery — and arming is atomic with finalize, so a race falls back
    to the in-turn report.
    """
    job.session_event.wait(timeout=_HANDLE_WAIT_S)

    if job.done_event.is_set() and job.result is not None:
        # Terminal before the handle was handed out (delivery never armed,
        # nothing enqueued): this tool result IS the report.
        return {**_jobs.trim_result(job.result), "status": job.status}

    if not job.session_event.is_set():
        # No session and not terminal — cursor-agent is wedged pre-handshake.
        # Cancel it (the worker's kill path takes over) and report.
        job.request_cancel()
        return {
            "success": False,
            "status": "failed",
            "error": (
                f"cursor-agent did not establish an ACP session within "
                f"{int(_HANDLE_WAIT_S)}s — cancelled the attempt"
            ),
            "repo": job.repo,
        }

    if not job.arm_delivery():
        # Finalize won the race while we held an un-armed job: report the
        # final result in-turn (nothing was enqueued).
        return {**_jobs.trim_result(job.result or {}), "status": job.status}

    return {
        "success": True,
        "status": "running",
        "session_id": job.cursor_session_id,
        "resumed": job.resumed,
        "model": job.model or (job.requested_model or ""),
        "repo": job.repo,
        "note": (
            "Cursor is working in the background — this turn can end now. "
            "Tell the user the task was dispatched; the final result "
            "(files changed with diffs, session_id) is delivered "
            "automatically as a new message on ANY outcome. Meanwhile "
            f"cursor_status(session_id='{job.cursor_session_id}') gives "
            "read-only progress, and cursor_send can steer the run."
        ),
    }


def _settle_job(job: "_jobs.CursorJob", wait_s: float = _INTERRUPT_WAIT_S) -> bool:
    """Cancel ``job`` (native session/cancel) and wait for it to settle.

    Delivery is suppressed first (mark_handled) because the caller reports
    the outcome in its own tool result. Returns True when the job reached a
    terminal state within the wait.
    """
    job.mark_handled()
    job.request_cancel()
    return job.done_event.wait(timeout=wait_s)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def cursor_start(
    task: str,
    repo: Optional[str] = None,
    model: Optional[str] = None,
    session_id: Optional[str] = None,
    timeout: float = _runner.DEFAULT_TIMEOUT_S,
    **_kwargs: Any,
) -> str:
    """Dispatch a cursor run; return the session handle immediately.

    ``session_id`` continues a prior cursor session via ACP ``session/load``
    (the run's ``resumed`` field reports whether the load actually
    succeeded — an expired/unknown handle falls back to a fresh session).
    ``model`` threads through to the cursor-agent invocation (see module
    docstring for the instrumentation).
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

    resume_id = (str(session_id).strip() or None) if session_id else None
    if resume_id:
        active = _jobs.registry.get_by_session(resume_id)
        if active is not None and active.status == "running":
            return json.dumps({
                "success": False,
                "status": "rejected",
                "reason": "that cursor session is still running",
                "session_id": resume_id,
                "note": (
                    "Use cursor_send(session_id, message) to steer the "
                    "active run, or cursor_stop(session_id) first."
                ),
            }, ensure_ascii=False)

    return json.dumps(
        _dispatch_run(
            task=str(task),
            workdir=str(workdir),
            session_id=resume_id,
            model=_resolve_model(model),
            timeout=timeout,
        ),
        ensure_ascii=False,
        default=str,
    )


def cursor_send(
    session_id: str,
    message: str,
    timeout: float = _runner.DEFAULT_TIMEOUT_S,
    **_kwargs: Any,
) -> str:
    """Steer / follow up on a cursor run (cancel-and-re-prompt semantics).

    Mid-flight: suppress the old run's delivery, cancel it natively, wait
    for it to settle, then re-prompt the SAME session (``session/load``)
    with ``message``. Already settled: just re-prompt the session. Unknown
    handle: if the handle table knows the repo, re-prompt anyway (ACP falls
    back to a fresh session when the id expired server-side — graceful,
    reported via ``resumed: false``); otherwise a clean error.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return json.dumps({"success": False, "error": "session_id is required"})
    if not str(message or "").strip():
        return json.dumps({"success": False, "error": "message is required"})

    job = _jobs.registry.get_by_session(sid)
    interrupted = False
    partial: Dict[str, Any] = {}

    if job is not None and job.status == "running":
        interrupted = True
        if not _settle_job(job):
            return json.dumps({
                "success": False,
                "status": "running",
                "session_id": sid,
                "error": (
                    "the running cursor prompt did not settle after a "
                    f"native cancel within {int(_INTERRUPT_WAIT_S)}s — "
                    "not re-prompting a possibly-live session; retry, or "
                    "use cursor_stop"
                ),
            }, ensure_ascii=False)
        if job.result:
            partial = {
                "interrupted_files_changed_count":
                    job.result.get("files_changed_count", 0),
            }
        repo = job.repo
        model = job.requested_model
    elif job is not None:
        repo = job.repo
        model = job.requested_model
    else:
        entry = _handles.get(sid)
        if not entry or not entry.get("repo"):
            return json.dumps({
                "success": False,
                "error": (
                    f"unknown cursor session '{sid}' — no active or "
                    "recorded run has that handle"
                ),
                "known_sessions": _known_sessions(),
                "note": "Start a new run with cursor_start.",
            }, ensure_ascii=False)
        repo = str(entry["repo"])
        model = entry.get("model")
        if not os.path.isdir(repo):
            return json.dumps({
                "success": False,
                "error": (
                    f"the repo recorded for session '{sid}' no longer "
                    f"exists: {repo}"
                ),
            }, ensure_ascii=False)

    result = _dispatch_run(
        task=str(message),
        workdir=repo,
        session_id=sid,
        model=_resolve_model(model),
        timeout=timeout,
    )
    if interrupted:
        result["interrupted_previous_prompt"] = True
        result.update(partial)
        if isinstance(result.get("note"), str):
            result["note"] = (
                "The previous prompt was interrupted (cursor has no queue "
                "— send cancels and re-prompts with full session context). "
                + result["note"]
            )
    return json.dumps(result, ensure_ascii=False, default=str)


def _event_log_extras(
    sid: str, offset: Optional[Any], limit: Optional[Any]
) -> Dict[str, Any]:
    """Additive ``event_log`` (+ ``events_page`` when paging) fields.

    ``event_log`` = {path, total_events} whenever the session has a
    persisted JSONL log; ``events_page`` = one page over it when the caller
    passed offset and/or limit. Both are additive — the compact-tail shape
    is unchanged.
    """
    extras: Dict[str, Any] = {}
    stats = _eventlog.stats(sid)
    if stats is not None:
        extras["event_log"] = stats
    if offset is not None or limit is not None:
        page = _eventlog.read_page(
            sid,
            offset=offset if offset is not None else 0,
            limit=limit if limit is not None else _eventlog.DEFAULT_PAGE_LIMIT,
        )
        extras["events_page"] = page if page is not None else {
            "events": [],
            "total_events": 0,
            "note": "no persisted event log for this session",
        }
    return extras


def cursor_status(
    session_id: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    scope: str = "session",
    **_kwargs: Any,
) -> str:
    """Read-only snapshot for a cursor run (see CURSOR_STATUS_SCHEMA).

    STRICTLY READ-ONLY: only copies job state under its lock. It never
    sends ``session/cancel``, never touches the cancel event, never joins
    or signals the worker — polling a running run cannot affect it.

    ``offset``/``limit`` page through the persisted JSONL event log
    (additive ``events_page`` field). ``scope`` only affects the
    ``known_sessions`` hint on an unknown handle; the explicit lookup
    itself always crosses sessions.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return json.dumps({"success": False, "error": "session_id is required"})

    job = _jobs.registry.get_by_session(sid)
    if job is not None:
        snap = job.snapshot()
        if not snap.get("session_id"):
            # Session not yet established for a just-dispatched resume.
            snap["session_id"] = sid
            snap["cursor_session_id"] = sid
        snap.update(_event_log_extras(str(snap.get("session_id") or sid), offset, limit))
        return json.dumps({"success": True, **snap}, ensure_ascii=False, default=str)

    entry = _handles.get(sid)
    if entry is not None:
        # Known handle, but no live job in this process (e.g. restart).
        # The JSONL event log persists, so history stays pageable here too.
        status = str(entry.get("status") or "unknown")
        if status == "running":
            # A dead process can't have left a live run behind.
            status = "unknown (recorded as running by a previous process)"
        return json.dumps({
            "success": True,
            "session_id": sid,
            "status": status,
            "repo": entry.get("repo", ""),
            "task": entry.get("task", ""),
            "model": entry.get("model", ""),
            **_event_log_extras(sid, offset, limit),
            "note": (
                "This handle is not tracked live in the current process; "
                "showing its persisted record. Pass it to cursor_send or "
                "cursor_start(session_id=...) to continue the session."
            ),
        }, ensure_ascii=False, default=str)

    return json.dumps({
        "success": False,
        "error": f"unknown cursor session '{sid}'",
        "known_sessions": _known_sessions(scope),
    }, ensure_ascii=False)


def cursor_stop(session_id: str, **_kwargs: Any) -> str:
    """Stop a cursor run gracefully; report final status + partial files.

    Graceful path: cursor's native ``session/cancel`` (the prompt resolves
    with stopReason "cancelled" ~immediately); the acp_runner escalates to
    a process-group SIGKILL only if cursor hangs past its grace window.
    Idempotent on finished runs.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return json.dumps({"success": False, "error": "session_id is required"})

    job = _jobs.registry.get_by_session(sid)
    if job is None:
        entry = _handles.get(sid)
        if entry is not None:
            status = str(entry.get("status") or "unknown")
            return json.dumps({
                "success": True,
                "session_id": sid,
                "status": status if status != "running" else "not running (stale record)",
                "note": "No live run with this handle in the current process — nothing to stop.",
            }, ensure_ascii=False)
        return json.dumps({
            "success": False,
            "error": f"unknown cursor session '{sid}'",
            "known_sessions": _known_sessions(),
        }, ensure_ascii=False)

    if job.status in _jobs.TERMINAL_STATUSES:
        result = _jobs.trim_result(job.result or {})
        return json.dumps({
            "success": True,
            "session_id": job.cursor_session_id or sid,
            "status": job.status,
            "already_finished": True,
            "files_changed": result.get("files_changed", []),
            "files_changed_count": result.get("files_changed_count", 0),
            "result": result,
        }, ensure_ascii=False, default=str)

    settled = _settle_job(job)
    if not settled:
        return json.dumps({
            "success": False,
            "session_id": sid,
            "status": "running",
            "error": (
                f"the run did not settle within {int(_INTERRUPT_WAIT_S)}s "
                "of a native cancel — it may be hung; the kill escalation "
                "is in progress, check again with cursor_status"
            ),
        }, ensure_ascii=False)

    result = _jobs.trim_result(job.result or {})
    return json.dumps({
        "success": True,
        "session_id": job.cursor_session_id or sid,
        "status": job.status,
        "stopped": True,
        "files_changed": result.get("files_changed", []),
        "files_changed_count": result.get("files_changed_count", 0),
        "result": result,
        "note": (
            "Run stopped. The session handle remains continuable — pass it "
            "to cursor_send or cursor_start(session_id=...) to pick the "
            "work back up with full context."
        ),
    }, ensure_ascii=False, default=str)


def _handle_cursor_start(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_start(
        task=args.get("task", ""),
        repo=args.get("repo"),
        model=args.get("model"),
        session_id=args.get("session_id"),
    )


def _handle_cursor_send(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_send(
        session_id=args.get("session_id", ""),
        message=args.get("message", ""),
    )


def _handle_cursor_status(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_status(
        session_id=args.get("session_id", ""),
        offset=args.get("offset"),
        limit=args.get("limit"),
        scope=args.get("scope", "session"),
    )


def _handle_cursor_stop(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_stop(session_id=args.get("session_id", ""))


def register(ctx) -> None:
    """Register the 4 cursor tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in (
        (START_TOOL_NAME, CURSOR_START_SCHEMA, _handle_cursor_start, "🖱️"),
        (SEND_TOOL_NAME, CURSOR_SEND_SCHEMA, _handle_cursor_send, "📨"),
        (STATUS_TOOL_NAME, CURSOR_STATUS_SCHEMA, _handle_cursor_status, "🛰️"),
        (STOP_TOOL_NAME, CURSOR_STOP_SCHEMA, _handle_cursor_stop, "🛑"),
    ):
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            check_fn=check_cursor_available,
            emoji=emoji,
        )
