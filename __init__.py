"""Ghost ⇄ Cursor delegation plugin — bundled, auto-loaded.

v0.4: explicit named sessions + plain-text tool output. Six tools in the
``ghost_cursor`` toolset:

* ``cursor_create_session(repo?, model?)`` — mint a named session handle
  (adjective-adjective-noun slug, e.g. ``playful-space-bunny``). LAZY: it
  dispatches nothing; the ACP process spawns on the first message.
* ``cursor_send_message(session, message)`` — ALL work goes through here.
  The first message on a fresh session is the task; later messages are
  follow-ups (or interrupt + re-prompt when the run is live — the ack says
  which). Cursor works in the background; the terminal result is delivered
  automatically on every outcome.
* ``cursor_status(session)`` — strictly read-only snapshot (never cancels).
* ``cursor_stop(session)`` — graceful native ``session/cancel``.
* ``cursor_events(session, offset=-1, limit=10, kind?)`` — dedicated pager
  over the per-session JSONL event log. Defaults = the last 10 events;
  negative offsets index from the end python-style; ``offset>=0`` pages
  forward by seq; ``kind`` filters (reasoning / file_diff / tool_result /
  tool_use / content / lifecycle); limit clamps at 500.
* ``cursor_list(scope='session'|'all')`` — TSV listing of session handles,
  scoped to the current Hermes session by default.

(v0.3's ``cursor_start``/``cursor_send`` are gone — create + send replaced
them outright; pre-launch, no deprecation shim.)

Handle model (v0.4)
-------------------
THE handle is the session name minted by ``cursor_create_session``
(collision-checked against the persistent handle table, ``names.py``). The
cursor ACP session UUID is recorded on the handle entry as an alias
(``handles.resolve``), so UUIDs from older runs / completion payloads still
resolve everywhere a name is accepted. The handle table (``handles.py``)
persists name → repo/status/model/cursor_session_id across restarts; the
in-process job table (``jobs.py``) keys live state by name; the JSONL event
log (``eventlog.py``) is also keyed by name, so one named session = one log
across resumes.

Output contract (v0.4)
----------------------
Tool returns are PLAIN TEXT rendered by pure f-string templates
(``render.py``, formats per /tmp/gc-v04-formats.md): labeled header lines,
plain english, raw fenced ```diff blocks (never JSON-escaped), TSV for
cursor_list. No ``success`` booleans; the ``status:`` line carries state.
Reasoning/thinking text never appears in status output; full event content
lives behind ``cursor_events`` (2 KB inline clip per event, ~20 KB response
cap, JSONL log path for everything else).

Completion delivery
-------------------
Background runs deliver a completion message into the session on ALL
terminal states via the shared ``process_registry.completion_queue`` — the
same rail ``delegate_task(background=true)`` uses. When ``cursor_stop`` /
``cursor_send_message`` settle the run in-turn, the outcome is in that tool
result and the duplicate delivery is suppressed (``CursorJob.mark_handled``).

Timeouts are INACTIVITY-based (``inactivity_timeout_s``: silence kills,
activity resets the clock) with an optional ``max_wall_s`` hard ceiling —
see ``acp_runner``. Model override precedence: explicit param >
``plugins.ghost_cursor.model`` in config.yaml > cursor-agent's default,
threaded as ``--model`` on the ``cursor-agent acp`` invocation.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from . import acp_runner as _acp
from . import eventlog as _eventlog
from . import events as _events
from . import handles as _handles
from . import jobs as _jobs
from . import names as _names
from . import render as _render
from . import runner as _runner

logger = logging.getLogger(__name__)

CREATE_TOOL_NAME = "cursor_create_session"
SEND_TOOL_NAME = "cursor_send_message"
STATUS_TOOL_NAME = "cursor_status"
STOP_TOOL_NAME = "cursor_stop"
EVENTS_TOOL_NAME = "cursor_events"
LIST_TOOL_NAME = "cursor_list"
TOOLSET = "ghost_cursor"

# Env var naming is Threshold's (the Ghost frontend), not HERMES_* — it points
# at the workspace repo Ghost delegates coding tasks into.
REPO_ENV_VAR = "THRESHOLD_WORKSPACE_REPO"

# Cap for per-file diff text carried in the FINAL result dict (persisted to
# the session DB via the completion message). Live file_diff progress
# envelopes carry more (see events.MAX_DIFF_CHARS).
_RESULT_DIFF_CHARS = 20_000

# How long dispatching tools block waiting for the ACP session to be
# established. Bounded by the acp_runner handshake timeouts (initialize 30s
# + session/load 60s) plus slack; a healthy run yields it in seconds.
_HANDLE_WAIT_S = 150.0

# How long cursor_send_message/cursor_stop wait for a cancelled run to
# settle. Native session/cancel resolves ~immediately; the ceiling covers
# the acp_runner CANCEL_GRACE_S (15s) + TERM_GRACE_S (10s) escalation.
_INTERRUPT_WAIT_S = 40.0

_SESSION_DOC = (
    "The session handle: the name returned by cursor_create_session (e.g. "
    "'playful-space-bunny'). A cursor session UUID from an older run also "
    "resolves as an alias."
)

# Shared watchdog params for the dispatching tools. Timeouts are
# inactivity-based: streamed progress keeps a run alive indefinitely unless
# the optional wall ceiling is set.
_TIMEOUT_PROPERTIES = {
    "inactivity_timeout_s": {
        "type": "number",
        "description": (
            "Optional. Abort the run only after this many seconds of "
            "SILENCE (no ACP events from cursor). Any streamed activity — "
            "reasoning, tool calls, content — resets the clock, so a long "
            "run that keeps making progress is never killed by this limit. "
            "0 disables the inactivity watchdog. Default: "
            "plugins.ghost_cursor.inactivity_timeout_s in config.yaml, "
            "else 600."
        ),
    },
    "max_wall_s": {
        "type": "number",
        "description": (
            "Optional hard ceiling on TOTAL run time in seconds — a "
            "safety net for runaway runs that keep streaming without "
            "finishing. 0 disables it. Default: "
            "plugins.ghost_cursor.max_wall_s in config.yaml, else 0 "
            "(disabled)."
        ),
    },
}

CURSOR_CREATE_SCHEMA = {
    "name": CREATE_TOOL_NAME,
    "description": (
        "Create a named Cursor session for delegating coding work into a "
        "repository. Returns a session name handle (e.g. "
        "'playful-space-bunny') and dispatches NOTHING — the cursor agent "
        "spawns lazily on the first cursor_send_message. Use this, then "
        "send the task as a message. Only one run may be active per repo "
        "at a time (different repos proceed in parallel)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
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
                    "Optional cursor-agent model override for this session "
                    "(e.g. 'composer-2.5', 'gpt-5.3-codex'). Omit to use "
                    "the configured/default model."
                ),
            },
        },
        "required": [],
    },
}

CURSOR_SEND_SCHEMA = {
    "name": SEND_TOOL_NAME,
    "description": (
        "Send work to a Cursor session. STRONGLY PREFER this for any "
        "request to write, modify, refactor, debug, or fix code in a repo "
        "— delegate instead of editing project files yourself. The FIRST "
        "message on a fresh session is the task; later messages are "
        "follow-ups with full prior context. HONEST SEMANTICS: cursor has "
        "no message queue — if the run is still working, this INTERRUPTS "
        "its current prompt (native cancel) and re-prompts the same "
        "session with your message (the ack says so). Returns immediately "
        "while cursor works in the background; the final result — files "
        "changed with diffs — is delivered automatically as a new message "
        "on ANY outcome. Track with cursor_status(session), page history "
        "with cursor_events(session), stop with cursor_stop(session)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DOC},
            "message": {
                "type": "string",
                "description": (
                    "The coding task or follow-up instruction, e.g. 'add a "
                    "multiply function to calc.py with tests'. Be specific; "
                    "include file names and acceptance criteria when known."
                ),
            },
            **_TIMEOUT_PROPERTIES,
        },
        "required": ["session", "message"],
    },
}

CURSOR_STATUS_SCHEMA = {
    "name": STATUS_TOOL_NAME,
    "description": (
        "Check a cursor session WITHOUT affecting it — strictly read-only; "
        "it never cancels, pauses, or otherwise touches the running "
        "session, so it is always safe to call mid-run. Returns the status "
        "(running / completed / failed / cancelled / timeout), what it is "
        "working on, files changed so far (paths + line counts, no diffs), "
        "recent activity, elapsed and last-activity times, and the event "
        "log location. Full history and diffs: cursor_events(session)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DOC},
            "scope": {
                "type": "string",
                "enum": ["session", "all"],
                "description": (
                    "Optional. Which sessions an unknown-handle error may "
                    "list: 'session' (default) = only this Hermes "
                    "session's, 'all' = every recorded session. Explicit "
                    "handle lookups always resolve regardless of scope."
                ),
            },
        },
        "required": ["session"],
    },
}

CURSOR_STOP_SCHEMA = {
    "name": STOP_TOOL_NAME,
    "description": (
        "Stop a running cursor session gracefully (cursor's native cancel; "
        "the process is only force-killed if it hangs). Reports the final "
        "status and any partial work. Idempotent on finished runs. The "
        "session stays continuable via cursor_send_message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DOC},
        },
        "required": ["session"],
    },
}

CURSOR_EVENTS_SCHEMA = {
    "name": EVENTS_TOOL_NAME,
    "description": (
        "Page through a cursor session's persisted event history (the "
        "JSONL log): reasoning, tool calls/results, file diffs, streamed "
        "content. Defaults (offset=-1, limit=10) return the LAST 10 "
        "events. Negative offset indexes from the end python-style "
        "(offset=-11, limit=10 = the previous page); offset>=0 pages "
        "forward from that event seq. `kind` filters to one event kind. "
        "Per-event inline content clips at 2KB and the whole response at "
        "~20KB; the JSONL log keeps full fidelity."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DOC},
            "offset": {
                "type": "integer",
                "description": (
                    "Event window position. Negative = from the end "
                    "(-1 = tail, the default); >=0 = forward from that "
                    "event seq."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max events in the window (default 10, max 500).",
            },
            "kind": {
                "type": "string",
                "description": (
                    "Optional filter: reasoning, file_diff, tool_result, "
                    "tool_use, content, or lifecycle."
                ),
            },
        },
        "required": ["session"],
    },
}

CURSOR_LIST_SCHEMA = {
    "name": LIST_TOOL_NAME,
    "description": (
        "List cursor session handles as a TSV table (session, repo, "
        "status, elapsed, files, last_activity). Default scope 'session' "
        "shows only sessions created from this Hermes session; "
        "scope='all' shows every recorded session."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["session", "all"],
                "description": (
                    "'session' (default) = this Hermes session's handles; "
                    "'all' = everything recorded."
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


def _configured_timeout(key: str) -> Optional[float]:
    """A numeric ``plugins.ghost_cursor.<key>`` config.yaml value, if set."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        val = cfg_get(read_raw_config(), "plugins", "ghost_cursor", key)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return None


def _resolve_inactivity_timeout(explicit: Optional[float]) -> float:
    """Inactivity threshold precedence: explicit param >
    ``plugins.ghost_cursor.inactivity_timeout_s`` in config >
    ``DEFAULT_INACTIVITY_TIMEOUT_S`` (600s of silence). 0 disables."""
    if explicit is not None:
        return float(explicit)
    configured = _configured_timeout("inactivity_timeout_s")
    return (
        configured
        if configured is not None
        else _acp.DEFAULT_INACTIVITY_TIMEOUT_S
    )


def _resolve_max_wall(explicit: Optional[float]) -> float:
    """Wall-ceiling precedence: explicit param >
    ``plugins.ghost_cursor.max_wall_s`` in config > ``DEFAULT_MAX_WALL_S``
    (0 = disabled)."""
    if explicit is not None:
        return float(explicit)
    configured = _configured_timeout("max_wall_s")
    return configured if configured is not None else _acp.DEFAULT_MAX_WALL_S


# ---------------------------------------------------------------------------
# Session-handle resolution (name-or-UUID → canonical name)
# ---------------------------------------------------------------------------

def _resolve_session(identifier: str) -> Optional[str]:
    """The canonical session name for a name-or-UUID identifier, or None."""
    ident = str(identifier or "").strip()
    if not ident:
        return None
    name = _handles.resolve(ident)
    if name:
        return name
    # A live job whose handle record hasn't landed yet (dispatch races).
    job = _jobs.registry.get_by_session(ident)
    if job is not None:
        return job.session_name or ident
    return None


def _live_job(name: str, entry: Optional[Dict[str, Any]]) -> Optional["_jobs.CursorJob"]:
    """The live job for a session, by name first, then by UUID alias."""
    job = _jobs.registry.get_by_name(name)
    if job is not None:
        return job
    sid = str((entry or {}).get("cursor_session_id") or "") or name
    return _jobs.registry.get_by_session(sid)


def _resume_sid(name: str, entry: Dict[str, Any]) -> Optional[str]:
    """The cursor ACP session id to resume, or None for a fresh session.

    A session that has run before carries its ``cursor_session_id`` alias.
    A pre-v0.4 entry keyed by the raw cursor UUID (no alias field, but a
    recorded run status) resumes by its own key. A freshly created (lazy,
    never-run) session starts a new ACP session.
    """
    sid = str(entry.get("cursor_session_id") or "").strip()
    if sid:
        return sid
    if str(entry.get("status") or "created") not in ("", "created"):
        return name
    return None


def _log_key(name: str, entry: Optional[Dict[str, Any]]) -> str:
    """The event-log key for a session: its name, else its legacy sid log."""
    if _eventlog.stats(name) is not None:
        return name
    sid = str((entry or {}).get("cursor_session_id") or "")
    if sid and _eventlog.stats(sid) is not None:
        return sid
    return name


# ---------------------------------------------------------------------------
# Run execution (worker-thread body)
# ---------------------------------------------------------------------------

def _fold_envelope(job: "_jobs.CursorJob", envelope: Dict[str, Any]) -> None:
    """Fold one canonical envelope into the job's aggregation state and
    append a compact line to the rolling buffer status views read."""
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
            job.segment_open = False
        elif kind == "content":
            delta = str(envelope.get("delta") or "")
            if delta:
                # Contiguous deltas extend the open block; anything else
                # (tool activity, reasoning) closed it, so a fresh block
                # starts here. summary_text() prefers the final block.
                if job.segment_open and job.assistant_segments:
                    job.assistant_segments[-1].append(delta)
                else:
                    job.assistant_segments.append([delta])
                    job.segment_open = True
        elif kind in ("tool_use", "tool_result"):
            job.segment_open = False
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
                job.segment_open = False


def _execute_cursor_run(job: "_jobs.CursorJob") -> Dict[str, Any]:
    """Run cursor-agent for ``job`` and build the final result dict.

    Runs on the job worker thread. Cancellation is the job's cancel event
    (set by cursor_stop / cursor_send_message), which triggers cursor's
    native ``session/cancel``. The cursor session id is persisted onto the
    session's handle entry (as the UUID alias) the instant ACP establishes
    it.
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
            inactivity_timeout_s=float(job.inactivity_timeout_s),
            max_wall_s=float(job.max_wall_s),
            cancel_check=job.cancel_event.is_set,
            session_id=job.requested_session_id,
            model=job.requested_model,
        ):
            job.last_event_at = time.time()
            if key == "acp.session":
                sid = str(obj.get("sessionId") or "")
                with job._lock:
                    job.cursor_session_id = sid
                    job.resumed = bool(obj.get("resumed"))
                    job.model = str(obj.get("model") or "")
                # Persist the handle the moment the ACP session exists —
                # keyed by the session NAME, with the UUID as an alias.
                _handles.record(
                    job.session_name or sid,
                    repo=workdir,
                    status="running",
                    task=job.task[:200],
                    model=job.model or job.requested_model,
                    session_key=job.session_key,
                    cursor_session_id=sid,
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
        # The FINAL content block (the wrap-up message) when the turn ended
        # on one; joined blocks otherwise — see CursorJob.summary_text.
        prose = job.summary_text()
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
        "session": job.session_name,
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
# Dispatch plumbing behind cursor_send_message
# ---------------------------------------------------------------------------

def _dispatch_run(
    task: str,
    workdir: str,
    session_name: str,
    session_id: Optional[str],
    model: Optional[str],
    inactivity_timeout_s: float,
    max_wall_s: float,
) -> Dict[str, Any]:
    """Dispatch a run and block only until the handle exists.

    Returns the internal result dict: the running shape once the ACP
    session is established, the final result if the run terminated before
    a session existed (handshake failure) or terminated very fast, or the
    same-repo rejection.
    """
    job, existing = _jobs.registry.dispatch(
        runner=_execute_cursor_run,
        task=str(task),
        repo=str(workdir),
        inactivity_timeout_s=float(inactivity_timeout_s),
        max_wall_s=float(max_wall_s),
        session_name=session_name,
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
            "session": existing.session_name,
            "session_id": existing.cursor_session_id,
            "repo": str(workdir),
        }
    return _await_handle(job)


def _await_handle(job: "_jobs.CursorJob") -> Dict[str, Any]:
    """Block until the run has a session handle (or died trying).

    Exactly-once outcome reporting: the job is dispatched with delivery
    DISARMED. If the run reaches a terminal state before this function
    returns the running shape (handshake failure, ultra-fast completion),
    the final result is returned right here, in-turn, and nothing is ever
    enqueued. Only once we are about to hand the running shape back do we
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
            "session": job.session_name,
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
        "session": job.session_name,
        "session_id": job.cursor_session_id,
        "resumed": job.resumed,
        "model": job.model or (job.requested_model or ""),
        "repo": job.repo,
        "note": (
            "Cursor is working in the background — this turn can end now. "
            "The final result is delivered automatically as a new message "
            f"on ANY outcome. cursor_status('{job.session_name}') gives "
            "read-only progress; cursor_send_message can steer the run."
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


def _send_to_session(
    name: str,
    entry: Dict[str, Any],
    message: str,
    inactivity_timeout_s: Optional[float],
    max_wall_s: Optional[float],
) -> Dict[str, Any]:
    """The shared send path: interrupt a live run if needed, then dispatch.

    Returns the internal result dict (running / terminal / rejected /
    error shapes) with ``interrupted_previous_prompt`` set when a live
    prompt was cancelled first.
    """
    repo = str(entry.get("repo") or "")
    if not repo:
        return {
            "success": False,
            "status": "failed",
            "error": f"session '{name}' has no repo recorded",
        }
    if not os.path.isdir(repo):
        return {
            "success": False,
            "status": "failed",
            "error": (
                f"the repo recorded for session '{name}' no longer "
                f"exists: {repo}"
            ),
        }

    job = _live_job(name, entry)
    interrupted = False
    if job is not None and job.status == "running":
        interrupted = True
        if not _settle_job(job):
            return {
                "success": False,
                "status": "running",
                "session": name,
                "error": (
                    "the running cursor prompt did not settle after a "
                    f"native cancel within {int(_INTERRUPT_WAIT_S)}s — "
                    "not re-prompting a possibly-live session; retry, or "
                    "use cursor_stop"
                ),
            }

    resume = (
        (job.cursor_session_id or None)
        if job is not None and job.cursor_session_id
        else _resume_sid(name, entry)
    )
    result = _dispatch_run(
        task=str(message),
        workdir=repo,
        session_name=name,
        session_id=resume,
        model=_resolve_model(entry.get("model")),
        inactivity_timeout_s=_resolve_inactivity_timeout(inactivity_timeout_s),
        max_wall_s=_resolve_max_wall(max_wall_s),
    )
    result.setdefault("session", name)
    if interrupted:
        result["interrupted_previous_prompt"] = True
    return result


# ---------------------------------------------------------------------------
# cursor_list row assembly (shared with unknown-session errors)
# ---------------------------------------------------------------------------

def _list_rows(scope: str = "session") -> List[Dict[str, str]]:
    """Presentation rows for cursor_list, freshest data first.

    Live jobs override the persisted record (status/elapsed/files/activity
    move while a run streams); entries without a live job render from what
    the handle table remembers.
    """
    now = time.time()
    rows: List[Dict[str, str]] = []
    for entry in _handles.entries(scope=scope, session_key=_resolve_session_key()):
        name = str(entry.get("session") or "")
        job = _live_job(name, entry)
        if job is not None and job.status == "running":
            with job._lock:
                files = len(job.files)
            last_event = job.last_event_at
            rows.append({
                "session": name,
                "repo": job.repo,
                "status": "running",
                "elapsed": _render.secs(now - job.created_at),
                "files": str(files),
                "last_activity": (
                    _render.secs(now - last_event) if last_event else "—"
                ),
            })
            continue
        status = str(entry.get("status") or "created")
        duration = entry.get("duration_s")
        files_count = entry.get("files_changed_count")
        rows.append({
            "session": name,
            "repo": str(entry.get("repo") or "—"),
            "status": status,
            "elapsed": _render.secs(duration) if duration is not None else "—",
            "files": str(files_count) if files_count is not None else "—",
            "last_activity": "—",
        })
    return rows


def _unknown_session_text(identifier: str) -> str:
    return _render.unknown_session(identifier, _list_rows("session"))


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def cursor_create_session(
    repo: Optional[str] = None,
    model: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Mint a named session handle. LAZY — no cursor process is spawned;
    the ACP session starts on the first cursor_send_message."""
    target_repo = (repo or "").strip() or _default_repo()
    if not target_repo:
        return (
            "no workspace repo resolvable — pass `repo` or set "
            f"{REPO_ENV_VAR}."
        )
    try:
        workdir = _runner.resolve_repo(target_repo)
    except _runner.HarnessError as exc:
        return f"cannot create session: {exc}"

    explicit_model = (str(model).strip() or None) if model else None
    name = _names.generate(
        taken=lambda n: _handles.resolve(n) is not None
    )
    _handles.record(
        name,
        repo=str(workdir),
        status="created",
        model=explicit_model,
        session_key=_resolve_session_key(),
    )
    return _render.create_session_ack(
        name, str(workdir), _resolve_model(explicit_model)
    )


def cursor_send_message(
    session: str,
    message: str,
    inactivity_timeout_s: Optional[float] = None,
    max_wall_s: Optional[float] = None,
    **_kwargs: Any,
) -> str:
    """Send work to a session (first message = the task; later = follow-up,
    interrupting a live prompt when necessary — the ack says which)."""
    ident = str(session or "").strip()
    if not ident:
        return "session is required — pass the name from cursor_create_session."
    if not str(message or "").strip():
        return "message is required — describe the coding task or follow-up."

    name = _resolve_session(ident)
    if name is None:
        return _unknown_session_text(ident)
    entry = _handles.get(name) or {}

    result = _send_to_session(
        name, entry, str(message), inactivity_timeout_s, max_wall_s
    )
    return _render_send_result(name, result)


def _render_send_result(name: str, result: Dict[str, Any]) -> str:
    """Internal dispatch dict → the plain-text send/completion rendering."""
    status = str(result.get("status") or "failed")
    if status == "running":
        return _render.send_ack(
            name, bool(result.get("interrupted_previous_prompt"))
        )
    if status == "rejected":
        return _render.repo_busy(
            str(result.get("session") or ""), str(result.get("repo") or "")
        )
    if status in _jobs.TERMINAL_STATUSES:
        # Ultra-fast in-turn finish (or handshake failure): this ack IS the
        # completion report.
        return _render.completion_text(
            name=name,
            status=status,
            elapsed_s=(result.get("duration_ms") or 0) / 1000.0,
            repo=str(result.get("repo") or ""),
            summary=str(result.get("summary") or ""),
            files=result.get("files_changed") or [],
            error=str(result.get("error") or ""),
        )
    # Undelivered/unsettled shapes degrade to their error sentence.
    return str(
        result.get("error")
        or f"cursor_send_message did not settle (status: {status})."
    )


def cursor_status(session: str, scope: str = "session", **_kwargs: Any) -> str:
    """Read-only snapshot for a cursor session (see CURSOR_STATUS_SCHEMA).

    STRICTLY READ-ONLY: only copies job state under its lock. It never
    sends ``session/cancel``, never touches the cancel event, never joins
    or signals the worker — polling a running run cannot affect it.
    """
    ident = str(session or "").strip()
    if not ident:
        return "session is required — pass the name from cursor_create_session."

    name = _resolve_session(ident)
    if name is None:
        return _render.unknown_session(ident, _list_rows(scope))

    entry = _handles.get(name)
    log_key = _log_key(name, entry)
    stats = _eventlog.stats(log_key)
    tail = _eventlog.read_events(log_key, offset=-1, limit=20)
    bullets = _render.recent_bullets((tail or {}).get("events") or [])

    job = _live_job(name, entry)
    if job is not None:
        snap = job.snapshot()
        terminal = snap.get("status") in _jobs.TERMINAL_STATUSES
        result = job.result or {}
        return _render.status_text(
            name=name,
            status=str(snap.get("status") or "unknown"),
            elapsed_s=snap.get("elapsed_s"),
            last_activity_s=snap.get("last_activity_s"),
            total_events=(stats or {}).get("total_events", 0),
            log_path=(stats or {}).get("path"),
            task=str(snap.get("task") or ""),
            files=snap.get("files_changed_so_far") or [],
            bullets=bullets,
            # Peek line only once the run settled; running output never
            # carries partial prose (and NEVER reasoning text).
            summary=str(snap.get("summary_so_far") or "") if terminal else "",
            error=str(result.get("error") or ""),
        )

    if entry is not None:
        # Known handle, but no live job in this process (e.g. restart).
        # The JSONL event log persists, so history stays visible.
        status = str(entry.get("status") or "unknown")
        if status == "running":
            # A dead process can't have left a live run behind.
            status = "unknown (recorded as running by a previous process)"
        return _render.status_text(
            name=name,
            status=status,
            elapsed_s=entry.get("duration_s"),
            last_activity_s=None,
            total_events=(stats or {}).get("total_events", 0),
            log_path=(stats or {}).get("path"),
            task=str(entry.get("task") or ""),
            files=[],
            bullets=bullets,
            note=(
                "not tracked live in this process — showing the persisted "
                "record. cursor_send_message continues the session."
            ),
        )

    return _render.unknown_session(ident, _list_rows(scope))


def cursor_stop(session: str, **_kwargs: Any) -> str:
    """Stop a session's run gracefully; report final status + partial work.

    Graceful path: cursor's native ``session/cancel`` (the prompt resolves
    with stopReason "cancelled" ~immediately); the acp_runner escalates to
    a process-group SIGKILL only if cursor hangs past its grace window.
    Idempotent on finished runs.
    """
    ident = str(session or "").strip()
    if not ident:
        return "session is required — pass the name from cursor_create_session."

    name = _resolve_session(ident)
    if name is None:
        return _unknown_session_text(ident)

    entry = _handles.get(name)
    job = _live_job(name, entry)
    if job is None:
        if entry is not None:
            status = str(entry.get("status") or "unknown")
            return _render.stop_text(
                name=name,
                status=status if status != "running" else "not running (stale record)",
                elapsed_s=entry.get("duration_s"),
                files=[],
                already_finished=True,
            )
        return _unknown_session_text(ident)

    if job.status in _jobs.TERMINAL_STATUSES:
        result = _jobs.trim_result(job.result or {})
        return _render.stop_text(
            name=name,
            status=job.status,
            elapsed_s=(result.get("duration_ms") or 0) / 1000.0,
            files=result.get("files_changed") or [],
            already_finished=True,
        )

    settled = _settle_job(job)
    if not settled:
        return (
            f"the run in session '{name}' did not settle within "
            f"{int(_INTERRUPT_WAIT_S)}s of a native cancel — it may be "
            "hung; the kill escalation is in progress. check again with "
            "cursor_status."
        )

    result = _jobs.trim_result(job.result or {})
    return _render.stop_text(
        name=name,
        status=job.status,
        elapsed_s=job.snapshot().get("elapsed_s"),
        files=result.get("files_changed") or [],
        already_finished=False,
    )


def cursor_events(
    session: str,
    offset: Optional[int] = None,
    limit: Optional[int] = None,
    kind: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Page the session's persisted JSONL event history (read-only)."""
    ident = str(session or "").strip()
    if not ident:
        return "session is required — pass the name from cursor_create_session."
    name = _resolve_session(ident)
    if name is None:
        return _unknown_session_text(ident)

    entry = _handles.get(name)
    page = _eventlog.read_events(
        _log_key(name, entry),
        offset=offset if offset is not None else -1,
        limit=limit if limit is not None else _eventlog.DEFAULT_EVENTS_LIMIT,
        kind=kind,
    )
    if page is None:
        return _render.no_event_log(name)
    return _render.events_text(name, page)


def cursor_list(scope: str = "session", **_kwargs: Any) -> str:
    """TSV listing of session handles (default: this Hermes session's)."""
    scope = scope if scope in _handles.VALID_SCOPES else "session"
    rows = _list_rows(scope)
    if not rows:
        return _render.empty_list(scope)
    return _render.list_text(rows)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _handle_cursor_create_session(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_create_session(
        repo=args.get("repo"),
        model=args.get("model"),
    )


def _handle_cursor_send_message(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_send_message(
        session=args.get("session", ""),
        message=args.get("message", ""),
        inactivity_timeout_s=args.get("inactivity_timeout_s"),
        max_wall_s=args.get("max_wall_s"),
    )


def _handle_cursor_status(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_status(
        session=args.get("session") or args.get("session_id", ""),
        scope=args.get("scope", "session"),
    )


def _handle_cursor_stop(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_stop(session=args.get("session") or args.get("session_id", ""))


def _handle_cursor_events(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_events(
        session=args.get("session") or args.get("session_id", ""),
        offset=args.get("offset"),
        limit=args.get("limit"),
        kind=args.get("kind"),
    )


def _handle_cursor_list(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_list(scope=args.get("scope", "session"))


def register(ctx) -> None:
    """Register the 6 cursor tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in (
        (CREATE_TOOL_NAME, CURSOR_CREATE_SCHEMA, _handle_cursor_create_session, "🆕"),
        (SEND_TOOL_NAME, CURSOR_SEND_SCHEMA, _handle_cursor_send_message, "📨"),
        (STATUS_TOOL_NAME, CURSOR_STATUS_SCHEMA, _handle_cursor_status, "🛰️"),
        (STOP_TOOL_NAME, CURSOR_STOP_SCHEMA, _handle_cursor_stop, "🛑"),
        (EVENTS_TOOL_NAME, CURSOR_EVENTS_SCHEMA, _handle_cursor_events, "📜"),
        (LIST_TOOL_NAME, CURSOR_LIST_SCHEMA, _handle_cursor_list, "📋"),
    ):
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            check_fn=check_cursor_available,
            emoji=emoji,
        )
