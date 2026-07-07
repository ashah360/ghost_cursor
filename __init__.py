"""Ghost ⇄ Cursor delegation plugin — bundled, auto-loaded.

v0.4: explicit named sessions + plain-text tool output. Seven tools in the
``ghost_cursor`` toolset:

* ``cursor_create_session(repo?, model?)`` — mint a named session handle
  (adjective-adjective-noun slug, e.g. ``playful-space-bunny``). LAZY: it
  dispatches nothing; the cursor agent spawns on the first message.
* ``cursor_send_message(session, message)`` — ALL work goes through here.
  The first message on a fresh session is the task; later messages are
  follow-ups (or interrupt + re-prompt when the run is live — the ack says
  which). Cursor works in the background; the terminal result is delivered
  automatically on every outcome.
* ``cursor_status(session)`` — strictly read-only snapshot (never cancels).
* ``cursor_stop(session)`` — graceful native ``run.cancel()``.
* ``cursor_events(session, offset=-1, limit=10, kind?)`` — dedicated pager
  over the per-session JSONL event log. Defaults = the last 10 events;
  negative offsets index from the end python-style; ``offset>=0`` pages
  forward by seq; ``kind`` filters (reasoning / file_diff / tool_result /
  tool_use / content / lifecycle); limit clamps at 500.
* ``cursor_list(scope='session'|'all')`` — TSV listing of session handles,
  scoped to the current Hermes session by default.
* ``cursor_subscribe(session, interval_s)`` — subscribe the CALLING
  hermes session to periodic progress digests while a run is active (see
  ``progress.py``; per-subscriber, multiple hermes sessions each get
  their own copy). ``cursor_send_message`` auto-subscribes the caller at
  ``update_interval_s`` (explicit > persisted > 180s default, 0 off).

(v0.3's ``cursor_start``/``cursor_send`` are gone — create + send replaced
them outright; pre-launch, no deprecation shim.)

Handle model (v0.4)
-------------------
THE handle is the session name minted by ``cursor_create_session``
(collision-checked against the persistent handle table, ``names.py``). The
cursor-sdk AGENT ID (``agent-<uuid>``) is recorded on the handle entry as an
alias (``handles.resolve``), so ids from older runs / completion payloads
still resolve everywhere a name is accepted. The handle table
(``handles.py``) persists name → repo/status/model/cursor_session_id (the
agent id) across restarts — ``Agent.resume(agent_id)`` continues the
conversation even across a plugin/gateway restart; the in-process job table
(``jobs.py``) keys live state by name; the JSONL event log (``eventlog.py``)
is also keyed by name, so one named session = one log across resumes.

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
see ``sdk_runner``. Model override precedence: explicit param >
``plugins.ghost_cursor.model`` in config.yaml > the plugin default,
threaded as ``model=`` on ``Agent.create``.

Transport (v0.5): the official ``cursor-sdk`` python package. One bridge
sidecar per workspace (reused across sessions on the same repo, closed at
process exit), agents resumed by persisted agent_id, event streams
re-attached transparently via ``run.observe(after_offset=...)`` when a
connection drops mid-run. Auth: the ``CURSOR_API_KEY`` env var.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from . import eventlog as _eventlog
from . import events as _events
from . import handles as _handles
from . import jobs as _jobs
from . import names as _names
from . import progress as _progress
from . import render as _render
from . import runner as _runner
from . import sdk_runner as _sdk

logger = logging.getLogger(__name__)

CREATE_TOOL_NAME = "cursor_create_session"
SEND_TOOL_NAME = "cursor_send_message"
STATUS_TOOL_NAME = "cursor_status"
STOP_TOOL_NAME = "cursor_stop"
EVENTS_TOOL_NAME = "cursor_events"
LIST_TOOL_NAME = "cursor_list"
SUBSCRIBE_TOOL_NAME = "cursor_subscribe"
TOOLSET = "ghost_cursor"

# Env var naming is Threshold's (the Ghost frontend), not HERMES_* — it points
# at the workspace repo Ghost delegates coding tasks into.
REPO_ENV_VAR = "THRESHOLD_WORKSPACE_REPO"

# Cap for per-file diff text carried in the FINAL result dict (persisted to
# the session DB via the completion message). Live file_diff progress
# envelopes carry more (see events.MAX_DIFF_CHARS).
_RESULT_DIFF_CHARS = 20_000

# How long dispatching tools block waiting for the cursor agent to be
# established. Bounded by the sdk_runner bridge-launch + create/resume
# retry budget plus slack; a healthy run yields it in seconds.
_HANDLE_WAIT_S = 150.0

# How long cursor_send_message's interrupt path waits for a cancelled run
# to settle before refusing to re-prompt. Native run.cancel() resolves
# ~immediately; the ceiling covers the sdk_runner CANCEL_GRACE_S escalation.
_INTERRUPT_WAIT_S = 40.0

# How long cursor_stop waits for OBSERVED termination after signalling
# cancel (issue #22). Matches sdk_runner.CANCEL_GRACE_S: a landed
# run.cancel() settles the job well within this; a job still live at the
# deadline most likely lost the signal (bridge RPC dropped / run between
# attempts), and cursor_stop must say so instead of acking "stopped".
# Module-level so tests can shrink it.
_STOP_WAIT_S = 15.0

_SESSION_DOC = (
    "The session handle: the name returned by cursor_create_session (e.g. "
    "'playful-space-bunny'). A cursor agent id from an older run also "
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
            "SILENCE (no stream events from cursor). Any streamed activity "
            "— reasoning, tool calls, content — resets the clock, so a "
            "long run that keeps making progress is never killed by this "
            "limit. 0 disables the inactivity watchdog. Default: "
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
                    "Optional cursor model override for this session "
                    "(e.g. 'composer-2.5', 'gpt-5.3-codex'). Omit to use "
                    "the configured/default model. Obviously-malformed "
                    "strings are rejected here; whether a well-formed id "
                    "exists in the model catalog is validated on the "
                    "first cursor_send_message (create stays lazy)."
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
            "update_interval_s": {
                "type": "number",
                "description": (
                    "Optional. Sending AUTO-SUBSCRIBES the calling Hermes "
                    "session to progress digests: while the run is active, "
                    "a compact digest (status header + new events) arrives "
                    "as a new message every this-many seconds. Explicit "
                    "value wins, else this Hermes session's persisted "
                    "subscription, else 180; 0 disables digests for THIS "
                    "session only (other subscribed sessions keep theirs). "
                    "Negative values are rejected; positive values are "
                    "clamped to the "
                    f"{_render.dur_compact(_progress.MIN_UPDATE_INTERVAL_S)}–"
                    f"{_render.dur_compact(_progress.MAX_UPDATE_INTERVAL_S)} "
                    "range (the ack notes any clamping). The subscription "
                    "persists per Hermes session (cursor_subscribe changes "
                    "it mid-run)."
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
        "Stop a running cursor session gracefully (cursor's native "
        "cancel). Acks 'stopped' only after the run is OBSERVED to reach "
        "a terminal state; if it hasn't settled within the bounded wait, "
        "the status honestly stays running — retry or check "
        "cursor_status. Reports the final status and any partial work. "
        "Idempotent on finished runs. The session stays continuable via "
        "cursor_send_message."
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

CURSOR_SUBSCRIBE_SCHEMA = {
    "name": SUBSCRIBE_TOOL_NAME,
    "description": (
        "Subscribe THIS Hermes session to a cursor session's progress "
        "digests (or retune/cancel its existing subscription — the "
        "subscription belongs to the calling Hermes session; other "
        "sessions' subscriptions are never affected). Each digest arrives "
        "as a new message in the subscribing session: a "
        "cursor_status-style header (status, elapsed, last activity, "
        "files so far, pending tool call) plus the events since the "
        "previous update. Every subscribed Hermes session gets its own "
        "copy of every digest at its own cadence. Takes effect on the "
        "next tick (sooner if the new interval is shorter); interval_s=0 "
        "removes only this session's subscription. Works whether or not "
        "a run is active — the subscription persists on the session and "
        "applies to the running/next run. Use this to watch a long run "
        "more closely, to follow a run dispatched from another Hermes "
        "session, or to silence a chatty one; the final result is "
        "delivered separately to every subscriber (and always to the "
        "dispatching session) regardless of this setting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DOC},
            "interval_s": {
                "type": "number",
                "description": (
                    "Seconds between progress digests for the calling "
                    "Hermes session. 0 unsubscribes this session only "
                    "(no digests here; other subscribers and completion "
                    "delivery are unaffected). Negative values are "
                    "rejected; positive values are clamped to the "
                    f"{_render.dur_compact(_progress.MIN_UPDATE_INTERVAL_S)}–"
                    f"{_render.dur_compact(_progress.MAX_UPDATE_INTERVAL_S)} "
                    "range (the ack notes any clamping)."
                ),
            },
        },
        "required": ["session", "interval_s"],
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
    """Tool gate: cursor-sdk importable + a workspace repo resolvable.

    Deliberately does NOT gate on CURSOR_API_KEY: a missing key surfaces as
    an actionable error on the first send instead of silently hiding the
    toolset.
    """
    try:
        return _sdk.sdk_available() and _default_repo() is not None
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
        else _sdk.DEFAULT_INACTIVITY_TIMEOUT_S
    )


def _resolve_max_wall(explicit: Optional[float]) -> float:
    """Wall-ceiling precedence: explicit param >
    ``plugins.ghost_cursor.max_wall_s`` in config > ``DEFAULT_MAX_WALL_S``
    (0 = disabled)."""
    if explicit is not None:
        return float(explicit)
    configured = _configured_timeout("max_wall_s")
    return configured if configured is not None else _sdk.DEFAULT_MAX_WALL_S


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


# Persisted onto a handle whose recorded run cannot exist anymore: runs are
# worker threads, so the restart that emptied this process's job registry
# also destroyed the run before it could finalize its record (issue #13).
_ORPHANED_NOTE = "orphaned: plugin process restarted mid-run"


def _reconcile_orphan(
    name: str, entry: Dict[str, Any], job: Optional["_jobs.CursorJob"]
) -> Dict[str, Any]:
    """Settle a persisted "running" record that has NO job behind it.

    Only the run's own worker thread ever moves a "running" handle to a
    terminal state — after a process restart nothing would, so cursor_list
    / cursor_status would claim "running" forever (issue #13). Any read
    that observes the contradiction repairs the record to "failed" with an
    explanatory ``status_note``. The session stays continuable
    (cursor_send_message resumes by the durable agent id — only the run
    died) and nothing is delivered: there is no run left to deliver for.
    A handle with ANY job in this process (running or just settled) is
    left to the normal lifecycle.
    """
    if job is not None or str(entry.get("status") or "") != "running":
        return entry
    _handles.record(name, status="failed", status_note=_ORPHANED_NOTE)
    return _handles.get(name) or entry


def _resume_sid(name: str, entry: Dict[str, Any]) -> Optional[str]:
    """The cursor agent id to resume, or None for a fresh agent.

    A session that has run before carries its ``cursor_session_id`` alias.
    A pre-v0.4 entry keyed by the raw cursor UUID (no alias field, but a
    recorded run status) resumes by its own key. A freshly created (lazy,
    never-run) session starts a fresh agent.
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

def _repair_terminal_handle(job: "_jobs.CursorJob") -> None:
    """Repair a persisted terminal status contradicted by a live run.

    Invariant (issue #22): the persisted handle status must never be
    terminal while the job is still live in this process — events folding
    for a session recorded as cancelled/completed means both read surfaces
    (cursor_status / cursor_list) are lying. Belt-and-braces at the point
    events fold: flip the record back to "running" and leave a lifecycle
    event in the log so the contradiction is visible, not papered over.
    """
    name = job.session_name
    if not name or job.status in _jobs.TERMINAL_STATUSES:
        return
    stale = str((_handles.get(name) or {}).get("status") or "")
    if stale not in _jobs.TERMINAL_STATUSES:
        return
    _handles.record(name, status="running")
    logger.warning(
        "ghost_cursor: session %s was persisted %r while its run is still "
        "streaming events — repaired to 'running'", name, stale,
    )
    job.append_progress(_events.lifecycle(
        "status.repaired",
        was=stale,
        note=(
            f"persisted status '{stale}' contradicted a live run still "
            "streaming events — repaired to 'running'"
        ),
    ))


def _fold_envelope(job: "_jobs.CursorJob", envelope: Dict[str, Any]) -> None:
    """Fold one canonical envelope into the job's aggregation state and
    append a compact line to the rolling buffer status views read."""
    _repair_terminal_handle(job)
    job.append_progress(envelope)

    kind = envelope.get("kind")
    with job._lock:
        # Progress evidence for the auto-retry gate (issue #17): every
        # non-lifecycle envelope counts toward the since-prompt total;
        # completed tool calls are tracked by id (replay-safe).
        if kind != "lifecycle":
            job.nonlifecycle_events += 1
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
        elif kind == "tool_use":
            job.segment_open = False
            tool = str(envelope.get("tool") or "tool")
            detail = str(
                envelope.get("command") or envelope.get("title") or ""
            ).strip()
            job.pending_tool = f"{tool} `{detail}`" if detail else tool
            job.pending_tool_since = time.time()
        elif kind == "tool_result":
            job.segment_open = False
            job.pending_tool = ""
            job.pending_tool_since = None
            if envelope.get("status") == _events.STATUS_DONE:
                job.completed_tool_ids.add(str(envelope.get("id") or "tool"))
        elif kind == "lifecycle":
            event = envelope.get("event")
            if event == "run.completed":
                job.completed = True
            elif event == "run.failed":
                job.run_error = str(envelope.get("error") or "run failed")
                job.timed_out = job.timed_out or bool(envelope.get("timeout"))
                job.cancelled = job.cancelled or bool(envelope.get("cancelled"))
                if "retryable" in envelope:
                    job.error_retryable = envelope.get("retryable")
                if "retry_after" in envelope:
                    job.error_retry_after = envelope.get("retry_after")
            elif event == "reasoning":
                text = str(envelope.get("text") or "")
                if text:
                    job.reasoning_tail = (job.reasoning_tail + text)[-4000:]
                job.segment_open = False


# Transparent zero-progress auto-retry. Live incident (2026-07-04): runs
# dispatched through a stale, hours-old cursor-sdk-bridge sidecar reached
# terminal "error" within seconds having produced ZERO meaningful events,
# while fresh bridges worked — killing the stale bridges fixed everything.
# A run matching that signature (terminal error + no progress since the
# prompt + retryable-or-unknown) is re-sent on the SAME agent up to
# _MAX_AUTO_RETRIES times with backoff, recycling the workspace bridge
# before the first retry. Nothing user-facing: the job stays "running"
# (digest numbering/subscription continue) and only jsonl lifecycle events
# ("sdk.autoretry") record it. A run WITH progress, a non-retryable error,
# or an exhausted budget surfaces the detailed failure instead — delivered
# immediately through the normal failure path.
_MAX_AUTO_RETRIES = 2
# Backoff ladder before retry N (1-based); a parseable server retry_after
# on the error overrides the step. Module-level so tests can zero it.
_AUTO_RETRY_BACKOFF_S = (15.0, 60.0)
# Tight first-event watchdog for retry attempts, independent of the user's
# inactivity_timeout_s (issue #17): a retried run that streams NOTHING in
# this window is settled failed and the failure delivered, instead of
# sitting as a silent "running" zombie until the (possibly 30-minute)
# inactivity timeout. Module-level so tests can shrink it.
_AUTO_RETRY_FIRST_EVENT_S = 75.0

# The progress gate's triviality threshold (issue #17): any file_diff or
# any COMPLETED tool call is progress outright; otherwise more than this
# many non-lifecycle envelopes since the prompt counts too. At or below
# it (e.g. a half-sentence of narration before a transient server error)
# the run is still the zero-progress signature and may be retried.
_TRIVIAL_PROGRESS_EVENTS = 3


def _made_progress(job: "_jobs.CursorJob") -> bool:
    """Durable since-prompt progress evidence for the auto-retry gate.

    Read from JOB-level aggregation folded by :func:`_fold_envelope` — a
    job is one prompt, so this accumulates across auto-retry attempts and
    is never reset by attempt-local stream state. Any file diff, any
    completed tool call (a run that committed/pushed work always has
    both), or a non-trivial number of non-lifecycle envelopes means the
    run did real work and must never be re-prompted automatically.
    """
    with job._lock:
        return bool(
            job.files
            or job.completed_tool_ids
            or job.nonlifecycle_events > _TRIVIAL_PROGRESS_EVENTS
        )


def _auto_retry_delay_s(state: Dict[str, Any], attempt: int) -> float:
    """Backoff before auto-retry ``attempt`` (1-based): the server-supplied
    retry_after when it parses as seconds, else the fixed ladder."""
    retry_after = state.get("retry_after")
    if retry_after:
        try:
            return max(float(str(retry_after)), 0.0)
        except (TypeError, ValueError):
            pass  # HTTP-date form — fall through to the ladder
    return _AUTO_RETRY_BACKOFF_S[min(attempt - 1, len(_AUTO_RETRY_BACKOFF_S) - 1)]


def _run_attempt(
    job: "_jobs.CursorJob",
    workdir: str,
    first_event_timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """One run_sdk pass for ``job``: stream, fold, persist the handle.

    ``first_event_timeout_s`` (retry attempts only) arms sdk_runner's tight
    first-event watchdog so a zero-event retry settles failed within the
    window instead of riding the user's inactivity timeout.

    Returns the attempt state the auto-retry decision needs:

    * ``preflight`` — a final result dict when the run never happened
      (bridge/auth/repo preflight failure); the caller returns it as-is.
    * ``terminal_error`` (+ ``retryable`` / ``retry_after``) — True when
      the run settled with SDK status "error" (the enriched ``sdk.error``
      payload, see sdk_runner).

    Progress evidence is NOT attempt state: it accumulates on the job via
    :func:`_fold_envelope` (see :func:`_made_progress`).
    """
    state: Dict[str, Any] = {
        "preflight": None,
        "terminal_error": False,
        "retryable": None,
        "retry_after": None,
    }
    normalizer = _events.SdkNormalizer()
    try:
        for key, obj in _sdk.run_sdk(
            job.task,
            workdir,
            inactivity_timeout_s=float(job.inactivity_timeout_s),
            max_wall_s=float(job.max_wall_s),
            cancel_check=job.cancel_event.is_set,
            # A retry re-sends on the SAME agent established by the first
            # attempt (its state survives the bridge recycle on disk).
            agent_id=job.cursor_session_id or job.requested_session_id,
            model=job.requested_model,
            # Kwarg only when armed: fakes/replays without the param keep
            # working for ordinary (non-retry) dispatches.
            **(
                {"first_event_timeout_s": float(first_event_timeout_s)}
                if first_event_timeout_s
                else {}
            ),
        ):
            job.last_event_at = time.time()
            if key == "sdk.session":
                sid = str(obj.get("agentId") or "")
                with job._lock:
                    job.cursor_session_id = sid
                    job.resumed = bool(obj.get("resumed"))
                    job.model = str(obj.get("model") or "")
                # Persist the handle the moment the agent exists — keyed by
                # the session NAME, with the agent id as an alias.
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
            elif key == "sdk.error" and obj.get("run_status") == "error":
                state["terminal_error"] = True
                state["retryable"] = obj.get("retryable")
                state["retry_after"] = obj.get("retry_after")
            for envelope in normalizer.normalize(key, obj):
                _fold_envelope(job, envelope)
    except _sdk.SdkRunnerError as exc:
        # Hard SDK failure (bridge/create/auth) — actionable error, no
        # silent regress.
        state["preflight"] = {"success": False, "error": str(exc)}
    except _runner.HarnessError as exc:
        state["preflight"] = {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.exception("cursor run failed")
        with job._lock:
            job.run_error = f"{type(exc).__name__}: {exc}"
    return state


def _execute_cursor_run(job: "_jobs.CursorJob") -> Dict[str, Any]:
    """Run cursor for ``job`` (via the cursor-sdk) and build the final result.

    Runs on the job worker thread. Cancellation is the job's cancel event
    (set by cursor_stop / cursor_send_message), which triggers the native
    ``run.cancel()``. The cursor agent id is persisted onto the session's
    handle entry (as the alias) the instant the agent exists, so
    ``Agent.resume`` keeps working across process restarts. Zero-progress
    terminal errors are transparently retried on the same agent (see
    _MAX_AUTO_RETRIES above) with a bridge recycle before the first retry.
    """
    started = time.monotonic()
    workdir = job.repo

    # Pre-run git snapshot: fuels the fallback that populates files_changed
    # when cursor edits through paths that emit no parseable diff content
    # in the stream (e.g. shell commands; tool payloads are unstable).
    git_before = _sdk.git_status_snapshot(workdir)

    attempt = 0  # auto-retries used so far
    while True:
        state = _run_attempt(
            job,
            workdir,
            # Retry attempts get the tight first-event watchdog (issue
            # #17); the user's inactivity_timeout_s still applies once the
            # retry starts streaming.
            first_event_timeout_s=_AUTO_RETRY_FIRST_EVENT_S if attempt else None,
        )
        if state["preflight"] is not None:
            return state["preflight"]
        if not (
            state["terminal_error"]
            # Durable since-prompt evidence, not attempt-local stream
            # state: a run that already did real work (committed/pushed,
            # edited files, completed tools) is never re-prompted — its
            # failure surfaces immediately instead (issue #17).
            and not _made_progress(job)
            and state["retryable"] is not False  # retryable or unknown
            and attempt < _MAX_AUTO_RETRIES
            and not job.cancel_event.is_set()
        ):
            break
        attempt += 1
        # The stale-bridge lever (see the incident note above): recycle the
        # workspace bridge once, before the FIRST retry.
        bridge_recycled = attempt == 1 and _sdk.recycle_bridge(workdir)
        with job._lock:
            reason = (
                "zero-progress terminal error: "
                f"{job.run_error or 'unknown error'}"
            )
        # Log-only signal (jsonl + rolling buffer) — nothing user-facing.
        _fold_envelope(job, _events.lifecycle(
            "sdk.autoretry",
            attempt=attempt,
            reason=reason,
            bridge_recycled=bool(bridge_recycled),
        ))
        logger.warning(
            "ghost_cursor auto-retry %d/%d for job %s (%s; bridge_recycled=%s)",
            attempt, _MAX_AUTO_RETRIES, job.job_id, reason, bridge_recycled,
        )
        if job.cancel_event.wait(_auto_retry_delay_s(state, attempt)):
            break  # cancelled during backoff — settle with what we have
        # The retry owns the failure reporting now: clear the previous
        # attempt's error state so a successful retry builds a clean result.
        with job._lock:
            job.run_error = None
            job.error_retryable = None
            job.error_retry_after = None

    # Git fallback: edits the stream carried no diff for (shell-driven
    # writes, kill-before-diff) still land in files_changed + progress.
    try:
        with job._lock:
            known_paths = set(job.files)
        for fb in _sdk.git_fallback_diffs(workdir, git_before):
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
        error_retryable = job.error_retryable
        error_retry_after = job.error_retry_after
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
        if error_retryable is not None:
            result["error_retryable"] = error_retryable
        if error_retry_after is not None:
            result["error_retry_after"] = error_retry_after
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

    Returns the internal result dict: the running shape once the cursor
    agent is established, the final result if the run terminated before
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
        # No agent and not terminal — the bridge is wedged pre-create.
        # Cancel it (the worker's kill path takes over) and report.
        job.request_cancel()
        return {
            "success": False,
            "status": "failed",
            "session": job.session_name,
            "error": (
                f"cursor did not establish an agent within "
                f"{int(_HANDLE_WAIT_S)}s — cancelled the attempt"
            ),
            "repo": job.repo,
        }

    if not job.arm_delivery():
        # Finalize won the race while we held an un-armed job: report the
        # final result in-turn (nothing was enqueued).
        return {**_jobs.trim_result(job.result or {}), "status": job.status}

    # The run is live and its outcome will arrive as a delivered message —
    # start one progress-digest timer per persisted subscriber alongside
    # (the dispatch path persisted the caller's auto-subscription before
    # dispatching, so the map is current; an empty map starts nothing).
    _progress.start_for_job(
        job, _handles.subscribers_of(_handles.get(job.session_name))
    )

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
    """Cancel ``job`` (native run.cancel()) and wait for it to settle.

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
    update_interval_s: Optional[float] = None,
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
            # Same honesty rule as cursor_stop (issue #22): the run is
            # still live, so re-arm the delivery mark_handled suppressed —
            # otherwise a run whose cancel was lost settles silently.
            job.arm_delivery()
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
    # "Events since prompt" marker: snapshot the log position BEFORE the
    # dispatch (events the new run appends must count as since-prompt), but
    # record it only for a send that actually prompted — a rejected send
    # (another session holds the repo) leaves the previous marker in place.
    prompt_seq = (
        _eventlog.stats(_log_key(name, entry)) or {}
    ).get("total_events", 0)
    # AUTO-SUBSCRIBE the calling hermes session at dispatch: explicit
    # param > this session's persisted subscriber interval > the 180s
    # default — whoever prompts is always watching. Persisted (in the
    # entry's per-subscriber map) so it survives restarts and carries
    # over interrupt-and-reprompt (digest numbering continues); OTHER
    # sessions' subscriptions are untouched by a send.
    caller_key = _resolve_session_key()
    interval = _progress.resolve_interval(entry, update_interval_s, caller_key)
    _handles.set_subscriber(name, caller_key, interval)
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
    # Model validation is deferred to this first send (create is lazy by
    # design) — so when the agent-create failure hits and this session
    # carries an explicit model, attribute the failure to the param chosen
    # at create instead of leaving a generic sdk error (issue #12).
    explicit_model = str(entry.get("model") or "")
    if explicit_model and "agent create failed" in str(result.get("error") or ""):
        result["error"] = (
            f"{result['error']} This session's model {explicit_model!r} "
            f"was set at {CREATE_TOOL_NAME} — if the model is the problem, "
            "create a new session with a valid model id (or omit model "
            "for the default)."
        )
    if str(result.get("status") or "") != "rejected":
        _handles.record(name, last_prompt_seq=prompt_seq)
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
        entry = _reconcile_orphan(name, entry, job)
        status = str(entry.get("status") or "created")
        status_note = str(entry.get("status_note") or "")
        duration = entry.get("duration_s")
        files_count = entry.get("files_changed_count")
        rows.append({
            "session": name,
            "repo": str(entry.get("repo") or "—"),
            "status": f"{status} ({status_note})" if status_note else status,
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
    """Mint a named session handle. LAZY — no cursor agent is created;
    the agent starts on the first cursor_send_message."""
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
    if explicit_model:
        # Shape-only check (issue #12) — create stays lazy (no sdk
        # contact), so obviously-malformed strings fail HERE, adjacent to
        # the param; whether a well-formed id actually exists in the
        # catalog is still validated on the first send.
        reason = _sdk.invalid_model_reason(explicit_model)
        if reason:
            return (
                f"cannot create session: {reason}. Pass a base model id "
                "(e.g. 'claude-fable-5'), optionally with a "
                "'-thinking[-<level>]' or '[param=value,...]' suffix — or "
                "omit model for the default."
            )
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
    update_interval_s: Optional[float] = None,
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

    # Same validation contract as cursor_subscribe (issue #14): negative
    # or non-numeric intervals are rejected before anything is dispatched
    # or persisted; out-of-range positives are clamped and the ack says so.
    try:
        interval, interval_note = (
            _progress.validate_interval(update_interval_s, "update_interval_s")
            if update_interval_s is not None
            else (None, None)
        )
    except ValueError as exc:
        return str(exc)

    result = _send_to_session(
        name, entry, str(message), inactivity_timeout_s, max_wall_s,
        interval,
    )
    text = _render_send_result(name, result)
    return f"{text}\nnote: {interval_note}" if interval_note else text


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
        entry = _handles.get(name)
        stats = _eventlog.stats(_log_key(name, entry))
        return _render.completion_text(
            name=name,
            status=status,
            elapsed_s=(result.get("duration_ms") or 0) / 1000.0,
            repo=str(result.get("repo") or ""),
            summary=str(result.get("summary") or ""),
            files=result.get("files_changed") or [],
            error=str(result.get("error") or ""),
            total_events=(stats or {}).get("total_events", 0),
            last_prompt_seq=_handles.last_prompt_seq(entry),
            retryable=result.get("error_retryable"),
            retry_after=result.get("error_retry_after"),
        )
    # Undelivered/unsettled shapes degrade to their error sentence.
    return str(
        result.get("error")
        or f"cursor_send_message did not settle (status: {status})."
    )


def cursor_status(session: str, scope: str = "session", **_kwargs: Any) -> str:
    """Read-only snapshot for a cursor session (see CURSOR_STATUS_SCHEMA).

    STRICTLY READ-ONLY: only copies job state under its lock. It never
    sends ``run.cancel()``, never touches the cancel event, never joins
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
            last_prompt_seq=_handles.last_prompt_seq(entry),
        )

    if entry is not None:
        # Known handle, but no live job in this process (e.g. restart).
        # The JSONL event log persists, so history stays visible.
        entry = _reconcile_orphan(name, entry, job)
        status = str(entry.get("status") or "unknown")
        if status == "running":
            # A dead process can't have left a live run behind. Reconciled
            # above; this fallback only fires if that write failed.
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
            error=str(entry.get("status_note") or ""),
            note=(
                "not tracked live in this process — showing the persisted "
                "record. cursor_send_message continues the session."
            ),
            last_prompt_seq=_handles.last_prompt_seq(entry),
        )

    return _render.unknown_session(ident, _list_rows(scope))


def cursor_stop(session: str, **_kwargs: Any) -> str:
    """Stop a session's run gracefully; report final status + partial work.

    Graceful path: the SDK's native ``run.cancel()`` (the run resolves
    with status "cancelled" ~immediately); the bridge owns the run, so
    there is no process to kill on our side. Idempotent on finished runs.
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

    settled = _settle_job(job, wait_s=_STOP_WAIT_S)
    if not settled and job.arm_delivery():
        # No terminal state was OBSERVED inside the window — the cancel
        # signal may have been lost (issue #22: the bridge owns the run;
        # our native run.cancel() is best-effort and there is no local
        # process to kill). Never claim "stopped" or persist a terminal
        # status on faith: the job stays running, the cancel event stays
        # signalled, and delivery is re-armed so the real outcome still
        # reaches the conversation whenever the run does settle.
        return (
            f"cancel signalled, but the run in session '{name}' has not "
            f"stopped within {int(_STOP_WAIT_S)}s — it is still executing. "
            "status stays running; nothing terminal was recorded. retry "
            "cursor_stop, or watch cursor_status; the final outcome will "
            "be delivered when the run actually settles."
        )

    # Either the settle wait succeeded, or finalize won the race with the
    # re-arm above (arm_delivery returned False, nothing was enqueued) —
    # both mean an observed terminal state to report in-turn.
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
    return _render.events_text(name, page, _handles.last_prompt_seq(entry))


def cursor_subscribe(session: str, interval_s: Any = None, **_kwargs: Any) -> str:
    """Subscribe/retune the CALLING hermes session's progress digests for
    a cursor session (see CURSOR_SUBSCRIBE_SCHEMA). Persists on the
    session's per-subscriber map; retunes (or starts, for a subscriber
    joining mid-run) the caller's live timer. interval_s=0 removes only
    the caller's subscription — other subscribers are untouched."""
    ident = str(session or "").strip()
    if not ident:
        return "session is required — pass the name from cursor_create_session."
    name = _resolve_session(ident)
    if name is None:
        return _unknown_session_text(ident)
    if interval_s is None:
        return "interval_s is required — seconds between digests (0 unsubscribes)."
    try:
        interval, note = _progress.validate_interval(interval_s, "interval_s")
    except ValueError as exc:
        return str(exc)

    job = _live_job(name, _handles.get(name))
    _progress.subscribe(
        name,
        _resolve_session_key(),
        interval,
        job=job if job is not None and job.status == "running" else None,
    )
    return _render.subscribe_ack(name, interval, note)


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
        update_interval_s=args.get("update_interval_s"),
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


def _handle_cursor_subscribe(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_subscribe(
        session=args.get("session") or args.get("session_id", ""),
        interval_s=args.get("interval_s"),
    )


def register(ctx) -> None:
    """Register the 7 cursor tools. Called once by the plugin loader.

    Also arranges clean bridge shutdown: the plugin loader has no unload
    hook, so the per-workspace cursor-sdk bridge sidecars are closed at
    process exit.
    """
    atexit.register(_sdk.shutdown_bridges)
    for name, schema, handler, emoji in (
        (CREATE_TOOL_NAME, CURSOR_CREATE_SCHEMA, _handle_cursor_create_session, "🆕"),
        (SEND_TOOL_NAME, CURSOR_SEND_SCHEMA, _handle_cursor_send_message, "📨"),
        (STATUS_TOOL_NAME, CURSOR_STATUS_SCHEMA, _handle_cursor_status, "🛰️"),
        (STOP_TOOL_NAME, CURSOR_STOP_SCHEMA, _handle_cursor_stop, "🛑"),
        (EVENTS_TOOL_NAME, CURSOR_EVENTS_SCHEMA, _handle_cursor_events, "📜"),
        (LIST_TOOL_NAME, CURSOR_LIST_SCHEMA, _handle_cursor_list, "📋"),
        (SUBSCRIBE_TOOL_NAME, CURSOR_SUBSCRIBE_SCHEMA, _handle_cursor_subscribe, "🔔"),
    ):
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            check_fn=check_cursor_available,
            emoji=emoji,
        )
