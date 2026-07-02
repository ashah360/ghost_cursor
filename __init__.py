"""Ghost ⇄ Cursor delegation plugin — bundled, auto-loaded.

Registers one tool, ``cursor_edit``, into the ``ghost_cursor`` toolset. The
tool runs the Cursor agent over ACP (``cursor-agent acp``, JSON-RPC over
stdio — see ``acp_runner.py``; the legacy ``--print`` stdout-scraping runner
is kept in ``runner.py`` as reference/fallback) inside a target repo, streams
per-edit progress (reasoning fragments + full file diffs) through the calling
agent's ``tool_progress_callback``, and returns a structured summary of the
files changed. Because the call is an ordinary Hermes tool call inside a real
session, the result persists in the session transcript and reloads for free.

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
persists.

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
from typing import Any, Callable, Dict, List, Optional

from plugins.ghost_cursor import acp_runner as _acp
from plugins.ghost_cursor import events as _events
from plugins.ghost_cursor import runner as _runner

logger = logging.getLogger(__name__)

TOOL_NAME = "cursor_edit"
TOOLSET = "ghost_cursor"

# Env var naming is Threshold's (the Ghost frontend), not HERMES_* — it points
# at the workspace repo Ghost delegates coding tasks into.
REPO_ENV_VAR = "THRESHOLD_WORKSPACE_REPO"

# Cap for per-file diff text carried in the FINAL tool result (persisted to
# the session DB). Live file_diff progress envelopes carry more (see
# events.MAX_DIFF_CHARS); the persisted summary stays lean.
_RESULT_DIFF_CHARS = 20_000

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
        "prior context instead of starting fresh."
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
        },
        "required": ["task"],
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
# Progress emission
# ---------------------------------------------------------------------------

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
# Tool handler
# ---------------------------------------------------------------------------

def cursor_edit(
    task: str,
    repo: Optional[str] = None,
    timeout: float = _runner.DEFAULT_TIMEOUT_S,
    progress_callback: Optional[Callable] = None,
    session_id: Optional[str] = None,
    **_kwargs: Any,
) -> str:
    """Run cursor-agent on ``task``, stream progress, return a JSON summary.

    ``progress_callback`` overrides the auto-resolved agent callback (used by
    tests and direct harness invocation). ``session_id`` continues a prior
    cursor session (multi-turn) via ACP ``session/load``; the result's
    ``session_id`` / ``resumed`` fields report the session actually used.
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

    started = time.monotonic()
    emitted = 0
    live_progress = pcb is not None

    # Aggregation state for the final summary.
    files: Dict[str, Dict[str, Any]] = {}
    assistant_parts: List[str] = []
    run_error: Optional[str] = None
    timed_out = False
    completed = False
    result_session_id = ""
    resumed = False

    def _fold(envelope: Dict[str, Any]) -> None:
        nonlocal emitted, completed, run_error, timed_out
        if _emit_progress(pcb, envelope):
            emitted += 1

        kind = envelope.get("kind")
        if kind == "file_diff":
            path = str(envelope.get("path") or "")
            entry = files.setdefault(
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
                assistant_parts.append(delta)
        elif kind == "lifecycle":
            event = envelope.get("event")
            if event == "run.completed":
                completed = True
            elif event == "run.failed":
                run_error = str(envelope.get("error") or "run failed")
                timed_out = timed_out or bool(envelope.get("timeout"))

    # Pre-run git snapshot: fuels the fallback that populates files_changed
    # when cursor edits through paths that emit no ACP diff content (e.g.
    # shell commands).
    git_before = _acp.git_status_snapshot(workdir)

    normalizer = _events.AcpNormalizer()
    try:
        for key, obj in _acp.run_acp(
            str(task), str(workdir), timeout=float(timeout), session_id=session_id
        ):
            if key == "acp.session":
                # The sessionId rides back to the caller for multi-turn
                # continuation; the normalizer only folds it into lifecycle.
                result_session_id = str(obj.get("sessionId") or "")
                resumed = bool(obj.get("resumed"))
            for envelope in normalizer.normalize(key, obj):
                _fold(envelope)
    except _acp.AcpError as exc:
        # Hard ACP failure (handshake) — actionable error, no silent regress.
        return json.dumps({"success": False, "error": str(exc)})
    except _runner.HarnessError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except Exception as exc:
        logger.exception("cursor_edit failed")
        run_error = f"{type(exc).__name__}: {exc}"

    # Git fallback: edits the ACP stream carried no diff for (shell-driven
    # writes, kill-before-diff) still land in files_changed + progress.
    try:
        for fb in _acp.git_fallback_diffs(workdir, git_before):
            if fb["path"] not in files:
                _fold(_events.file_diff(**fb))
    except Exception:
        logger.debug("cursor_edit git fallback failed", exc_info=True)

    duration_ms = int((time.monotonic() - started) * 1000)
    files_changed = sorted(files.values(), key=lambda f: f["path"])
    prose = "".join(assistant_parts).strip()

    result: Dict[str, Any] = {
        "success": completed and run_error is None,
        "status": "timeout" if timed_out else ("completed" if completed and not run_error else "failed"),
        "repo": str(workdir),
        "summary": prose or ("(no assistant summary)" if not run_error else ""),
        "files_changed": files_changed,
        "files_changed_count": len(files_changed),
        "duration_ms": duration_ms,
        "live_progress": live_progress,
        "progress_events_emitted": emitted,
        "session_id": result_session_id,
        "resumed": resumed,
    }
    if run_error:
        result["error"] = run_error
        if files_changed:
            result["partial"] = True
    return json.dumps(result, ensure_ascii=False)


def _handle_cursor_edit(args: Dict[str, Any], **kwargs: Any) -> str:
    return cursor_edit(
        task=args.get("task", ""),
        repo=args.get("repo"),
        session_id=args.get("session_id"),
    )


def register(ctx) -> None:
    """Register the cursor_edit tool. Called once by the plugin loader."""
    ctx.register_tool(
        name=TOOL_NAME,
        toolset=TOOLSET,
        schema=CURSOR_EDIT_SCHEMA,
        handler=_handle_cursor_edit,
        check_fn=check_cursor_edit_available,
        emoji="🖱️",
    )
