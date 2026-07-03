"""Normalization of cursor-agent events into canonical envelopes.

Two normalizers share the same envelope builders:

* :func:`normalize_harness` — the legacy ``--print --output-format
  stream-json`` stdout events (kept for the fallback/reference runner).
* :class:`AcpNormalizer` — ACP ``session/update`` notifications from
  ``cursor-agent acp`` (the current runner). Stateful, because ACP
  ``tool_call_update`` events carry only ``toolCallId`` + delta fields; the
  ``kind``/``title`` arrive on the initial ``tool_call`` event.

Adapted from the Threshold bridge (``app/bridge/events.py`` — the
``normalize_harness`` block), so the envelopes ``cursor_edit`` emits as tool
progress are byte-compatible with what the Threshold frontend already folds
into its transcript: ``content`` / ``tool_use`` / ``tool_result`` /
``lifecycle`` plus the ``file_diff`` kind carrying full before/after content
per completed edit.

Canonical envelope::

    {"source": "ghost", "kind": <"content"|"tool_use"|"tool_result"
                                 |"lifecycle"|"file_diff">, ...}
"""

from __future__ import annotations

import difflib
import time
from typing import Any, Dict, List, Tuple

SOURCE = "ghost"

TOOL_SHELL = "shell"
TOOL_FILE_EDIT = "file-edit"

STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"

# Full-file before/after payloads ride inside tool-progress events; cap each
# field so a single giant file can't balloon one SSE frame past what the
# gateway (and the persisted session row for the final result) can absorb.
MAX_CONTENT_CHARS = 200_000
MAX_DIFF_CHARS = 100_000
# Shell/tool output cap for the canonical envelope. Deliberately generous:
# the envelope is the full-fidelity record that lands in the per-session
# JSONL spill log (eventlog.py); the compact views (rolling buffer, status
# tails, paged events) apply their own much smaller inline clips.
MAX_OUTPUT_CHARS = 200_000


def _clip(text: Any, limit: int) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n… [truncated {len(s) - limit} chars]"


def _envelope(kind: str, **payload: Any) -> Dict[str, Any]:
    return {"source": SOURCE, "kind": kind, **payload}


def content_delta(delta: str, done: bool = False) -> Dict[str, Any]:
    """A streamed text fragment destined for a TextBlock."""
    return _envelope("content", delta=delta, done=done)


def lifecycle(event: str, **payload: Any) -> Dict[str, Any]:
    """A run-level signal (start/complete/fail/reasoning)."""
    return _envelope("lifecycle", event=event, **payload)


def file_diff(
    path: str,
    before: str,
    after: str,
    diff: str,
    added: int = 0,
    removed: int = 0,
    status: str | None = None,
) -> Dict[str, Any]:
    """A completed file edit with full before/after content.

    ``status`` follows git porcelain: "A" (added — no prior content),
    "M" (modified), "D" (deleted — no content after). Inferred when omitted.
    """
    if status is None:
        if not before:
            status = "A"
        elif not after:
            status = "D"
        else:
            status = "M"
    return _envelope(
        "file_diff",
        path=path,
        before=_clip(before, MAX_CONTENT_CHARS),
        after=_clip(after, MAX_CONTENT_CHARS),
        diff=_clip(diff, MAX_DIFF_CHARS),
        added=int(added or 0),
        removed=int(removed or 0),
        status=status,
    )


# cursor-agent tool_call payloads key the call by its kind: exactly one of
# these (or an unknown *ToolCall) is present under obj["tool_call"].
_EDIT_TOOLS = {"editToolCall", "writeToolCall", "deleteToolCall"}


def _tool_call_payload(data: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """Extract (raw_tool_key, payload) from a tool_call event, e.g.
    ("editToolCall", {"args": ..., "result": ...})."""
    tc = data.get("tool_call")
    if isinstance(tc, dict):
        for key, val in tc.items():
            if key.endswith("ToolCall") and isinstance(val, dict):
                return key, val
    return "", {}


def _tool_kind(raw_key: str) -> str:
    return TOOL_FILE_EDIT if raw_key in _EDIT_TOOLS else TOOL_SHELL


def _tool_title(raw_key: str, kind: str, args: Dict[str, Any]) -> str:
    path = str(args.get("path") or "")
    if kind == TOOL_FILE_EDIT:
        return f"Editing {path}" if path else "File edit"
    if raw_key == "readToolCall":
        return f"Reading {path}" if path else "Read file"
    cmd = str(args.get("command") or args.get("cmd") or "")
    if cmd:
        return cmd.strip().splitlines()[0][:120]
    name = raw_key[: -len("ToolCall")] if raw_key.endswith("ToolCall") else raw_key
    return name or "Tool"


def _duration_ms(tc_payload: Dict[str, Any], data: Dict[str, Any]) -> int | None:
    """durationMs = completedAtMs - startedAtMs when both are present.

    cursor-agent puts these on the tool_call wrapper (as STRINGS), sibling to
    the *ToolCall payload.
    """
    wrapper = data.get("tool_call") if isinstance(data.get("tool_call"), dict) else {}
    started = wrapper.get("startedAtMs") or tc_payload.get("startedAtMs")
    completed = wrapper.get("completedAtMs") or tc_payload.get("completedAtMs")
    try:
        return int(completed) - int(started)
    except (TypeError, ValueError):
        return None


def _assistant_text(data: Dict[str, Any]) -> str:
    """Concatenate text parts from a user/assistant message event."""
    message = data.get("message")
    if not isinstance(message, dict):
        return ""
    parts: List[str] = []
    for part in message.get("content") or []:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif isinstance(part, str):
            parts.append(part)
    return "".join(parts)


def _tool_started(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_key, payload = _tool_call_payload(data)
    kind = _tool_kind(raw_key)
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    call_id = str(data.get("call_id") or "tool")
    env = _envelope(
        "tool_use",
        id=call_id,
        tool=kind,
        status=STATUS_RUNNING,
        title=_tool_title(raw_key, kind, args),
    )
    if kind == TOOL_FILE_EDIT:
        env["path"] = str(args.get("path") or "")
        env["additions"] = 0
        env["deletions"] = 0
        stream = args.get("streamContent")
        if isinstance(stream, str) and stream:
            env["preview"] = stream[:4000]
    else:
        env["command"] = str(
            args.get("command") or args.get("cmd") or args.get("path") or ""
        )
    return [env]


def _tool_completed(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_key, payload = _tool_call_payload(data)
    kind = _tool_kind(raw_key)
    call_id = str(data.get("call_id") or "tool")
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    success = result.get("success") if isinstance(result.get("success"), dict) else None
    is_error = success is None and ("error" in result or "failure" in result)

    res_env: Dict[str, Any] = _envelope(
        "tool_result",
        id=call_id,
        status=STATUS_ERROR if is_error else STATUS_DONE,
    )
    dur = _duration_ms(payload, data)
    if dur is not None:
        res_env["durationMs"] = dur

    envelopes = [res_env]

    if raw_key == "editToolCall" and success is not None:
        added = int(success.get("linesAdded") or 0)
        removed = int(success.get("linesRemoved") or 0)
        res_env["additions"] = added
        res_env["deletions"] = removed
        envelopes.append(
            file_diff(
                path=str(success.get("path") or ""),
                before=str(success.get("beforeFullFileContent") or ""),
                after=str(success.get("afterFullFileContent") or ""),
                diff=str(success.get("diffString") or ""),
                added=added,
                removed=removed,
            )
        )
    elif kind == TOOL_SHELL and success is not None:
        out = success.get("content") or success.get("output") or success.get("stdout")
        if isinstance(out, str) and out:
            res_env["output"] = _clip(out, MAX_OUTPUT_CHARS)
    elif is_error:
        err = result.get("error") or result.get("failure")
        if err:
            res_env["output"] = _clip(str(err), MAX_OUTPUT_CHARS)

    return envelopes


def normalize_harness(event_key: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Translate one cursor-agent stream-json event into canonical envelopes.

    Args:
        event_key: ``"{type}.{subtype}"`` or bare ``"{type}"`` as produced by
            :func:`runner.event_key` (e.g. ``"tool_call.completed"``).
        data: The parsed JSON object for that stream line.

    Returns:
        Zero or more canonical envelopes. A completed edit yields TWO: the
        ``tool_result`` card plus a ``file_diff`` for the code pane.
    """
    key = (event_key or "").strip()
    data = data if isinstance(data, dict) else {}

    if key == "system.init":
        return [
            lifecycle(
                "run.started",
                model=data.get("model"),
                cwd=data.get("cwd"),
                harness_session_id=data.get("session_id"),
            )
        ]

    if key == "user" or key.startswith("user."):
        return []  # task echo — the caller already knows its own input

    if key.startswith("thinking"):
        text = data.get("text")
        if isinstance(text, str) and text:
            return [lifecycle("reasoning", text=text)]
        return []

    if key == "tool_call.started":
        return _tool_started(data)

    if key == "tool_call.completed":
        return _tool_completed(data)

    if key == "assistant" or key.startswith("assistant."):
        text = _assistant_text(data)
        return [content_delta(text)] if text else []

    if key.startswith("result"):
        if data.get("is_error"):
            return [
                lifecycle(
                    "run.failed",
                    status="failed",
                    error=data.get("result"),
                    duration_ms=data.get("duration_ms"),
                    usage=data.get("usage"),
                )
            ]
        return [
            lifecycle(
                "run.completed",
                status="completed",
                duration_ms=data.get("duration_ms"),
                usage=data.get("usage"),
            )
        ]

    if key == "harness.error":
        return [
            lifecycle(
                "run.failed",
                status="failed",
                error=data.get("error") or "harness error",
                timeout=bool(data.get("timeout")),
            )
        ]

    # Unknown event: opaque passthrough so nothing is silently dropped.
    return [lifecycle("passthrough", name=key, data=data)]


# ---------------------------------------------------------------------------
# ACP (Agent Client Protocol) normalization — cursor-agent acp
# ---------------------------------------------------------------------------
# Field names below match CAPTURED session/update payloads from a real
# `cursor-agent acp` run (2026-07-02, cursor-agent 2026.07.01-777f564), not
# the spec. Observed variants (`sessionUpdate` discriminator):
#
#   available_commands_update {availableCommands: [...]}          → (noise)
#   session_info_update       {title}                             → (noise)
#   agent_thought_chunk       {content: {type: "text", text}}     → reasoning
#   agent_message_chunk       {content: {type: "text", text}}     → content
#   tool_call        {toolCallId, title, kind: "read"|"edit"|"execute",
#                     status: "pending", rawInput: {command?}}    → tool_use
#   tool_call_update {toolCallId, status: "in_progress"}          → (skip)
#   tool_call_update {toolCallId, status: "completed",
#                     rawOutput: {content} | {exitCode, stdout, stderr},
#                     content: [{type: "diff", path, oldText, newText}]}
#                                            → tool_result (+ file_diff)
#
# New-file quirk (observed): the diff content for a brand-new file arrives as
# oldText="-- /dev/null\n" and newText prefixed with a "++ b/<path>" line —
# fragments of a diff header leaking into the before/after fields. Stripped
# by _clean_acp_diff().

_ACP_EDIT_KINDS = {"edit", "write", "delete"}
_ACP_NOISE_UPDATES = {"available_commands_update", "session_info_update", "plan"}


def unified_diff_text(before: str, after: str, path: str) -> Tuple[str, int, int]:
    """Unified diff text plus (added, removed) line counts."""
    rel = str(path).lstrip("/")
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return "".join(lines), added, removed


def _clean_acp_diff(old_text: str, new_text: str) -> Tuple[str, str]:
    """Strip the observed diff-header fragments from new-file diff content."""
    old = str(old_text or "")
    if old.strip() in ("-- /dev/null", "--- /dev/null"):
        old = ""
    new = str(new_text or "")
    first, _, rest = new.partition("\n")
    if first.startswith(("++ b/", "+++ b/")):
        new = rest
    return old, new


def _acp_text(data: Dict[str, Any]) -> str:
    """Text from an agent_message_chunk / agent_thought_chunk content block."""
    content = data.get("content")
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    if isinstance(content, str):
        return content
    return ""


class AcpNormalizer:
    """Stateful session/update → canonical-envelope mapper (one per run)."""

    def __init__(self) -> None:
        # toolCallId → {kind, title, command, started (monotonic)}
        self._calls: Dict[str, Dict[str, Any]] = {}

    # -- event entry point ---------------------------------------------------

    def normalize(self, event_key: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Translate one acp_runner event into canonical envelopes.

        Args:
            event_key: ``"acp.session" | "acp.update" | "acp.result" |
                "acp.error"`` as yielded by :func:`acp_runner.run_acp`.
            data: For ``acp.update``, the raw ``params.update`` payload.
        """
        data = data if isinstance(data, dict) else {}

        if event_key == "acp.session":
            return [
                lifecycle(
                    "run.started",
                    model=data.get("model"),
                    cwd=data.get("cwd"),
                    harness_session_id=data.get("sessionId"),
                )
            ]

        if event_key == "acp.result":
            stop = str(data.get("stopReason") or "")
            if stop == "end_turn":
                return [lifecycle("run.completed", status="completed", stop_reason=stop)]
            if stop == "cancelled":
                return [
                    lifecycle(
                        "run.failed",
                        status="failed",
                        error="run cancelled (session/cancel)",
                        cancelled=True,
                    )
                ]
            return [
                lifecycle(
                    "run.failed",
                    status="failed",
                    error=f"cursor-agent stopped: {stop or 'unknown stopReason'}",
                    stop_reason=stop,
                )
            ]

        if event_key == "acp.error":
            return [
                lifecycle(
                    "run.failed",
                    status="failed",
                    error=data.get("error") or "ACP error",
                    timeout=bool(data.get("timeout")),
                )
            ]

        if event_key == "acp.update":
            return self._update(data)

        # Unknown runner event: opaque passthrough (mirrors normalize_harness).
        return [lifecycle("passthrough", name=event_key, data=data)]

    # -- session/update variants ----------------------------------------------

    def _update(self, upd: Dict[str, Any]) -> List[Dict[str, Any]]:
        variant = str(upd.get("sessionUpdate") or "")

        if variant in _ACP_NOISE_UPDATES:
            return []

        if variant == "agent_thought_chunk":
            text = _acp_text(upd)
            return [lifecycle("reasoning", text=text)] if text else []

        if variant == "agent_message_chunk":
            text = _acp_text(upd)
            return [content_delta(text)] if text else []

        if variant == "tool_call":
            return self._tool_call(upd)

        if variant == "tool_call_update":
            return self._tool_call_update(upd)

        # Unknown variant: opaque passthrough so nothing is silently dropped.
        return [lifecycle("passthrough", name=f"acp.{variant or 'update'}", data=upd)]

    def _tool_call(self, upd: Dict[str, Any]) -> List[Dict[str, Any]]:
        call_id = str(upd.get("toolCallId") or "tool")
        raw_kind = str(upd.get("kind") or "")
        kind = TOOL_FILE_EDIT if raw_kind in _ACP_EDIT_KINDS else TOOL_SHELL
        raw_input = upd.get("rawInput") if isinstance(upd.get("rawInput"), dict) else {}
        title = str(upd.get("title") or "").strip() or (raw_kind or "Tool")

        state = {
            "kind": kind,
            "title": title,
            "command": str(raw_input.get("command") or ""),
            "started": time.monotonic(),
        }
        self._calls[call_id] = state

        env = _envelope(
            "tool_use",
            id=call_id,
            tool=kind,
            status=STATUS_RUNNING,
            title=title,
        )
        if kind == TOOL_FILE_EDIT:
            env["path"] = str(raw_input.get("path") or "")
            env["additions"] = 0
            env["deletions"] = 0
        else:
            env["command"] = state["command"]

        envelopes = [env]
        # Defensive: a tool_call that arrives already terminal.
        if str(upd.get("status") or "") in ("completed", "failed"):
            envelopes.extend(self._tool_call_update(upd))
        return envelopes

    def _tool_call_update(self, upd: Dict[str, Any]) -> List[Dict[str, Any]]:
        status = str(upd.get("status") or "")
        if status not in ("completed", "failed"):
            return []  # pending / in_progress — tool_use already emitted

        call_id = str(upd.get("toolCallId") or "tool")
        state = self._calls.get(call_id) or {}
        kind = state.get("kind") or TOOL_SHELL

        res_env: Dict[str, Any] = _envelope(
            "tool_result",
            id=call_id,
            status=STATUS_ERROR if status == "failed" else STATUS_DONE,
        )
        started = state.get("started")
        if isinstance(started, float):
            res_env["durationMs"] = int((time.monotonic() - started) * 1000)

        envelopes = [res_env]

        # File edits carry [{type: "diff", path, oldText, newText}] blocks.
        diff_blocks = [
            b
            for b in (upd.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "diff"
        ]
        total_added = total_removed = 0
        for block in diff_blocks:
            path = str(block.get("path") or "")
            before, after = _clean_acp_diff(block.get("oldText"), block.get("newText"))
            diff_text, added, removed = unified_diff_text(before, after, path)
            total_added += added
            total_removed += removed
            envelopes.append(
                file_diff(
                    path=path,
                    before=before,
                    after=after,
                    diff=diff_text,
                    added=added,
                    removed=removed,
                )
            )
        if diff_blocks:
            res_env["additions"] = total_added
            res_env["deletions"] = total_removed

        # Shell / read output for the tool_result card.
        raw_out = upd.get("rawOutput") if isinstance(upd.get("rawOutput"), dict) else {}
        if not diff_blocks and raw_out:
            out_parts = [
                str(raw_out.get(k))
                for k in ("stdout", "stderr", "content", "output")
                if isinstance(raw_out.get(k), str) and raw_out.get(k)
            ]
            if out_parts:
                res_env["output"] = _clip("\n".join(out_parts), MAX_OUTPUT_CHARS)
            if raw_out.get("exitCode") not in (None, 0):
                res_env["status"] = STATUS_ERROR

        return envelopes
