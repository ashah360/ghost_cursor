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
import re
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
# cursor-sdk normalization — sdk_runner.run_sdk event tuples
# ---------------------------------------------------------------------------
# SDKMessage shapes follow the official cursor-sdk docs (type discriminator:
# system / user / assistant / thinking / tool_call / status / task / request
# / usage). The envelope fields (type, call_id, name, status) are stable;
# tool_call ``args`` and ``result`` payloads are EXPLICITLY UNSTABLE upstream
# — everything below parses them defensively and never raises on an
# unexpected shape.

# Message types that never produce envelopes: the task echo, handshake
# metadata, and transient status pings.
_SDK_NOISE_TYPES = {"system", "user", "request", "status"}

# tool_call names that mean a file edit (vs shell/read). Names are unstable
# upstream, so this is a substring match, not an enum.
_SDK_EDIT_NAME_HINTS = ("edit", "write", "delete", "apply_patch", "applypatch")

_SDK_TERMINAL_TOOL_STATUSES = {"completed", "error", "failed", "cancelled"}


def _first_str(data: Dict[str, Any], *keys: str) -> str:
    """The first non-empty string value among ``keys``, else ""."""
    for key in keys:
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _sdk_tool_kind(name: str) -> str:
    lowered = str(name or "").lower()
    if any(hint in lowered for hint in _SDK_EDIT_NAME_HINTS):
        return TOOL_FILE_EDIT
    return TOOL_SHELL


def _sdk_tool_title(name: str, kind: str, args: Dict[str, Any]) -> str:
    path = _first_str(args, "path", "file_path", "filePath")
    if kind == TOOL_FILE_EDIT:
        return f"Editing {path}" if path else "File edit"
    cmd = _first_str(args, "command", "cmd")
    if cmd:
        return cmd.strip().splitlines()[0][:120]
    if "read" in str(name or "").lower() and path:
        return f"Reading {path}"
    return str(name or "").strip() or "Tool"


def _sdk_assistant_text(msg: Dict[str, Any]) -> str:
    """Concatenated text blocks from an assistant message, shape-tolerant."""
    message = msg.get("message")
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts: List[str] = []
    for block in content or []:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def _sdk_diff_entries(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort file_diff envelopes mined from a tool_call result.

    The payload schema is unstable, so this probes the shapes seen from
    cursor's edit tools (before/after full content, oldText/newText blocks,
    pre-rendered diff strings) and returns [] rather than guessing. Runs
    whose edits slip through entirely still land in files_changed via the
    git fallback (sdk_runner.git_fallback_diffs).
    """
    candidates: List[Dict[str, Any]] = [result]
    for key in ("success", "edit", "data"):
        nested = result.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)
    for key in ("content", "diffs", "edits", "files"):
        nested = result.get(key)
        if isinstance(nested, list):
            candidates.extend(b for b in nested if isinstance(b, dict))

    entries: List[Dict[str, Any]] = []
    for cand in candidates:
        path = _first_str(cand, "path", "file_path", "filePath")
        if not path:
            continue
        before = _first_str(cand, "beforeFullFileContent", "oldText", "before")
        after = _first_str(cand, "afterFullFileContent", "newText", "after")
        pre_rendered = _first_str(cand, "diffString", "diff")
        if not (before or after or pre_rendered):
            continue
        if before or after:
            diff_text, added, removed = unified_diff_text(before, after, path)
        else:
            diff_text = pre_rendered
            added = sum(
                1 for l in pre_rendered.splitlines()
                if l.startswith("+") and not l.startswith("+++")
            )
            removed = sum(
                1 for l in pre_rendered.splitlines()
                if l.startswith("-") and not l.startswith("---")
            )
        if not diff_text:
            continue
        entries.append(
            file_diff(
                path=path,
                before=before,
                after=after,
                diff=diff_text,
                added=added,
                removed=removed,
            )
        )
    return entries


def _sdk_output_text(result: Any) -> str:
    """Shell/read output mined from a tool_call result, shape-tolerant."""
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return "" if result is None else str(result)
    parts = [
        str(result.get(k))
        for k in ("stdout", "stderr", "content", "output", "text", "error")
        if isinstance(result.get(k), str) and result.get(k)
    ]
    return "\n".join(parts)


class SdkNormalizer:
    """Stateful sdk_runner event → canonical-envelope mapper (one per run).

    Stateful because tool_call completion messages must inherit the
    kind/title resolved when the call started, and because durations are
    measured client-side.
    """

    def __init__(self) -> None:
        # call_id → {kind, title, command, started (monotonic)}
        self._calls: Dict[str, Dict[str, Any]] = {}
        # call_ids whose "running" tool_use envelope was already emitted
        # (the SDK may re-emit running-status messages as args accumulate).
        self._started: set = set()
        # Tail of the FINAL contiguous content segment: assistant deltas
        # accumulate here and any tool/thinking activity resets it, so at a
        # "finished" result it holds only what the agent streamed last.
        # Scanned for transport-error signatures (see _TRANSPORT_ERROR_RE) —
        # observed live under ACP and kept under the SDK: a dying model
        # stream can be narrated as ordinary content before a clean finish.
        self._final_content_tail = ""

    # -- event entry point ---------------------------------------------------

    def normalize(self, event_key: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Translate one sdk_runner event into canonical envelopes.

        Args:
            event_key: ``"sdk.session" | "sdk.message" | "sdk.reattached" |
                "sdk.result" | "sdk.error"`` as yielded by
                :func:`sdk_runner.run_sdk`.
            data: For ``sdk.message``, the SDKMessage as a plain dict.
        """
        data = data if isinstance(data, dict) else {}

        if event_key == "sdk.session":
            return [
                lifecycle(
                    "run.started",
                    model=data.get("model"),
                    cwd=data.get("cwd"),
                    harness_session_id=data.get("agentId"),
                )
            ]

        if event_key == "sdk.reattached":
            # Transparent stream recovery: JSONL-log signal only. The fold
            # ignores unknown lifecycle events, so nothing user-visible
            # changes — no synthetic messages, no re-prompt.
            return [
                lifecycle(
                    "stream.reattached",
                    offset=data.get("offset"),
                    attempt=data.get("attempt"),
                )
            ]

        if event_key == "sdk.result":
            status = str(data.get("status") or "")
            if status == "finished":
                transport_error = _transport_error_line(self._final_content_tail)
                if transport_error:
                    # The "clean" finish is a lie: the stream died and the
                    # error was narrated as the last message. Classify the
                    # run failed, error first-class.
                    return [
                        lifecycle(
                            "run.failed",
                            status="failed",
                            error=transport_error,
                            transport_error=True,
                            run_status=status,
                        )
                    ]
                return [lifecycle("run.completed", status="completed",
                                  run_status=status)]
            if status == "cancelled":
                return [
                    lifecycle(
                        "run.failed",
                        status="failed",
                        error="run cancelled (run.cancel)",
                        cancelled=True,
                    )
                ]
            if status == "expired":
                return [
                    lifecycle(
                        "run.failed",
                        status="failed",
                        error="cursor run expired",
                        timeout=True,
                        run_status=status,
                    )
                ]
            return [
                lifecycle(
                    "run.failed",
                    status="failed",
                    error=str(data.get("error") or "")
                    or f"cursor run ended with status: {status or 'unknown'}",
                    run_status=status,
                )
            ]

        if event_key == "sdk.error":
            return [
                lifecycle(
                    "run.failed",
                    status="failed",
                    error=data.get("error") or "cursor-sdk error",
                    timeout=bool(data.get("timeout")),
                )
            ]

        if event_key == "sdk.message":
            return self._message(data)

        # Unknown runner event: opaque passthrough so nothing is dropped.
        return [lifecycle("passthrough", name=event_key, data=data)]

    # -- SDKMessage types ------------------------------------------------------

    def _message(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        mtype = str(msg.get("type") or "")

        if mtype in _SDK_NOISE_TYPES:
            return []

        if mtype == "thinking":
            self._final_content_tail = ""
            text = msg.get("text")
            if isinstance(text, str) and text:
                return [lifecycle("reasoning", text=text)]
            return []

        if mtype == "assistant":
            text = _sdk_assistant_text(msg)
            if text:
                self._final_content_tail = (
                    self._final_content_tail + text
                )[-_TRANSPORT_SCAN_CHARS:]
            return [content_delta(text)] if text else []

        if mtype == "tool_call":
            self._final_content_tail = ""
            return self._tool_call(msg)

        if mtype == "usage":
            # Log-only: token accounting lands in the JSONL event log.
            return [lifecycle("usage", usage=msg.get("usage"))]

        if mtype == "task":
            return [lifecycle("task", status=msg.get("status"),
                              text=msg.get("text"))]

        # Unknown message type: opaque passthrough.
        return [lifecycle("passthrough", name=f"sdk.{mtype or 'message'}", data=msg)]

    def _tool_call(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        call_id = str(msg.get("call_id") or "tool")
        status = str(msg.get("status") or "")
        if status in _SDK_TERMINAL_TOOL_STATUSES:
            return self._tool_completed(call_id, status, msg)
        return self._tool_started(call_id, msg)

    def _tool_started(self, call_id: str, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        if call_id in self._started:
            return []  # re-emitted running message (args accumulating)
        self._started.add(call_id)

        name = str(msg.get("name") or "")
        args = msg.get("args") if isinstance(msg.get("args"), dict) else {}
        kind = _sdk_tool_kind(name)
        title = _sdk_tool_title(name, kind, args)
        state = {
            "kind": kind,
            "title": title,
            "command": _first_str(args, "command", "cmd"),
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
            env["path"] = _first_str(args, "path", "file_path", "filePath")
            env["additions"] = 0
            env["deletions"] = 0
        else:
            env["command"] = state["command"]
        return [env]

    def _tool_completed(
        self, call_id: str, status: str, msg: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        state = self._calls.get(call_id) or {}
        kind = state.get("kind") or _sdk_tool_kind(str(msg.get("name") or ""))

        envelopes: List[Dict[str, Any]] = []
        if call_id not in self._started:
            # Terminal message with no prior running message (observed on
            # very fast calls): synthesize the tool_use so the pair renders.
            envelopes.extend(self._tool_started(call_id, msg))

        is_error = status in ("error", "failed", "cancelled")
        res_env: Dict[str, Any] = _envelope(
            "tool_result",
            id=call_id,
            status=STATUS_ERROR if is_error else STATUS_DONE,
        )
        started = state.get("started")
        if isinstance(started, float):
            res_env["durationMs"] = int((time.monotonic() - started) * 1000)

        envelopes.append(res_env)

        result = msg.get("result")
        result_dict = result if isinstance(result, dict) else {}

        diffs = _sdk_diff_entries(result_dict) if kind == TOOL_FILE_EDIT else []
        if diffs:
            res_env["additions"] = sum(d["added"] for d in diffs)
            res_env["deletions"] = sum(d["removed"] for d in diffs)
            envelopes.extend(diffs)
        else:
            out = _sdk_output_text(result)
            if out:
                res_env["output"] = _clip(out, MAX_OUTPUT_CHARS)
            exit_code = result_dict.get("exitCode", result_dict.get("exit_code"))
            if exit_code not in (None, 0):
                res_env["status"] = STATUS_ERROR

        return envelopes


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

# Transport-level failure signatures. Observed live (2026-07-03): when the
# model stream behind cursor-agent dies mid-run, cursor-agent streams the
# error text as an ordinary ``agent_message_chunk`` (e.g. "RetriableError:
# [canceled] http/2 stream closed with error code CANCEL (0x8)") and STILL
# resolves ``session/prompt`` with ``stopReason: "end_turn"`` — a clean
# completion from ACP's point of view. Detect those signatures in the FINAL
# contiguous content segment (anything the agent streamed after its last
# tool/thought activity) so the run is classified failed instead of
# completed. A signature that appears earlier and is followed by more
# activity means cursor recovered — that run stays a normal completion.
_TRANSPORT_ERROR_RE = re.compile(
    r"(?:\bRetriableError\b|\bConnectError\b|\bECONNRESET\b"
    r"|http/2 stream closed|stream closed with error code"
    r"|\bconnection (?:closed|reset|refused|lost)\b|socket hang up)",
    re.IGNORECASE,
)
# How much trailing content to keep for the transport-error scan; error
# chunks are short, this only needs to survive delta splitting.
_TRANSPORT_SCAN_CHARS = 4_000


def _transport_error_line(text: str) -> str:
    """The line of ``text`` carrying a transport-error signature, or ""."""
    match = _TRANSPORT_ERROR_RE.search(text)
    if not match:
        return ""
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    return text[start : end if end >= 0 else len(text)].strip()


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
        # Tail of the FINAL contiguous content segment: agent_message_chunk
        # deltas accumulate here and any tool/thought activity resets it, so
        # at end_turn it holds only what the agent streamed last. Scanned
        # for transport-error signatures (see _TRANSPORT_ERROR_RE).
        self._final_content_tail = ""

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
                transport_error = _transport_error_line(self._final_content_tail)
                if transport_error:
                    # The "clean" end_turn is a lie: the stream died and
                    # cursor-agent narrated the transport error as its last
                    # message. Classify the run failed, error first-class.
                    return [
                        lifecycle(
                            "run.failed",
                            status="failed",
                            error=transport_error,
                            transport_error=True,
                            stop_reason=stop,
                        )
                    ]
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
            self._final_content_tail = ""  # thought after content: not final
            text = _acp_text(upd)
            return [lifecycle("reasoning", text=text)] if text else []

        if variant == "agent_message_chunk":
            text = _acp_text(upd)
            if text:
                self._final_content_tail = (
                    self._final_content_tail + text
                )[-_TRANSPORT_SCAN_CHARS:]
            return [content_delta(text)] if text else []

        if variant == "tool_call":
            self._final_content_tail = ""  # tool activity: content wasn't final
            return self._tool_call(upd)

        if variant == "tool_call_update":
            self._final_content_tail = ""
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
