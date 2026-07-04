"""Plain-text renderings for every ghost_cursor tool result (v0.4).

Target formats: ``/tmp/gc-v04-formats.md`` (schematic T-050, approved
2026-07-02). Construction is pure python f-string templating — no LLM calls
anywhere in this module (the aux-model peek line is deliberately absent; see
the TODO at the ``working on:`` line).

House rules (from the spec):

* no ``success`` booleans — the ``status:`` line carries the state;
* labeled header lines, then plain english, then raw fenced ```` ```diff ````
  blocks — a diff is NEVER JSON-escaped;
* reasoning/thinking content NEVER appears in status output; peek/summary
  lines clip hard; full text only via ``cursor_events`` (and even there each
  event's inline text clips at 2 KB with a pointer to the JSONL log);
* every clip carries an explicit marker + where to find the full content;
* TSV (header row, aligned) for ``cursor_list`` rows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import eventlog as _eventlog

# ---------------------------------------------------------------------------
# Bounds (all from the format spec)
# ---------------------------------------------------------------------------

# 'working on:' line — first 100 chars of the task string.
WORKING_ON_CHARS = 100
# status peek lines (summary of a finished run shown by cursor_status).
STATUS_PEEK_CHARS = 200
# cursor's final summary in a completion message.
SUMMARY_CHARS = 2000
# 'recent' bullets: last ~5 event titles, each clipped to 120 chars.
RECENT_BULLETS = 5
BULLET_CHARS = 120
# completion diff bounding: full diffs only when <=5 files AND <=200 total
# diff lines; otherwise counts+paths for all + diffs for the largest 3.
FULL_DIFF_MAX_FILES = 5
FULL_DIFF_MAX_LINES = 200
LARGEST_DIFF_COUNT = 3
# cursor_events: per-event inline content clip + total response cap.
EVENT_INLINE_CLIP = 2048
EVENTS_RESPONSE_CAP = 20 * 1024
EVENTS_TRUNCATION_NOTE = "page truncated at 20KB — narrow with limit/kind"
# progress digest: events-since-last-tick body caps (~5 lines / ~1KB).
DIGEST_MAX_EVENTS = 5
DIGEST_BODY_CAP = 1024

_STATUS_WORDS = {"A": "added", "M": "modified", "D": "deleted"}


# ---------------------------------------------------------------------------
# Small shared pieces
# ---------------------------------------------------------------------------

def secs(seconds: Any) -> str:
    """'43s' — whole seconds, template-friendly."""
    try:
        return f"{int(float(seconds))}s"
    except (TypeError, ValueError):
        return "—"


def clip(text: Any, limit: int, where: str = "cursor_events") -> str:
    """Hard clip with an explicit marker + where the full content lives."""
    s = str(text or "")
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [+{len(s) - limit} chars — full text via {where}]"


def _clip_kb(text: str, limit: int = EVENT_INLINE_CLIP) -> str:
    """The 2 KB per-event inline clip, marker per the format spec."""
    if len(text) <= limit:
        return text
    clipped_k = max(1, round((len(text) - limit) / 1024))
    return text[:limit] + (
        f"… [clipped {clipped_k}k — full event in the JSONL log]"
    )


def _one_line(text: Any) -> str:
    return " ".join(str(text or "").split())


def files_inline(files: List[Dict[str, Any]], max_files: int = 8) -> str:
    """'calc.py +4 −0, test.py +12 −0' (counts only, never diffs)."""
    parts = [
        f"{f.get('path', '?')} +{f.get('added', 0)} −{f.get('removed', 0)}"
        for f in files[:max_files]
    ]
    if len(files) > max_files:
        parts.append(f"… {len(files) - max_files} more")
    return ", ".join(parts)


def _file_rows(files: List[Dict[str, Any]]) -> str:
    """One row per file: 'calc.py  +4 −0  modified'."""
    return "\n".join(
        f"{f.get('path', '?')}  +{f.get('added', 0)} −{f.get('removed', 0)}  "
        f"{_STATUS_WORDS.get(str(f.get('status') or 'M'), str(f.get('status') or 'modified'))}"
        for f in files
    )


def _diff_fence(diff: str) -> str:
    """A raw fenced diff block. NEVER JSON-escape a diff."""
    body = diff.rstrip("\n")
    return f"```diff\n{body}\n```"


# ---------------------------------------------------------------------------
# cursor_create_session / cursor_send_message
# ---------------------------------------------------------------------------

def create_session_ack(name: str, repo: str, model: Optional[str]) -> str:
    return (
        f"session: {name}\n"
        f"repo: {repo} · model: {model or 'default'}\n"
        "created. send work with cursor_send_message."
    )


def send_ack(name: str, interrupted: bool = False) -> str:
    lines = [
        f"sent to {name} · running in background",
        "result auto-delivers; cursor_status polls; a second send "
        "interrupts + re-prompts.",
    ]
    if interrupted:
        lines.append(
            "note: interrupted mid-run — in-flight step discarded, "
            "continuing with context."
        )
    return "\n".join(lines)


def repo_busy(active_session: str, repo: str) -> str:
    active = active_session or "(handle pending)"
    return (
        f"cannot start: session '{active}' is already running in {repo} and "
        "two cursor runs on one working tree corrupt it. send the work to "
        f"that session with cursor_send_message('{active}', ...), watch it "
        f"with cursor_status('{active}'), or stop it first with "
        f"cursor_stop('{active}')."
    )


# ---------------------------------------------------------------------------
# cursor_status
# ---------------------------------------------------------------------------

def recent_bullets(events: List[Dict[str, Any]]) -> List[str]:
    """Template-rendered bullets from the last event titles, newest last.

    Reasoning and raw content chunks are excluded entirely (status output
    never carries thinking text); bullets clip at 120 chars.
    """
    bullets: List[str] = []
    for record in events:
        kind = _eventlog.display_kind(record)
        if kind == "tool_use":
            if str(record.get("tool") or "") == "shell":
                cmd = _one_line(record.get("command") or record.get("title"))
                bullets.append(f"ran `{cmd}`")
            else:
                title = _one_line(record.get("title") or record.get("path"))
                bullets.append(title or "tool call")
        elif kind == "file_diff":
            bullets.append(
                f"edited {record.get('path', '?')} "
                f"+{record.get('added', 0)} −{record.get('removed', 0)}"
            )
    return [clip(b, BULLET_CHARS) for b in bullets[-RECENT_BULLETS:]]


def status_text(
    *,
    name: str,
    status: str,
    elapsed_s: Any,
    last_activity_s: Any,
    total_events: int,
    log_path: Optional[str],
    task: str,
    files: List[Dict[str, Any]],
    bullets: List[str],
    summary: str = "",
    error: str = "",
    note: str = "",
) -> str:
    """The read-only status view. NO inline diffs, NO reasoning content."""
    log = log_path or "(none yet)"
    last = f"{secs(last_activity_s)} ago" if last_activity_s is not None else "—"
    lines = [
        f"status: {status}",
        f"session: {name} · elapsed: {secs(elapsed_s)} · "
        f"last activity: {last}",
        f"events: {total_events} total · log: {log}",
        "",
        # TODO(peek design, T-050 §aux-model): replace this template-only
        # line with the aux-model one-line peek once that lands. Until then
        # it is the task string verbatim — never LLM-generated.
        f"working on: {clip(_one_line(task), WORKING_ON_CHARS)}",
    ]
    if summary:
        lines += ["", f"summary: {clip(_one_line(summary), STATUS_PEEK_CHARS)}"]
    if error:
        lines += ["", f"error: {_one_line(error)}"]
    if files:
        lines += [
            "",
            f"files so far ({len(files)}): {files_inline(files)}",
            f"diffs: cursor_events('{name}', kind='file_diff')",
        ]
    if bullets:
        lines += ["", "recent:"] + [f"- {b}" for b in bullets]
    if note:
        lines += ["", note]
    return "\n".join(lines)


def subscribe_ack(name: str, interval_s: float) -> str:
    """The 1-line cursor_subscribe ack."""
    if interval_s <= 0:
        return (
            f"session '{name}': progress updates off (completion delivery "
            "unaffected)."
        )
    return (
        f"session '{name}': progress updates every {secs(interval_s)} "
        "(applies to the running/next run; takes effect on the next tick)."
    )


# ---------------------------------------------------------------------------
# progress digest (periodic subscription updates)
# ---------------------------------------------------------------------------

def digest_text(
    *,
    name: str,
    n: int,
    status: str,
    elapsed_s: Any,
    last_activity_s: Any,
    files: List[Dict[str, Any]],
    pending_tool: str = "",
    pending_tool_s: Any = None,
    events: Optional[List[Dict[str, Any]]] = None,
    new_count: int = 0,
) -> str:
    """One periodic progress digest: the cursor_status-style header plus
    the events since the previous tick (cursor_events-style lines, capped
    at DIGEST_MAX_EVENTS lines / DIGEST_BODY_CAP chars). Tagged with the
    session name and the digest number so concurrent sessions stay
    distinguishable."""
    last = f"{secs(last_activity_s)} ago" if last_activity_s is not None else "—"
    lines = [
        f"cursor session '{name}' — progress update {n}",
        f"status: {status} · elapsed: {secs(elapsed_s)} · last activity: {last}",
    ]
    if files:
        lines.append(f"files so far ({len(files)}): {files_inline(files)}")
    if pending_tool:
        since = f" ({secs(pending_tool_s)})" if pending_tool_s is not None else ""
        lines.append(f"pending tool call: {pending_tool}{since}")
    lines.append("")
    if not events:
        lines.append("no new events since last update")
        return "\n".join(lines)

    lines.append(f"new events since last update ({new_count}):")
    used = 0
    rows = 0
    for record in events[-DIGEST_MAX_EVENTS:]:
        row = clip(
            f"{record.get('seq')}  {_eventlog.display_kind(record):<11}  "
            f"{_event_summary(record)}",
            200,
        )
        if used + len(row) + 1 > DIGEST_BODY_CAP:
            break
        lines.append(row)
        used += len(row) + 1
        rows += 1
    if new_count > rows:
        lines.append(f"… {new_count - rows} more — cursor_events('{name}')")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# completion message (all terminal states) + cursor_stop
# ---------------------------------------------------------------------------

def _bounded_diff_blocks(name: str, files: List[Dict[str, Any]]) -> List[str]:
    """Fenced diff blocks per the bounding rule (see module docstring)."""
    with_diffs = [f for f in files if str(f.get("diff") or "").strip()]
    total_lines = sum(
        len(str(f.get("diff") or "").splitlines()) for f in with_diffs
    )
    if len(files) <= FULL_DIFF_MAX_FILES and total_lines <= FULL_DIFF_MAX_LINES:
        return [_diff_fence(str(f["diff"])) for f in with_diffs]

    largest = sorted(
        with_diffs,
        key=lambda f: len(str(f.get("diff") or "").splitlines()),
        reverse=True,
    )[:LARGEST_DIFF_COUNT]
    blocks = [
        f"largest {len(largest)} diffs of {len(files)} files:"
    ] if largest else []
    blocks += [_diff_fence(str(f["diff"])) for f in largest]
    blocks.append(
        f"remaining diffs: cursor_events('{name}', kind='file_diff') "
        "or the event log above."
    )
    return blocks


def completion_text(
    *,
    name: str,
    status: str,
    elapsed_s: Any,
    repo: str,
    summary: str,
    files: List[Dict[str, Any]],
    error: str = "",
) -> str:
    """The terminal-state report (delivered message / in-turn fast finish)."""
    lines = [
        f"status: {status}",
        f"session: {name} · elapsed: {secs(elapsed_s)} · repo: {repo}",
    ]
    if error:
        lines += [
            "",
            f"run {status}: {_one_line(error)}. working-tree changes are "
            "intact. send another message to the same session to continue "
            "— transient failures usually succeed on retry.",
        ]
    if summary:
        clipped = summary[:SUMMARY_CHARS]
        note = (
            f"\n… [+{len(summary) - SUMMARY_CHARS} chars — full summary via "
            f"cursor_events('{name}', kind='content')]"
            if len(summary) > SUMMARY_CHARS
            else ""
        )
        lines += ["", "cursor's summary:", clipped + note]
    if files:
        lines += ["", f"files changed ({len(files)}):", _file_rows(files)]
        blocks = _bounded_diff_blocks(name, files)
        if blocks:
            lines += [""] + blocks
    elif not error:
        lines += ["", "no files were changed."]
    return "\n".join(lines)


def stop_text(
    *,
    name: str,
    status: str,
    elapsed_s: Any,
    files: List[Dict[str, Any]],
    already_finished: bool = False,
) -> str:
    if already_finished:
        second = (
            f"session: {name} · already finished ({secs(elapsed_s)} run) — "
            "nothing to stop"
        )
    else:
        second = (
            f"session: {name} · stopped after {secs(elapsed_s)} "
            "(native cancel)"
        )
    if files:
        work = (
            f"partial work: {len(files)} "
            f"file{'s' if len(files) != 1 else ''} — {files_inline(files)} "
            "(diffs in event log)"
        )
    else:
        work = "partial work: none"
    return "\n".join([
        f"status: {status}",
        second,
        work,
        "the session stays continuable — cursor_send_message picks the "
        "work back up with full context.",
    ])


# ---------------------------------------------------------------------------
# cursor_events
# ---------------------------------------------------------------------------

def _event_summary(record: Dict[str, Any]) -> str:
    kind = _eventlog.display_kind(record)
    if kind == "tool_use":
        tool = str(record.get("tool") or "tool")
        detail = _one_line(record.get("command") or record.get("title") or "")
        return f"{tool} `{detail}`" if detail else tool
    if kind == "tool_result":
        status = str(record.get("status") or "done")
        dur = record.get("durationMs")
        suffix = f" ({dur / 1000:.1f}s)" if isinstance(dur, (int, float)) else ""
        head = _one_line(record.get("output") or "")
        head = f" {head}" if head else ""
        return clip(f"→ {status}{head}{suffix}", 160)
    if kind == "file_diff":
        return (
            f"{record.get('path', '?')} +{record.get('added', 0)} "
            f"−{record.get('removed', 0)}"
        )
    if kind == "reasoning":
        return clip(f'"{_one_line(record.get("text"))}"', 160)
    if kind == "content":
        return clip(f'"{_one_line(record.get("delta"))}"', 160)
    if kind == "lifecycle":
        event = str(record.get("event") or "lifecycle")
        err = _one_line(record.get("error") or "")
        return clip(f"{event}: {err}" if err else event, 160)
    return kind


def _event_body(record: Dict[str, Any]) -> str:
    """Full event content (clipped at 2 KB), below the summary line.

    Diffs render as raw fenced blocks; long text/output as plain text. Short
    single-line content already fits the summary line and renders nothing.
    """
    kind = _eventlog.display_kind(record)
    if kind == "file_diff":
        diff = str(record.get("diff") or "")
        return _diff_fence(_clip_kb(diff)) if diff.strip() else ""
    field = {
        "reasoning": "text",
        "content": "delta",
        "tool_result": "output",
    }.get(kind)
    if not field:
        return ""
    text = str(record.get(field) or "")
    if len(text) <= 100 and "\n" not in text:
        return ""  # already fully visible in the summary line
    return _clip_kb(text)


def events_text(name: str, page: Dict[str, Any]) -> str:
    events = page.get("events") or []
    total = int(page.get("total_events") or 0)
    log = str(page.get("log_path") or "")
    kind = page.get("kind")

    if not events:
        scope = f" of kind '{kind}'" if kind else ""
        return (
            f"no events{scope} in the requested window for session "
            f"'{name}' ({total} events total · log: {log})."
        )

    first, last = events[0].get("seq"), events[-1].get("seq")
    matching = int(page.get("total_matching") or total)
    of = (
        f"{matching} matching (kind={kind}) of {total}" if kind else f"{total}"
    )
    chunks = [f"events {first}–{last} of {of} (log: {log})"]

    for gap in page.get("gaps") or []:
        gfirst, glast = gap.get("first_dropped_seq"), gap.get("last_dropped_seq")
        chunks.append(f"gap: seqs {gfirst}–{glast} compacted away")

    body_chunks: List[str] = []
    for record in events:
        seq = record.get("seq")
        line = f"{seq}  {_eventlog.display_kind(record):<11}  {_event_summary(record)}"
        body = _event_body(record)
        body_chunks.append(f"{line}\n{body}" if body else line)

    # Total response cap: drop whole trailing events once past ~20 KB.
    out: List[str] = list(chunks)
    used = sum(len(c) + 1 for c in out)
    truncated = False
    for chunk in body_chunks:
        if used + len(chunk) + 1 > EVENTS_RESPONSE_CAP:
            truncated = True
            break
        out.append(chunk)
        used += len(chunk) + 1
    if truncated:
        out.append(EVENTS_TRUNCATION_NOTE)
    return "\n".join(out)


def no_event_log(name: str) -> str:
    return (
        f"no events recorded for session '{name}' yet — the log appears "
        "with the first cursor_send_message activity."
    )


# ---------------------------------------------------------------------------
# cursor_list + unknown-session errors
# ---------------------------------------------------------------------------

_LIST_COLUMNS = ("session", "repo", "status", "elapsed", "files", "last_activity")


def list_text(rows: List[Dict[str, str]]) -> str:
    """TSV with a header row; fields space-padded so columns align."""
    table = [dict(zip(_LIST_COLUMNS, _LIST_COLUMNS))] + [
        {c: str(r.get(c, "—")) for c in _LIST_COLUMNS} for r in rows
    ]
    widths = {
        c: max(len(row[c]) for row in table) for c in _LIST_COLUMNS
    }
    return "\n".join(
        "\t".join(row[c].ljust(widths[c]) for c in _LIST_COLUMNS).rstrip()
        for row in table
    )


def empty_list(scope: str) -> str:
    if scope == "all":
        return "no cursor sessions exist yet. create one with cursor_create_session."
    return (
        "no cursor sessions in this hermes session. create one with "
        "cursor_create_session (scope='all' shows every session)."
    )


def unknown_session(identifier: str, rows: List[Dict[str, str]]) -> str:
    if rows:
        return (
            f"no session named '{identifier}'. sessions in this hermes "
            f"session:\n{list_text(rows)}\n(scope='all' shows every session)"
        )
    return (
        f"no session named '{identifier}', and this hermes session has no "
        "sessions yet. create one with cursor_create_session, or pass "
        "scope='all' to cursor_list to see every session."
    )
