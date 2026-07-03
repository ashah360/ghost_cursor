"""Tests for the ghost_cursor plugin (v0.4 named-session surface).

Seven tools: ``cursor_create_session`` / ``cursor_send_message`` /
``cursor_status`` / ``cursor_stop`` / ``cursor_events`` / ``cursor_list``
plus the deprecated ``cursor_start`` shim — all keyed on adjective-adjective-
noun session names (cursor UUIDs resolve as aliases). Every tool returns
plain text (labeled headers, prose, raw fenced diffs, TSV) — never JSON.
Covered here:

* ``events.AcpNormalizer`` — session/update → canonical envelope mapping
  against a CAPTURED fixture (``fixtures/acp_session_updates.jsonl``, real
  ``cursor-agent acp`` notifications, 2026-07-02).
* The tool handlers — session lifecycle (create → lazy first send → status
  → stop/follow-up), the read-only guarantee of ``cursor_status``, the
  same-repo concurrency guard, name minting + collision retry + UUID alias
  resolution, ``cursor_events`` paging (tail defaults, negative offsets,
  kind filter, 2KB inline clip, 20KB response cap), ``cursor_list`` TSV +
  scoping, model threading, completion delivery on the shared
  async-delegation rail, and actionable prose errors for bogus/expired
  handles. The ACP layer is replayed with fast deterministic fakes (no live
  cursor-agent).
* ``handles.py`` — the persistent handle table (explicit lookup only; the
  v0.2 auto-resume heuristic is gone by design).
* ``acp_runner.run_acp`` — real subprocess round-trips against a fake ACP
  server (happy path, native session/cancel, silent-hang→inactivity kill,
  slow-but-active run surviving past the old wall limit, max-wall ceiling
  for runaway streams, dead binary, session/load resume + fallback).
* The legacy ``--print`` runner + ``normalize_harness`` mapping (kept as
  fallback/reference — must stay importable and correct).

No live cursor-agent runs.
"""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import plugins.ghost_cursor as gc
from plugins.ghost_cursor import eventlog as gc_eventlog
from plugins.ghost_cursor import (
    CREATE_TOOL_NAME,
    CURSOR_CREATE_SCHEMA,
    CURSOR_EVENTS_SCHEMA,
    CURSOR_LIST_SCHEMA,
    CURSOR_SEND_SCHEMA,
    CURSOR_START_SCHEMA,
    CURSOR_STATUS_SCHEMA,
    CURSOR_STOP_SCHEMA,
    EVENTS_TOOL_NAME,
    LIST_TOOL_NAME,
    SEND_TOOL_NAME,
    START_TOOL_NAME,
    STATUS_TOOL_NAME,
    STOP_TOOL_NAME,
    TOOLSET,
    _handle_cursor_create_session,
    _handle_cursor_events,
    _handle_cursor_list,
    _handle_cursor_send_message,
    _handle_cursor_start,
    _handle_cursor_status,
    _handle_cursor_stop,
    check_cursor_available,
    cursor_create_session,
    cursor_events,
    cursor_list,
    cursor_send_message,
    cursor_start,
    cursor_status,
    cursor_stop,
    register,
)
from plugins.ghost_cursor import acp_runner as gc_acp
from plugins.ghost_cursor import events as gc_events
from plugins.ghost_cursor import handles as gc_handles
from plugins.ghost_cursor import jobs as gc_jobs
from plugins.ghost_cursor import names as gc_names
from plugins.ghost_cursor import render as gc_render
from plugins.ghost_cursor import runner as gc_runner

FIXTURE = Path(__file__).parent / "fixtures" / "cursor_stream.jsonl"
ACP_FIXTURE = Path(__file__).parent / "fixtures" / "acp_session_updates.jsonl"


def _prepend_path_env(stub_dir):
    """subprocess_env replacement that puts a stub bin dir FIRST on PATH
    (keeping the rest, so /bin utilities inside stub scripts still work)."""
    import os

    def _env():
        env = dict(os.environ)
        env["PATH"] = f"{stub_dir}:{env.get('PATH', '')}"
        return env

    return _env


# ---------------------------------------------------------------------------
# ACP fixture replay helpers
# ---------------------------------------------------------------------------

def _acp_fixture_updates():
    """The raw session/update payloads (params.update) from the capture."""
    return [
        json.loads(line)["params"]["update"]
        for line in ACP_FIXTURE.read_text().splitlines()
        if line.strip()
    ]


def _acp_replay_events():
    """A full run's worth of acp_runner events built from the capture."""
    events = [(
        "acp.session",
        {"sessionId": "s-fixture", "cwd": "/tmp/acp_probe/repo", "model": "fake-model"},
    )]
    events.extend(("acp.update", upd) for upd in _acp_fixture_updates())
    events.append(("acp.result", {"stopReason": "end_turn"}))
    return events


def _replay_acp(*_args, **_kwargs):
    yield from _acp_replay_events()


def _normalize_all(events):
    normalizer = gc_events.AcpNormalizer()
    envs = []
    for key, obj in events:
        envs.extend(normalizer.normalize(key, obj))
    return envs


# ---------------------------------------------------------------------------
# events.AcpNormalizer — captured session/update payloads → envelopes
# ---------------------------------------------------------------------------

class TestAcpNormalizer:
    def _all_envelopes(self):
        return _normalize_all(_acp_replay_events())

    def test_every_envelope_carries_ghost_source(self):
        envs = self._all_envelopes()
        assert envs, "fixture produced no envelopes"
        assert all(e.get("source") == "ghost" for e in envs)

    def test_noise_updates_produce_no_envelopes(self):
        normalizer = gc_events.AcpNormalizer()
        for upd in _acp_fixture_updates():
            if upd["sessionUpdate"] in ("available_commands_update", "session_info_update"):
                assert normalizer.normalize("acp.update", upd) == []

    def test_agent_thought_chunk_maps_to_reasoning(self):
        reasoning = [
            e for e in self._all_envelopes()
            if e["kind"] == "lifecycle" and e.get("event") == "reasoning"
        ]
        assert reasoning
        joined = "".join(e["text"] for e in reasoning)
        assert "notes.txt" in joined  # real captured thought text

    def test_agent_message_chunk_maps_to_content(self):
        content = [e for e in self._all_envelopes() if e["kind"] == "content"]
        assert any("subtract" in e["delta"] for e in content)

    def test_tool_call_maps_to_tool_use_running(self):
        uses = [e for e in self._all_envelopes() if e["kind"] == "tool_use"]
        # capture: read + edit (run 1), execute + edit (run 2)
        assert len(uses) == 4
        assert all(u["status"] == "running" for u in uses)
        edits = [u for u in uses if u["tool"] == "file-edit"]
        shells = [u for u in uses if u["tool"] == "shell"]
        assert len(edits) == 2 and len(shells) == 2
        assert any(u.get("command") == "ls -la" for u in shells)

    def test_tool_call_update_completed_maps_to_tool_result(self):
        results = [e for e in self._all_envelopes() if e["kind"] == "tool_result"]
        assert len(results) == 4
        assert all(r["status"] == "done" for r in results)
        # execute output surfaced from rawOutput.stdout
        assert any("calc.py" in str(r.get("output", "")) for r in results)

    def test_edit_diff_content_yields_file_diff_modified(self):
        diffs = [e for e in self._all_envelopes() if e["kind"] == "file_diff"]
        assert len(diffs) == 2
        calc = next(d for d in diffs if d["path"].endswith("calc.py"))
        assert calc["status"] == "M"
        assert "multiply" in calc["after"]
        assert "multiply" not in calc["before"]
        assert calc["added"] == 8 and calc["removed"] == 0
        assert "+def multiply(a, b):" in calc["diff"]

    def test_new_file_diff_header_quirk_is_stripped(self):
        # Captured new-file diff arrives as oldText="-- /dev/null\n" and
        # newText prefixed with "++ b/<path>\n" — must clean to A-status.
        diffs = [e for e in self._all_envelopes() if e["kind"] == "file_diff"]
        notes = next(d for d in diffs if d["path"].endswith("notes.txt"))
        assert notes["status"] == "A"
        assert notes["before"] == ""
        assert notes["after"] == "hello from acp"
        assert "/dev/null" not in notes["after"]
        assert "++ b/" not in notes["after"]

    def test_in_progress_update_emits_nothing(self):
        normalizer = gc_events.AcpNormalizer()
        assert normalizer.normalize(
            "acp.update",
            {"sessionUpdate": "tool_call_update", "toolCallId": "x", "status": "in_progress"},
        ) == []

    def test_session_event_maps_to_run_started(self):
        envs = gc_events.AcpNormalizer().normalize(
            "acp.session", {"sessionId": "s1", "cwd": "/w", "model": "m"}
        )
        assert envs == [{
            "source": "ghost", "kind": "lifecycle", "event": "run.started",
            "model": "m", "cwd": "/w", "harness_session_id": "s1",
        }]

    def test_end_turn_maps_to_run_completed(self):
        envs = gc_events.AcpNormalizer().normalize("acp.result", {"stopReason": "end_turn"})
        assert envs[0]["event"] == "run.completed"

    def test_cancelled_stop_reason_maps_to_run_failed(self):
        envs = gc_events.AcpNormalizer().normalize("acp.result", {"stopReason": "cancelled"})
        assert envs[0]["event"] == "run.failed"
        assert envs[0]["cancelled"] is True
        assert "cancel" in envs[0]["error"]

    def test_acp_error_timeout_maps_to_run_failed(self):
        envs = gc_events.AcpNormalizer().normalize(
            "acp.error", {"error": "ACP run timed out after 600s", "timeout": True}
        )
        assert envs[0]["event"] == "run.failed"
        assert envs[0]["timeout"] is True

    def test_unknown_update_variant_passes_through(self):
        envs = gc_events.AcpNormalizer().normalize(
            "acp.update", {"sessionUpdate": "mystery_update", "x": 1}
        )
        assert len(envs) == 1
        assert envs[0]["event"] == "passthrough"
        assert envs[0]["name"] == "acp.mystery_update"

    def test_failed_tool_status_maps_to_error_result(self):
        normalizer = gc_events.AcpNormalizer()
        normalizer.normalize("acp.update", {
            "sessionUpdate": "tool_call", "toolCallId": "t9",
            "title": "`boom`", "kind": "execute", "status": "pending",
            "rawInput": {"command": "boom"},
        })
        envs = normalizer.normalize("acp.update", {
            "sessionUpdate": "tool_call_update", "toolCallId": "t9",
            "status": "failed",
        })
        assert envs[0]["kind"] == "tool_result"
        assert envs[0]["status"] == "error"

    def test_nonzero_exit_code_marks_result_error(self):
        normalizer = gc_events.AcpNormalizer()
        normalizer.normalize("acp.update", {
            "sessionUpdate": "tool_call", "toolCallId": "t8",
            "title": "`false`", "kind": "execute", "status": "pending",
            "rawInput": {"command": "false"},
        })
        envs = normalizer.normalize("acp.update", {
            "sessionUpdate": "tool_call_update", "toolCallId": "t8",
            "status": "completed", "rawOutput": {"exitCode": 1, "stdout": "", "stderr": "nope"},
        })
        assert envs[0]["status"] == "error"
        assert "nope" in envs[0]["output"]


# ---------------------------------------------------------------------------
# Tool-test plumbing — deterministic gated fakes for the ACP layer
# ---------------------------------------------------------------------------

def _drain_completion_queue():
    """Pop and return every event currently on the shared completion queue."""
    from tools.process_registry import process_registry

    events = []
    while not process_registry.completion_queue.empty():
        try:
            events.append(process_registry.completion_queue.get_nowait())
        except Exception:
            break
    return events


@pytest.fixture
def clean_state(monkeypatch):
    """Fresh job registry + handle table + event-log writer state + drained
    completion queue."""
    gc_jobs.registry._reset_for_tests()
    _drain_completion_queue()
    monkeypatch.setattr(gc_handles, "_table", {})
    monkeypatch.setattr(gc_handles, "_loaded", False)
    gc_eventlog._reset_for_tests()
    yield gc_jobs.registry
    gc_jobs.registry._reset_for_tests()
    _drain_completion_queue()
    gc_eventlog._reset_for_tests()


def _wait_until(cond, timeout=10.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


def _gated_replay_factory(release, sid="s-run", early_edit=True, late_edit=True):
    """A cancel-aware replay held open on ``release``.

    Yields the ACP session (the handle), optionally one early file edit,
    then blocks until ``release`` is set — honoring ``cancel_check`` the way
    the real ``run_acp`` does (a cancel mid-run resolves with
    ``stopReason: "cancelled"``). After release: an optional second edit,
    a summary chunk, and a clean end_turn.
    """

    def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
               cancel_check=None, session_id=None, model=None):
        yield ("acp.session", {
            "sessionId": sid, "cwd": str(workdir),
            "model": model or "fake-model", "resumed": bool(session_id),
        })
        if early_edit:
            yield ("acp.update", {
                "sessionUpdate": "tool_call", "toolCallId": "t1",
                "title": "Edit File", "kind": "edit", "status": "pending",
                "rawInput": {},
            })
            yield ("acp.update", {
                "sessionUpdate": "tool_call_update", "toolCallId": "t1",
                "status": "completed",
                "content": [{"type": "diff", "path": f"{workdir}/f1.py",
                             "oldText": "a\n", "newText": "a\nb\n"}],
            })
        while not release.is_set():
            if cancel_check and cancel_check():
                yield ("acp.result", {"stopReason": "cancelled"})
                return
            time.sleep(0.01)
        if late_edit:
            yield ("acp.update", {
                "sessionUpdate": "tool_call", "toolCallId": "t2",
                "title": "Edit File", "kind": "edit", "status": "pending",
                "rawInput": {},
            })
            yield ("acp.update", {
                "sessionUpdate": "tool_call_update", "toolCallId": "t2",
                "status": "completed",
                "content": [{"type": "diff", "path": f"{workdir}/f2.py",
                             "oldText": "", "newText": "new\n"}],
            })
        yield ("acp.update", {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "all done"},
        })
        yield ("acp.result", {"stopReason": "end_turn"})

    return replay


class _AcpSequence:
    """Route successive run_acp calls to successive replay factories,
    recording each call's kwargs for assertion."""

    def __init__(self, *factories):
        self._factories = list(factories)
        self.calls = []

    def __call__(self, task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                 cancel_check=None, session_id=None, model=None):
        self.calls.append({
            "task": task, "workdir": str(workdir),
            "session_id": session_id, "model": model,
            "inactivity_timeout_s": inactivity_timeout_s,
            "max_wall_s": max_wall_s,
        })
        factory = self._factories.pop(0)
        return factory(task, workdir, inactivity_timeout_s=inactivity_timeout_s,
                       max_wall_s=max_wall_s, cancel_check=cancel_check,
                       session_id=session_id, model=model)


def _resolved(tmp_path):
    """The repo key the tools use (resolved workdir)."""
    return str(gc_runner.resolve_repo(str(tmp_path)))


def _job_for(sid):
    job = gc_jobs.registry.get_by_session(sid)
    assert job is not None, f"no job for handle {sid!r}"
    return job


def _session_name(sid):
    """The v0.4 session NAME behind a cursor sid (alias or live job)."""
    name = gc_handles.resolve(sid)
    if name:
        return name
    return _job_for(sid).session_name


def _assert_running_ack(ack):
    assert "running in background" in ack, f"not a running ack: {ack!r}"
    return ack


# ---------------------------------------------------------------------------
# cursor_start — new runs: handle out, running status, delivery on completion
# ---------------------------------------------------------------------------

class TestCursorStartShim:
    """cursor_start is a deprecated create+send shim — same background-run
    plumbing, session-name handles, plain-text acks."""

    def test_new_run_returns_running_ack_and_registers_named_handle(
        self, clean_state, monkeypatch, tmp_path
    ):
        release = threading.Event()
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(release, sid="s-new"))

        ack = cursor_start("add multiply", repo=str(tmp_path))
        try:
            _assert_running_ack(ack)
            assert "deprecated" in ack

            # Live job table is addressable by the cursor sid alias...
            job = _job_for("s-new")
            assert job.status == "running"
            assert job.deliver is True  # armed: outcome must arrive as a message
            # ...the ack names the minted session...
            name = job.session_name
            assert name and f"sent to {name}" in ack
            # ...and the persistent handle table has the entry immediately,
            # keyed by name, with the cursor sid recorded as the alias.
            entry = gc_handles.get(name)
            assert entry is not None
            assert entry["repo"] == _resolved(tmp_path)
            assert entry["status"] == "running"
            assert entry["cursor_session_id"] == "s-new"
            assert gc_handles.get("s-new") == entry  # UUID alias resolves
        finally:
            release.set()
        assert job.done_event.wait(10)

    def test_completion_is_delivered_with_full_result_and_session_id(
        self, clean_state, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:test:1")
        release = threading.Event()
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(release, sid="s-done"))

        _assert_running_ack(cursor_start("t", repo=str(tmp_path)))
        job = _job_for("s-done")
        release.set()
        assert job.done_event.wait(10)

        events = _drain_completion_queue()
        assert len(events) == 1
        evt = events[0]
        assert evt["type"] == "async_delegation"
        assert evt["delegation_id"] == job.session_name
        assert evt["session_key"] == "gw:test:1"  # routes back to the caller
        assert evt["status"] == "completed"
        assert evt["cursor_session_id"] == "s-done"
        # Continuation survives the async boundary: full result in payload.
        assert evt["result"]["success"] is True
        assert evt["result"]["session"] == job.session_name
        assert evt["result"]["session_id"] == "s-done"
        assert evt["result"]["files_changed_count"] == 2
        # The delivered text is the v0.4 completion format.
        assert "status: completed" in evt["summary"]
        assert f"session: {job.session_name}" in evt["summary"]
        assert "```diff" in evt["summary"]
        # The shared formatter renders it (this is what re-enters the chat).
        from tools.process_registry import format_process_notification
        text = format_process_notification(evt)
        assert job.session_name in text
        assert "completed" in text
        # The handle table settled to the terminal status.
        assert gc_handles.get("s-done")["status"] == "completed"

    def test_fixture_replay_run_aggregates_files_and_summary(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The captured ACP stream flows through the job aggregation: both
        edits land in files_changed with diffs, prose lands in summary."""
        monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)
        cursor_start("add multiply", repo=str(tmp_path))

        job = _job_for("s-fixture")
        assert job.done_event.wait(10)
        result = job.result
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["session_id"] == "s-fixture"
        assert result["resumed"] is False
        assert result["files_changed_count"] == 2
        by_path = {f["path"]: f for f in result["files_changed"]}
        calc = by_path["/tmp/acp_probe/repo/calc.py"]
        assert calc["added"] == 8 and calc["removed"] == 0
        assert calc["status"] == "M"
        assert "multiply" in calc["diff"]
        assert by_path["/tmp/acp_probe/repo/notes.txt"]["status"] == "A"
        assert "subtract" in result["summary"]

    def test_handshake_failure_reports_in_turn_and_never_delivers(
        self, clean_state, monkeypatch, tmp_path
    ):
        def failing(*_a, **_k):
            raise gc_acp.AcpError(
                "ACP initialize handshake with cursor-agent failed (boom)"
            )

        monkeypatch.setattr(gc_acp, "run_acp", failing)
        out = cursor_start("t", repo=str(tmp_path))
        assert "status: failed" in out
        assert "handshake" in out
        # Errors are prose sentences with a next step, not codes.
        assert "send another message" in out
        # Exactly-once: the tool result IS the report; nothing enqueued.
        assert _drain_completion_queue() == []

    def test_wedged_prehandshake_run_is_cancelled_and_reported(
        self, clean_state, monkeypatch, tmp_path
    ):
        """cursor-agent that never establishes a session gets cancelled and
        the tool returns an actionable failure instead of blocking forever."""
        monkeypatch.setattr(gc, "_HANDLE_WAIT_S", 0.3)

        def never_session(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                          cancel_check=None, session_id=None, model=None):
            while not (cancel_check and cancel_check()):
                time.sleep(0.01)
            return
            yield  # pragma: no cover — make it a generator

        monkeypatch.setattr(gc_acp, "run_acp", never_session)
        out = cursor_start("t", repo=str(tmp_path))
        assert "status: failed" in out
        assert "did not establish" in out
        job = gc_jobs.registry.list_jobs()[-1]
        assert job.cancel_event.is_set()
        assert job.done_event.wait(10)
        # Never armed → no delivery for the wedged attempt.
        assert _drain_completion_queue() == []

    def test_empty_task_is_a_clean_error(self, clean_state):
        assert "task is required" in cursor_start("   ")

    def test_missing_repo_is_a_clean_error(self, clean_state, monkeypatch):
        monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)
        assert "not an existing directory" in cursor_start(
            "t", repo="/definitely/not/a/dir"
        )

    def test_no_resolvable_workspace_repo_is_a_clean_error(
        self, clean_state, monkeypatch
    ):
        monkeypatch.setattr(gc, "_default_repo", lambda: None)
        out = cursor_start("t")
        assert "no workspace repo resolvable" in out
        assert gc.REPO_ENV_VAR in out

    def test_handler_maps_all_args(self, clean_state, monkeypatch, tmp_path):
        gc_handles.record("s-prior", repo=str(tmp_path), status="completed")
        seq = _AcpSequence(_gated_replay_factory(_preset_event(), sid="s-h"))
        monkeypatch.setattr(gc_acp, "run_acp", seq)
        out = _handle_cursor_start({
            "task": "handler task", "repo": str(tmp_path),
            "model": "handler-model", "session_id": "s-prior",
        })
        assert "no session named" not in out
        assert seq.calls[0]["task"] == "handler task"
        assert seq.calls[0]["session_id"] == "s-prior"
        assert seq.calls[0]["model"] == "handler-model"
        _job_for("s-h").done_event.wait(10)


def _preset_event():
    evt = threading.Event()
    evt.set()
    return evt


# ---------------------------------------------------------------------------
# cursor_start(session_id=...) — explicit resume (session/load), no heuristics
# ---------------------------------------------------------------------------

class TestCursorStartResume:
    def test_explicit_session_threads_the_resume_id_to_run_acp(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A pre-v0.4 handle (keyed by the raw cursor sid, no alias field)
        resumes by its own key via ACP session/load."""
        gc_handles.record("s-prior", repo=_resolved(tmp_path), status="completed")
        release = threading.Event()
        seq = _AcpSequence(_gated_replay_factory(release, sid="s-prior"))
        monkeypatch.setattr(gc_acp, "run_acp", seq)

        ack = cursor_start("continue it", repo=str(tmp_path), session="s-prior")
        try:
            _assert_running_ack(ack)
            assert "sent to s-prior" in ack
            # The resume id reached the ACP layer (session/load path).
            assert seq.calls[0]["session_id"] == "s-prior"
        finally:
            release.set()
        assert _job_for("s-prior").done_event.wait(10)

    def test_start_into_a_running_session_interrupts_like_send(
        self, clean_state, monkeypatch, tmp_path
    ):
        """v0.4: the shim is create+send, so targeting a live session takes
        send's interrupt + re-prompt semantics (the ack says so)."""
        release2 = threading.Event()
        seq = _AcpSequence(
            _gated_replay_factory(threading.Event(), sid="s-live"),
            _gated_replay_factory(release2, sid="s-live", early_edit=False),
        )
        monkeypatch.setattr(gc_acp, "run_acp", seq)
        _assert_running_ack(cursor_start("task A", repo=str(tmp_path)))
        first_job = _job_for("s-live")
        try:
            ack = cursor_start("task B", repo=str(tmp_path), session="s-live")
            _assert_running_ack(ack)
            assert "interrupted mid-run" in ack
            assert first_job.status == "cancelled"
            assert seq.calls[1]["session_id"] == "s-live"
            assert seq.calls[1]["task"] == "task B"
        finally:
            release2.set()
        assert gc_jobs.registry.get_by_session("s-live").done_event.wait(10)

    def test_expired_handle_falls_back_to_fresh_session(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The ACP layer falls back to session/new for an expired id; the
        session's alias is updated to the fresh sid — no crash, no hard
        failure."""

        def fallback_replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                            cancel_check=None, session_id=None, model=None):
            # Simulates acp_runner's session/load → session/new fallback.
            yield ("acp.session", {"sessionId": "s-fresh", "cwd": str(workdir),
                                   "model": "m", "resumed": False})
            yield ("acp.result", {"stopReason": "end_turn"})

        gc_handles.record("s-expired", repo=_resolved(tmp_path), status="completed")
        monkeypatch.setattr(gc_acp, "run_acp", fallback_replay)
        cursor_start("t", repo=str(tmp_path), session="s-expired")
        job = _job_for("s-fresh")
        assert job.done_event.wait(10)
        assert job.result["session_id"] == "s-fresh"
        assert job.result["resumed"] is False
        # The name still resolves, now aliased to the fresh cursor sid.
        assert gc_handles.get("s-expired")["cursor_session_id"] == "s-fresh"
        assert gc_handles.resolve("s-fresh") == "s-expired"


# ---------------------------------------------------------------------------
# Model override — explicit > config > cursor default
# ---------------------------------------------------------------------------

class TestModelParam:
    def _run_and_capture(self, monkeypatch, tmp_path, sid, **start_kwargs):
        seq = _AcpSequence(_gated_replay_factory(_preset_event(), sid=sid))
        monkeypatch.setattr(gc_acp, "run_acp", seq)
        cursor_start("t", repo=str(tmp_path), **start_kwargs)
        _job_for(sid).done_event.wait(10)
        return seq.calls[0]

    def test_explicit_model_reaches_the_runner(self, clean_state, monkeypatch, tmp_path):
        monkeypatch.setattr(gc, "_configured_model", lambda: None)
        call = self._run_and_capture(monkeypatch, tmp_path, "s-m1",
                                     model="composer-2.5")
        assert call["model"] == "composer-2.5"

    def test_configured_model_is_the_fallback(self, clean_state, monkeypatch, tmp_path):
        monkeypatch.setattr(gc, "_configured_model", lambda: "cfg-model")
        call = self._run_and_capture(monkeypatch, tmp_path, "s-m2")
        assert call["model"] == "cfg-model"

    def test_explicit_model_beats_config(self, clean_state, monkeypatch, tmp_path):
        monkeypatch.setattr(gc, "_configured_model", lambda: "cfg-model")
        call = self._run_and_capture(monkeypatch, tmp_path, "s-m3",
                                     model="explicit-model")
        assert call["model"] == "explicit-model"

    def test_no_model_anywhere_passes_none(self, clean_state, monkeypatch, tmp_path):
        monkeypatch.setattr(gc, "_configured_model", lambda: None)
        call = self._run_and_capture(monkeypatch, tmp_path, "s-m4")
        assert call["model"] is None

    def test_session_model_lands_on_the_job_and_the_handle(
        self, clean_state, monkeypatch, tmp_path
    ):
        release = threading.Event()
        monkeypatch.setattr(gc, "_configured_model", lambda: None)
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(release, sid="s-m5"))
        ack = cursor_start("t", repo=str(tmp_path), model="composer-x")
        try:
            _assert_running_ack(ack)
            job = _job_for("s-m5")
            assert job.model == "composer-x"  # reported by acp.session
            assert gc_handles.get("s-m5")["model"] == "composer-x"
        finally:
            release.set()
        assert _job_for("s-m5").done_event.wait(10)


# ---------------------------------------------------------------------------
# cursor_send — interrupt + re-prompt on the same handle
# ---------------------------------------------------------------------------

class TestCursorSendMessage:
    def test_send_midflight_interrupts_and_reprompts_same_session(
        self, clean_state, monkeypatch, tmp_path
    ):
        release1 = threading.Event()  # never set: run 1 ends only via cancel
        release2 = threading.Event()
        seq = _AcpSequence(
            _gated_replay_factory(release1, sid="s-send"),
            _gated_replay_factory(release2, sid="s-send", early_edit=False),
        )
        monkeypatch.setattr(gc_acp, "run_acp", seq)

        _assert_running_ack(cursor_start("task A", repo=str(tmp_path)))
        first_job = _job_for("s-send")
        name = first_job.session_name
        # Let the early edit land so the interrupted partial work is real.
        assert _wait_until(lambda: first_job.files)

        ack = cursor_send_message("s-send", "also add subtract")
        try:
            _assert_running_ack(ack)
            assert f"sent to {name}" in ack
            # The ack SAYS the live prompt was interrupted.
            assert "interrupted mid-run" in ack
            assert "continuing with context" in ack

            # The old run was settled via native cancel...
            assert first_job.status == "cancelled"
            assert first_job.result["files_changed_count"] == 1
            # ...its delivery suppressed (the send ack reported it) ...
            assert _drain_completion_queue() == []
            # ...and the re-prompt continued the SAME session with the message.
            assert seq.calls[1]["session_id"] == "s-send"
            assert seq.calls[1]["task"] == "also add subtract"
        finally:
            release2.set()

        second_job = gc_jobs.registry.get_by_session("s-send")
        assert second_job is not first_job
        assert second_job.session_name == name  # same named session
        assert second_job.done_event.wait(10)
        # The re-prompted run delivers its own completion normally.
        events = _drain_completion_queue()
        assert len(events) == 1
        assert events[0]["result"]["session_id"] == "s-send"
        assert events[0]["delegation_id"] == name

    def test_send_after_run_settled_is_a_plain_followup(
        self, clean_state, monkeypatch, tmp_path
    ):
        release2 = threading.Event()
        seq = _AcpSequence(
            _gated_replay_factory(_preset_event(), sid="s-f"),
            _gated_replay_factory(release2, sid="s-f", early_edit=False),
        )
        monkeypatch.setattr(gc_acp, "run_acp", seq)

        cursor_start("task A", repo=str(tmp_path))
        first_job = _job_for("s-f")
        assert first_job.done_event.wait(10)
        _drain_completion_queue()

        ack = cursor_send_message("s-f", "refine it")
        try:
            _assert_running_ack(ack)
            # No interruption happened — the run had already settled.
            assert "interrupted mid-run" not in ack
            assert seq.calls[1]["session_id"] == "s-f"
            assert seq.calls[1]["task"] == "refine it"
        finally:
            release2.set()
        assert gc_jobs.registry.get_by_session("s-f").done_event.wait(10)

    def test_first_message_on_a_fresh_session_is_the_task(
        self, clean_state, monkeypatch, tmp_path
    ):
        """cursor_create_session dispatches NOTHING; the first send spawns
        the ACP session with the message as the task."""
        seq = _AcpSequence(_gated_replay_factory(_preset_event(), sid="s-lazy"))
        monkeypatch.setattr(gc_acp, "run_acp", seq)

        ack = cursor_create_session(repo=str(tmp_path))
        name = ack.splitlines()[0].split("session: ")[1]
        assert seq.calls == []  # lazy: nothing dispatched yet
        assert gc_jobs.registry.get_by_name(name) is None
        assert gc_handles.get(name)["status"] == "created"

        cursor_send_message(name, "build the thing")
        assert seq.calls[0]["task"] == "build the thing"
        assert seq.calls[0]["session_id"] is None  # fresh, not a resume
        job = gc_jobs.registry.get_by_name(name)
        assert job is not None
        assert job.done_event.wait(10)
        assert gc_handles.get(name)["cursor_session_id"] == "s-lazy"

    def test_send_resolves_handle_from_persisted_table_after_restart(
        self, clean_state, monkeypatch, tmp_path
    ):
        """No live job (process restart) but the handle table knows the repo:
        send re-prompts the session instead of erroring."""
        gc_handles.record("s-old", repo=_resolved(tmp_path), status="cancelled",
                          model="recorded-model")
        release = threading.Event()
        seq = _AcpSequence(_gated_replay_factory(release, sid="s-old",
                                                 early_edit=False))
        monkeypatch.setattr(gc, "_configured_model", lambda: None)
        monkeypatch.setattr(gc_acp, "run_acp", seq)

        ack = cursor_send_message("s-old", "pick it back up")
        try:
            _assert_running_ack(ack)
            assert "sent to s-old" in ack
            assert seq.calls[0]["session_id"] == "s-old"
            # The recorded model is reused for the continuation.
            assert seq.calls[0]["model"] == "recorded-model"
        finally:
            release.set()
        assert _job_for("s-old").done_event.wait(10)

    def test_unknown_session_is_actionable_prose(self, clean_state):
        out = cursor_send_message("s-nope", "hello")
        assert "no session named 's-nope'" in out
        assert "cursor_create_session" in out

    def test_unknown_session_error_lists_scoped_sessions_as_tsv(
        self, clean_state, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:me")
        gc_handles.record("mine-alpha-fox", repo=str(tmp_path),
                          status="completed", session_key="gw:me")
        gc_handles.record("theirs-beta-owl", repo=str(tmp_path),
                          status="completed", session_key="gw:other")
        out = cursor_send_message("s-nope", "hello")
        assert "no session named 's-nope'" in out
        assert "mine-alpha-fox" in out
        assert "theirs-beta-owl" not in out  # scoped by default
        assert "scope='all'" in out

    def test_recorded_repo_gone_is_a_graceful_error(self, clean_state, tmp_path):
        gone = tmp_path / "was-here"
        gc_handles.record("s-gone", repo=str(gone), status="cancelled")
        out = cursor_send_message("s-gone", "hello")
        assert "no longer exists" in out

    def test_missing_args_are_clean_errors(self, clean_state):
        assert "session is required" in cursor_send_message("", "hello")
        assert "message is required" in cursor_send_message("s-x", "   ")

    def test_handler_maps_args(self, clean_state):
        out = _handle_cursor_send_message({"session": "s-nope", "message": "m"})
        assert "no session named 's-nope'" in out


# ---------------------------------------------------------------------------
# cursor_status — STRICTLY read-only (the whole point of the tool)
# ---------------------------------------------------------------------------

class TestCursorStatusReadOnly:
    def test_polling_a_live_run_never_cancels_it(self, clean_state, monkeypatch, tmp_path):
        """THE critical property: asking "how's it going?" must not kill the
        run — no cancel, no mutation, and the run still completes normally."""
        release = threading.Event()
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(release, sid="s-ro"))

        _assert_running_ack(cursor_start("long task", repo=str(tmp_path)))
        job = _job_for("s-ro")
        name = job.session_name
        assert _wait_until(lambda: job.files)  # first diff landed

        s1 = cursor_status("s-ro")
        assert s1.startswith("status: running")
        assert f"session: {name}" in s1
        assert "working on: long task" in s1
        # Files as path + counts ONLY — no inline diffs in status output.
        assert "files so far (1)" in s1
        assert "f1.py +1 −0" in s1
        assert "+b" not in s1
        assert "cursor_events" in s1  # pointer to the diffs
        assert "elapsed:" in s1 and "last activity:" in s1

        # Poll repeatedly — still running, still untouched.
        for _ in range(3):
            assert cursor_status("s-ro").startswith("status: running")
        assert not job.cancel_event.is_set()
        assert job.deliver is True  # delivery not suppressed either

        # Read-only proof: the run keeps going and produces its normal,
        # complete result with its delivery intact.
        release.set()
        assert job.done_event.wait(10)
        assert job.status == "completed"
        assert job.result["files_changed_count"] == 2
        assert job.result["session_id"] == "s-ro"
        events = _drain_completion_queue()
        assert len(events) == 1 and events[0]["status"] == "completed"

    def test_finished_job_status_shows_summary_peek_without_diffs(
        self, clean_state, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(_preset_event(),
                                                                     sid="s-fin"))
        cursor_start("t", repo=str(tmp_path))
        job = _job_for("s-fin")
        assert job.done_event.wait(10)

        status = cursor_status("s-fin")
        assert status.startswith("status: completed")
        assert "summary: all done" in status  # the ~200-char peek line
        assert "files so far (2)" in status
        assert "```diff" not in status  # never inline diffs in status
        assert "reasoning" not in status  # never thinking content either

    def test_status_reports_last_activity_seconds(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The header carries last activity — seconds since the last ACP
        event — so a caller can spot a silent run without touching it.
        Fresh while events stream; frozen at finished_at once terminal."""
        release = threading.Event()
        monkeypatch.setattr(gc_acp, "run_acp",
                            _gated_replay_factory(release, sid="s-act"))

        cursor_start("t", repo=str(tmp_path))
        job = _job_for("s-act")
        assert _wait_until(lambda: job.last_event_at is not None)

        live = cursor_status("s-act")
        assert live.startswith("status: running")
        header = live.splitlines()[1]
        seconds = int(header.split("last activity: ")[1].split("s ago")[0])
        assert 0 <= seconds < 10  # events just streamed — fresh, not silent

        release.set()
        assert job.done_event.wait(10)
        done = cursor_status("s-act")
        assert done.startswith("status: completed")
        # Terminal runs freeze the clock at finished_at: repeated polls of a
        # finished run must not report ever-growing silence.
        time.sleep(0.3)
        done_line = done.splitlines()[1].split("last activity: ")[1]
        again_line = cursor_status("s-act").splitlines()[1].split("last activity: ")[1]
        assert again_line == done_line

    def test_status_addresses_a_dispatched_resume_before_its_session_event(
        self, clean_state, tmp_path
    ):
        """A just-dispatched continuation is addressable by the requested
        handle in the window before acp.session fires."""
        release = threading.Event()

        def runner(job):
            release.wait(10)
            return {"success": True, "status": "completed"}

        job, existing = gc_jobs.registry.dispatch(
            runner=runner, task="t", repo=str(tmp_path), inactivity_timeout_s=60,
            requested_session_id="s-req",
        )
        assert existing is None
        try:
            snap = cursor_status("s-req")
            assert snap.startswith("status: running")
            assert not job.cancel_event.is_set()
        finally:
            release.set()
        assert job.done_event.wait(10)

    def test_persisted_handle_without_live_job_reports_its_record(
        self, clean_state, tmp_path
    ):
        gc_handles.record("s-hist", repo=str(tmp_path), status="completed",
                          task="old task", model="m")
        status = cursor_status("s-hist")
        assert status.startswith("status: completed")
        assert "working on: old task" in status
        assert "not tracked live" in status
        assert "cursor_send_message" in status

    def test_stale_running_record_is_reported_as_unknown(self, clean_state, tmp_path):
        # A dead process can't have left a live run behind.
        gc_handles.record("s-stale", repo=str(tmp_path), status="running")
        status = cursor_status("s-stale")
        assert status.startswith("status: unknown")

    def test_unknown_session_is_actionable_prose(self, clean_state):
        out = cursor_status("s-nope")
        assert "no session named 's-nope'" in out

    def test_empty_session_is_a_clean_error(self, clean_state):
        assert "session is required" in cursor_status("")

    def test_handler_maps_args(self, clean_state):
        out = _handle_cursor_status({"session": "s-nope"})
        assert "no session named 's-nope'" in out


# ---------------------------------------------------------------------------
# cursor_stop — graceful native cancel; idempotent on finished runs
# ---------------------------------------------------------------------------

class TestCursorStop:
    def test_stop_cancels_a_running_job_and_reports_partials(
        self, clean_state, monkeypatch, tmp_path
    ):
        release = threading.Event()  # never set: only the cancel ends the run
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(release, sid="s-stop"))

        _assert_running_ack(cursor_start("t", repo=str(tmp_path)))
        job = _job_for("s-stop")
        name = job.session_name
        assert _wait_until(lambda: job.files)  # partial edit landed

        out = cursor_stop("s-stop")
        assert out.startswith("status: cancelled")
        assert f"session: {name} · stopped after" in out
        assert "(native cancel)" in out
        # Partial work is surfaced as counts, diffs stay in the event log...
        assert "partial work: 1 file" in out
        assert "f1.py +1 −0" in out
        assert "diffs in event log" in out
        # ...the session stays continuable...
        assert "continuable" in out
        # ...and the outcome was reported in-turn: delivery suppressed.
        assert job.status == "cancelled"
        assert _drain_completion_queue() == []
        # Handle table settled too.
        assert gc_handles.get("s-stop")["status"] == "cancelled"

    def test_stop_on_finished_run_is_graceful_and_idempotent(
        self, clean_state, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(_preset_event(),
                                                                     sid="s-idem"))
        cursor_start("t", repo=str(tmp_path))
        job = _job_for("s-idem")
        assert job.done_event.wait(10)
        _drain_completion_queue()

        out = cursor_stop("s-idem")
        assert out.startswith("status: completed")
        assert "already finished" in out and "nothing to stop" in out
        assert "2 files" in out
        # Stopping a finished run must not re-deliver anything.
        assert _drain_completion_queue() == []
        # Truly idempotent: a second stop reports the same.
        assert "nothing to stop" in cursor_stop("s-idem")

    def test_stop_on_persisted_handle_without_live_job_is_graceful(
        self, clean_state, tmp_path
    ):
        gc_handles.record("s-past", repo=str(tmp_path), status="completed")
        out = cursor_stop("s-past")
        assert out.startswith("status: completed")
        assert "nothing to stop" in out

    def test_stop_on_stale_running_record_is_graceful(self, clean_state, tmp_path):
        gc_handles.record("s-ghost", repo=str(tmp_path), status="running")
        out = cursor_stop("s-ghost")
        assert out.startswith("status: not running (stale record)")

    def test_unknown_session_is_actionable_prose(self, clean_state):
        out = cursor_stop("s-nope")
        assert "no session named 's-nope'" in out

    def test_empty_session_is_a_clean_error(self, clean_state):
        assert "session is required" in cursor_stop("")

    def test_handler_maps_args(self, clean_state):
        out = _handle_cursor_stop({"session": "s-nope"})
        assert "no session named 's-nope'" in out


# ---------------------------------------------------------------------------
# Same-repo concurrency guard
# ---------------------------------------------------------------------------

class TestSameRepoConcurrency:
    """Two cursor agents on one working tree = corruption. Reject."""

    def test_second_start_on_same_repo_is_rejected_with_existing_handle(
        self, clean_state, monkeypatch, tmp_path
    ):
        release = threading.Event()
        monkeypatch.setattr(gc_acp, "run_acp", _gated_replay_factory(release, sid="s-a"))
        first = cursor_start("task A", repo=str(tmp_path))
        try:
            _assert_running_ack(first)
            name = _job_for("s-a").session_name

            second = cursor_start("task B", repo=str(tmp_path))
            # Actionable prose: names the ACTIVE session so the caller can
            # steer/inspect it instead.
            assert "cannot start" in second
            assert f"'{name}' is already running" in second
            assert "cursor_send_message" in second
            assert "cursor_stop" in second
        finally:
            release.set()

        job = _job_for("s-a")
        assert job.done_event.wait(10)
        # Only the first job ever ran and delivered.
        assert len(_drain_completion_queue()) == 1

    def test_different_repos_run_concurrently(self, clean_state, monkeypatch, tmp_path):
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        release = threading.Event()
        seq = _AcpSequence(
            _gated_replay_factory(release, sid="s-ra", early_edit=False),
            _gated_replay_factory(release, sid="s-rb", early_edit=False),
        )
        monkeypatch.setattr(gc_acp, "run_acp", seq)

        res_a = cursor_start("a", repo=str(repo_a))
        res_b = cursor_start("b", repo=str(repo_b))
        try:
            _assert_running_ack(res_a)
            _assert_running_ack(res_b)
            assert _job_for("s-ra").session_name != _job_for("s-rb").session_name
        finally:
            release.set()
        for sid in ("s-ra", "s-rb"):
            assert _job_for(sid).done_event.wait(10)

    def test_finished_run_releases_the_repo(self, clean_state, monkeypatch, tmp_path):
        seq = _AcpSequence(
            _gated_replay_factory(_preset_event(), sid="s-one"),
            _gated_replay_factory(_preset_event(), sid="s-two"),
        )
        monkeypatch.setattr(gc_acp, "run_acp", seq)
        cursor_start("t", repo=str(tmp_path))
        assert _job_for("s-one").done_event.wait(10)

        second = cursor_start("t2", repo=str(tmp_path))
        assert "already running" not in second  # no rejection once settled
        assert _job_for("s-two").done_event.wait(10)


# ---------------------------------------------------------------------------
# handles.py — persistent handle table (explicit lookup only)
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_handles(monkeypatch):
    """Fresh in-memory handle table (HERMES_HOME is already a temp dir via
    the autouse _isolate_hermes_home fixture, so the JSON file is too)."""
    monkeypatch.setattr(gc_handles, "_table", {})
    monkeypatch.setattr(gc_handles, "_loaded", False)
    return gc_handles


class TestHandleTable:
    def test_record_and_get_roundtrip(self, clean_handles):
        gc_handles.record("s-1", repo="/w/repo", status="running", task="t", model="m")
        entry = gc_handles.get("s-1")
        assert entry["repo"] == "/w/repo"
        assert entry["status"] == "running"
        assert entry["model"] == "m"
        assert entry["updated_at"] > 0

    def test_record_merges_and_skips_none_fields(self, clean_handles):
        gc_handles.record("s-1", repo="/w/repo", status="running", model="m")
        gc_handles.record("s-1", status="completed", model=None)
        entry = gc_handles.get("s-1")
        assert entry["status"] == "completed"
        assert entry["repo"] == "/w/repo"  # merged, not replaced
        assert entry["model"] == "m"       # None did not clobber

    def test_get_returns_a_copy(self, clean_handles):
        gc_handles.record("s-1", status="running")
        gc_handles.get("s-1")["status"] = "mutated"
        assert gc_handles.get("s-1")["status"] == "running"

    def test_record_without_session_id_is_a_noop(self, clean_handles):
        gc_handles.record(None, repo="/w", status="running")
        gc_handles.record("", repo="/w", status="running")
        assert gc_handles._table == {}

    def test_unknown_or_missing_handle_returns_none(self, clean_handles):
        assert gc_handles.get("s-nope") is None
        assert gc_handles.get(None) is None
        assert gc_handles.get("") is None

    def test_persists_across_process_restart(self, clean_handles, monkeypatch):
        gc_handles.record("s-1", repo="/w/repo", status="cancelled")
        # Simulate a fresh process: wipe the in-memory dict, force a reload.
        monkeypatch.setattr(gc_handles, "_table", {})
        monkeypatch.setattr(gc_handles, "_loaded", False)
        assert gc_handles.get("s-1")["status"] == "cancelled"

    def test_corrupt_file_never_raises_and_recovers(self, clean_handles):
        path = gc_handles._state_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json at all", "utf-8")
        assert gc_handles.get("s-1") is None
        # record still works and repairs the file.
        gc_handles.record("s-1", status="running")
        assert gc_handles.get("s-1")["status"] == "running"
        assert json.loads(path.read_text("utf-8"))["s-1"]["status"] == "running"

    def test_unwritable_state_file_degrades_to_memory(self, clean_handles, monkeypatch):
        monkeypatch.setattr(gc_handles, "_state_file", lambda: None)
        gc_handles.record("s-1", status="running")
        assert gc_handles.get("s-1")["status"] == "running"  # in-memory

    def test_known_handles_most_recent_first(self, clean_handles):
        gc_handles.record("s-old", status="completed")
        gc_handles._table["s-old"]["updated_at"] = time.time() - 100
        gc_handles.record("s-new", status="running")
        assert gc_handles.known_handles() == ["s-new", "s-old"]
        assert gc_handles.known_handles(limit=1) == ["s-new"]

    def test_table_prunes_oldest_beyond_cap(self, clean_handles, monkeypatch):
        monkeypatch.setattr(gc_handles, "MAX_ENTRIES", 3)
        for i in range(5):
            gc_handles.record(f"s-{i}", status="completed")
            gc_handles._table[f"s-{i}"]["updated_at"] = float(i)
        gc_handles.record("s-final", status="running")
        assert len(gc_handles._table) <= 3
        assert "s-final" in gc_handles._table
        assert "s-0" not in gc_handles._table

    def test_no_auto_resume_surface_exists(self):
        """v0.3 contract: lookup is by exact handle only — the repo+timestamp
        auto-resume heuristic (get_recent) must not come back."""
        assert not hasattr(gc_handles, "get_recent")


# ---------------------------------------------------------------------------
# Git fallback — files_changed for edits the ACP stream carried no diff for
# ---------------------------------------------------------------------------

def _git(repo, *args):
    import subprocess

    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=str(repo), check=True, capture_output=True,
    )


class TestGitFallback:
    @pytest.fixture
    def git_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "tool.txt").write_text("orig\n")
        (repo / "pre.txt").write_text("pre\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-qm", "init")
        return repo

    def test_shell_driven_edit_lands_via_git_fallback(
        self, clean_state, monkeypatch, git_repo
    ):
        # Pre-existing dirty file must NOT be attributed to the run.
        (git_repo / "pre.txt").write_text("pre dirty before run\n")

        def shell_edit_replay(*_a, **_k):
            yield ("acp.session", {"sessionId": "s-git", "cwd": str(git_repo), "model": "m"})
            # Simulates cursor editing through a shell command: no diff
            # content ever appears on the ACP stream.
            (git_repo / "tool.txt").write_text("orig\nedited by shell\n")
            (git_repo / "new.txt").write_text("brand new\n")
            yield ("acp.result", {"stopReason": "end_turn"})

        monkeypatch.setattr(gc_acp, "run_acp", shell_edit_replay)
        cursor_start("edit via shell", repo=str(git_repo))
        job = _job_for("s-git")
        assert job.done_event.wait(10)

        result = job.result
        by_path = {Path(f["path"]).name: f for f in result["files_changed"]}
        assert set(by_path) == {"tool.txt", "new.txt"}  # pre.txt excluded
        assert by_path["tool.txt"]["status"] == "M"
        assert by_path["tool.txt"]["added"] == 1
        assert "edited by shell" in by_path["tool.txt"]["diff"]
        assert by_path["new.txt"]["status"] == "A"

    def test_acp_stream_diff_wins_over_fallback(self, clean_state, monkeypatch, git_repo):
        """A file already captured from the ACP stream is not re-added."""

        def stream_and_shell(*_a, **_k):
            yield ("acp.session", {"sessionId": "s-win", "cwd": str(git_repo), "model": "m"})
            (git_repo / "tool.txt").write_text("orig\nedited\n")
            yield ("acp.update", {
                "sessionUpdate": "tool_call", "toolCallId": "t1",
                "title": "Edit File", "kind": "edit", "status": "pending", "rawInput": {},
            })
            yield ("acp.update", {
                "sessionUpdate": "tool_call_update", "toolCallId": "t1",
                "status": "completed",
                "content": [{
                    "type": "diff", "path": str(git_repo / "tool.txt"),
                    "oldText": "orig\n", "newText": "orig\nedited\n",
                }],
            })
            yield ("acp.result", {"stopReason": "end_turn"})

        monkeypatch.setattr(gc_acp, "run_acp", stream_and_shell)
        cursor_start("t", repo=str(git_repo))
        job = _job_for("s-win")
        assert job.done_event.wait(10)
        assert job.result["files_changed_count"] == 1
        assert job.result["files_changed"][0]["added"] == 1

    def test_unified_diff_text_counts(self):
        diff, added, removed = gc_events.unified_diff_text("a\nb\n", "a\nc\nd\n", "/w/f.txt")
        assert added == 2 and removed == 1
        assert "--- a/w/f.txt" in diff and "+++ b/w/f.txt" in diff


# ---------------------------------------------------------------------------
# All terminal states deliver a completion message — never a silent death
# ---------------------------------------------------------------------------

class TestAllTerminalStatesDeliver:
    def _run_armed(self, monkeypatch, tmp_path, sid, terminal_event):
        """Dispatch through cursor_start so delivery gets armed; the replay
        emits ``terminal_event`` only after the running handle was returned
        (deterministic — the run cannot finalize before arming)."""
        release = threading.Event()

        def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                   cancel_check=None, session_id=None, model=None):
            yield ("acp.session", {"sessionId": sid, "cwd": str(workdir), "model": "m"})
            release.wait(10)
            yield terminal_event

        monkeypatch.setattr(gc_acp, "run_acp", replay)
        res = cursor_start("t", repo=str(tmp_path))
        assert "running in background" in res, f"run never armed: {res}"
        job = _job_for(sid)
        release.set()
        assert job.done_event.wait(10)
        events = _drain_completion_queue()
        assert len(events) == 1, f"expected exactly one delivery, got {events}"
        return job, events[0]

    def test_completed_delivers(self, clean_state, monkeypatch, tmp_path):
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-ok",
            ("acp.result", {"stopReason": "end_turn"}),
        )
        assert job.status == "completed"
        assert evt["status"] == "completed"
        assert evt["result"]["success"] is True

    def test_cancelled_run_delivers(self, clean_state, monkeypatch, tmp_path):
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-c",
            ("acp.result", {"stopReason": "cancelled"}),
        )
        # The JOB status names the real terminal state...
        assert job.status == "cancelled"
        assert evt["status"] == "cancelled"
        assert "cancel" in evt["error"]
        # ...while the result dict keeps the run's exact vocabulary (a
        # native cancel is result status "failed" — unchanged from v0.2).
        assert evt["result"]["status"] == "failed"

    def test_timeout_delivers(self, clean_state, monkeypatch, tmp_path):
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-t",
            ("acp.error", {"error": "ACP run timed out after 5s", "timeout": True}),
        )
        assert job.status == "timeout"
        assert evt["status"] == "timeout"
        assert "timed out" in evt["error"]
        assert evt["result"]["status"] == "timeout"

    def test_midrun_failure_delivers(self, clean_state, monkeypatch, tmp_path):
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-x",
            ("acp.error", {"error": "ACP connection failed mid-run: boom"}),
        )
        assert job.status == "failed"
        assert evt["status"] == "failed"
        assert "boom" in evt["error"]


# ---------------------------------------------------------------------------
# acp_runner.run_acp — real subprocess round-trips against a fake ACP server
# ---------------------------------------------------------------------------

_FAKE_ACP = '''#!/usr/bin/env python3
import json, sys, time
MODE = "__MODE__"

def send(o):
    sys.stdout.write(json.dumps(o) + "\\n")
    sys.stdout.flush()

def update(upd):
    send({"jsonrpc": "2.0", "method": "session/update",
          "params": {"sessionId": "s-test", "update": upd}})

if MODE == "die":
    sys.exit(1)

prompt_id = None
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    m = msg.get("method")
    if m == "initialize":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": {"protocolVersion": 1}})
    elif m == "session/new":
        send({"jsonrpc": "2.0", "id": msg["id"],
              "result": {"sessionId": "s-test", "models": {"currentModelId": "fake-model"}}})
    elif m == "session/load":
        if MODE == "loadfail":
            send({"jsonrpc": "2.0", "id": msg["id"],
                  "error": {"code": -32602, "message": "Invalid params",
                            "data": {"message": "Session not found"}}})
        else:
            # Replay of prior history precedes the load response (observed
            # live 2026-07-02), then the same result shape as session/new
            # minus sessionId.
            update({"sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "prior task"}})
            update({"sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "prior answer"}})
            send({"jsonrpc": "2.0", "id": msg["id"],
                  "result": {"models": {"currentModelId": "fake-model"}}})
    elif m == "session/prompt":
        prompt_id = msg["id"]
        update({"sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "working"}})
        if MODE == "slow":
            # Slow but ACTIVE: keep streaming updates with gaps well under
            # the inactivity threshold, for a total well past it.
            for i in range(5):
                time.sleep(0.5)
                update({"sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "step %d" % i}})
            send({"jsonrpc": "2.0", "id": prompt_id,
                  "result": {"stopReason": "end_turn"}})
        elif MODE == "chatty":
            # Runaway: streams updates forever, never finishes, never
            # reads stdin again (so session/cancel goes unanswered).
            while True:
                time.sleep(0.2)
                update({"sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "still going"}})
        elif MODE in ("happy", "load", "loadfail"):
            update({"sessionUpdate": "tool_call", "toolCallId": "t1",
                    "title": "Edit File", "kind": "edit", "status": "pending",
                    "rawInput": {}})
            update({"sessionUpdate": "tool_call_update", "toolCallId": "t1",
                    "status": "completed",
                    "content": [{"type": "diff", "path": "/w/f.py",
                                 "oldText": "a\\n", "newText": "a\\nb\\n"}]})
            send({"jsonrpc": "2.0", "id": prompt_id,
                  "result": {"stopReason": "end_turn"}})
        # cancel / hang: keep reading
    elif m == "session/cancel":
        if MODE == "cancel":
            send({"jsonrpc": "2.0", "id": prompt_id,
                  "result": {"stopReason": "cancelled"}})
        # hang: ignore the cancel entirely
'''


def _install_fake_acp(tmp_path, monkeypatch, mode):
    stub = tmp_path / "cursor-agent"
    stub.write_text(_FAKE_ACP.replace("__MODE__", mode))
    stub.chmod(0o755)
    monkeypatch.setattr(gc_acp, "subprocess_env", _prepend_path_env(tmp_path))


class TestAcpRunner:
    def test_happy_path_yields_session_updates_result(self, tmp_path, monkeypatch):
        _install_fake_acp(tmp_path, monkeypatch, "happy")
        events = list(gc_acp.run_acp("do it", str(tmp_path),
                                     inactivity_timeout_s=30.0,
                                     cancel_check=lambda: False))
        keys = [k for k, _ in events]
        assert keys[0] == "acp.session"
        assert events[0][1]["sessionId"] == "s-test"
        assert "acp.update" in keys
        assert keys[-1] == "acp.result"
        assert events[-1][1]["stopReason"] == "end_turn"
        updates = [o for k, o in events if k == "acp.update"]
        assert any(u.get("sessionUpdate") == "tool_call_update" and u.get("content")
                   for u in updates)

    def test_cancel_check_sends_session_cancel(self, tmp_path, monkeypatch):
        _install_fake_acp(tmp_path, monkeypatch, "cancel")
        cancelled = {"flag": False}
        events = []
        t0 = time.monotonic()
        for key, obj in gc_acp.run_acp("do it", str(tmp_path),
                                       inactivity_timeout_s=30.0,
                                       cancel_check=lambda: cancelled["flag"]):
            events.append((key, obj))
            if key == "acp.update":
                cancelled["flag"] = True  # interrupt lands mid-run
        elapsed = time.monotonic() - t0
        assert events[-1] == ("acp.result", {"stopReason": "cancelled"})
        assert elapsed < 15, f"native cancel did not resolve promptly ({elapsed:.1f}s)"

    def test_silent_hang_is_killed_at_the_inactivity_limit(self, tmp_path, monkeypatch):
        """A run with ZERO activity is hung: the inactivity watchdog kills
        it and the error names the limit that fired."""
        _install_fake_acp(tmp_path, monkeypatch, "hang")
        monkeypatch.setattr(gc_acp, "CANCEL_GRACE_S", 1.0)
        t0 = time.monotonic()
        events = list(gc_acp.run_acp("do it", str(tmp_path),
                                     inactivity_timeout_s=1.0,
                                     cancel_check=lambda: False))
        elapsed = time.monotonic() - t0
        assert events[-1][0] == "acp.error"
        assert events[-1][1]["timeout"] is True
        assert "no activity for 1s" in events[-1][1]["error"]
        assert elapsed < 20, f"timeout kill did not tear down ({elapsed:.1f}s)"

    def test_slow_but_active_run_is_not_killed(self, tmp_path, monkeypatch):
        """Inactivity semantics: events keep arriving past the old wall
        limit (total >> inactivity_timeout_s), so the run must complete —
        activity resets the watchdog clock."""
        _install_fake_acp(tmp_path, monkeypatch, "slow")
        t0 = time.monotonic()
        # Gaps between updates are 0.5s; total streaming is ~2.5s. Under the
        # old wall-clock semantics a 1.5s timeout would have killed it.
        events = list(gc_acp.run_acp("do it", str(tmp_path),
                                     inactivity_timeout_s=1.5,
                                     cancel_check=lambda: False))
        elapsed = time.monotonic() - t0
        assert elapsed > 1.5, "run finished before the old wall limit — inconclusive"
        assert events[-1] == ("acp.result", {"stopReason": "end_turn"})
        assert not any(k == "acp.error" for k, _ in events)

    def test_max_wall_ceiling_kills_a_runaway_stream(self, tmp_path, monkeypatch):
        """A run that streams forever never trips the inactivity watchdog —
        the max_wall_s hard ceiling is the safety net, and the error says so."""
        _install_fake_acp(tmp_path, monkeypatch, "chatty")
        monkeypatch.setattr(gc_acp, "CANCEL_GRACE_S", 1.0)
        t0 = time.monotonic()
        events = list(gc_acp.run_acp("do it", str(tmp_path),
                                     inactivity_timeout_s=30.0,
                                     max_wall_s=1.5,
                                     cancel_check=lambda: False))
        elapsed = time.monotonic() - t0
        assert events[-1][0] == "acp.error"
        assert events[-1][1]["timeout"] is True
        assert "exceeded max wall time (1s)" in events[-1][1]["error"]
        assert elapsed < 20, f"wall-ceiling kill did not tear down ({elapsed:.1f}s)"

    def test_legacy_timeout_kwarg_is_an_inactivity_alias(self, tmp_path, monkeypatch):
        """Backward compat: the old ``timeout=`` kwarg still works, now with
        inactivity semantics."""
        _install_fake_acp(tmp_path, monkeypatch, "hang")
        monkeypatch.setattr(gc_acp, "CANCEL_GRACE_S", 1.0)
        events = list(gc_acp.run_acp("do it", str(tmp_path), timeout=1.0,
                                     cancel_check=lambda: False))
        assert events[-1][0] == "acp.error"
        assert events[-1][1]["timeout"] is True
        assert "no activity for 1s" in events[-1][1]["error"]

    def test_dead_binary_raises_actionable_acp_error(self, tmp_path, monkeypatch):
        _install_fake_acp(tmp_path, monkeypatch, "die")
        with pytest.raises(gc_acp.AcpError) as exc_info:
            list(gc_acp.run_acp("do it", str(tmp_path),
                                inactivity_timeout_s=10.0,
                                cancel_check=lambda: False))
        assert "handshake" in str(exc_info.value)

    def test_empty_task_raises(self, tmp_path):
        with pytest.raises(gc_runner.HarnessError):
            list(gc_acp.run_acp("  ", str(tmp_path)))

    def test_bad_repo_raises(self):
        with pytest.raises(gc_runner.HarnessError):
            list(gc_acp.run_acp("t", "/nope/nothing/here"))

    def test_session_id_resumes_via_session_load(self, tmp_path, monkeypatch):
        """session_id → session/load path: the resume id is kept (not the
        fake server's session/new id) and resumed=True, with the replayed
        history flowing through as ordinary acp.update events."""
        _install_fake_acp(tmp_path, monkeypatch, "load")
        events = list(gc_acp.run_acp("continue it", str(tmp_path),
                                     inactivity_timeout_s=30.0,
                                     cancel_check=lambda: False,
                                     session_id="s-prior"))
        # Replayed history arrives BEFORE session/load resolves, so the
        # acp.session event follows the replay updates in the stream.
        sessions = [o for k, o in events if k == "acp.session"]
        assert len(sessions) == 1
        assert sessions[0]["sessionId"] == "s-prior"
        assert sessions[0]["resumed"] is True
        replayed = [o for k, o in events if k == "acp.update"
                    and o.get("sessionUpdate") == "user_message_chunk"]
        assert replayed, "session/load replay did not flow through acp.update"
        assert events[-1] == ("acp.result", {"stopReason": "end_turn"})

    def test_session_load_failure_falls_back_to_session_new(self, tmp_path, monkeypatch):
        """Expired/unknown session id must not hard-fail: fall back to a
        fresh session/new and report resumed=False."""
        _install_fake_acp(tmp_path, monkeypatch, "loadfail")
        events = list(gc_acp.run_acp("continue it", str(tmp_path),
                                     inactivity_timeout_s=30.0,
                                     cancel_check=lambda: False,
                                     session_id="s-expired"))
        assert events[0][0] == "acp.session"
        assert events[0][1]["sessionId"] == "s-test"  # fresh session
        assert events[0][1]["resumed"] is False
        assert events[-1] == ("acp.result", {"stopReason": "end_turn"})

    def test_no_session_id_creates_fresh_session(self, tmp_path, monkeypatch):
        """Omitting session_id keeps the one-shot behavior byte-identical."""
        _install_fake_acp(tmp_path, monkeypatch, "happy")
        events = list(gc_acp.run_acp("do it", str(tmp_path),
                                     inactivity_timeout_s=30.0,
                                     cancel_check=lambda: False))
        assert events[0][1]["sessionId"] == "s-test"
        assert events[0][1]["resumed"] is False


# ---------------------------------------------------------------------------
# _AcpClient._establish_session — request-layer method selection
# ---------------------------------------------------------------------------

class TestEstablishSession:
    def _run(self, session_id, responder):
        """Drive _establish_session with a mocked _request; returns
        (methods_called, params_by_method, (sess, resumed), client)."""
        import asyncio
        import queue

        client = gc_acp._AcpClient(
            task="t", workdir=Path("/w"), out_q=queue.Queue(),
            cancel_requested=threading.Event(),
            session_id=session_id,
        )
        calls = []

        async def fake_request(method, params, timeout=None):
            calls.append((method, params))
            return responder(method)

        client._request = fake_request
        result = asyncio.run(client._establish_session())
        return calls, result, client

    def test_session_id_uses_session_load_not_session_new(self):
        calls, (sess, resumed), client = self._run(
            "s-prior",
            lambda m: {"models": {"currentModelId": "m"}},
        )
        assert [m for m, _ in calls] == ["session/load"]
        assert calls[0][1]["sessionId"] == "s-prior"
        assert calls[0][1]["mcpServers"] == []
        assert resumed is True
        assert client._session_id == "s-prior"

    def test_load_failure_falls_back_to_session_new(self):
        def responder(method):
            if method == "session/load":
                raise RuntimeError("ACP error -32602: Invalid params")
            return {"sessionId": "s-fresh", "models": {}}

        calls, (sess, resumed), client = self._run("s-gone", responder)
        assert [m for m, _ in calls] == ["session/load", "session/new"]
        assert resumed is False
        assert client._session_id == "s-fresh"

    def test_no_session_id_goes_straight_to_session_new(self):
        calls, (sess, resumed), client = self._run(
            None, lambda m: {"sessionId": "s-fresh", "models": {}}
        )
        assert [m for m, _ in calls] == ["session/new"]
        assert resumed is False
        assert client._session_id == "s-fresh"


# ---------------------------------------------------------------------------
# Registration + availability gate
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_wires_seven_tools_into_ghost_cursor_toolset(self):
        calls = []
        ctx = SimpleNamespace(register_tool=lambda **kw: calls.append(kw))
        register(ctx)

        by_name = {c["name"]: c for c in calls}
        assert set(by_name) == {
            CREATE_TOOL_NAME, SEND_TOOL_NAME, STATUS_TOOL_NAME,
            STOP_TOOL_NAME, EVENTS_TOOL_NAME, LIST_TOOL_NAME,
            START_TOOL_NAME,
        }
        for name, schema, required in (
            (CREATE_TOOL_NAME, CURSOR_CREATE_SCHEMA, []),
            (SEND_TOOL_NAME, CURSOR_SEND_SCHEMA, ["session", "message"]),
            (STATUS_TOOL_NAME, CURSOR_STATUS_SCHEMA, ["session"]),
            (STOP_TOOL_NAME, CURSOR_STOP_SCHEMA, ["session"]),
            (EVENTS_TOOL_NAME, CURSOR_EVENTS_SCHEMA, ["session"]),
            (LIST_TOOL_NAME, CURSOR_LIST_SCHEMA, []),
            (START_TOOL_NAME, CURSOR_START_SCHEMA, ["task"]),
        ):
            entry = by_name[name]
            assert entry["toolset"] == TOOLSET
            assert entry["check_fn"] is check_cursor_available
            assert entry["schema"] is schema
            assert entry["schema"]["parameters"]["required"] == required
            assert callable(entry["handler"])

    def test_send_schema_steers_delegation(self):
        desc = CURSOR_SEND_SCHEMA["description"].lower()
        assert "prefer this" in desc
        assert "delegate" in desc

    def test_create_schema_promises_lazy_spawn(self):
        desc = CURSOR_CREATE_SCHEMA["description"].lower()
        assert "dispatches nothing" in desc
        assert "lazily" in desc

    def test_start_schema_is_marked_deprecated(self):
        desc = CURSOR_START_SCHEMA["description"]
        assert desc.startswith("DEPRECATED")
        assert "cursor_create_session" in desc
        assert "cursor_send_message" in desc
        props = CURSOR_START_SCHEMA["parameters"]["properties"]
        for optional in ("session", "model", "repo"):
            assert props[optional]["type"] == "string"
            assert optional not in CURSOR_START_SCHEMA["parameters"]["required"]

    def test_events_schema_documents_paging_semantics(self):
        desc = CURSOR_EVENTS_SCHEMA["description"]
        assert "LAST 10" in desc
        assert "kind" in desc
        assert "2KB" in desc and "20KB" in desc

    def test_status_schema_promises_read_only(self):
        desc = CURSOR_STATUS_SCHEMA["description"].lower()
        assert "read-only" in desc
        assert "never cancels" in desc

    def test_send_schema_states_interrupt_semantics(self):
        desc = CURSOR_SEND_SCHEMA["description"].lower()
        assert "interrupt" in desc
        assert "re-prompt" in desc

    def test_check_fn_false_without_binary(self, monkeypatch):
        monkeypatch.setattr(gc_runner, "cursor_agent_available", lambda: False)
        assert check_cursor_available() is False

    def test_check_fn_false_without_resolvable_repo(self, monkeypatch):
        monkeypatch.setattr(gc_runner, "cursor_agent_available", lambda: True)
        monkeypatch.setattr(gc, "_default_repo", lambda: None)
        assert check_cursor_available() is False

    def test_check_fn_true_with_binary_and_repo(self, monkeypatch):
        monkeypatch.setattr(gc_runner, "cursor_agent_available", lambda: True)
        # _default_repo falls back to os.getcwd(), which always exists.
        assert check_cursor_available() is True

    def test_check_fn_never_raises(self, monkeypatch):
        def boom():
            raise RuntimeError("probe failed")

        monkeypatch.setattr(gc_runner, "cursor_agent_available", boom)
        assert check_cursor_available() is False


# ---------------------------------------------------------------------------
# Legacy --print runner + normalize_harness (kept as fallback/reference)
# ---------------------------------------------------------------------------

def _fixture_events():
    out = []
    for line in FIXTURE.read_text().splitlines():
        obj = json.loads(line)
        out.append((gc_runner.event_key(obj), obj))
    return out


class TestNormalizeHarness:
    def _all_envelopes(self):
        envs = []
        for key, obj in _fixture_events():
            envs.extend(gc_events.normalize_harness(key, obj))
        return envs

    def test_every_envelope_carries_ghost_source(self):
        envs = self._all_envelopes()
        assert envs, "fixture produced no envelopes"
        assert all(e.get("source") == "ghost" for e in envs)

    def test_edit_tool_call_yields_file_diff(self):
        diffs = [e for e in self._all_envelopes() if e["kind"] == "file_diff"]
        assert len(diffs) == 1
        d = diffs[0]
        assert d["path"] == "/tmp/ph2probe/calc.py"
        assert d["added"] == 3
        assert d["removed"] == 0
        assert d["status"] == "M"  # existing file modified
        assert "multiply" in d["after"]
        assert "multiply" not in d["before"]
        assert d["diff"]  # diffString captured

    def test_thinking_maps_to_reasoning_lifecycle(self):
        reasoning = [
            e for e in self._all_envelopes()
            if e["kind"] == "lifecycle" and e.get("event") == "reasoning"
        ]
        assert reasoning
        joined = "".join(e["text"] for e in reasoning)
        assert "calc.py" in joined

    def test_harness_error_timeout_maps_to_run_failed(self):
        envs = gc_events.normalize_harness(
            "harness.error", {"error": "harness timed out after 600s", "timeout": True}
        )
        assert len(envs) == 1
        assert envs[0]["event"] == "run.failed"
        assert envs[0]["timeout"] is True

    def test_unknown_event_passes_through(self):
        envs = gc_events.normalize_harness("mystery.event", {"x": 1})
        assert envs == [
            {"source": "ghost", "kind": "lifecycle", "event": "passthrough",
             "name": "mystery.event", "data": {"x": 1}}
        ]

    def test_file_diff_truncates_oversized_content(self):
        big = "x" * (gc_events.MAX_CONTENT_CHARS + 500)
        env = gc_events.file_diff("f.py", before=big, after="y", diff="d")
        assert len(env["before"]) < len(big)
        assert "truncated" in env["before"]


class TestLegacyRunner:
    def test_event_key_composition(self):
        assert gc_runner.event_key({"type": "tool_call", "subtype": "completed"}) == "tool_call.completed"
        assert gc_runner.event_key({"type": "assistant"}) == "assistant"

    def test_empty_task_raises(self, tmp_path):
        with pytest.raises(gc_runner.HarnessError):
            list(gc_runner.run_harness("  ", str(tmp_path)))

    def test_bad_repo_raises(self):
        with pytest.raises(gc_runner.HarnessError):
            gc_runner.resolve_repo("/nope/nothing/here")

    def test_local_bin_prepended_to_path(self):
        env = gc_runner.subprocess_env()
        assert str(Path.home() / ".local" / "bin") in env["PATH"].split(":")

    def test_run_harness_skips_noise_and_yields_json(self, tmp_path, monkeypatch):
        """Drive run_harness with a fake 'cursor-agent' that prints noise +
        JSON lines, verifying parse/skip behavior end-to-end via a real
        subprocess."""
        fake = tmp_path / "cursor-agent"
        fake.write_text(
            "#!/bin/sh\n"
            "echo 'not json banner'\n"
            "echo '{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\","
            "\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}}'\n"
        )
        fake.chmod(0o755)
        monkeypatch.setattr(
            gc_runner, "subprocess_env", _prepend_path_env(tmp_path),
        )
        events = list(gc_runner.run_harness("do it", str(tmp_path)))
        assert events == [
            ("assistant", {
                "type": "assistant",
                "message": {"role": "assistant",
                            "content": [{"type": "text", "text": "hi"}]},
            })
        ]

    def test_run_harness_timeout_kills_and_reports(self, tmp_path, monkeypatch):
        # `sleep 30` is a grandchild holding the stdout pipe — the group kill
        # must reap it too, or readline blocks for the full 30s (regression
        # caught by this test before start_new_session + killpg were added).
        fake = tmp_path / "cursor-agent"
        fake.write_text("#!/bin/sh\nsleep 30\n")
        fake.chmod(0o755)
        monkeypatch.setattr(
            gc_runner, "subprocess_env", _prepend_path_env(tmp_path),
        )
        t0 = time.monotonic()
        events = list(gc_runner.run_harness("do it", str(tmp_path), timeout=1.0))
        elapsed = time.monotonic() - t0
        assert events[-1][0] == "harness.error"
        assert events[-1][1]["timeout"] is True
        assert elapsed < 15, f"group kill did not tear down the pipe ({elapsed:.1f}s)"


# ---------------------------------------------------------------------------
# CursorJob progress buffer — rolling, bounded, line-safe
# ---------------------------------------------------------------------------

class TestProgressBuffer:
    def test_buffer_accumulates_and_rolls(self, clean_state):
        job = gc_jobs.CursorJob(job_id="cursor_test", task="t", repo="/r",
                                inactivity_timeout_s=60)
        monkey_cap = 500
        old_cap = gc_jobs.MAX_PROGRESS_CHARS
        gc_jobs.MAX_PROGRESS_CHARS = monkey_cap
        try:
            for i in range(100):
                job.append_progress({"kind": "lifecycle", "event": "reasoning",
                                     "text": f"chunk {i:03d} " + "x" * 40})
            assert len(job.progress_buffer) <= monkey_cap
            assert job.progress_events == 100
            # Oldest entries rolled out, newest survive.
            assert "chunk 099" in job.progress_buffer
            assert "chunk 000" not in job.progress_buffer
            # Rolling trims at a line boundary — every line stays valid JSON.
            for line in job.progress_buffer.strip().splitlines():
                json.loads(line)
        finally:
            gc_jobs.MAX_PROGRESS_CHARS = old_cap

    def test_buffer_drops_full_file_content_but_keeps_diff(self, clean_state):
        job = gc_jobs.CursorJob(job_id="cursor_test2", task="t", repo="/r",
                                inactivity_timeout_s=60)
        job.append_progress({
            "kind": "file_diff", "path": "/r/f.py",
            "before": "B" * 10_000, "after": "A" * 10_000,
            "diff": "+line\n" * 1_000, "added": 1000, "removed": 0, "status": "M",
        })
        line = json.loads(job.progress_buffer.strip())
        assert "before" not in line and "after" not in line
        assert line["path"] == "/r/f.py"
        assert len(line["diff"]) <= gc_jobs._BUFFER_DIFF_CHARS + 40


# ---------------------------------------------------------------------------
# eventlog.py — per-session JSONL spill log: creation, pagination, compaction
# ---------------------------------------------------------------------------

def _log_lines(sid):
    path = gc_eventlog.log_path(sid)
    assert path is not None and path.is_file(), f"no spill log for {sid!r}"
    return [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]


class TestEventLog:
    def test_append_creates_jsonl_under_hermes_home(
        self, clean_state, _isolate_hermes_home
    ):
        gc_eventlog.append("s-log", {"kind": "content", "delta": "hi"})
        path = gc_eventlog.log_path("s-log")
        assert path == (
            _isolate_hermes_home / "state" / "ghost_cursor" / "logs" / "s-log.jsonl"
        )
        lines = _log_lines("s-log")
        assert len(lines) == 1
        assert lines[0]["seq"] == 0
        assert lines[0]["ts"] > 0
        assert lines[0]["kind"] == "content"
        assert lines[0]["delta"] == "hi"

    def test_seq_is_monotonic_and_survives_a_process_restart(self, clean_state):
        for i in range(3):
            gc_eventlog.append("s-seq", {"kind": "content", "delta": f"e{i}"})
        # Simulate a fresh process: the writer must re-derive next_seq from
        # the file tail, not restart at 0.
        gc_eventlog._reset_for_tests()
        gc_eventlog.append("s-seq", {"kind": "content", "delta": "e3"})
        assert [l["seq"] for l in _log_lines("s-seq")] == [0, 1, 2, 3]

    def test_unsafe_handle_characters_cannot_escape_the_logs_dir(self, clean_state):
        gc_eventlog.append("../../evil/sid", {"kind": "content", "delta": "x"})
        path = gc_eventlog.log_path("../../evil/sid")
        assert path.parent == gc_eventlog.logs_dir()
        assert path.is_file()

    def test_append_never_raises_without_a_session_id(self, clean_state):
        gc_eventlog.append("", {"kind": "content", "delta": "x"})
        gc_eventlog.append(None, {"kind": "content", "delta": "x"})

    def test_stats_reports_path_and_total_events(self, clean_state):
        assert gc_eventlog.stats("s-none") is None
        for i in range(5):
            gc_eventlog.append("s-st", {"kind": "content", "delta": str(i)})
        stats = gc_eventlog.stats("s-st")
        assert stats["path"] == str(gc_eventlog.log_path("s-st"))
        assert stats["total_events"] == 5

    def test_read_page_slices_by_seq(self, clean_state):
        for i in range(30):
            gc_eventlog.append("s-page", {"kind": "content", "delta": f"event {i:02d}"})

        page = gc_eventlog.read_page("s-page", offset=10, limit=5)
        assert [e["seq"] for e in page["events"]] == [10, 11, 12, 13, 14]
        assert page["events"][0]["delta"] == "event 10"
        assert page["offset"] == 10 and page["limit"] == 5
        assert page["total_events"] == 30
        assert page["log_path"] == str(gc_eventlog.log_path("s-page"))

        # Last (partial) page, then past-the-end.
        assert [e["seq"] for e in gc_eventlog.read_page(
            "s-page", offset=28, limit=10)["events"]] == [28, 29]
        assert gc_eventlog.read_page("s-page", offset=100, limit=10)["events"] == []

    def test_read_page_defaults_and_bad_params_are_forgiving(self, clean_state):
        for i in range(3):
            gc_eventlog.append("s-def", {"kind": "content", "delta": str(i)})
        page = gc_eventlog.read_page("s-def")
        assert len(page["events"]) == 3
        page = gc_eventlog.read_page("s-def", offset="junk", limit="junk")
        assert page["offset"] == 0 and page["limit"] == gc_eventlog.DEFAULT_PAGE_LIMIT
        assert gc_eventlog.read_page("s-def", offset=-5, limit=0)["limit"] == 1
        assert gc_eventlog.read_page(
            "s-def", limit=10_000)["limit"] == gc_eventlog.MAX_PAGE_LIMIT
        assert gc_eventlog.read_page("s-missing") is None

    def test_oversized_fields_clip_inline_but_stay_full_in_the_jsonl(
        self, clean_state
    ):
        big = "x" * (gc_eventlog.PAGE_FIELD_CHARS * 3)
        gc_eventlog.append("s-big", {"kind": "tool_result", "id": "t1",
                                     "status": "done", "output": big})
        # Full fidelity on disk...
        assert _log_lines("s-big")[0]["output"] == big
        # ...clipped inline on the paged view, with a pointer to the log.
        paged = gc_eventlog.read_page("s-big", offset=0, limit=1)["events"][0]
        assert len(paged["output"]) < len(big)
        assert paged["output"].startswith("x" * gc_eventlog.PAGE_FIELD_CHARS)
        assert "truncated" in paged["output"]
        assert paged["status"] == "done"  # small fields untouched

    def test_log_compacts_to_head_plus_tail_at_the_byte_cap(
        self, clean_state, monkeypatch
    ):
        monkeypatch.setattr(gc_eventlog, "MAX_LOG_BYTES", 20_000)
        monkeypatch.setattr(gc_eventlog, "HEAD_RETAIN_BYTES", 5_000)
        monkeypatch.setattr(gc_eventlog, "TAIL_RETAIN_BYTES", 10_000)

        n = 300  # ~100 bytes/line → several compactions past the 20KB cap
        for i in range(n):
            gc_eventlog.append("s-cap", {"kind": "content",
                                         "delta": f"event {i:03d} " + "p" * 60})

        path = gc_eventlog.log_path("s-cap")
        assert path.stat().st_size <= 20_000

        lines = _log_lines("s-cap")
        seqs = [l["seq"] for l in lines if isinstance(l.get("seq"), int)]
        markers = [l for l in lines if l.get("kind") == "log_compaction"]
        # Head (oldest) and tail (newest) both survive; the middle is gone.
        assert seqs[0] == 0
        assert seqs[-1] == n - 1
        assert len(seqs) < n
        assert markers, "compaction must leave a marker in the gap"
        marker = markers[-1]
        assert marker["dropped_events"] > 0
        assert 0 < marker["first_dropped_seq"] <= marker["last_dropped_seq"] < n - 1
        # Every retained line is still valid JSON with its ORIGINAL seq —
        # compaction never renumbers.
        assert seqs == sorted(seqs)
        # total_events counts everything ever appended, dropped included.
        assert gc_eventlog.stats("s-cap")["total_events"] == n

    def test_read_page_reports_compaction_gaps_in_range(
        self, clean_state, monkeypatch
    ):
        monkeypatch.setattr(gc_eventlog, "MAX_LOG_BYTES", 20_000)
        monkeypatch.setattr(gc_eventlog, "HEAD_RETAIN_BYTES", 5_000)
        monkeypatch.setattr(gc_eventlog, "TAIL_RETAIN_BYTES", 10_000)
        for i in range(300):
            gc_eventlog.append("s-gap", {"kind": "content",
                                         "delta": f"event {i:03d} " + "p" * 60})
        marker = [l for l in _log_lines("s-gap")
                  if l.get("kind") == "log_compaction"][-1]
        dropped_seq = marker["first_dropped_seq"]

        page = gc_eventlog.read_page("s-gap", offset=dropped_seq, limit=5)
        assert page["gaps"], "a page over dropped seqs must disclose the gap"
        assert "compaction" in page["note"]
        # A page fully inside the retained tail has no gap disclosure.
        tail_page = gc_eventlog.read_page("s-gap", offset=295, limit=5)
        assert len(tail_page["events"]) == 5
        assert "gaps" not in tail_page


# ---------------------------------------------------------------------------
# Spill integration — runs stream to the JSONL log; cursor_status pages it
# ---------------------------------------------------------------------------

class TestEventLogIntegration:
    def _run_to_completion(self, monkeypatch, tmp_path, sid="s-spill"):
        monkeypatch.setattr(
            gc_acp, "run_acp", _gated_replay_factory(_preset_event(), sid=sid)
        )
        cursor_start("t", repo=str(tmp_path))
        job = _job_for(sid)
        assert job.done_event.wait(10)
        return job

    def test_run_streams_full_envelopes_to_the_session_log(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The log is keyed by the session NAME — one named session, one log."""
        job = self._run_to_completion(monkeypatch, tmp_path)
        lines = _log_lines(job.session_name)
        # Every folded envelope landed, in order, seq from 0.
        assert [l["seq"] for l in lines] == list(range(len(lines)))
        assert len(lines) == job.progress_events
        kinds = [l["kind"] for l in lines]
        assert "lifecycle" in kinds and "tool_result" in kinds
        # FULL fidelity: file_diff lines keep before/after content that the
        # in-memory rolling buffer strips.
        diffs = [l for l in lines if l["kind"] == "file_diff"]
        assert diffs and all("before" in d and "after" in d for d in diffs)

    def test_cursor_status_reports_log_path_and_total_event_count(
        self, clean_state, monkeypatch, tmp_path
    ):
        job = self._run_to_completion(monkeypatch, tmp_path)
        status = cursor_status("s-spill")
        path = str(gc_eventlog.log_path(job.session_name))
        assert f"events: {job.progress_events} total · log: {path}" in status

    def test_cursor_events_pages_the_persisted_log_forward(
        self, clean_state, monkeypatch, tmp_path
    ):
        job = self._run_to_completion(monkeypatch, tmp_path)
        total = job.progress_events

        first = cursor_events("s-spill", offset=0, limit=2)
        assert f"events 0–1 of {total}" in first

        rest = cursor_events("s-spill", offset=2, limit=500)
        assert f"events 2–{total - 1} of {total}" in rest

    def test_events_page_across_a_process_restart(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The log outlives the process: a persisted handle with no live job
        still pages, and status still shows the log location."""
        job = self._run_to_completion(monkeypatch, tmp_path)
        name = job.session_name
        gc_jobs.registry._reset_for_tests()
        gc_eventlog._reset_for_tests()

        status = cursor_status("s-spill")
        assert "not tracked live" in status
        assert str(gc_eventlog.log_path(name)) in status

        page = cursor_events("s-spill", offset=0, limit=3)
        assert "events 0–2 of" in page

    def test_paging_an_unknown_log_is_graceful(self, clean_state, tmp_path):
        gc_handles.record("s-nolog", repo=str(tmp_path), status="completed")
        out = cursor_events("s-nolog", offset=0, limit=5)
        assert "no events recorded for session 's-nolog'" in out

    def test_oversized_tool_output_full_in_log_clipped_on_page(
        self, clean_state, monkeypatch, tmp_path
    ):
        """Massive per-event output: full in the JSONL, clipped to 2KB with
        an explicit marker when paged back through cursor_events."""
        big = "y" * 50_000

        def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                   cancel_check=None, session_id=None, model=None):
            yield ("acp.session", {"sessionId": "s-bigout", "cwd": str(workdir),
                                   "model": "m", "resumed": False})
            yield ("acp.update", {
                "sessionUpdate": "tool_call", "toolCallId": "t1",
                "title": "big", "kind": "execute", "status": "pending",
                "rawInput": {"command": "generate"},
            })
            yield ("acp.update", {
                "sessionUpdate": "tool_call_update", "toolCallId": "t1",
                "status": "completed", "rawOutput": {"stdout": big},
            })
            yield ("acp.result", {"stopReason": "end_turn"})

        monkeypatch.setattr(gc_acp, "run_acp", replay)
        cursor_start("t", repo=str(tmp_path))
        job = _job_for("s-bigout")
        assert job.done_event.wait(10)

        on_disk = [l for l in _log_lines(job.session_name)
                   if l["kind"] == "tool_result"][0]
        assert on_disk["output"] == big
        paged = cursor_events("s-bigout", offset=on_disk["seq"], limit=1)
        assert len(paged) < 4_000  # 2KB body + headers, nowhere near 50KB
        assert "y" * 100 in paged
        assert "full event in the JSONL log" in paged


# ---------------------------------------------------------------------------
# Handle scoping — session_key isolation, scope='all', explicit-id crossing
# ---------------------------------------------------------------------------

class TestHandleScoping:
    def _seed_two_sessions(self):
        gc_handles.record("s-alice", repo="/r/a", status="running",
                          session_key="gw:alice")
        gc_handles.record("s-bob", repo="/r/b", status="completed",
                          session_key="gw:bob")

    def test_session_scope_isolates_two_session_keys(self, clean_handles):
        self._seed_two_sessions()
        assert gc_handles.known_handles(
            scope="session", session_key="gw:alice") == ["s-alice"]
        assert gc_handles.known_handles(
            scope="session", session_key="gw:bob") == ["s-bob"]

    def test_scope_all_sees_both(self, clean_handles):
        self._seed_two_sessions()
        assert set(gc_handles.known_handles(scope="all", session_key="gw:alice")) == {
            "s-alice", "s-bob",
        }

    def test_explicit_id_lookup_crosses_scopes(self, clean_handles):
        self._seed_two_sessions()
        # get() takes no scope at all — an explicit handle is explicit intent.
        assert gc_handles.get("s-bob")["session_key"] == "gw:bob"
        assert gc_handles.get("s-alice")["repo"] == "/r/a"

    def test_legacy_entries_without_session_key_belong_to_cli(self, clean_handles):
        gc_handles.record("s-legacy", repo="/r", status="completed")  # no key
        assert gc_handles.known_handles(scope="session", session_key="") == ["s-legacy"]
        assert gc_handles.known_handles(scope="session", session_key="gw:x") == []

    def test_unknown_scope_falls_back_to_session(self, clean_handles):
        self._seed_two_sessions()
        assert gc_handles.known_handles(
            scope="everything", session_key="gw:alice") == ["s-alice"]

    def test_dispatch_records_the_hermes_session_key_on_the_handle(
        self, clean_state, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:alice")
        monkeypatch.setattr(
            gc_acp, "run_acp", _gated_replay_factory(_preset_event(), sid="s-mine")
        )
        cursor_start("t", repo=str(tmp_path))
        assert _job_for("s-mine").done_event.wait(10)
        assert gc_handles.get("s-mine")["session_key"] == "gw:alice"

    def test_unknown_handle_hints_are_scoped_by_default(
        self, clean_state, monkeypatch
    ):
        """Two Hermes sessions share the table: session A's error hints must
        not leak session B's handles unless scope='all' is requested."""
        gc_handles.record("s-mine", repo="/r", status="completed",
                          session_key="gw:alice")
        gc_handles.record("s-theirs", repo="/r", status="completed",
                          session_key="gw:bob")
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:alice")

        for out in (
            cursor_status("s-nope"),
            cursor_send_message("s-nope", "hi"),
            cursor_stop("s-nope"),
            cursor_events("s-nope"),
        ):
            assert "no session named 's-nope'" in out
            assert "s-mine" in out
            assert "s-theirs" not in out

        opened = cursor_status("s-nope", scope="all")
        assert "s-mine" in opened and "s-theirs" in opened

    def test_explicit_status_lookup_crosses_hermes_sessions(
        self, clean_state, monkeypatch
    ):
        gc_handles.record("s-theirs", repo="/r/b", status="completed",
                          session_key="gw:bob", task="their task")
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:alice")
        status = cursor_status("s-theirs")
        assert status.startswith("status: completed")
        assert "working on: their task" in status


# ---------------------------------------------------------------------------
# Handle pruning — age out old terminal entries; cap spares live handles
# ---------------------------------------------------------------------------

class TestHandlePrune:
    def test_old_terminal_handles_age_out_on_write(self, clean_handles):
        gc_handles.record("s-old-done", status="completed")
        gc_handles._table["s-old-done"]["updated_at"] = (
            time.time() - gc_handles.PRUNE_TERMINAL_AFTER_S - 60
        )
        gc_handles.record("s-fresh", status="running")
        assert gc_handles.get("s-old-done") is None
        assert gc_handles.get("s-fresh") is not None

    def test_old_running_handles_are_not_aged_out(self, clean_handles):
        """Age-based pruning only touches TERMINAL states — a long-running
        (or stale-but-unresolved) handle is not silently forgotten."""
        gc_handles.record("s-old-run", status="running")
        gc_handles._table["s-old-run"]["updated_at"] = (
            time.time() - gc_handles.PRUNE_TERMINAL_AFTER_S - 60
        )
        gc_handles.record("s-fresh", status="running")
        assert gc_handles.get("s-old-run") is not None

    def test_recent_terminal_handles_survive(self, clean_handles):
        gc_handles.record("s-done", status="completed")
        gc_handles.record("s-fresh", status="running")
        assert gc_handles.get("s-done") is not None

    def test_cap_evicts_terminal_entries_before_running_ones(
        self, clean_handles, monkeypatch
    ):
        monkeypatch.setattr(gc_handles, "MAX_ENTRIES", 3)
        now = time.time()
        gc_handles.record("s-run-old", status="running")
        gc_handles._table["s-run-old"]["updated_at"] = now - 300
        gc_handles.record("s-done-new", status="completed")
        gc_handles._table["s-done-new"]["updated_at"] = now - 10
        gc_handles.record("s-run-new", status="running")
        gc_handles._table["s-run-new"]["updated_at"] = now - 5
        gc_handles.record("s-final", status="running")
        # Over cap by one: the TERMINAL entry goes, even though a running
        # entry is older — finished runs never push out live handles.
        assert len(gc_handles._table) <= 3
        assert gc_handles.get("s-done-new") is None
        assert gc_handles.get("s-run-old") is not None
        assert gc_handles.get("s-run-new") is not None
        assert gc_handles.get("s-final") is not None

    def test_cap_still_enforced_when_everything_is_running(
        self, clean_handles, monkeypatch
    ):
        monkeypatch.setattr(gc_handles, "MAX_ENTRIES", 2)
        for i in range(3):
            gc_handles.record(f"s-run-{i}", status="running")
            gc_handles._table[f"s-run-{i}"]["updated_at"] = float(i)
        gc_handles.record("s-run-final", status="running")
        assert len(gc_handles._table) <= 2
        assert "s-run-final" in gc_handles._table
        assert "s-run-0" not in gc_handles._table


# ---------------------------------------------------------------------------
# names.py — adjective-adjective-noun slugs, collision retry, suffix fallback
# ---------------------------------------------------------------------------

class TestSessionNames:
    def test_generated_slug_is_mood_modifier_creature(self):
        name = gc_names.generate(taken=lambda n: False)
        mood, modifier, creature = name.split("-")
        assert mood in gc_names.MOODS
        assert modifier in gc_names.MODIFIERS
        assert creature in gc_names.CREATURES

    def test_vocabulary_is_50x50x100(self):
        assert len(set(gc_names.MOODS)) == 50
        assert len(set(gc_names.MODIFIERS)) == 50
        assert len(set(gc_names.CREATURES)) == 100

    def test_collision_retries_until_a_free_slug(self):
        rng = random.Random(7)
        first = gc_names.generate(taken=lambda n: False, rng=random.Random(7))
        # The same rng seed with the first draw claimed must yield a
        # DIFFERENT (second-draw) name, not the taken one.
        second = gc_names.generate(taken=lambda n: n == first, rng=rng)
        assert second != first
        assert second.split("-")[0] in gc_names.MOODS

    def test_exhausted_draws_fall_back_to_a_numeric_suffix(self):
        def is_taken(n):
            return n.count("-") == 2  # every plain slug is claimed

        # The fallback must kick in and still terminate, with a suffix.
        name = gc_names.generate(taken=is_taken, rng=random.Random(1))
        mood, modifier, creature, suffix = name.split("-")
        assert suffix == "2"
        assert mood in gc_names.MOODS
        assert creature in gc_names.CREATURES

    def test_suffix_increments_past_taken_suffixes(self):
        def is_taken(n):
            # Plain slugs and '-2' both claimed → must land on '-3'.
            return n.count("-") == 2 or n.endswith("-2")

        name = gc_names.generate(taken=is_taken, rng=random.Random(1))
        assert name.endswith("-3")


# ---------------------------------------------------------------------------
# cursor_create_session — lazy named handles + UUID alias resolution
# ---------------------------------------------------------------------------

def _created_name(ack):
    assert ack.startswith("session: "), f"not a create ack: {ack!r}"
    return ack.splitlines()[0].split("session: ", 1)[1]


class TestCursorCreateSession:
    def test_creates_a_named_handle_and_dispatches_nothing(
        self, clean_state, monkeypatch, tmp_path
    ):
        boom = lambda *a, **k: pytest.fail("create must not spawn ACP")
        monkeypatch.setattr(gc_acp, "run_acp", boom)

        ack = cursor_create_session(repo=str(tmp_path))
        name = _created_name(ack)
        # The exact ack format from the spec: 2 headers + instruction.
        assert ack == (
            f"session: {name}\n"
            f"repo: {_resolved(tmp_path)} · model: {gc._resolve_model(None) or 'default'}\n"
            "created. send work with cursor_send_message."
        )
        # Named like playful-space-bunny, from the embedded word lists.
        mood, modifier, creature = name.split("-")
        assert mood in gc_names.MOODS and creature in gc_names.CREATURES
        # LAZY: nothing running, but the handle exists as 'created'.
        assert gc_jobs.registry.list_jobs() == []
        entry = gc_handles.get(name)
        assert entry["status"] == "created"
        assert entry["repo"] == _resolved(tmp_path)

    def test_minted_names_avoid_existing_handles(self, clean_state, tmp_path):
        names = {
            _created_name(cursor_create_session(repo=str(tmp_path)))
            for _ in range(5)
        }
        assert len(names) == 5  # collision-checked against the table

    def test_explicit_model_is_recorded_on_the_handle(
        self, clean_state, tmp_path
    ):
        ack = cursor_create_session(repo=str(tmp_path), model="composer-x")
        name = _created_name(ack)
        assert "model: composer-x" in ack
        assert gc_handles.get(name)["model"] == "composer-x"

    def test_unresolvable_repo_is_actionable_prose(self, clean_state, monkeypatch):
        monkeypatch.setattr(gc, "_default_repo", lambda: None)
        out = cursor_create_session()
        assert "no workspace repo resolvable" in out
        assert gc.REPO_ENV_VAR in out

    def test_missing_repo_dir_is_actionable_prose(self, clean_state):
        out = cursor_create_session(repo="/definitely/not/a/dir")
        assert "cannot create session" in out
        assert "not an existing directory" in out

    def test_handler_maps_args(self, clean_state, tmp_path):
        ack = _handle_cursor_create_session({
            "repo": str(tmp_path), "model": "m-x",
        })
        assert ack.startswith("session: ")
        assert "model: m-x" in ack

    def test_uuid_alias_resolves_everywhere_a_name_does(
        self, clean_state, monkeypatch, tmp_path
    ):
        """After the first send binds the cursor UUID, name and UUID are
        interchangeable across status/stop/events/send."""
        monkeypatch.setattr(
            gc_acp, "run_acp",
            _gated_replay_factory(_preset_event(), sid="11111111-2222-3333"),
        )
        name = _created_name(cursor_create_session(repo=str(tmp_path)))
        cursor_send_message(name, "do the thing")
        assert gc_jobs.registry.get_by_name(name).done_event.wait(10)

        assert gc_handles.resolve("11111111-2222-3333") == name
        by_name, by_uuid = cursor_status(name), cursor_status("11111111-2222-3333")
        # Identical view, however addressed (allow the elapsed clock to tick).
        assert by_uuid.splitlines()[0] == by_name.splitlines()[0]
        assert f"session: {name}" in by_uuid
        assert f"session: {name}" in cursor_stop("11111111-2222-3333")
        assert "events" in cursor_events("11111111-2222-3333")


# ---------------------------------------------------------------------------
# cursor_events — tail defaults, negative offsets, kind filter, clips, cap
# ---------------------------------------------------------------------------

def _seed_session_log(name, envelopes, repo="/r"):
    """A persisted handle + synthetic JSONL history for paging tests."""
    gc_handles.record(name, repo=repo, status="completed")
    for env in envelopes:
        gc_eventlog.append(name, env)


class TestCursorEventsPaging:
    def test_bare_call_defaults_to_the_last_10_events(self, clean_state):
        _seed_session_log("tail-y-log", [
            {"kind": "content", "delta": f"e{i}"} for i in range(25)
        ])
        out = cursor_events("tail-y-log")
        assert "events 15–24 of 25" in out
        assert '"e15"' in out and '"e24"' in out
        assert '"e14"' not in out

    def test_negative_offset_pages_backwards_python_style(self, clean_state):
        _seed_session_log("neg-off-log", [
            {"kind": "content", "delta": f"e{i}"} for i in range(25)
        ])
        # offset=-11 = window ENDING at the 11th-from-last event.
        out = cursor_events("neg-off-log", offset=-11, limit=10)
        assert "events 5–14 of 25" in out
        # More negative than the history: clamps to the start, no crash.
        assert "events 0–1 of 25" in cursor_events(
            "neg-off-log", offset=-24, limit=2
        )

    def test_positive_offset_pages_forward_from_that_seq(self, clean_state):
        _seed_session_log("fwd-log", [
            {"kind": "content", "delta": f"e{i}"} for i in range(25)
        ])
        out = cursor_events("fwd-log", offset=3, limit=4)
        assert "events 3–6 of 25" in out

    def test_kind_filter_selects_only_that_kind(self, clean_state):
        envs = []
        for i in range(6):
            envs.append({"kind": "content", "delta": f"prose{i}"})
            envs.append({"kind": "lifecycle", "event": "reasoning",
                         "text": f"thinking hard {i}"})
        _seed_session_log("kind-log", envs)

        out = cursor_events("kind-log", kind="reasoning")
        assert "6 matching (kind=reasoning) of 12" in out
        assert "thinking hard 5" in out
        assert "prose" not in out

        # The window applies to the FILTERED sequence.
        two = cursor_events("kind-log", kind="reasoning", offset=-1, limit=2)
        assert "thinking hard 4" in two and "thinking hard 5" in two
        assert "thinking hard 3" not in two

    def test_limit_clamps_at_500(self, clean_state):
        _seed_session_log("clamp-log", [
            {"kind": "content", "delta": f"e{i}"} for i in range(510)
        ])
        out = cursor_events("clamp-log", offset=0, limit=99999)
        assert "events 0–499 of 510" in out  # 500, not 510

    def test_reasoning_text_clips_inline_at_2kb_with_log_pointer(
        self, clean_state
    ):
        wall = "reason " * 2000  # ~14KB of thinking
        _seed_session_log("clip-log", [
            {"kind": "lifecycle", "event": "reasoning", "text": wall},
        ])
        out = cursor_events("clip-log", kind="reasoning")
        assert len(out) < 4_000
        assert "full event in the JSONL log" in out
        # The full text stayed intact on disk.
        assert _log_lines("clip-log")[0]["text"] == wall

    def test_total_response_caps_at_20kb_with_truncation_note(
        self, clean_state
    ):
        # 30 events x ~1.9KB inline (each under the 2KB per-event clip) —
        # only the response-level cap can bound this page.
        _seed_session_log("cap-log", [
            {"kind": "lifecycle", "event": "reasoning", "text": f"e{i} " + "x" * 1900}
            for i in range(30)
        ])
        out = cursor_events("cap-log", offset=0, limit=30)
        assert len(out) <= gc_render.EVENTS_RESPONSE_CAP + 200
        assert gc_render.EVENTS_TRUNCATION_NOTE in out
        assert "e0 " in out          # the page kept its head...
        assert "e29 " not in out     # ...and dropped whole trailing events

    def test_empty_window_names_the_filter(self, clean_state):
        _seed_session_log("empty-log", [{"kind": "content", "delta": "hi"}])
        out = cursor_events("empty-log", kind="file_diff")
        assert "no events of kind 'file_diff'" in out

    def test_unknown_session_is_actionable_prose(self, clean_state):
        assert "no session named 's-nope'" in cursor_events("s-nope")

    def test_handler_maps_args(self, clean_state):
        _seed_session_log("handler-log", [
            {"kind": "content", "delta": f"e{i}"} for i in range(5)
        ])
        out = _handle_cursor_events({
            "session": "handler-log", "offset": 0, "limit": 2,
        })
        assert "events 0–1 of 5" in out


# ---------------------------------------------------------------------------
# cursor_list — TSV shape + hermes-session scoping
# ---------------------------------------------------------------------------

class TestCursorList:
    def test_empty_table_is_helpful_prose(self, clean_state, monkeypatch):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:me")
        assert "no cursor sessions in this hermes session" in cursor_list()
        assert "no cursor sessions exist yet" in cursor_list(scope="all")

    def test_tsv_has_header_row_and_tab_separated_columns(
        self, clean_state, monkeypatch
    ):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:me")
        gc_handles.record(
            "brave-jade-owl", repo="/r/a", status="completed",
            session_key="gw:me", files_changed_count=3, duration_s=61.2,
        )
        out = cursor_list()
        lines = out.splitlines()
        header = [c.strip() for c in lines[0].split("\t")]
        assert header == ["session", "repo", "status", "elapsed",
                          "files", "last_activity"]
        row = [c.strip() for c in lines[1].split("\t")]
        assert row == ["brave-jade-owl", "/r/a", "completed", "61s", "3", "—"]

    def test_default_scope_is_the_current_hermes_session(
        self, clean_state, monkeypatch
    ):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:me")
        gc_handles.record("mine-fox", repo="/a", status="completed",
                          session_key="gw:me")
        gc_handles.record("theirs-owl", repo="/b", status="completed",
                          session_key="gw:other")
        mine = cursor_list()
        assert "mine-fox" in mine and "theirs-owl" not in mine
        everyone = cursor_list(scope="all")
        assert "mine-fox" in everyone and "theirs-owl" in everyone

    def test_running_session_renders_live_state(
        self, clean_state, monkeypatch, tmp_path
    ):
        release = threading.Event()
        monkeypatch.setattr(gc_acp, "run_acp",
                            _gated_replay_factory(release, sid="s-live-list"))
        cursor_start("t", repo=str(tmp_path))
        job = _job_for("s-live-list")
        try:
            assert _wait_until(lambda: job.files)
            row = [
                l for l in cursor_list(scope="all").splitlines()
                if job.session_name in l
            ][0]
            cells = [c.strip() for c in row.split("\t")]
            assert cells[2] == "running"
            assert cells[4] == "1"  # live files count, not the stale record
        finally:
            release.set()
        assert job.done_event.wait(10)

    def test_handler_maps_scope(self, clean_state, monkeypatch):
        monkeypatch.setattr(gc, "_resolve_session_key", lambda: "gw:me")
        gc_handles.record("theirs-owl", repo="/b", status="completed",
                          session_key="gw:other")
        assert "theirs-owl" in _handle_cursor_list({"scope": "all"})
        assert "theirs-owl" not in _handle_cursor_list({})
