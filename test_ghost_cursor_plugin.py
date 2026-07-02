"""Tests for the ghost_cursor plugin (cursor_edit tool).

The tool now runs cursor-agent over ACP (``acp_runner.py``). Covered here:

* ``events.AcpNormalizer`` — session/update → canonical envelope mapping
  against a CAPTURED fixture (``fixtures/acp_session_updates.jsonl``, real
  ``cursor-agent acp`` notifications, 2026-07-02).
* The ``cursor_edit`` handler — progress emission + summary building with the
  ACP runner replayed from the fixture; timeout / cancel / handshake-failure
  paths; git fallback for shell-driven edits.
* ``acp_runner.run_acp`` — real subprocess round-trips against a fake ACP
  server (happy path, native session/cancel, hang→timeout kill, dead binary).
* The legacy ``--print`` runner + ``normalize_harness`` mapping (kept as
  fallback/reference — must stay importable and correct).

No live cursor-agent runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.ghost_cursor import (
    CURSOR_EDIT_SCHEMA,
    TOOL_NAME,
    TOOLSET,
    _resolve_progress_callback,
    check_cursor_edit_available,
    cursor_edit,
    register,
)
from plugins.ghost_cursor import acp_runner as gc_acp
from plugins.ghost_cursor import events as gc_events
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
# cursor_edit handler — progress emission + summary (ACP replayed)
# ---------------------------------------------------------------------------

class TestCursorEditHandler:
    def test_emits_progress_and_returns_summary(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)

        seen = []

        def pcb(event_type, tool_name, preview, args, **kwargs):
            seen.append((event_type, tool_name, json.loads(preview)))

        result = json.loads(
            cursor_edit("add multiply", repo=str(tmp_path), progress_callback=pcb)
        )

        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["live_progress"] is True
        assert result["files_changed_count"] == 2
        by_path = {f["path"]: f for f in result["files_changed"]}
        calc = by_path["/tmp/acp_probe/repo/calc.py"]
        assert calc["added"] == 8 and calc["removed"] == 0
        assert calc["status"] == "M"
        assert "multiply" in calc["diff"]
        notes = by_path["/tmp/acp_probe/repo/notes.txt"]
        assert notes["status"] == "A"
        assert "subtract" in result["summary"]

        # Every emission uses the shape the api_server forwards as
        # event: tool.progress — reasoning.available + tool_name marker.
        assert seen
        assert all(et == "reasoning.available" for et, _, _ in seen)
        assert all(tn == TOOL_NAME for _, tn, _ in seen)
        assert result["progress_events_emitted"] == len(seen)
        # file_diff envelopes rode through progress with full content.
        diffs = [p for _, _, p in seen if p.get("kind") == "file_diff"]
        assert len(diffs) == 2
        assert any("multiply" in d["after"] for d in diffs)
        # reasoning fragments streamed too.
        assert any(
            p.get("kind") == "lifecycle" and p.get("event") == "reasoning"
            for _, _, p in seen
        )

    def test_works_without_progress_callback(self, monkeypatch, tmp_path):
        """No resolvable agent → no live progress, but diffs still persist."""
        monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)
        monkeypatch.setattr(
            "plugins.ghost_cursor._resolve_progress_callback", lambda: None
        )

        result = json.loads(cursor_edit("add multiply", repo=str(tmp_path)))
        assert result["success"] is True
        assert result["live_progress"] is False
        assert result["progress_events_emitted"] == 0
        by_path = {f["path"]: f for f in result["files_changed"]}
        assert by_path["/tmp/acp_probe/repo/calc.py"]["added"] == 8

    def test_progress_callback_errors_never_break_the_tool(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)

        def broken(*a, **k):
            raise RuntimeError("consumer died")

        result = json.loads(
            cursor_edit("t", repo=str(tmp_path), progress_callback=broken)
        )
        assert result["success"] is True
        assert result["progress_events_emitted"] == 0

    def test_timeout_returns_partial_error(self, monkeypatch, tmp_path):
        def timing_out(*_a, **_k):
            # One real edit lands, then the run times out.
            for key, obj in _acp_replay_events():
                yield key, obj
                if key == "acp.update" and obj.get("sessionUpdate") == "tool_call_update" \
                        and obj.get("status") == "completed" and obj.get("content"):
                    break
            yield ("acp.error", {"error": "ACP run timed out after 5s", "timeout": True})

        monkeypatch.setattr(gc_acp, "run_acp", timing_out)
        result = json.loads(
            cursor_edit("t", repo=str(tmp_path), progress_callback=None)
        )
        assert result["success"] is False
        assert result["status"] == "timeout"
        assert "timed out" in result["error"]
        # Partial progress is preserved in the result.
        assert result["partial"] is True
        assert result["files_changed_count"] == 1

    def test_cancel_returns_partial_failed(self, monkeypatch, tmp_path):
        def cancelling(*_a, **_k):
            for key, obj in _acp_replay_events():
                if key == "acp.result":
                    break
                yield key, obj
            yield ("acp.result", {"stopReason": "cancelled"})

        monkeypatch.setattr(gc_acp, "run_acp", cancelling)
        result = json.loads(
            cursor_edit("t", repo=str(tmp_path), progress_callback=None)
        )
        assert result["success"] is False
        assert result["status"] == "failed"
        assert "cancel" in result["error"]
        assert result["partial"] is True
        assert result["files_changed_count"] == 2

    def test_acp_handshake_failure_is_actionable_error(self, monkeypatch, tmp_path):
        def failing(*_a, **_k):
            raise gc_acp.AcpError(
                "ACP initialize handshake with cursor-agent failed (boom)"
            )
            yield  # pragma: no cover — make it a generator

        monkeypatch.setattr(gc_acp, "run_acp", failing)
        result = json.loads(cursor_edit("t", repo=str(tmp_path)))
        assert result["success"] is False
        assert "handshake" in result["error"]
        assert "files_changed" not in result  # no run happened

    def test_missing_repo_is_a_clean_error(self, monkeypatch):
        monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)
        result = json.loads(cursor_edit("t", repo="/definitely/not/a/dir"))
        assert result["success"] is False
        assert "not an existing directory" in result["error"]

    def test_empty_task_is_a_clean_error(self):
        result = json.loads(cursor_edit("   "))
        assert result["success"] is False
        assert "task is required" in result["error"]


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

    def test_shell_driven_edit_lands_via_git_fallback(self, monkeypatch, git_repo):
        # Pre-existing dirty file must NOT be attributed to the run.
        (git_repo / "pre.txt").write_text("pre dirty before run\n")

        def shell_edit_replay(*_a, **_k):
            yield ("acp.session", {"sessionId": "s", "cwd": str(git_repo), "model": "m"})
            # Simulates cursor editing through a shell command: no diff
            # content ever appears on the ACP stream.
            (git_repo / "tool.txt").write_text("orig\nedited by shell\n")
            (git_repo / "new.txt").write_text("brand new\n")
            yield ("acp.result", {"stopReason": "end_turn"})

        monkeypatch.setattr(gc_acp, "run_acp", shell_edit_replay)
        seen = []

        def pcb(event_type, tool_name, preview, args, **kwargs):
            seen.append(json.loads(preview))

        result = json.loads(
            cursor_edit("edit via shell", repo=str(git_repo), progress_callback=pcb)
        )
        assert result["success"] is True
        by_path = {Path(f["path"]).name: f for f in result["files_changed"]}
        assert set(by_path) == {"tool.txt", "new.txt"}  # pre.txt excluded
        assert by_path["tool.txt"]["status"] == "M"
        assert by_path["tool.txt"]["added"] == 1
        assert "edited by shell" in by_path["tool.txt"]["diff"]
        assert by_path["new.txt"]["status"] == "A"
        # Fallback diffs also went out through the progress callback.
        fallback_diffs = [e for e in seen if e.get("kind") == "file_diff"]
        assert len(fallback_diffs) == 2

    def test_acp_stream_diff_wins_over_fallback(self, monkeypatch, git_repo):
        """A file already captured from the ACP stream is not re-added."""

        def stream_and_shell(*_a, **_k):
            yield ("acp.session", {"sessionId": "s", "cwd": str(git_repo), "model": "m"})
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
        result = json.loads(cursor_edit("t", repo=str(git_repo), progress_callback=None))
        assert result["files_changed_count"] == 1
        assert result["files_changed"][0]["added"] == 1

    def test_unified_diff_text_counts(self):
        diff, added, removed = gc_events.unified_diff_text("a\nb\n", "a\nc\nd\n", "/w/f.txt")
        assert added == 2 and removed == 1
        assert "--- a/w/f.txt" in diff and "+++ b/w/f.txt" in diff


# ---------------------------------------------------------------------------
# acp_runner.run_acp — real subprocess round-trips against a fake ACP server
# ---------------------------------------------------------------------------

_FAKE_ACP = '''#!/usr/bin/env python3
import json, sys
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
    elif m == "session/prompt":
        prompt_id = msg["id"]
        update({"sessionUpdate": "agent_message_chunk",
                "content": {"type": "text", "text": "working"}})
        if MODE == "happy":
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
        events = list(gc_acp.run_acp("do it", str(tmp_path), timeout=30.0,
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
        for key, obj in gc_acp.run_acp("do it", str(tmp_path), timeout=30.0,
                                       cancel_check=lambda: cancelled["flag"]):
            events.append((key, obj))
            if key == "acp.update":
                cancelled["flag"] = True  # interrupt lands mid-run
        elapsed = time.monotonic() - t0
        assert events[-1] == ("acp.result", {"stopReason": "cancelled"})
        assert elapsed < 15, f"native cancel did not resolve promptly ({elapsed:.1f}s)"

    def test_hang_times_out_and_kills(self, tmp_path, monkeypatch):
        _install_fake_acp(tmp_path, monkeypatch, "hang")
        monkeypatch.setattr(gc_acp, "CANCEL_GRACE_S", 1.0)
        t0 = time.monotonic()
        events = list(gc_acp.run_acp("do it", str(tmp_path), timeout=1.0,
                                     cancel_check=lambda: False))
        elapsed = time.monotonic() - t0
        assert events[-1][0] == "acp.error"
        assert events[-1][1]["timeout"] is True
        assert elapsed < 20, f"timeout kill did not tear down ({elapsed:.1f}s)"

    def test_dead_binary_raises_actionable_acp_error(self, tmp_path, monkeypatch):
        _install_fake_acp(tmp_path, monkeypatch, "die")
        with pytest.raises(gc_acp.AcpError) as exc_info:
            list(gc_acp.run_acp("do it", str(tmp_path), timeout=10.0,
                                cancel_check=lambda: False))
        assert "handshake" in str(exc_info.value)

    def test_empty_task_raises(self, tmp_path):
        with pytest.raises(gc_runner.HarnessError):
            list(gc_acp.run_acp("  ", str(tmp_path)))

    def test_bad_repo_raises(self):
        with pytest.raises(gc_runner.HarnessError):
            list(gc_acp.run_acp("t", "/nope/nothing/here"))


# ---------------------------------------------------------------------------
# Agent auto-resolution via the thread-local activity callback
# ---------------------------------------------------------------------------

class TestProgressCallbackResolution:
    def test_resolves_agent_from_thread_local_activity_callback(self, monkeypatch, tmp_path):
        """tool_executor installs agent._touch_activity thread-locally before
        dispatch; the plugin walks __self__ back to the agent and uses its
        tool_progress_callback."""
        from tools.environments.base import set_activity_callback

        seen = []

        class FakeAgent:
            def __init__(self):
                self.tool_progress_callback = (
                    lambda *a, **k: seen.append((a, k))
                )

            def _touch_activity(self, desc):
                pass

        agent = FakeAgent()
        set_activity_callback(agent._touch_activity)
        try:
            pcb = _resolve_progress_callback()
            assert pcb is agent.tool_progress_callback

            monkeypatch.setattr(gc_acp, "run_acp", _replay_acp)
            result = json.loads(cursor_edit("t", repo=str(tmp_path)))
            assert result["live_progress"] is True
            assert result["progress_events_emitted"] == len(seen) > 0
        finally:
            set_activity_callback(None)

    def test_no_thread_local_and_no_cli_returns_none(self):
        from tools.environments.base import set_activity_callback

        set_activity_callback(None)
        # _cli_ref is None outside an interactive CLI run (gateway/tests).
        assert _resolve_progress_callback() is None


# ---------------------------------------------------------------------------
# Registration + availability gate
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_wires_tool_into_ghost_cursor_toolset(self):
        captured = {}

        ctx = SimpleNamespace(
            register_tool=lambda **kw: captured.update(kw)
        )
        register(ctx)

        assert captured["name"] == TOOL_NAME
        assert captured["toolset"] == TOOLSET
        assert captured["check_fn"] is check_cursor_edit_available
        assert captured["schema"] is CURSOR_EDIT_SCHEMA
        assert captured["schema"]["parameters"]["required"] == ["task"]
        # The handler returns a JSON string (registry contract).
        assert callable(captured["handler"])

    def test_schema_steers_delegation(self):
        desc = CURSOR_EDIT_SCHEMA["description"].lower()
        assert "prefer this tool" in desc
        assert "delegate" in desc

    def test_check_fn_false_without_binary(self, monkeypatch):
        monkeypatch.setattr(gc_runner, "cursor_agent_available", lambda: False)
        assert check_cursor_edit_available() is False

    def test_check_fn_true_with_binary_and_repo(self, monkeypatch):
        monkeypatch.setattr(gc_runner, "cursor_agent_available", lambda: True)
        # _default_repo falls back to os.getcwd(), which always exists.
        assert check_cursor_edit_available() is True


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
