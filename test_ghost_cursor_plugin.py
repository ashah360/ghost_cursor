"""Tests for the ghost_cursor plugin (v0.5: cursor-sdk transport).

Six tools: ``cursor_create_session`` / ``cursor_send_message`` /
``cursor_status`` / ``cursor_stop`` / ``cursor_events`` / ``cursor_list`` —
all keyed on adjective-adjective-noun session names (cursor agent ids
resolve as aliases). Every tool returns plain text (labeled headers, prose,
raw fenced diffs, TSV) — never JSON. Covered here:

* ``sdk_runner.run_sdk`` — the cursor-sdk transport against a fully faked
  bridge/client (happy path, native run.cancel, inactivity watchdog with
  pending-tool-call suspension, max-wall ceiling, Agent.resume + fresh
  fallback, bounded is_retryable retries, observe(after_offset) stream
  re-attach, bridge caching/shutdown, missing-key/missing-sdk preflight).
* ``events.SdkNormalizer`` — SDKMessage dicts → canonical envelope mapping,
  including defensive parsing of the (explicitly unstable) tool_call
  payload shapes, plus a full-run fixture replay
  (``fixtures/sdk_stream.jsonl``).
* The tool handlers — session lifecycle (create → lazy first send → status
  → stop/follow-up), the read-only guarantee of ``cursor_status``, the
  same-repo concurrency guard, name minting + collision retry + agent-id
  alias resolution, ``cursor_events`` paging (tail defaults, negative
  offsets, kind filter, 2KB inline clip, 20KB response cap), ``cursor_list``
  TSV + scoping, model threading, completion delivery on the shared
  async-delegation rail, and actionable prose errors for bogus/expired
  handles. The SDK layer is replayed with fast deterministic fakes (no
  live bridge, no network).
* ``handles.py`` — the persistent handle table (explicit lookup only; the
  v0.2 auto-resume heuristic is gone by design).
* The legacy ``--print`` runner + ``normalize_harness`` mapping (kept as
  fallback/reference — must stay importable and correct).

No live cursor runs.
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
    CURSOR_STATUS_SCHEMA,
    CURSOR_STOP_SCHEMA,
    CURSOR_SUBSCRIBE_SCHEMA,
    EVENTS_TOOL_NAME,
    LIST_TOOL_NAME,
    SEND_TOOL_NAME,
    STATUS_TOOL_NAME,
    STOP_TOOL_NAME,
    SUBSCRIBE_TOOL_NAME,
    TOOLSET,
    _handle_cursor_create_session,
    _handle_cursor_events,
    _handle_cursor_list,
    _handle_cursor_send_message,
    _handle_cursor_status,
    _handle_cursor_stop,
    _handle_cursor_subscribe,
    check_cursor_available,
    cursor_create_session,
    cursor_events,
    cursor_list,
    cursor_send_message,
    cursor_status,
    cursor_stop,
    cursor_subscribe,
    register,
)
from plugins.ghost_cursor import events as gc_events
from plugins.ghost_cursor import sdk_runner as gc_sdk
from plugins.ghost_cursor import handles as gc_handles
from plugins.ghost_cursor import jobs as gc_jobs
from plugins.ghost_cursor import names as gc_names
from plugins.ghost_cursor import progress as gc_progress
from plugins.ghost_cursor import render as gc_render
from plugins.ghost_cursor import runner as gc_runner

FIXTURE = Path(__file__).parent / "fixtures" / "cursor_stream.jsonl"
SDK_FIXTURE = Path(__file__).parent / "fixtures" / "sdk_stream.jsonl"
# Raw tool_call SDKMessages captured VERBATIM from a real cursor-sdk run
# (2026-07-03, model gpt-5.4-nano) that created + committed a file — the
# reproduction of the "completion said no files were changed" blind spot.
SDK_EDIT_FIXTURE = Path(__file__).parent / "fixtures" / "sdk_edit_tool_call.jsonl"


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
# SDK fixture replay helpers
# ---------------------------------------------------------------------------

def _sdk_fixture_messages():
    """The SDKMessage dicts from the fixture stream."""
    return [
        json.loads(line)
        for line in SDK_FIXTURE.read_text().splitlines()
        if line.strip()
    ]


def _sdk_replay_events():
    """A full run's worth of sdk_runner events built from the fixture."""
    events = [(
        "sdk.session",
        {"agentId": "agent-fixture", "cwd": "/tmp/sdk_probe/repo",
         "model": "fake-model", "resumed": False},
    )]
    events.extend(("sdk.message", msg) for msg in _sdk_fixture_messages())
    events.append(("sdk.result", {"status": "finished"}))
    return events


def _replay_sdk(*_args, **_kwargs):
    yield from _sdk_replay_events()


def _normalize_all(events):
    normalizer = gc_events.SdkNormalizer()
    envs = []
    for key, obj in events:
        envs.extend(normalizer.normalize(key, obj))
    return envs


# ---------------------------------------------------------------------------
# Tool-test plumbing — deterministic gated fakes for the SDK layer
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
    completion queue + no live progress tickers."""
    gc_progress._reset_for_tests()
    gc_jobs.registry._reset_for_tests()
    _drain_completion_queue()
    monkeypatch.setattr(gc_handles, "_table", {})
    monkeypatch.setattr(gc_handles, "_loaded", False)
    gc_eventlog._reset_for_tests()
    yield gc_jobs.registry
    gc_progress._reset_for_tests()
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


def _gated_replay_factory(release, sid="agent-run", early_edit=True, late_edit=True):
    """A cancel-aware replay held open on ``release``.

    Yields the sdk.session event (the handle), optionally one early file
    edit, then blocks until ``release`` is set — honoring ``cancel_check``
    the way the real ``run_sdk`` does (a cancel mid-run resolves with
    ``status: "cancelled"``). After release: an optional second edit, a
    summary message, and a clean finished result.
    """

    def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
               cancel_check=None, agent_id=None, model=None):
        yield ("sdk.session", {
            "agentId": sid, "cwd": str(workdir),
            "model": model or "fake-model", "resumed": bool(agent_id),
        })
        if early_edit:
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "edit_file",
                "status": "running", "args": {"path": f"{workdir}/f1.py"},
            })
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "edit_file",
                "status": "completed",
                "result": {"path": f"{workdir}/f1.py",
                           "oldText": "a\n", "newText": "a\nb\n"},
            })
        while not release.is_set():
            if cancel_check and cancel_check():
                yield ("sdk.result", {"status": "cancelled"})
                return
            time.sleep(0.01)
        if late_edit:
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t2", "name": "edit_file",
                "status": "running", "args": {"path": f"{workdir}/f2.py"},
            })
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t2", "name": "edit_file",
                "status": "completed",
                "result": {"path": f"{workdir}/f2.py",
                           "oldText": "", "newText": "new\n"},
            })
        yield ("sdk.message", {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "all done"}]},
        })
        yield ("sdk.result", {"status": "finished"})

    return replay


class _SdkSequence:
    """Route successive run_sdk calls to successive replay factories,
    recording each call's kwargs for assertion."""

    def __init__(self, *factories):
        self._factories = list(factories)
        self.calls = []

    def __call__(self, task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                 cancel_check=None, agent_id=None, model=None):
        self.calls.append({
            "task": task, "workdir": str(workdir),
            "agent_id": agent_id, "model": model,
            "inactivity_timeout_s": inactivity_timeout_s,
            "max_wall_s": max_wall_s,
        })
        factory = self._factories.pop(0)
        return factory(task, workdir, inactivity_timeout_s=inactivity_timeout_s,
                       max_wall_s=max_wall_s, cancel_check=cancel_check,
                       agent_id=agent_id, model=model)


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


def _created_name(ack):
    assert ack.startswith("session: "), f"not a create ack: {ack!r}"
    return ack.splitlines()[0].split("session: ", 1)[1]


def _start_run(task, repo, model=None):
    """Test setup shorthand: create a session + send the first message."""
    name = _created_name(cursor_create_session(repo=repo, model=model))
    return cursor_send_message(name, task)


# ---------------------------------------------------------------------------
# create + first send — new runs: ack out, running status, delivery on done
# ---------------------------------------------------------------------------

class TestFirstSendRun:
    """The create + first-send flow: background-run plumbing, session-name
    handles, plain-text acks."""

    def test_new_run_returns_running_ack_and_registers_named_handle(
        self, clean_state, monkeypatch, tmp_path
    ):
        release = threading.Event()
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(release, sid="s-new"))

        ack = _start_run("add multiply", repo=str(tmp_path))
        try:
            _assert_running_ack(ack)

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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(release, sid="s-done"))

        _assert_running_ack(_start_run("t", repo=str(tmp_path)))
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
        """The fixture SDK stream flows through the job aggregation: both
        edits land in files_changed with diffs, and the summary is the
        FINAL content block of the turn (the wrap-up message), not every
        interstitial fragment fused together."""
        monkeypatch.setattr(gc_sdk, "run_sdk", _replay_sdk)
        _start_run("add multiply", repo=str(tmp_path))

        job = _job_for("agent-fixture")
        assert job.done_event.wait(10)
        result = job.result
        assert result["success"] is True
        assert result["status"] == "completed"
        assert result["session_id"] == "agent-fixture"
        assert result["resumed"] is False
        assert result["files_changed_count"] == 2
        by_path = {f["path"]: f for f in result["files_changed"]}
        calc = by_path["/tmp/sdk_probe/repo/calc.py"]
        assert calc["added"] == 8 and calc["removed"] == 0
        assert calc["status"] == "M"
        assert "multiply" in calc["diff"]
        assert by_path["/tmp/sdk_probe/repo/notes.txt"]["status"] == "A"
        # The capture's final message block is the notes.txt wrap-up.
        assert result["summary"].startswith("Both done")
        assert "not deleted or touched" in result["summary"]
        # Earlier narration blocks ("Added `subtract`…", "I'll run `ls -…")
        # are NOT concatenated into the summary.
        assert "subtract" not in result["summary"]
        assert "I'll run" not in result["summary"]

    def test_handshake_failure_reports_in_turn_and_never_delivers(
        self, clean_state, monkeypatch, tmp_path
    ):
        def failing(*_a, **_k):
            raise gc_sdk.SdkRunnerError(
                "failed to create cursor agent via cursor-sdk (boom)"
            )

        monkeypatch.setattr(gc_sdk, "run_sdk", failing)
        out = _start_run("t", repo=str(tmp_path))
        assert "status: failed" in out
        assert "failed to create cursor agent" in out
        # Errors are prose sentences with a next step, not codes.
        assert "send another message" in out
        # Exactly-once: the tool result IS the report; nothing enqueued.
        assert _drain_completion_queue() == []

    def test_wedged_prehandshake_run_is_cancelled_and_reported(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A run that never establishes an agent gets cancelled and
        the tool returns an actionable failure instead of blocking forever."""
        monkeypatch.setattr(gc, "_HANDLE_WAIT_S", 0.3)

        def never_session(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                          cancel_check=None, agent_id=None, model=None):
            while not (cancel_check and cancel_check()):
                time.sleep(0.01)
            return
            yield  # pragma: no cover — make it a generator

        monkeypatch.setattr(gc_sdk, "run_sdk", never_session)
        out = _start_run("t", repo=str(tmp_path))
        assert "status: failed" in out
        assert "did not establish" in out
        job = gc_jobs.registry.list_jobs()[-1]
        assert job.cancel_event.is_set()
        assert job.done_event.wait(10)
        # Never armed → no delivery for the wedged attempt.
        assert _drain_completion_queue() == []


def _preset_event():
    evt = threading.Event()
    evt.set()
    return evt


# ---------------------------------------------------------------------------
# completion summary — the final content block, not fused narration
# ---------------------------------------------------------------------------

def _narration_chunk(text):
    return ("sdk.message", {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": text}]},
    })


def _tool_round(call_id):
    return [
        ("sdk.message", {
            "type": "tool_call", "call_id": call_id, "name": "read_file",
            "status": "running", "args": {"path": "/tmp/x"},
        }),
        ("sdk.message", {
            "type": "tool_call", "call_id": call_id, "name": "read_file",
            "status": "completed", "result": {},
        }),
    ]


class TestCompletionSummary:
    """Live repro: the delivered summary was every interstitial narration
    fragment glued with no separators — "…the leftover spike file.Now let
    me explore the relevant code…" — instead of the final wrap-up message."""

    def _run_replay(self, monkeypatch, tmp_path, sid, events):
        def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                   cancel_check=None, agent_id=None, model=None):
            yield ("sdk.session", {"agentId": sid, "cwd": str(workdir),
                                   "model": "m"})
            yield from events

        monkeypatch.setattr(gc_sdk, "run_sdk", replay)
        _start_run("t", repo=str(tmp_path))
        job = _job_for(sid)
        assert job.done_event.wait(10)
        return job

    def test_summary_is_the_final_content_block_not_fused_narration(
        self, clean_state, monkeypatch, tmp_path
    ):
        events = [
            _narration_chunk(
                "I'll start by reading the brief and the leftover spike file."
            ),
            *_tool_round("t1"),
            _narration_chunk("Now let me explore the relevant code..."),
            *_tool_round("t2"),
            _narration_chunk("Now let me read the actual services..."),
            *_tool_round("t3"),
            # The final wrap-up streams as multiple deltas of ONE message —
            # those must join raw, not with injected separators.
            _narration_chunk("Implemented the retry wrapper in servi"),
            _narration_chunk("ces/http.py and added regression tests."),
            ("sdk.result", {"status": "finished"}),
        ]
        job = self._run_replay(monkeypatch, tmp_path, "s-sum", events)

        summary = job.result["summary"]
        assert summary == (
            "Implemented the retry wrapper in services/http.py "
            "and added regression tests."
        )
        # None of the interstitial narration fragments leak in.
        assert "I'll start by reading" not in summary
        assert "Now let me explore" not in summary
        # And in particular nothing fuses sentence-to-sentence.
        assert "file.Now" not in summary

        # The delivered completion carries the same final-block summary.
        events_out = _drain_completion_queue()
        assert len(events_out) == 1
        delivered = events_out[0]["summary"]
        assert "cursor's summary:" in delivered
        assert "Implemented the retry wrapper" in delivered
        assert "file.Now" not in delivered

    def test_no_final_message_falls_back_to_separated_blocks(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A turn that does not END on a content block has no wrap-up to
        prefer: the fallback keeps every block, joined with blank lines so
        sentences never fuse."""
        events = [
            _narration_chunk("Reading the brief."),
            *_tool_round("t1"),
            _narration_chunk("Exploring the services."),
            *_tool_round("t2"),  # turn ends right after tool activity
            ("sdk.result", {"status": "finished"}),
        ]
        job = self._run_replay(monkeypatch, tmp_path, "s-fall", events)

        summary = job.result["summary"]
        assert summary == "Reading the brief.\n\nExploring the services."
        assert "brief.Exploring" not in summary


# ---------------------------------------------------------------------------
# resume via cursor_send_message — explicit session/load, no heuristics
# ---------------------------------------------------------------------------

class TestSendMessageResume:
    def test_pre_v04_handle_threads_the_resume_id_to_run_sdk(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A pre-v0.4 handle (keyed by the raw cursor sid, no alias field)
        resumes by its own key via Agent.resume."""
        gc_handles.record("s-prior", repo=_resolved(tmp_path), status="completed")
        release = threading.Event()
        seq = _SdkSequence(_gated_replay_factory(release, sid="s-prior"))
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        ack = cursor_send_message("s-prior", "continue it")
        try:
            _assert_running_ack(ack)
            assert "sent to s-prior" in ack
            # The resume id reached the SDK layer (Agent.resume path).
            assert seq.calls[0]["agent_id"] == "s-prior"
        finally:
            release.set()
        assert _job_for("s-prior").done_event.wait(10)

    def test_legacy_acp_model_record_is_sanitized_on_resume(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A pre-swap handle recorded the ACP-era bracket model string
        verbatim; re-sending must translate it to base id + params before
        Agent.resume (passing it straight through was a live
        BadRequestError). End-to-end through cursor_send_message with the
        REAL run_sdk against the fake bridge client."""
        legacy = "claude-fable-5[thinking=true,context=300k,effort=high]"
        gc_handles.record(
            "s-legacy", repo=_resolved(tmp_path), status="completed",
            model=legacy,
        )
        agent = _FakeAgent(agent_id="s-legacy",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)

        cursor_send_message("s-legacy", "continue the work")
        job = _job_for("s-legacy")
        assert job.done_event.wait(10)

        assert client.agents.resume_calls[0]["options"] == {
            "model": _FABLE_BRACKET_SELECTION
        }
        # The handle heals: sdk.session reports the base id, which is what
        # gets recorded for the next resume.
        assert gc_handles.get("s-legacy")["model"] == "claude-fable-5"

    def test_expired_handle_falls_back_to_fresh_session(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The SDK layer falls back to a fresh agent for an expired id; the
        session's alias is updated to the fresh sid — no crash, no hard
        failure."""

        def fallback_replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                            cancel_check=None, agent_id=None, model=None):
            # Simulates sdk_runner's resume → fresh-agent fallback.
            yield ("sdk.session", {"agentId": "s-fresh", "cwd": str(workdir),
                                   "model": "m", "resumed": False})
            yield ("sdk.result", {"status": "finished"})

        gc_handles.record("s-expired", repo=_resolved(tmp_path), status="completed")
        monkeypatch.setattr(gc_sdk, "run_sdk", fallback_replay)
        cursor_send_message("s-expired", "t")
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
        seq = _SdkSequence(_gated_replay_factory(_preset_event(), sid=sid))
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)
        _start_run("t", repo=str(tmp_path), **start_kwargs)
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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(release, sid="s-m5"))
        ack = _start_run("t", repo=str(tmp_path), model="composer-x")
        try:
            _assert_running_ack(ack)
            job = _job_for("s-m5")
            assert job.model == "composer-x"  # reported by sdk.session
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
        seq = _SdkSequence(
            _gated_replay_factory(release1, sid="s-send"),
            _gated_replay_factory(release2, sid="s-send", early_edit=False),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        _assert_running_ack(_start_run("task A", repo=str(tmp_path)))
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
            assert seq.calls[1]["agent_id"] == "s-send"
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
        seq = _SdkSequence(
            _gated_replay_factory(_preset_event(), sid="s-f"),
            _gated_replay_factory(release2, sid="s-f", early_edit=False),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        _start_run("task A", repo=str(tmp_path))
        first_job = _job_for("s-f")
        assert first_job.done_event.wait(10)
        _drain_completion_queue()

        ack = cursor_send_message("s-f", "refine it")
        try:
            _assert_running_ack(ack)
            # No interruption happened — the run had already settled.
            assert "interrupted mid-run" not in ack
            assert seq.calls[1]["agent_id"] == "s-f"
            assert seq.calls[1]["task"] == "refine it"
        finally:
            release2.set()
        assert gc_jobs.registry.get_by_session("s-f").done_event.wait(10)

    def test_first_message_on_a_fresh_session_is_the_task(
        self, clean_state, monkeypatch, tmp_path
    ):
        """cursor_create_session dispatches NOTHING; the first send creates
        the cursor agent with the message as the task."""
        seq = _SdkSequence(_gated_replay_factory(_preset_event(), sid="s-lazy"))
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        ack = cursor_create_session(repo=str(tmp_path))
        name = ack.splitlines()[0].split("session: ")[1]
        assert seq.calls == []  # lazy: nothing dispatched yet
        assert gc_jobs.registry.get_by_name(name) is None
        assert gc_handles.get(name)["status"] == "created"

        cursor_send_message(name, "build the thing")
        assert seq.calls[0]["task"] == "build the thing"
        assert seq.calls[0]["agent_id"] is None  # fresh, not a resume
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
        seq = _SdkSequence(_gated_replay_factory(release, sid="s-old",
                                                 early_edit=False))
        monkeypatch.setattr(gc, "_configured_model", lambda: None)
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        ack = cursor_send_message("s-old", "pick it back up")
        try:
            _assert_running_ack(ack)
            assert "sent to s-old" in ack
            assert seq.calls[0]["agent_id"] == "s-old"
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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(release, sid="s-ro"))

        _assert_running_ack(_start_run("long task", repo=str(tmp_path)))
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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(_preset_event(),
                                                                     sid="s-fin"))
        _start_run("t", repo=str(tmp_path))
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
        """The header carries last activity — seconds since the last stream
        event — so a caller can spot a silent run without touching it.
        Fresh while events stream; frozen at finished_at once terminal."""
        release = threading.Event()
        monkeypatch.setattr(gc_sdk, "run_sdk",
                            _gated_replay_factory(release, sid="s-act"))

        _start_run("t", repo=str(tmp_path))
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
        handle in the window before sdk.session fires."""
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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(release, sid="s-stop"))

        _assert_running_ack(_start_run("t", repo=str(tmp_path)))
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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(_preset_event(),
                                                                     sid="s-idem"))
        _start_run("t", repo=str(tmp_path))
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
        monkeypatch.setattr(gc_sdk, "run_sdk", _gated_replay_factory(release, sid="s-a"))
        first = _start_run("task A", repo=str(tmp_path))
        try:
            _assert_running_ack(first)
            name = _job_for("s-a").session_name

            second = _start_run("task B", repo=str(tmp_path))
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
        seq = _SdkSequence(
            _gated_replay_factory(release, sid="s-ra", early_edit=False),
            _gated_replay_factory(release, sid="s-rb", early_edit=False),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        res_a = _start_run("a", repo=str(repo_a))
        res_b = _start_run("b", repo=str(repo_b))
        try:
            _assert_running_ack(res_a)
            _assert_running_ack(res_b)
            assert _job_for("s-ra").session_name != _job_for("s-rb").session_name
        finally:
            release.set()
        for sid in ("s-ra", "s-rb"):
            assert _job_for(sid).done_event.wait(10)

    def test_finished_run_releases_the_repo(self, clean_state, monkeypatch, tmp_path):
        seq = _SdkSequence(
            _gated_replay_factory(_preset_event(), sid="s-one"),
            _gated_replay_factory(_preset_event(), sid="s-two"),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)
        _start_run("t", repo=str(tmp_path))
        assert _job_for("s-one").done_event.wait(10)

        second = _start_run("t2", repo=str(tmp_path))
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
# Git fallback — files_changed for edits the SDK stream carried no diff for
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
            yield ("sdk.session", {"agentId": "s-git", "cwd": str(git_repo), "model": "m"})
            # Simulates cursor editing through a shell command: no diff
            # content ever appears on the SDK stream.
            (git_repo / "tool.txt").write_text("orig\nedited by shell\n")
            (git_repo / "new.txt").write_text("brand new\n")
            yield ("sdk.result", {"status": "finished"})

        monkeypatch.setattr(gc_sdk, "run_sdk", shell_edit_replay)
        _start_run("edit via shell", repo=str(git_repo))
        job = _job_for("s-git")
        assert job.done_event.wait(10)

        result = job.result
        by_path = {Path(f["path"]).name: f for f in result["files_changed"]}
        assert set(by_path) == {"tool.txt", "new.txt"}  # pre.txt excluded
        assert by_path["tool.txt"]["status"] == "M"
        assert by_path["tool.txt"]["added"] == 1
        assert "edited by shell" in by_path["tool.txt"]["diff"]
        assert by_path["new.txt"]["status"] == "A"

    def test_sdk_stream_diff_wins_over_fallback(self, clean_state, monkeypatch, git_repo):
        """A file already captured from the SDK stream is not re-added."""

        def stream_and_shell(*_a, **_k):
            yield ("sdk.session", {"agentId": "s-win", "cwd": str(git_repo), "model": "m"})
            (git_repo / "tool.txt").write_text("orig\nedited\n")
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "edit_file",
                "status": "running", "args": {"path": str(git_repo / "tool.txt")},
            })
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "edit_file",
                "status": "completed",
                "result": {
                    "path": str(git_repo / "tool.txt"),
                    "oldText": "orig\n", "newText": "orig\nedited\n",
                },
            })
            yield ("sdk.result", {"status": "finished"})

        monkeypatch.setattr(gc_sdk, "run_sdk", stream_and_shell)
        _start_run("t", repo=str(git_repo))
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
    def _run_armed(self, monkeypatch, tmp_path, sid, terminal_event,
                   pre_events=()):
        """Dispatch through create + first send so delivery gets armed; the replay
        emits ``pre_events`` + ``terminal_event`` only after the running
        handle was returned (deterministic — the run cannot finalize before
        arming)."""
        release = threading.Event()

        def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                   cancel_check=None, agent_id=None, model=None):
            yield ("sdk.session", {"agentId": sid, "cwd": str(workdir), "model": "m"})
            release.wait(10)
            yield from pre_events
            yield terminal_event

        monkeypatch.setattr(gc_sdk, "run_sdk", replay)
        res = _start_run("t", repo=str(tmp_path))
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
            ("sdk.result", {"status": "finished"}),
        )
        assert job.status == "completed"
        assert evt["status"] == "completed"
        assert evt["result"]["success"] is True

    def test_cancelled_run_delivers(self, clean_state, monkeypatch, tmp_path):
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-c",
            ("sdk.result", {"status": "cancelled"}),
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
            ("sdk.error", {"error": "cursor run timed out: no activity for 5s", "timeout": True}),
        )
        assert job.status == "timeout"
        assert evt["status"] == "timeout"
        assert "timed out" in evt["error"]
        assert evt["result"]["status"] == "timeout"

    def test_midrun_failure_delivers(self, clean_state, monkeypatch, tmp_path):
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-x",
            ("sdk.error", {"error": "cursor-sdk stream failed mid-run: boom"}),
        )
        assert job.status == "failed"
        assert evt["status"] == "failed"
        assert "boom" in evt["error"]

    def test_terminal_error_detail_reaches_the_completion_summary(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A terminal-error sdk.error with typed detail (retryable /
        retry_after) renders it on the failure line — not the bare
        'cursor run ended with status: error'. (The run streamed content
        first, so the zero-progress auto-retry stays out of the way.)"""
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-err-detail",
            ("sdk.error", {
                "error": "ServerError: upstream 502 from the agent backend",
                "retryable": True,
                "retry_after": "30",
                "run_status": "error",
            }),
            pre_events=[_narration_chunk("started digging in")],
        )
        assert job.status == "failed"
        assert evt["error"] == "ServerError: upstream 502 from the agent backend"
        assert evt["result"]["error_retryable"] is True
        assert evt["result"]["error_retry_after"] == "30"
        assert (
            "run failed: ServerError: upstream 502 from the agent backend "
            "(retryable, retry after 30s)" in evt["summary"]
        )

    def test_terminal_error_without_detail_stays_generic(
        self, clean_state, monkeypatch, tmp_path
    ):
        """No typed detail available: the generic terminal-error text is
        delivered with NO retry parenthetical (None = unknown, never
        rendered as 'not retryable')."""
        job, evt = self._run_armed(
            monkeypatch, tmp_path, "s-err-bare",
            ("sdk.error", {
                "error": "cursor run ended with status: error",
                "retryable": None,
                "retry_after": None,
                "run_status": "error",
            }),
            pre_events=[_narration_chunk("started digging in")],
        )
        assert job.status == "failed"
        assert "run failed: cursor run ended with status: error." in evt["summary"]
        assert "retryable" not in evt["summary"]
        assert "error_retryable" not in evt["result"]

    def test_transport_drop_delivers_failed_not_completed(
        self, clean_state, monkeypatch, tmp_path
    ):
        """Live repro (2026-07-03, ACP era): the stream died with a
        RetriableError narrated as the final message chunks, then a "clean"
        finish. The delivered completion must say failed with the error
        first-class in the header — not completed with the error buried in
        the summary. Kept under the SDK transport: the classification is
        content-based and transport-agnostic."""
        release = threading.Event()

        def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                   cancel_check=None, agent_id=None, model=None):
            yield ("sdk.session", {"agentId": "s-drop", "cwd": str(workdir),
                                   "model": "m"})
            release.wait(10)
            yield ("sdk.message", {
                "type": "assistant",
                "message": {"content": [{
                    "type": "text",
                    "text": "Now let me explore the relevant code\n",
                }]},
            })
            yield ("sdk.message", {
                "type": "assistant",
                "message": {"content": [{
                    "type": "text",
                    "text": "RetriableError: [canceled] http/2 stream "
                            "closed with error code CANCEL (0x8)",
                }]},
            })
            yield ("sdk.result", {"status": "finished"})

        monkeypatch.setattr(gc_sdk, "run_sdk", replay)
        res = _start_run("t", repo=str(tmp_path))
        assert "running in background" in res, f"run never armed: {res}"
        job = _job_for("s-drop")
        release.set()
        assert job.done_event.wait(10)

        events = _drain_completion_queue()
        assert len(events) == 1
        evt = events[0]
        assert job.status == "failed"
        assert evt["status"] == "failed"
        assert "http/2 stream closed with error code CANCEL (0x8)" in evt["error"]
        assert evt["result"]["success"] is False
        assert evt["result"]["status"] == "failed"
        assert "http/2 stream closed" in evt["result"]["error"]
        # The delivered text names the failure in the header, not just the
        # stitched summary body.
        assert "status: failed" in evt["summary"]
        assert "run failed:" in evt["summary"]
        assert "http/2 stream closed" in evt["summary"].split("cursor's summary:")[0]


# ---------------------------------------------------------------------------
# Progress subscriptions — periodic digests on the completion queue
# ---------------------------------------------------------------------------

def _digest_events(events):
    """The progress digests among drained completion-queue events."""
    return [e for e in events if e.get("cursor_progress_update")]


def _completion_events(events):
    """The terminal completions among drained completion-queue events."""
    return [e for e in events if not e.get("cursor_progress_update")]


def _collect_queue(collected, min_digests=1, timeout=5.0):
    """Keep draining the completion queue into ``collected`` until it holds
    at least ``min_digests`` digest events."""
    def _pump():
        collected.extend(_drain_completion_queue())
        return len(_digest_events(collected)) >= min_digests

    return _wait_until(_pump, timeout=timeout)


class TestProgressSubscriptions:
    """cursor_send_message(update_interval_s) + cursor_subscribe: periodic
    digests ride the same completion_queue rail as terminal completions,
    numbered per session, and never outlive the run."""

    def _held_run(self, monkeypatch, tmp_path, sid="agent-digest", **send_kw):
        """A run held open on the returned release event, dispatched via
        create + send so subscription plumbing runs end-to-end."""
        release = threading.Event()
        monkeypatch.setattr(
            gc_sdk, "run_sdk", _gated_replay_factory(release, sid=sid)
        )
        name = _created_name(cursor_create_session(repo=str(tmp_path)))
        ack = cursor_send_message(name, "task", **send_kw)
        _assert_running_ack(ack)
        return name, _job_for(sid), release

    # -- subscription set at dispatch --------------------------------------

    def test_default_send_sets_180s_subscription(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(monkeypatch, tmp_path)
        try:
            entry = gc_handles.get(name)
            assert entry["update_interval_s"] == 180.0
            ticker = gc_progress._tickers.get(name)
            assert ticker is not None
            assert ticker.interval_s == 180.0
        finally:
            release.set()
            assert job.done_event.wait(10)

    def test_explicit_interval_used_and_persisted(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=45
        )
        try:
            assert gc_handles.get(name)["update_interval_s"] == 45.0
            assert gc_progress._tickers[name].interval_s == 45.0
        finally:
            release.set()
            assert job.done_event.wait(10)

    def test_zero_interval_means_no_digests(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0
        )
        try:
            assert gc_handles.get(name)["update_interval_s"] == 0.0
            assert name not in gc_progress._tickers
            time.sleep(0.15)
            assert _digest_events(_drain_completion_queue()) == []
        finally:
            release.set()
            assert job.done_event.wait(10)
        events = _drain_completion_queue()
        assert _digest_events(events) == []
        assert len(_completion_events(events)) == 1

    # -- digest delivery + content -----------------------------------------

    def test_digest_rides_async_delegation_rail_with_status_and_events(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0.05
        )
        collected = []
        try:
            assert _collect_queue(collected, min_digests=1)
        finally:
            release.set()
            assert job.done_event.wait(10)

        digest = _digest_events(collected)[0]
        # The exact event shape every hermes-core consumer differentiates
        # on: type field, unique delegation_id (TUI dedup), session_key
        # routing — and NOT deregistering anything is the producer's job.
        assert digest["type"] == "async_delegation"
        assert digest["delegation_id"] == f"{name}#progress-1"
        assert digest["status"] == "running"
        assert digest["cursor_progress_update"] == 1
        assert "NOT the final result" in digest["goal"]

        text = digest["summary"]
        assert f"cursor session '{name}' — progress update 1" in text
        assert "status: running" in text
        assert "elapsed:" in text and "last activity:" in text
        # The early edit of the gated replay is visible either as the
        # files-so-far header count or in the events-since-last-tick body.
        assert "f1.py" in text

    def test_quiet_tick_says_no_new_events(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0.05
        )
        collected = []
        try:
            # By the second digest the held run has gone quiet — no events
            # stream while the replay blocks on the release gate.
            assert _collect_queue(collected, min_digests=2)
        finally:
            release.set()
            assert job.done_event.wait(10)

        quiet = _digest_events(collected)[-1]
        assert "no new events since last update" in quiet["summary"]
        # Header still carries the signal for a quiet run.
        assert "last activity:" in quiet["summary"]

    def test_digest_numbering_increments_and_tags_session(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0.05
        )
        collected = []
        try:
            assert _collect_queue(collected, min_digests=3)
        finally:
            release.set()
            assert job.done_event.wait(10)

        digests = _digest_events(collected)[:3]
        assert [d["cursor_progress_update"] for d in digests] == [1, 2, 3]
        for i, d in enumerate(digests, start=1):
            assert f"progress update {i}" in d["summary"]
            assert f"'{name}'" in d["summary"]

    # -- cursor_subscribe ---------------------------------------------------

    def test_subscribe_changes_interval_mid_run(
        self, clean_state, monkeypatch, tmp_path
    ):
        # Start slow (no tick due for 60s), then shorten mid-run: the
        # pending timer must be rescheduled immediately.
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=60
        )
        collected = []
        try:
            ack = cursor_subscribe(name, 0.05)
            assert ack == gc_render.subscribe_ack(name, 0.05)
            assert "\n" not in ack  # 1-line plain-text ack
            assert name in ack
            assert gc_handles.get(name)["update_interval_s"] == 0.05
            assert _collect_queue(collected, min_digests=1)
        finally:
            release.set()
            assert job.done_event.wait(10)

    def test_digest_next_update_line_reflects_midrun_interval_change(
        self, clean_state, monkeypatch, tmp_path
    ):
        """The ticker hands its CURRENT interval to the digest at tick
        time: a run dispatched at 60s and retuned mid-run via
        cursor_subscribe renders the NEW interval in the next digest,
        not the dispatch-time one."""
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=60
        )
        collected = []
        try:
            cursor_subscribe(name, 0.05)
            assert _collect_queue(collected, min_digests=1)
        finally:
            release.set()
            assert job.done_event.wait(10)

        digest = _digest_events(collected)[0]
        # 0.05s rounds up to the 1s floor; the old 60s would say "1m".
        assert "next update in 1s" in digest["summary"]
        assert "next update in 1m" not in digest["summary"]

    def test_subscribe_zero_unsubscribes_mid_run(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0.05
        )
        collected = []
        try:
            assert _collect_queue(collected, min_digests=1)
            ack = cursor_subscribe(name, 0)
            assert "off" in ack and name in ack
            assert gc_handles.get(name)["update_interval_s"] == 0.0
            assert name not in gc_progress._tickers
            _drain_completion_queue()
            time.sleep(0.2)
            assert _digest_events(_drain_completion_queue()) == []
        finally:
            release.set()
            assert job.done_event.wait(10)
        # The terminal completion is unaffected by unsubscribing.
        events = _drain_completion_queue()
        assert len(_completion_events(events)) == 1

    def test_subscribe_without_run_persists_for_next_run(
        self, clean_state, monkeypatch, tmp_path
    ):
        name = _created_name(cursor_create_session(repo=str(tmp_path)))
        ack = cursor_subscribe(name, 0.05)
        assert name in ack
        assert gc_handles.get(name)["update_interval_s"] == 0.05

        release = threading.Event()
        monkeypatch.setattr(
            gc_sdk, "run_sdk", _gated_replay_factory(release, sid="agent-persist")
        )
        collected = []
        _assert_running_ack(cursor_send_message(name, "task"))
        job = _job_for("agent-persist")
        try:
            # No update_interval_s on send — the persisted 0.05 drives it.
            assert gc_progress._tickers[name].interval_s == 0.05
            assert _collect_queue(collected, min_digests=1)
        finally:
            release.set()
            assert job.done_event.wait(10)

    def test_subscribe_unknown_session_is_actionable(self, clean_state):
        out = cursor_subscribe("no-such-session", 30)
        assert "no session named 'no-such-session'" in out

    def test_subscribe_rejects_negative_interval(
        self, clean_state, monkeypatch, tmp_path
    ):
        name = _created_name(cursor_create_session(repo=str(tmp_path)))
        assert ">= 0" in cursor_subscribe(name, -5)

    # -- terminal-state guarantees -------------------------------------------

    def test_digest_never_fires_after_terminal_state(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0.05
        )
        collected = []
        assert _collect_queue(collected, min_digests=1)
        release.set()
        assert job.done_event.wait(10)
        # Everything already enqueued at settle time is legal; nothing may
        # be added after it — the pending timer was cancelled at finalize.
        collected.extend(_drain_completion_queue())
        time.sleep(0.25)
        late = _drain_completion_queue()
        assert late == [], f"digest arrived after terminal state: {late}"
        # And on the collected sequence the completion is the LAST event.
        assert not _digest_events([collected[-1]])
        assert job.status == "completed"

    def test_completion_delivered_exactly_once_with_digests_active(
        self, clean_state, monkeypatch, tmp_path
    ):
        name, job, release = self._held_run(
            monkeypatch, tmp_path, update_interval_s=0.05
        )
        collected = []
        assert _collect_queue(collected, min_digests=2)
        release.set()
        assert job.done_event.wait(10)
        collected.extend(_drain_completion_queue())

        completions = _completion_events(collected)
        assert len(completions) == 1
        assert completions[0]["status"] == "completed"
        assert completions[0]["delegation_id"] == name
        assert len(_digest_events(collected)) >= 2

    def test_interrupt_and_reprompt_carries_subscription_and_numbering(
        self, clean_state, monkeypatch, tmp_path
    ):
        release1, release2 = threading.Event(), threading.Event()
        seq = _SdkSequence(
            _gated_replay_factory(release1, sid="agent-ir"),
            _gated_replay_factory(release2, sid="agent-ir"),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)
        name = _created_name(cursor_create_session(repo=str(tmp_path)))
        _assert_running_ack(cursor_send_message(name, "task", update_interval_s=0.05))
        collected = []
        assert _collect_queue(collected, min_digests=1)
        first_n = max(d["cursor_progress_update"] for d in _digest_events(collected))

        # Interrupt + re-prompt: the new run inherits the subscription (no
        # explicit interval on this send) and numbering continues.
        ack = cursor_send_message(name, "follow-up")
        _assert_running_ack(ack)
        assert "interrupted" in ack
        job2 = _job_for("agent-ir")
        try:
            assert gc_progress._tickers[name].interval_s == 0.05
            collected2 = []
            assert _collect_queue(collected2, min_digests=1)
            nums = [d["cursor_progress_update"] for d in _digest_events(collected2)]
            assert min(nums) == first_n + 1, (
                f"numbering restarted: {nums} after {first_n}"
            )
        finally:
            release1.set()
            release2.set()
            assert job2.done_event.wait(10)

    # -- multiple sessions ---------------------------------------------------

    def test_multiple_sessions_with_different_intervals(
        self, clean_state, monkeypatch, tmp_path
    ):
        repo_a, repo_b = tmp_path / "a", tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        release_a, release_b = threading.Event(), threading.Event()
        seq = _SdkSequence(
            _gated_replay_factory(release_a, sid="agent-a"),
            _gated_replay_factory(release_b, sid="agent-b"),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        name_a = _created_name(cursor_create_session(repo=str(repo_a)))
        name_b = _created_name(cursor_create_session(repo=str(repo_b)))
        _assert_running_ack(
            cursor_send_message(name_a, "task a", update_interval_s=0.05)
        )
        _assert_running_ack(
            cursor_send_message(name_b, "task b", update_interval_s=60)
        )
        job_a, job_b = _job_for("agent-a"), _job_for("agent-b")
        collected = []
        try:
            assert gc_progress._tickers[name_a].interval_s == 0.05
            assert gc_progress._tickers[name_b].interval_s == 60.0
            assert _collect_queue(collected, min_digests=2)
        finally:
            release_a.set()
            release_b.set()
            assert job_a.done_event.wait(10)
            assert job_b.done_event.wait(10)

        digests = _digest_events(collected)
        # Only the fast session ticked; every digest is tagged with ITS name.
        assert all(f"'{name_a}'" in d["summary"] for d in digests)
        assert all(d["delegation_id"].startswith(f"{name_a}#progress-") for d in digests)
        assert not any(f"'{name_b}'" in d["summary"] for d in digests)

    # -- digest rendering (pure) ----------------------------------------------

    def test_digest_text_header_body_and_caps(self):
        events = [
            {"seq": 40 + i, "kind": "tool_use", "tool": "shell",
             "command": f"pytest test_{i}.py"}
            for i in range(8)
        ]
        text = gc_render.digest_text(
            name="busy-bee",
            n=4,
            status="running",
            elapsed_s=843,
            last_activity_s=12,
            files=[{"path": "calc.py", "added": 4, "removed": 0}],
            pending_tool="shell `pytest -q`",
            pending_tool_s=41,
            events=events,
            new_count=8,
            next_update_s=180,
        )
        assert "cursor session 'busy-bee' — progress update 4" in text
        assert (
            "status: running · elapsed: 843s · last activity: 12s ago · "
            "next update in 3m" in text
        )
        assert "files so far (1): calc.py +4 −0" in text
        assert "pending tool call: shell `pytest -q` (41s)" in text
        assert "new events since last update (8):" in text
        # Body capped at DIGEST_MAX_EVENTS lines + an omission pointer.
        assert text.count("pytest test_") == gc_render.DIGEST_MAX_EVENTS
        assert "3 more — cursor_events('busy-bee')" in text
        assert len(text) < 2048

    def test_digest_text_quiet_tick(self):
        text = gc_render.digest_text(
            name="quiet-owl",
            n=2,
            status="running",
            elapsed_s=360,
            last_activity_s=181,
            files=[],
            events=[],
            new_count=0,
        )
        assert "progress update 2" in text
        assert "no new events since last update" in text
        # No interval provided → the fragment is omitted entirely.
        assert "next update in" not in text

    def test_digest_text_next_update_reflects_changed_interval(self):
        """The 'next update in' fragment renders whatever interval the
        ticker holds AT TICK TIME — a mid-run retune shows immediately."""
        kwargs = dict(
            name="retuned-fox", n=3, status="running", elapsed_s=100,
            last_activity_s=5, files=[], events=[], new_count=0,
        )
        before = gc_render.digest_text(**kwargs, next_update_s=180)
        after = gc_render.digest_text(**kwargs, next_update_s=45)
        assert "next update in 3m" in before
        assert "next update in 45s" in after
        assert "next update in 3m" not in after

    def test_pending_tool_call_visible_in_digest(
        self, clean_state, monkeypatch, tmp_path
    ):
        """A run stuck inside one long tool call shows it in the header —
        the 'long quiet tool call vs stall' signal from the spec."""
        release = threading.Event()

        def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
                   cancel_check=None, agent_id=None, model=None):
            yield ("sdk.session", {"agentId": "agent-pending",
                                   "cwd": str(workdir), "model": "m"})
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t9", "name": "shell",
                "status": "running",
                "args": {"command": "sleep 999"},
            })
            while not release.is_set():
                if cancel_check and cancel_check():
                    yield ("sdk.result", {"status": "cancelled"})
                    return
                time.sleep(0.01)
            yield ("sdk.result", {"status": "finished"})

        monkeypatch.setattr(gc_sdk, "run_sdk", replay)
        name = _created_name(cursor_create_session(repo=str(tmp_path)))
        _assert_running_ack(
            cursor_send_message(name, "task", update_interval_s=0.05)
        )
        job = _job_for("agent-pending")
        collected = []
        try:
            assert _collect_queue(collected, min_digests=1)
        finally:
            release.set()
            assert job.done_event.wait(10)
        assert "pending tool call:" in _digest_events(collected)[0]["summary"]


# ---------------------------------------------------------------------------
# Zero-progress auto-retry — stale-bridge recovery (live incident 2026-07-04)
# ---------------------------------------------------------------------------

def _terminal_error_replay(sid="agent-zp", retryable=True, retry_after=None,
                           meaningful=False, release=None):
    """A replay that settles with terminal status "error" (the enriched
    sdk.error payload from sdk_runner), optionally after one meaningful
    tool round, optionally held open on ``release`` first."""

    def replay(task, workdir, inactivity_timeout_s=0.0, max_wall_s=0.0,
               cancel_check=None, agent_id=None, model=None):
        yield ("sdk.session", {"agentId": sid, "cwd": str(workdir),
                               "model": "m", "resumed": bool(agent_id)})
        if release is not None:
            while not release.is_set():
                if cancel_check and cancel_check():
                    yield ("sdk.result", {"status": "cancelled"})
                    return
                time.sleep(0.01)
        if meaningful:
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "shell",
                "status": "running", "args": {"command": "ls"},
            })
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "shell",
                "status": "completed",
                "result": {"exitCode": 0, "stdout": "ok"},
            })
        yield ("sdk.error", {
            "error": "ServerError: bridge went stale",
            "retryable": retryable,
            "retry_after": retry_after,
            "run_status": "error",
        })

    return replay


def _lifecycle_trail(name):
    """(event, record) for every lifecycle event in a session's jsonl log,
    in seq order."""
    page = gc_eventlog.read_events(name, offset=0, limit=500)
    return [
        (e.get("event"), e)
        for e in (page or {}).get("events") or []
        if e.get("kind") == "lifecycle"
    ]


class TestZeroProgressAutoRetry:
    """A terminal-error run with ZERO meaningful events (the stale-bridge
    signature, live incident 2026-07-04) is transparently re-sent on the
    same agent — bridge recycled before the first retry, jsonl-only
    lifecycle signal, no user-facing failure. Meaningful progress, a
    non-retryable error, or an exhausted budget surfaces the detailed
    failure from the error-observability path instead."""

    def _fast_retries(self, monkeypatch):
        """Zero the backoff ladder and stub the bridge recycle, returning
        the recorded recycle calls."""
        monkeypatch.setattr(gc, "_AUTO_RETRY_BACKOFF_S", (0.0, 0.0))
        recycles = []
        monkeypatch.setattr(
            gc_sdk, "recycle_bridge",
            lambda workspace: recycles.append(workspace) or True,
        )
        return recycles

    def test_zero_progress_error_recycles_bridge_and_retry_succeeds(
        self, clean_state, monkeypatch, tmp_path
    ):
        recycles = self._fast_retries(monkeypatch)
        release = threading.Event()
        seq = _SdkSequence(
            _terminal_error_replay(sid="agent-zp"),
            _gated_replay_factory(release, sid="agent-zp"),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        _assert_running_ack(_start_run("t", repo=str(tmp_path)))
        job = _job_for("agent-zp")
        # Same job across the retry — the digest subscription (default
        # 180s) keeps its ticker.
        assert gc_progress._tickers[job.session_name].job is job
        release.set()
        assert job.done_event.wait(10)

        # One job, retried in place, clean success — no user-facing failure.
        assert job.status == "completed"
        assert job.result["success"] is True
        assert "error" not in job.result
        assert len(seq.calls) == 2
        assert seq.calls[1]["agent_id"] == "agent-zp"  # SAME agent resumed
        assert recycles == [job.repo]

        events = _drain_completion_queue()
        completions = _completion_events(events)
        assert len(completions) == 1
        assert completions[0]["status"] == "completed"
        assert completions[0]["error"] is None

        # The jsonl log shows the transparent recovery, in order: failed
        # first run → autoretry marker (bridge recycled) → clean second run.
        trail = _lifecycle_trail(job.session_name)
        marks = [n for n, _ in trail
                 if n in ("run.started", "run.failed", "sdk.autoretry",
                          "run.completed")]
        assert marks == ["run.started", "run.failed", "sdk.autoretry",
                         "run.started", "run.completed"]
        autoretry = next(e for n, e in trail if n == "sdk.autoretry")
        assert autoretry["attempt"] == 1
        assert autoretry["bridge_recycled"] is True
        assert "zero-progress" in autoretry["reason"]
        assert "ServerError: bridge went stale" in autoretry["reason"]

    def test_error_after_meaningful_progress_does_not_auto_retry(
        self, clean_state, monkeypatch, tmp_path
    ):
        recycles = self._fast_retries(monkeypatch)
        release = threading.Event()
        seq = _SdkSequence(
            _terminal_error_replay(sid="agent-mp", meaningful=True,
                                   release=release),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        _assert_running_ack(_start_run("t", repo=str(tmp_path)))
        job = _job_for("agent-mp")
        release.set()
        assert job.done_event.wait(10)

        assert len(seq.calls) == 1  # no re-send
        assert recycles == []
        assert job.status == "failed"
        assert not [n for n, _ in _lifecycle_trail(job.session_name)
                    if n == "sdk.autoretry"]
        completions = _completion_events(_drain_completion_queue())
        assert len(completions) == 1
        evt = completions[0]
        assert evt["status"] == "failed"
        assert evt["error"] == "ServerError: bridge went stale"
        # The detailed failure from the error-observability path.
        assert ("run failed: ServerError: bridge went stale (retryable)"
                in evt["summary"])

    def test_retries_exhausted_surface_the_detailed_failure(
        self, clean_state, monkeypatch, tmp_path
    ):
        recycles = self._fast_retries(monkeypatch)
        release = threading.Event()
        seq = _SdkSequence(
            _terminal_error_replay(sid="agent-ex", release=release,
                                   retry_after="0"),
            _terminal_error_replay(sid="agent-ex"),
            _terminal_error_replay(sid="agent-ex"),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        _assert_running_ack(_start_run("t", repo=str(tmp_path)))
        job = _job_for("agent-ex")
        release.set()
        assert job.done_event.wait(10)

        assert len(seq.calls) == 3  # the send + both retries
        assert recycles == [job.repo]  # recycled ONCE, on the first retry
        assert job.status == "failed"
        trail = [e for n, e in _lifecycle_trail(job.session_name)
                 if n == "sdk.autoretry"]
        assert [e["attempt"] for e in trail] == [1, 2]
        assert [e["bridge_recycled"] for e in trail] == [True, False]
        completions = _completion_events(_drain_completion_queue())
        assert len(completions) == 1
        evt = completions[0]
        assert evt["error"] == "ServerError: bridge went stale"
        assert evt["result"]["error_retryable"] is True
        assert ("run failed: ServerError: bridge went stale (retryable)"
                in evt["summary"])

    def test_non_retryable_zero_progress_error_does_not_retry(
        self, clean_state, monkeypatch, tmp_path
    ):
        recycles = self._fast_retries(monkeypatch)
        release = threading.Event()
        seq = _SdkSequence(
            _terminal_error_replay(sid="agent-nr", retryable=False,
                                   release=release),
        )
        monkeypatch.setattr(gc_sdk, "run_sdk", seq)

        _assert_running_ack(_start_run("t", repo=str(tmp_path)))
        job = _job_for("agent-nr")
        release.set()
        assert job.done_event.wait(10)

        assert len(seq.calls) == 1
        assert recycles == []
        assert job.status == "failed"
        assert not [n for n, _ in _lifecycle_trail(job.session_name)
                    if n == "sdk.autoretry"]
        completions = _completion_events(_drain_completion_queue())
        assert len(completions) == 1
        assert ("run failed: ServerError: bridge went stale (not retryable)"
                in completions[0]["summary"])


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
            SUBSCRIBE_TOOL_NAME,
        }
        for name, schema, required in (
            (CREATE_TOOL_NAME, CURSOR_CREATE_SCHEMA, []),
            (SEND_TOOL_NAME, CURSOR_SEND_SCHEMA, ["session", "message"]),
            (STATUS_TOOL_NAME, CURSOR_STATUS_SCHEMA, ["session"]),
            (STOP_TOOL_NAME, CURSOR_STOP_SCHEMA, ["session"]),
            (EVENTS_TOOL_NAME, CURSOR_EVENTS_SCHEMA, ["session"]),
            (LIST_TOOL_NAME, CURSOR_LIST_SCHEMA, []),
            (SUBSCRIBE_TOOL_NAME, CURSOR_SUBSCRIBE_SCHEMA, ["session", "interval_s"]),
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

    def test_check_fn_false_without_sdk_package(self, monkeypatch):
        monkeypatch.setattr(gc_sdk, "sdk_available", lambda: False)
        assert check_cursor_available() is False

    def test_check_fn_false_without_resolvable_repo(self, monkeypatch):
        monkeypatch.setattr(gc_sdk, "sdk_available", lambda: True)
        monkeypatch.setattr(gc, "_default_repo", lambda: None)
        assert check_cursor_available() is False

    def test_check_fn_true_with_sdk_and_repo(self, monkeypatch):
        monkeypatch.setattr(gc_sdk, "sdk_available", lambda: True)
        # _default_repo falls back to os.getcwd(), which always exists.
        assert check_cursor_available() is True

    def test_check_fn_never_raises(self, monkeypatch):
        def boom():
            raise RuntimeError("probe failed")

        monkeypatch.setattr(gc_sdk, "sdk_available", boom)
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
            gc_sdk, "run_sdk", _gated_replay_factory(_preset_event(), sid=sid)
        )
        _start_run("t", repo=str(tmp_path))
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
                   cancel_check=None, agent_id=None, model=None):
            yield ("sdk.session", {"agentId": "s-bigout", "cwd": str(workdir),
                                   "model": "m", "resumed": False})
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "shell",
                "status": "running", "args": {"command": "generate"},
            })
            yield ("sdk.message", {
                "type": "tool_call", "call_id": "t1", "name": "shell",
                "status": "completed", "result": {"exitCode": 0, "stdout": big},
            })
            yield ("sdk.result", {"status": "finished"})

        monkeypatch.setattr(gc_sdk, "run_sdk", replay)
        _start_run("t", repo=str(tmp_path))
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
            gc_sdk, "run_sdk", _gated_replay_factory(_preset_event(), sid="s-mine")
        )
        _start_run("t", repo=str(tmp_path))
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

class TestCursorCreateSession:
    def test_creates_a_named_handle_and_dispatches_nothing(
        self, clean_state, monkeypatch, tmp_path
    ):
        boom = lambda *a, **k: pytest.fail("create must not start a run")
        monkeypatch.setattr(gc_sdk, "run_sdk", boom)

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
            gc_sdk, "run_sdk",
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
        monkeypatch.setattr(gc_sdk, "run_sdk",
                            _gated_replay_factory(release, sid="s-live-list"))
        _start_run("t", repo=str(tmp_path))
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


# ---------------------------------------------------------------------------
# sdk_runner.run_sdk — faked cursor-sdk bridge/client (no network, no bridge)
# ---------------------------------------------------------------------------

def _sdk_msg(**kw):
    """A fake SDKMessage: plain-attribute object, converted defensively by
    the runner (real messages are frozen dataclasses — same shape)."""
    return SimpleNamespace(**kw)


class _FakeStream:
    """A scriptable run event stream.

    ``script`` items are either fake SDKMessages (wrapped in RunStreamEvent
    envelopes with sequential offsets), Exception instances (raised from
    next() — a stream drop), or callables (invoked, e.g. to block on a
    gate). ``observe(after_offset=...)`` resumes after the given offset,
    skipping any Exception items at the resume point (the drop is gone).
    """

    def __init__(self, run, script):
        self._run = run
        self._script = list(script)

    def _iter(self, start_idx):
        i = start_idx
        while i < len(self._script):
            if self._run.cancel_event.is_set():
                self._run.status = "cancelled"
                return
            item = self._script[i]
            i += 1
            if isinstance(item, Exception):
                raise item
            if callable(item):
                item = item()
            if item is None:
                continue
            if self._run.cancel_event.is_set():
                self._run.status = "cancelled"
                return
            yield SimpleNamespace(
                kind="message", offset=str(i - 1), sdk_message=item
            )
        self._run.status = (
            "cancelled" if self._run.cancel_event.is_set()
            else self._run.final_status
        )

    def events(self):
        return self._iter(0)

    def observe(self, after_offset=None):
        self._run.observe_calls.append(after_offset)
        start = 0 if after_offset is None else int(after_offset) + 1
        # The dropped stream's poison pill is not replayed on re-attach.
        while start < len(self._script) and isinstance(self._script[start], Exception):
            start += 1
        return self._iter(start)


class _FakeRun:
    def __init__(self, script, final_status="finished"):
        self.status = "running"
        self.final_status = final_status
        self.cancel_event = threading.Event()
        self.observe_calls = []
        self._stream = _FakeStream(self, script)

    def events(self):
        return self._stream.events()

    def observe(self, after_offset=None):
        return self._stream.observe(after_offset=after_offset)

    def cancel(self):
        self.cancel_event.set()


class _FakeAgent:
    def __init__(self, agent_id="agent-fake-1", model_id="fake-model", runs=None):
        self.agent_id = agent_id
        self.model = SimpleNamespace(id=model_id) if model_id else None
        self.sent = []
        self._runs = list(runs or [])

    def send(self, message, *a, **kw):
        # Real bridge behavior (verified in the vendored @cursor/sdk source,
        # sendImpl): a LOCAL agent handle with no model rejects every send
        # with a non-retryable error — there is no conversation-model
        # fallback. A resumed handle only has a model if the resume options
        # carried one (see _FakeAgents.resume), so a model-less resume makes
        # every follow-up send fail exactly like the live bridge did.
        if self.model is None:
            raise _sdk_error(
                "Local SDK agents require an explicit `model`. Pass "
                '`model: { id: "<model-id>" }` to Agent.create() or to '
                "send(), or run this agent in cloud mode.",
                is_retryable=False,
            )
        self.sent.append(message)
        return self._runs.pop(0)


def _sdk_error(text, is_retryable=False, retry_after=None):
    err = RuntimeError(text)
    err.is_retryable = is_retryable
    if retry_after is not None:
        err.retry_after = retry_after
    return err


def _model_selection_ns(model):
    """Mimic the real bridge's model normalization: a string or raw-dict
    ModelSelection becomes a typed selection whose ``.id`` is the base id."""
    if not model:
        return None
    if isinstance(model, dict):
        return SimpleNamespace(id=model.get("id"), params=model.get("params") or [])
    return SimpleNamespace(id=model)


class _FakeAgents:
    def __init__(self, agent, resume_error=None, create_error=None):
        self._agent = agent
        self.resume_calls = []
        self.create_calls = []
        self.resume_error = resume_error
        self.create_error = create_error

    def resume(self, agent_id, options=None, *a, **kw):
        self.resume_calls.append({"agent_id": agent_id, "options": options})
        if self.resume_error is not None:
            raise self.resume_error
        # Real bridge behavior (verified): the resumed handle's model comes
        # ONLY from the resume options — the stored conversation model is
        # NOT rehydrated ("agent.model is None on resume unless you pass
        # model again", SDK docs).
        self._agent.model = _model_selection_ns((options or {}).get("model"))
        return self._agent

    def create(self, **kw):
        self.create_calls.append(kw)
        if self.create_error is not None:
            err, self.create_error = self.create_error, None
            raise err
        return self._agent


class _FakeClient:
    def __init__(self, agent, **kw):
        self.agents = _FakeAgents(agent, **kw)


def _install_fake_sdk(monkeypatch, client):
    """Route run_sdk at a fake bridge client, offline-safe."""
    monkeypatch.setenv("CURSOR_API_KEY", "crsr_test_key")
    monkeypatch.setattr(gc_sdk, "sdk_available", lambda: True)
    monkeypatch.setattr(gc_sdk, "get_bridge", lambda workspace: client)
    monkeypatch.setattr(gc_sdk, "_agents", {})  # isolate the handle cache


def _happy_script(workdir="/w"):
    """assistant text + one tool call round-trip + wrap-up text."""
    return [
        _sdk_msg(type="assistant",
                 message=SimpleNamespace(content=[
                     SimpleNamespace(type="text", text="working on it")])),
        _sdk_msg(type="tool_call", call_id="t1", name="shell",
                 status="running", args={"command": "ls -la"}, result=None),
        _sdk_msg(type="tool_call", call_id="t1", name="shell",
                 status="completed", args={"command": "ls -la"},
                 result={"exitCode": 0, "stdout": "calc.py"}),
        _sdk_msg(type="assistant",
                 message=SimpleNamespace(content=[
                     SimpleNamespace(type="text", text="all done")])),
    ]


class TestSdkRunner:
    def test_happy_path_yields_session_messages_result(self, tmp_path, monkeypatch):
        agent = _FakeAgent(runs=[_FakeRun(_happy_script())])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk("do it", str(tmp_path),
                                     inactivity_timeout_s=30.0,
                                     cancel_check=lambda: False))
        keys = [k for k, _ in events]
        assert keys[0] == "sdk.session"
        assert events[0][1]["agentId"] == "agent-fake-1"
        assert events[0][1]["model"] == "fake-model"
        assert events[0][1]["resumed"] is False
        messages = [o for k, o in events if k == "sdk.message"]
        assert [m["type"] for m in messages] == [
            "assistant", "tool_call", "tool_call", "assistant"
        ]
        # Message payloads arrive as plain dicts, nested objects included.
        assert messages[0]["message"]["content"][0]["text"] == "working on it"
        assert messages[2]["result"]["stdout"] == "calc.py"
        assert keys[-1] == "sdk.result"
        assert events[-1][1]["status"] == "finished"
        assert agent.sent == ["do it"]

    def test_missing_api_key_raises_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CURSOR_API_KEY", raising=False)
        monkeypatch.setattr(gc_sdk, "sdk_available", lambda: True)
        with pytest.raises(gc_sdk.SdkRunnerError) as err:
            list(gc_sdk.run_sdk("t", str(tmp_path)))
        assert "CURSOR_API_KEY" in str(err.value)

    def test_missing_sdk_package_raises_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURSOR_API_KEY", "crsr_test_key")
        monkeypatch.setattr(gc_sdk, "sdk_available", lambda: False)
        with pytest.raises(gc_sdk.SdkRunnerError) as err:
            list(gc_sdk.run_sdk("t", str(tmp_path)))
        assert "pip install cursor-sdk" in str(err.value)

    def test_empty_task_and_bad_repo_preflight(self, tmp_path, monkeypatch):
        _install_fake_sdk(monkeypatch, _FakeClient(_FakeAgent()))
        with pytest.raises(gc_runner.HarnessError):
            list(gc_sdk.run_sdk("   ", str(tmp_path)))
        with pytest.raises(gc_runner.HarnessError):
            list(gc_sdk.run_sdk("t", str(tmp_path / "nope")))

    def test_bridge_launch_failure_raises_sdk_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CURSOR_API_KEY", "crsr_test_key")
        monkeypatch.setattr(gc_sdk, "sdk_available", lambda: True)

        def boom(workspace):
            raise RuntimeError("no bridge binary")

        monkeypatch.setattr(gc_sdk, "get_bridge", boom)
        with pytest.raises(gc_sdk.SdkRunnerError) as err:
            list(gc_sdk.run_sdk("t", str(tmp_path)))
        assert "bridge" in str(err.value)

    def test_nonretryable_create_failure_raises_actionable_error(
        self, tmp_path, monkeypatch
    ):
        client = _FakeClient(
            _FakeAgent(), create_error=RuntimeError("invalid model")
        )
        _install_fake_sdk(monkeypatch, client)
        with pytest.raises(gc_sdk.SdkRunnerError) as err:
            list(gc_sdk.run_sdk("t", str(tmp_path)))
        assert "CURSOR_API_KEY" in str(err.value)
        assert "invalid model" in str(err.value)

    def test_native_cancel_resolves_run_cancelled(self, tmp_path, monkeypatch):
        run = _FakeRun([])
        blocker = lambda: run.cancel_event.wait(10) and None  # noqa: E731
        run._stream._script[:] = [_sdk_msg(type="thinking", text="hm"), blocker]
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        polls = []

        def cancel_after_two_polls():
            polls.append(1)
            return len(polls) > 2

        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=cancel_after_two_polls,
        ))
        keys = [k for k, _ in events]
        assert "sdk.session" in keys
        assert events[-1] == ("sdk.result", {"status": "cancelled"})
        assert run.cancel_event.is_set(), "cancel must reach run.cancel()"

    def test_inactivity_watchdog_fires_on_true_silence(self, tmp_path, monkeypatch):
        run = _FakeRun([])
        blocker = lambda: run.cancel_event.wait(10)  # noqa: E731
        run._stream._script[:] = [_sdk_msg(type="thinking", text="hm"), blocker]
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=0.4, cancel_check=lambda: False,
        ))
        key, obj = events[-1]
        assert key == "sdk.error"
        assert obj["timeout"] is True
        assert "no activity" in obj["error"]
        assert run.cancel_event.is_set(), "watchdog must cancel the live run"

    def test_pending_tool_call_suspends_inactivity_clock(self, tmp_path, monkeypatch):
        def slow_tool_result():
            time.sleep(1.0)  # silent, but a tool call is in flight
            return _sdk_msg(type="tool_call", call_id="t-slow", name="shell",
                            status="completed",
                            result={"exitCode": 0, "stdout": "ok"})

        run = _FakeRun([
            _sdk_msg(type="tool_call", call_id="t-slow", name="shell",
                     status="running", args={"command": "npx tsc"}),
            slow_tool_result,
            _sdk_msg(type="assistant",
                     message=SimpleNamespace(content=[
                         SimpleNamespace(type="text", text="done")])),
        ])
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=0.4, cancel_check=lambda: False,
        ))
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_finished_tool_call_does_not_suspend_the_clock(
        self, tmp_path, monkeypatch
    ):
        run = _FakeRun([])
        blocker = lambda: run.cancel_event.wait(10)  # noqa: E731
        run._stream._script[:] = [
            _sdk_msg(type="tool_call", call_id="t-done", name="shell",
                     status="running", args={"command": "ls"}),
            _sdk_msg(type="tool_call", call_id="t-done", name="shell",
                     status="completed", result={"exitCode": 0, "stdout": ""}),
            blocker,  # true silence, no pending call
        ]
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=0.4, cancel_check=lambda: False,
        ))
        key, obj = events[-1]
        assert key == "sdk.error" and obj["timeout"] is True

    def test_max_wall_ceiling_kills_runaway_streams(self, tmp_path, monkeypatch):
        def chatty():
            time.sleep(0.05)
            return _sdk_msg(type="thinking", text="still going")

        run = _FakeRun([chatty] * 1000)
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        started = time.monotonic()
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, max_wall_s=0.6,
            cancel_check=lambda: False,
        ))
        assert time.monotonic() - started < 10
        key, obj = events[-1]
        assert key == "sdk.error"
        assert obj["timeout"] is True
        assert "max wall time" in obj["error"]

    def test_resume_uses_persisted_agent_id(self, tmp_path, monkeypatch):
        agent = _FakeAgent(agent_id="agent-prior",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "follow up", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-prior", model="gpt-5.3-codex",
        ))
        assert [c["agent_id"] for c in client.agents.resume_calls] == ["agent-prior"]
        assert client.agents.create_calls == []
        assert events[0][1]["resumed"] is True
        assert events[0][1]["agentId"] == "agent-prior"

    def test_resume_resupplies_the_model_so_followup_sends_work(
        self, tmp_path, monkeypatch
    ):
        """Live-bridge regression (e2e test_followup_send_carries_context):
        a resumed LOCAL agent handle carries NO model unless the resume
        options pass one again, and a model-less handle rejects every send
        with the non-retryable "Local SDK agents require an explicit
        `model`" error. Two sequential sends on one agent — create+send,
        then resume+send — must both finish."""
        agent = _FakeAgent(agent_id="agent-multi",
                           runs=[_FakeRun(_happy_script()),
                                 _FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)

        first = list(gc_sdk.run_sdk(
            "Create calc.py with add(a, b).", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            model="gpt-5.4-nano",
        ))
        assert first[-1] == ("sdk.result", {"status": "finished"})

        second = list(gc_sdk.run_sdk(
            "Add subtract(a, b) to calc.py.", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-multi", model="gpt-5.4-nano",
        ))
        # The resume re-supplied the model on its options...
        assert client.agents.resume_calls == [
            {"agent_id": "agent-multi", "options": {"model": "gpt-5.4-nano"}}
        ]
        # ...so the follow-up send succeeded instead of failing the run.
        assert not [o for k, o in second if k == "sdk.error"]
        assert second[-1] == ("sdk.result", {"status": "finished"})
        assert agent.sent == [
            "Create calc.py with add(a, b).",
            "Add subtract(a, b) to calc.py.",
        ]

    def test_followup_send_reuses_the_live_agent_handle_without_resume(
        self, tmp_path, monkeypatch
    ):
        """A follow-up in the SAME process reuses the live Agent handle
        (the SDK's canonical multi-turn flow) instead of re-resuming: a
        resume of an agent still registered on the live bridge makes the
        bridge async-dispose the old handle, and that disposal path can
        crash the bridge process (observed live 2026-07-03 as "peer closed
        connection" then "connection refused" on every re-attach)."""
        agent = _FakeAgent(agent_id="agent-live", model_id="fake-model",
                           runs=[_FakeRun(_happy_script()),
                                 _FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)

        first = list(gc_sdk.run_sdk(
            "task one", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        assert first[-1] == ("sdk.result", {"status": "finished"})

        # Same requested model as the live handle → no resume RPC at all.
        second = list(gc_sdk.run_sdk(
            "task two", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-live", model="fake-model",
        ))
        assert client.agents.resume_calls == []
        assert len(client.agents.create_calls) == 1
        assert second[0][1]["resumed"] is True
        assert second[-1] == ("sdk.result", {"status": "finished"})
        assert agent.sent == ["task one", "task two"]

    def test_followup_without_model_also_reuses_the_live_handle(
        self, tmp_path, monkeypatch
    ):
        """No model requested on the follow-up: the live handle (which has
        one) is reused as-is — never a model-less resume."""
        agent = _FakeAgent(agent_id="agent-live",
                           runs=[_FakeRun(_happy_script()),
                                 _FakeRun(_happy_script())])
        _install_fake_sdk(monkeypatch, (client := _FakeClient(agent)))

        list(gc_sdk.run_sdk(
            "task one", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        second = list(gc_sdk.run_sdk(
            "task two", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-live",
        ))
        assert client.agents.resume_calls == []
        assert second[-1] == ("sdk.result", {"status": "finished"})

    def test_resume_without_recorded_model_falls_back_to_default(
        self, tmp_path, monkeypatch
    ):
        """No model threaded on the follow-up (nothing recorded on the
        handle, no config): the resume still must carry SOME model — the
        same DEFAULT_MODEL fallback the create path uses — or the send is
        rejected by the bridge."""
        agent = _FakeAgent(agent_id="agent-prior",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "follow up", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-prior",
        ))
        assert client.agents.resume_calls[0]["options"] == {
            "model": gc_runner.DEFAULT_MODEL
        }
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_failed_resume_falls_back_to_fresh_agent(self, tmp_path, monkeypatch):
        agent = _FakeAgent(agent_id="agent-fresh",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent, resume_error=RuntimeError("unknown agent"))
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "follow up", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-gone",
        ))
        assert [c["agent_id"] for c in client.agents.resume_calls] == ["agent-gone"]
        assert len(client.agents.create_calls) == 1
        assert events[0][1]["resumed"] is False
        assert events[0][1]["agentId"] == "agent-fresh"
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_model_and_cwd_thread_into_create(self, tmp_path, monkeypatch):
        agent = _FakeAgent(model_id="gpt-5.3-codex",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            model="gpt-5.3-codex",
        ))
        call = client.agents.create_calls[0]
        assert call["model"] == "gpt-5.3-codex"
        assert call["local"]["cwd"] == str(gc_runner.resolve_repo(str(tmp_path)))
        assert events[0][1]["model"] == "gpt-5.3-codex"

    def test_retryable_create_error_is_retried(self, tmp_path, monkeypatch):
        agent = _FakeAgent(runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent, create_error=_RetryableError("bridge hiccup"))
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        assert len(client.agents.create_calls) == 2  # failed + retried
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_retryable_send_error_is_retried(self, tmp_path, monkeypatch):
        run = _FakeRun(_happy_script())

        class _FlakySendAgent(_FakeAgent):
            def send(self, message, *a, **kw):
                if not self.sent:
                    self.sent.append(message)
                    raise _RetryableError("http/2 stream reset")
                return super().send(message, *a, **kw)

        agent = _FlakySendAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        assert agent.sent == ["t", "t"]
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_dropped_stream_reattaches_via_observe_after_offset(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(gc_sdk, "_REATTACH_BACKOFF_S", 0.0)
        run = _FakeRun([
            _sdk_msg(type="thinking", text="before "),
            _sdk_msg(type="thinking", text="the drop"),
            ConnectionError("http/2 stream closed with error code CANCEL"),
            _sdk_msg(type="thinking", text="after the drop"),
            _sdk_msg(type="assistant",
                     message=SimpleNamespace(content=[
                         SimpleNamespace(type="text", text="all done")])),
        ])
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        # Reconnected exactly where it left off — nothing lost, nothing
        # duplicated, no synthetic user-visible messages.
        assert run.observe_calls == ["1"]
        texts = [o.get("text") for k, o in events
                 if k == "sdk.message" and o.get("type") == "thinking"]
        assert texts == ["before ", "the drop", "after the drop"]
        reattached = [o for k, o in events if k == "sdk.reattached"]
        assert len(reattached) == 1
        assert reattached[0]["offset"] == "1"
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_reattach_budget_exhaustion_fails_the_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gc_sdk, "_REATTACH_BACKOFF_S", 0.0)

        class _DeadStreamRun(_FakeRun):
            def observe(self, after_offset=None):
                self.observe_calls.append(after_offset)
                raise ConnectionError("bridge gone")

        run = _DeadStreamRun([
            _sdk_msg(type="thinking", text="hm"),
            ConnectionError("stream dropped"),
            _sdk_msg(type="thinking", text="never seen"),
        ])
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        assert len(run.observe_calls) == gc_sdk.MAX_STREAM_REATTACHES
        key, obj = events[-1]
        assert key == "sdk.error"
        assert "stream failed mid-run" in obj["error"]

    def test_bridge_is_cached_per_workspace_and_shutdown_closes(
        self, tmp_path, monkeypatch
    ):
        launches = []

        class _CloseableClient:
            def __init__(self, ws):
                self.ws = ws
                self.closed = False

            def close(self):
                self.closed = True

        def fake_launch_bridge(workspace=None, **kw):
            client = _CloseableClient(workspace)
            launches.append(client)
            return client

        monkeypatch.setattr(gc_sdk, "_bridges", {})
        fake_mod = SimpleNamespace(
            CursorClient=SimpleNamespace(launch_bridge=fake_launch_bridge)
        )
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "cursor_sdk", fake_mod)

        a1 = gc_sdk.get_bridge("/repo/a")
        a2 = gc_sdk.get_bridge("/repo/a")
        b1 = gc_sdk.get_bridge("/repo/b")
        assert a1 is a2 and a1 is not b1
        assert len(launches) == 2

        gc_sdk.shutdown_bridges()
        assert all(c.closed for c in launches)
        assert gc_sdk.get_bridge("/repo/a") is not a1  # relaunches after shutdown

    # -- bridge recycling (stale-bridge recovery lever) ------------------------

    def test_recycle_bridge_closes_cached_client_and_relaunches(
        self, monkeypatch
    ):
        closed = []

        class _Client:
            def __init__(self, ws):
                self.ws = ws

            def close(self):
                closed.append(self)

        launches = []

        def fake_launch_bridge(workspace=None, **kw):
            client = _Client(workspace)
            launches.append(client)
            return client

        import sys as _sys
        monkeypatch.setattr(gc_sdk, "_bridges", {})
        monkeypatch.setattr(gc_sdk, "_bridge_launched_at", {})
        monkeypatch.setattr(gc_sdk, "_agents", {})
        monkeypatch.setitem(_sys.modules, "cursor_sdk", SimpleNamespace(
            CursorClient=SimpleNamespace(launch_bridge=fake_launch_bridge)
        ))

        old = gc_sdk.get_bridge("/repo/a")
        gc_sdk.cache_agent("/repo/a", "agent-1", object())

        assert gc_sdk.recycle_bridge("/repo/a") is True
        # The stale client was closed and its agent handles dropped with it
        # (the agent itself survives on disk, resumable by agent_id)...
        assert closed == [old]
        assert gc_sdk.get_cached_agent("/repo/a", "agent-1") is None
        # ...and a FRESH bridge was launched eagerly.
        assert len(launches) == 2
        assert gc_sdk.get_bridge("/repo/a") is launches[-1]

    def test_recycle_bridge_without_cached_client_is_a_noop(self, monkeypatch):
        monkeypatch.setattr(gc_sdk, "_bridges", {})
        monkeypatch.setattr(gc_sdk, "_bridge_launched_at", {})
        assert gc_sdk.recycle_bridge("/repo/none") is False

    # -- terminal-error detail mining ----------------------------------------

    def test_terminal_error_with_typed_detail_emits_enriched_sdk_error(
        self, tmp_path, monkeypatch
    ):
        """A run settling with status "error" mines the typed
        CursorAgentError fields off the handle and emits them on sdk.error
        instead of the bare status."""
        run = _FakeRun([_sdk_msg(type="thinking", text="hm")],
                       final_status="error")
        run.error = _typed_sdk_error(
            "ServerError", "upstream 502", is_retryable=True, retry_after="30"
        )
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        key, obj = events[-1]
        assert key == "sdk.error"
        assert obj["error"] == "ServerError: upstream 502"
        assert obj["retryable"] is True
        assert obj["retry_after"] == "30"
        assert obj["run_status"] == "error"

    def test_terminal_error_detail_mined_from_wait_raise(
        self, tmp_path, monkeypatch
    ):
        """No error attribute on the handle, but run.wait() raises the
        typed error (the SDK's documented no-streaming path) — still mined."""

        class _WaitRaisesRun(_FakeRun):
            def wait(self):
                raise _typed_sdk_error(
                    "RateLimitError", "usage limits exceeded",
                    is_retryable=True, retry_after="120",
                )

        run = _WaitRaisesRun([_sdk_msg(type="thinking", text="hm")],
                             final_status="error")
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        key, obj = events[-1]
        assert key == "sdk.error"
        assert obj["error"] == "RateLimitError: usage limits exceeded"
        assert obj["retryable"] is True
        assert obj["retry_after"] == "120"

    def test_terminal_error_without_detail_falls_back_to_generic(
        self, tmp_path, monkeypatch
    ):
        """Nothing error-shaped recoverable off the handle: the payload
        keeps the generic text with unknown (None) retry fields — the run
        still settles as an error, never raises."""
        run = _FakeRun([_sdk_msg(type="thinking", text="hm")],
                       final_status="error")
        agent = _FakeAgent(runs=[run])
        _install_fake_sdk(monkeypatch, _FakeClient(agent))
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
        ))
        key, obj = events[-1]
        assert key == "sdk.error"
        assert obj["error"] == "cursor run ended with status: error"
        assert obj["retryable"] is None
        assert obj["retry_after"] is None
        assert obj["run_status"] == "error"


def _typed_sdk_error(name, message, is_retryable=None, retry_after=None):
    """An exception duck-typing the CursorAgentError surface, with a
    controllable type name (the payload renders '<TypeName>: <message>')."""
    cls = type(name, (Exception,), {})
    err = cls(message)
    err.message = message
    if is_retryable is not None:
        err.is_retryable = is_retryable
    if retry_after is not None:
        err.retry_after = retry_after
    return err


class _RetryableError(Exception):
    """Duck-typed CursorAgentError: retryable, no server retry_after."""

    is_retryable = True
    retry_after = "0"


# ---------------------------------------------------------------------------
# translate_model — legacy ACP-era model strings → SDK id + params
# ---------------------------------------------------------------------------

# The exact params the legacy forms below must translate to (parameter ids
# verified live against Cursor.models.list() for claude-fable-5).
_FABLE_BRACKET_SELECTION = {
    "id": "claude-fable-5",
    "params": [
        {"id": "thinking", "value": "true"},
        {"id": "context", "value": "300k"},
        {"id": "effort", "value": "high"},
    ],
}
_FABLE_THINKING_HIGH_SELECTION = {
    "id": "claude-fable-5",
    "params": [
        {"id": "thinking", "value": "true"},
        {"id": "effort", "value": "high"},
    ],
}


class TestModelTranslation:
    """The sdk model catalog has BASE ids only — combined slugs like
    "claude-fable-5-thinking-high" (the old CLI shorthand, previously our
    DEFAULT_MODEL) and ACP-era handle records like
    "claude-fable-5[thinking=true,context=300k,effort=high]" are rejected
    with BadRequestError. translate_model maps both onto id + params."""

    def test_default_model_is_a_base_catalog_id(self):
        assert gc_runner.DEFAULT_MODEL == "claude-fable-5"
        assert "[" not in gc_runner.DEFAULT_MODEL
        assert "-thinking" not in gc_runner.DEFAULT_MODEL

    def test_plain_base_id_passes_through(self):
        assert gc_sdk.translate_model("gpt-5.3-codex") == ("gpt-5.3-codex", None)

    def test_none_and_blank_pass_through(self):
        assert gc_sdk.translate_model(None) == (None, None)
        assert gc_sdk.translate_model("   ") == (None, None)

    def test_thinking_level_suffix_becomes_params(self):
        value, warning = gc_sdk.translate_model("claude-fable-5-thinking-high")
        assert warning is None
        assert value == _FABLE_THINKING_HIGH_SELECTION

    def test_bare_thinking_suffix_becomes_thinking_param(self):
        value, warning = gc_sdk.translate_model("claude-sonnet-5-thinking")
        assert warning is None
        assert value == {
            "id": "claude-sonnet-5",
            "params": [{"id": "thinking", "value": "true"}],
        }

    def test_extra_high_level_maps_to_catalog_xhigh(self):
        value, warning = gc_sdk.translate_model(
            "claude-fable-5-thinking-extra-high"
        )
        assert warning is None
        assert {"id": "effort", "value": "xhigh"} in value["params"]

    def test_bracket_suffix_becomes_params(self):
        value, warning = gc_sdk.translate_model(
            "claude-fable-5[thinking=true,context=300k,effort=high]"
        )
        assert warning is None
        assert value == _FABLE_BRACKET_SELECTION

    def test_empty_bracket_reduces_to_base_id(self):
        assert gc_sdk.translate_model("claude-fable-5[]") == (
            "claude-fable-5", None,
        )

    def test_unparseable_bracket_falls_back_to_default_with_warning(self):
        value, warning = gc_sdk.translate_model("claude-fable-5[thinking")
        assert value == gc_runner.DEFAULT_MODEL
        assert warning and "claude-fable-5[thinking" in warning
        assert gc_runner.DEFAULT_MODEL in warning

    def test_malformed_bracket_pair_falls_back_with_warning(self):
        value, warning = gc_sdk.translate_model("m[thinking=]")
        assert value == gc_runner.DEFAULT_MODEL
        assert warning

    def test_unknown_thinking_level_falls_back_with_warning(self):
        value, warning = gc_sdk.translate_model("m-thinking-banana")
        assert value == gc_runner.DEFAULT_MODEL
        assert warning

    def test_dash_suffix_threads_params_into_create(self, tmp_path, monkeypatch):
        agent = _FakeAgent(model_id="claude-fable-5",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            model="claude-fable-5-thinking-high",
        ))
        assert client.agents.create_calls[0]["model"] == (
            _FABLE_THINKING_HIGH_SELECTION
        )
        assert events[0][1]["model"] == "claude-fable-5"
        assert not [o for k, o in events if k == "sdk.model_warning"]
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_legacy_bracket_record_threads_params_into_resume(
        self, tmp_path, monkeypatch
    ):
        """The exact model string a pre-swap handle recorded must never
        reach Agent.resume verbatim (BadRequestError live)."""
        agent = _FakeAgent(agent_id="agent-prior",
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "follow up", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-prior",
            model="claude-fable-5[thinking=true,context=300k,effort=high]",
        ))
        assert client.agents.resume_calls[0]["options"] == {
            "model": _FABLE_BRACKET_SELECTION
        }
        assert events[0][1]["model"] == "claude-fable-5"
        assert not [o for k, o in events if k == "sdk.model_warning"]
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_unparseable_model_warns_and_uses_default(self, tmp_path, monkeypatch):
        agent = _FakeAgent(model_id=gc_runner.DEFAULT_MODEL,
                           runs=[_FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        events = list(gc_sdk.run_sdk(
            "t", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            model="claude-fable-5[borked",
        ))
        # The warning is the FIRST event, so the substitution lands in the
        # event log before any run activity.
        key, obj = events[0]
        assert key == "sdk.model_warning"
        assert obj["requested"] == "claude-fable-5[borked"
        assert obj["using"] == gc_runner.DEFAULT_MODEL
        assert client.agents.create_calls[0]["model"] == gc_runner.DEFAULT_MODEL
        assert events[-1] == ("sdk.result", {"status": "finished"})

    def test_same_base_id_with_params_reuses_the_live_handle(
        self, tmp_path, monkeypatch
    ):
        """A params-only difference must not force a resume of an agent
        still registered on the live bridge (the disposal-crash path) —
        base-id comparison decides handle reuse."""
        agent = _FakeAgent(agent_id="agent-live", model_id="claude-fable-5",
                           runs=[_FakeRun(_happy_script()),
                                 _FakeRun(_happy_script())])
        client = _FakeClient(agent)
        _install_fake_sdk(monkeypatch, client)
        list(gc_sdk.run_sdk(
            "task one", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            model="claude-fable-5",
        ))
        second = list(gc_sdk.run_sdk(
            "task two", str(tmp_path),
            inactivity_timeout_s=30.0, cancel_check=lambda: False,
            agent_id="agent-live", model="claude-fable-5-thinking-high",
        ))
        assert client.agents.resume_calls == []
        assert second[-1] == ("sdk.result", {"status": "finished"})


# ---------------------------------------------------------------------------
# events.SdkNormalizer — SDKMessage dicts → canonical envelopes
# ---------------------------------------------------------------------------

class TestSdkNormalizer:
    def _norm(self):
        return gc_events.SdkNormalizer()

    def test_session_event_maps_to_run_started(self):
        envs = self._norm().normalize(
            "sdk.session",
            {"agentId": "agent-1", "cwd": "/w", "model": "m", "resumed": False},
        )
        assert envs == [{
            "source": "ghost", "kind": "lifecycle", "event": "run.started",
            "model": "m", "cwd": "/w", "harness_session_id": "agent-1",
        }]

    def test_reattached_maps_to_log_only_lifecycle(self):
        envs = self._norm().normalize(
            "sdk.reattached", {"offset": "41", "attempt": 1}
        )
        assert envs[0]["event"] == "stream.reattached"
        assert envs[0]["offset"] == "41"

    def test_finished_maps_to_run_completed(self):
        envs = self._norm().normalize("sdk.result", {"status": "finished"})
        assert envs[0]["event"] == "run.completed"
        assert envs[0]["status"] == "completed"

    def test_cancelled_maps_to_run_failed_cancelled(self):
        envs = self._norm().normalize("sdk.result", {"status": "cancelled"})
        assert envs[0]["event"] == "run.failed"
        assert envs[0]["cancelled"] is True
        assert "cancel" in envs[0]["error"]

    def test_expired_maps_to_run_failed_timeout(self):
        envs = self._norm().normalize("sdk.result", {"status": "expired"})
        assert envs[0]["event"] == "run.failed"
        assert envs[0]["timeout"] is True

    def test_error_status_maps_to_run_failed(self):
        envs = self._norm().normalize("sdk.result", {"status": "error"})
        assert envs[0]["event"] == "run.failed"
        assert "error" in envs[0]["error"]

    def test_sdk_error_maps_to_run_failed(self):
        envs = self._norm().normalize(
            "sdk.error", {"error": "cursor run timed out: no activity for 600s",
                          "timeout": True}
        )
        assert envs[0]["event"] == "run.failed"
        assert envs[0]["timeout"] is True

    def test_assistant_message_maps_to_content(self):
        envs = self._norm().normalize("sdk.message", {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "hello "},
                                    {"type": "text", "text": "world"}]},
        })
        assert envs == [{
            "source": "ghost", "kind": "content", "delta": "hello world",
            "done": False,
        }]

    def test_thinking_maps_to_reasoning(self):
        envs = self._norm().normalize("sdk.message", {
            "type": "thinking", "text": "pondering notes.txt",
        })
        assert envs[0]["event"] == "reasoning"
        assert envs[0]["text"] == "pondering notes.txt"

    def test_noise_types_produce_no_envelopes(self):
        norm = self._norm()
        for mtype in ("system", "user", "request", "status"):
            assert norm.normalize("sdk.message", {"type": mtype}) == []

    def test_shell_tool_call_round_trip(self):
        norm = self._norm()
        started = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t1", "name": "shell",
            "status": "running", "args": {"command": "ls -la"},
        })
        assert started == [{
            "source": "ghost", "kind": "tool_use", "id": "t1",
            "tool": "shell", "status": "running", "title": "ls -la",
            "command": "ls -la",
        }]
        done = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t1", "name": "shell",
            "status": "completed",
            "result": {"exitCode": 0, "stdout": "calc.py"},
        })
        assert done[0]["kind"] == "tool_result"
        assert done[0]["status"] == "done"
        assert "calc.py" in done[0]["output"]
        assert isinstance(done[0]["durationMs"], int)

    def test_repeated_running_messages_are_deduped(self):
        norm = self._norm()
        first = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t1", "name": "shell",
            "status": "running", "args": {},
        })
        assert len(first) == 1
        again = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t1", "name": "shell",
            "status": "running", "args": {"command": "ls"},
        })
        assert again == []

    def test_nonzero_exit_code_marks_result_error(self):
        norm = self._norm()
        norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t8", "name": "shell",
            "status": "running", "args": {"command": "false"},
        })
        envs = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t8", "name": "shell",
            "status": "completed",
            "result": {"exitCode": 1, "stdout": "", "stderr": "nope"},
        })
        assert envs[0]["status"] == "error"
        assert "nope" in envs[0]["output"]

    def test_error_status_tool_call_maps_to_error_result(self):
        norm = self._norm()
        norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t9", "name": "shell",
            "status": "running", "args": {"command": "boom"},
        })
        envs = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "t9", "name": "shell",
            "status": "error", "result": "command not found: boom",
        })
        assert envs[0]["kind"] == "tool_result"
        assert envs[0]["status"] == "error"
        assert "not found" in envs[0]["output"]

    def test_edit_tool_full_content_yields_file_diff(self):
        norm = self._norm()
        norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "e1", "name": "edit_file",
            "status": "running", "args": {"path": "/w/calc.py"},
        })
        envs = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "e1", "name": "edit_file",
            "status": "completed",
            "result": {"path": "/w/calc.py",
                       "beforeFullFileContent": "a\n",
                       "afterFullFileContent": "a\nb\n"},
        })
        assert envs[0]["kind"] == "tool_result"
        assert envs[0]["additions"] == 1 and envs[0]["deletions"] == 0
        diff = envs[1]
        assert diff["kind"] == "file_diff"
        assert diff["path"] == "/w/calc.py"
        assert diff["status"] == "M"
        assert "+b" in diff["diff"]

    def test_edit_tool_old_new_text_blocks_yield_file_diff(self):
        norm = self._norm()
        norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "e2", "name": "write",
            "status": "running", "args": {"path": "/w/notes.txt"},
        })
        envs = norm.normalize("sdk.message", {
            "type": "tool_call", "call_id": "e2", "name": "write",
            "status": "completed",
            "result": {"content": [{"path": "/w/notes.txt",
                                    "oldText": "", "newText": "hello\n"}]},
        })
        diffs = [e for e in envs if e["kind"] == "file_diff"]
        assert len(diffs) == 1
        assert diffs[0]["status"] == "A"
        assert diffs[0]["after"] == "hello\n"

    def test_real_sdk_edit_payload_yields_file_diff(self):
        """Regression for the live blind spot (2026-07-03): a REAL run
        created + committed a file but the completion said "no files were
        changed". The real edit tool's result wraps its payload in
        {"status": "success", "value": {linesAdded, linesRemoved,
        diffString}} and carries the path ONLY in the call's args.
        Payloads in the fixture were captured verbatim from the
        reproduction (model gpt-5.4-nano)."""
        msgs = [
            json.loads(line)
            for line in SDK_EDIT_FIXTURE.read_text().splitlines()
            if line.strip()
        ]
        norm = self._norm()
        envs = []
        for msg in msgs:
            envs.extend(norm.normalize("sdk.message", msg))

        diffs = [e for e in envs if e["kind"] == "file_diff"]
        assert len(diffs) == 1
        assert diffs[0]["path"] == "/private/tmp/gc-probe/repo/hello.txt"
        assert diffs[0]["status"] == "A"  # "--- /dev/null" diff header
        assert "+hello from sdk probe" in diffs[0]["diff"]
        assert diffs[0]["added"] == 1  # linesAdded from the payload

        edit_result = next(
            e for e in envs
            if e["kind"] == "tool_result" and e["id"] == msgs[0]["call_id"]
        )
        assert edit_result["additions"] == 1

        # The shell result rides the same {"status", "value"} envelope:
        # output and the non-zero exit code must still be mined from it.
        shell_result = [e for e in envs if e["kind"] == "tool_result"][-1]
        assert shell_result is not edit_result
        assert shell_result["status"] == "error"  # exitCode 128 in "value"
        assert "not a git repository" in shell_result["output"]

    def test_model_warning_maps_to_lifecycle(self):
        envs = self._norm().normalize("sdk.model_warning", {
            "warning": "requested model 'x[y' has an unparseable bracket "
                       "suffix — falling back to 'claude-fable-5'",
            "requested": "x[y", "using": "claude-fable-5",
        })
        assert envs[0]["kind"] == "lifecycle"
        assert envs[0]["event"] == "model.warning"
        assert envs[0]["requested"] == "x[y"
        assert envs[0]["using"] == "claude-fable-5"

    def test_terminal_tool_call_without_start_synthesizes_tool_use(self):
        envs = self._norm().normalize("sdk.message", {
            "type": "tool_call", "call_id": "fast", "name": "shell",
            "status": "completed", "args": {"command": "true"},
            "result": {"exitCode": 0, "stdout": ""},
        })
        assert [e["kind"] for e in envs] == ["tool_use", "tool_result"]
        assert envs[0]["status"] == "running"
        assert envs[1]["status"] == "done"

    def test_unstable_tool_payload_shapes_never_crash(self):
        norm = self._norm()
        for weird in (None, 42, "text", ["a", 1], {"nested": {"deep": object()}}):
            envs = norm.normalize("sdk.message", {
                "type": "tool_call", "call_id": f"w-{id(weird)}",
                "name": "mystery", "status": "completed", "result": weird,
            })
            assert envs, "terminal tool_call must always render"
            assert envs[-1]["kind"] == "tool_result"

    def test_usage_maps_to_log_only_lifecycle(self):
        envs = self._norm().normalize("sdk.message", {
            "type": "usage", "usage": {"total_tokens": 1234},
        })
        assert envs[0]["event"] == "usage"
        assert envs[0]["usage"]["total_tokens"] == 1234

    def test_unknown_message_type_passes_through(self):
        envs = self._norm().normalize("sdk.message", {"type": "mystery", "x": 1})
        assert envs[0]["event"] == "passthrough"
        assert envs[0]["name"] == "sdk.mystery"

    def test_unknown_runner_event_passes_through(self):
        envs = self._norm().normalize("sdk.wat", {"x": 1})
        assert envs[0]["event"] == "passthrough"
        assert envs[0]["name"] == "sdk.wat"
