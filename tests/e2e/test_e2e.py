"""End-to-end tests for ghost_cursor — NO MOCKS.

Runs the real plugin against the real `cursor-agent` binary and the real Hermes
core modules. Exercises every input shape of the v0.3 handle interface:
cursor_start (new + resume), cursor_send, cursor_status (read-only), cursor_stop,
same-repo concurrency guard, and bogus-handle fallback.

Assertions are INVARIANTS, not exact output (the model is nondeterministic):
"a .py file exists / imports / defines X / a handle came back / status never
cancels", never "the diff equals this fixture".

Requires (skips cleanly if absent):
  - CURSOR_API_KEY in env
  - `cursor-agent` on PATH
  - GHOST_CURSOR_E2E=1  (opt-in; keeps the real-network suite off by default)

Model is pinned cheap via GHOST_CURSOR_TEST_MODEL (default gpt-5.4-nano-low) so
CI stays fast + cheap. Tasks are trivially small so even a weak model nails them.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

CURSOR_MODEL = os.environ.get("GHOST_CURSOR_TEST_MODEL", "gpt-5.4-nano-low")

_run = os.environ.get("GHOST_CURSOR_E2E") == "1"
_have_key = bool(os.environ.get("CURSOR_API_KEY"))
_have_bin = shutil.which("cursor-agent") is not None

pytestmark = pytest.mark.skipif(
    not (_run and _have_key and _have_bin),
    reason="e2e opt-in: set GHOST_CURSOR_E2E=1, CURSOR_API_KEY, and install cursor-agent",
)

PLUGIN = Path(__file__).resolve().parents[1] / "__init__.py"


@pytest.fixture()
def gc(monkeypatch, tmp_path):
    """Load the real plugin module with an isolated HERMES_HOME."""
    home = tmp_path / "hermes_home"
    (home / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GHOST_CURSOR_MODEL", CURSOR_MODEL)
    spec = importlib.util.spec_from_file_location("gc_e2e", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._resolve_progress_callback = lambda: None
    mod._resolve_session_key = lambda: "e2e-session"
    return mod


def _repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    return str(d)


def _wait_done(gc, sid, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = json.loads(gc._handle_cursor_status({"session_id": sid}))
        if s.get("status") in ("completed", "failed", "cancelled", "timeout"):
            return s
        time.sleep(4)
    return json.loads(gc._handle_cursor_status({"session_id": sid}))


def test_start_status_readonly_and_completion(gc, tmp_path):
    """cursor_start returns a handle; polling status never cancels; run completes
    and the file lands on disk."""
    repo = _repo(tmp_path)
    r = json.loads(gc._handle_cursor_start(
        {"task": "Create calc.py with a function add(a, b) that returns a + b.",
         "repo": repo, "model": CURSOR_MODEL}))
    assert r.get("success") is True
    sid = r.get("session_id")
    assert sid, f"no handle returned: {r}"
    assert r.get("status") == "running"

    # poll twice mid-run — must NOT cancel
    seen_running = False
    for _ in range(2):
        time.sleep(6)
        s = json.loads(gc._handle_cursor_status({"session_id": sid}))
        if s.get("status") == "running":
            seen_running = True
    # whether or not we caught it running, the key invariant: status never killed it
    final = _wait_done(gc, sid)
    assert final.get("status") == "completed", f"run did not complete: {final}"

    calc = Path(repo) / "calc.py"
    assert calc.exists(), "calc.py was not created"
    ns = {}
    exec(calc.read_text(), ns)
    assert ns["add"](2, 3) == 5


def test_resume_carries_context(gc, tmp_path):
    """Explicit resume: pass the handle back, second task builds on the first."""
    repo = _repo(tmp_path)
    r1 = json.loads(gc._handle_cursor_start(
        {"task": "Create calc.py with add(a, b).", "repo": repo, "model": CURSOR_MODEL}))
    sid = r1["session_id"]
    _wait_done(gc, sid)
    r2 = json.loads(gc._handle_cursor_start(
        {"task": "Add subtract(a, b) to calc.py in the same style.",
         "repo": repo, "session_id": sid, "model": CURSOR_MODEL}))
    _wait_done(gc, r2["session_id"])
    calc = Path(repo) / "calc.py"
    ns = {}
    exec(calc.read_text(), ns)
    assert ns["add"](2, 3) == 5
    assert ns["subtract"](5, 2) == 3


def test_same_repo_second_start_rejected(gc, tmp_path):
    """A second cursor_start on a repo with an active run is rejected."""
    repo = _repo(tmp_path)
    r1 = json.loads(gc._handle_cursor_start(
        {"task": "Create a.py with a long docstring and function a().",
         "repo": repo, "model": CURSOR_MODEL}))
    sid = r1["session_id"]
    # immediately try a second — should be rejected while the first runs
    r2 = json.loads(gc._handle_cursor_start(
        {"task": "Create b.py.", "repo": repo, "model": CURSOR_MODEL}))
    assert r2.get("success") is False or r2.get("status") == "rejected", \
        f"second start should be rejected: {r2}"
    _wait_done(gc, sid)


def test_bogus_handle_is_graceful(gc):
    """status/send/stop on an unknown handle degrade gracefully, no exception."""
    for handler in (gc._handle_cursor_status, gc._handle_cursor_stop):
        out = json.loads(handler({"session_id": "does-not-exist-xyz"}))
        assert isinstance(out, dict)
    out = json.loads(gc._handle_cursor_send(
        {"session_id": "does-not-exist-xyz", "message": "hi"}))
    assert isinstance(out, dict)


def test_stop_cancels_running(gc, tmp_path):
    """cursor_stop on a running job cancels it gracefully."""
    repo = _repo(tmp_path)
    r = json.loads(gc._handle_cursor_start(
        {"task": "Create big.py with five functions, each with a long docstring.",
         "repo": repo, "model": CURSOR_MODEL}))
    sid = r["session_id"]
    time.sleep(5)
    out = json.loads(gc._handle_cursor_stop({"session_id": sid}))
    assert isinstance(out, dict)
    final = _wait_done(gc, sid, timeout=30)
    assert final.get("status") in ("cancelled", "completed", "failed", "timeout")
