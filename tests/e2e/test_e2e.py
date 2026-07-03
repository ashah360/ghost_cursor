"""End-to-end tests for ghost_cursor — NO MOCKS.

Runs the real plugin against the real `cursor-agent` binary and the real Hermes
core modules. Exercises every input shape of the v0.4 named-session interface:
cursor_create_session (lazy), cursor_send_message (first task + resume),
cursor_status (read-only), cursor_stop, cursor_events, cursor_list, the
same-repo concurrency guard, and bogus-handle fallback.

Assertions are INVARIANTS, not exact output (the model is nondeterministic):
"a .py file exists / imports / defines X / a named session came back / the
status header says completed / status never cancels", never "the diff equals
this fixture". Tool output is plain TEXT in v0.4 — assertions match header
lines like 'status: completed', never JSON keys.

Requires (skips cleanly if absent):
  - CURSOR_API_KEY in env
  - `cursor-agent` on PATH
  - GHOST_CURSOR_E2E=1  (opt-in; keeps the real-network suite off by default)

Model is pinned cheap via GHOST_CURSOR_TEST_MODEL (default gpt-5.4-nano-low) so
CI stays fast + cheap. Tasks are trivially small so even a weak model nails them.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

CURSOR_MODEL = os.environ.get("GHOST_CURSOR_TEST_MODEL", "gpt-5.4-nano-low")

# The Hermes test suite's autouse `_hermetic_environment` fixture scrubs
# API-key env vars (so unit tests can't hit real APIs). This is the opt-in
# REAL-network suite, so capture the key at import time — before any fixture
# runs — and restore it per-test.
_REAL_CURSOR_KEY = os.environ.get("CURSOR_API_KEY")

_run = os.environ.get("GHOST_CURSOR_E2E") == "1"
_have_key = bool(_REAL_CURSOR_KEY)
_have_bin = shutil.which("cursor-agent") is not None

pytestmark = pytest.mark.skipif(
    not (_run and _have_key and _have_bin),
    reason="e2e opt-in: set GHOST_CURSOR_E2E=1, CURSOR_API_KEY, and install cursor-agent",
)

TERMINAL = ("completed", "failed", "cancelled", "timeout")


def _find_plugin_init() -> Path:
    """Locate the ghost_cursor package __init__.py robustly.

    Works whether the e2e file lives beside the plugin (repo layout:
    <repo>/tests/e2e/, plugin at <repo>/__init__.py) or is copied into a
    Hermes tree (plugin at <hermes>/plugins/ghost_cursor/__init__.py).
    Override with GHOST_CURSOR_PLUGIN_INIT.
    """
    override = os.environ.get("GHOST_CURSOR_PLUGIN_INIT")
    if override and Path(override).is_file():
        return Path(override)
    here = Path(__file__).resolve()
    candidates = [
        # repo layout: tests/e2e/test_e2e.py -> <repo>/__init__.py
        here.parents[2] / "__init__.py",
        here.parents[1] / "__init__.py",
        # installed into a hermes tree
        Path("plugins/ghost_cursor/__init__.py").resolve(),
    ]
    # also search PYTHONPATH roots for plugins/ghost_cursor/__init__.py
    for root in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if root:
            candidates.append(Path(root) / "plugins" / "ghost_cursor" / "__init__.py")
    for c in candidates:
        # a real plugin init defines cursor_create_session; the empty
        # tests/plugins/__init__ does not — verify content, not just existence
        try:
            if c.is_file() and "def cursor_create_session" in c.read_text(encoding="utf-8"):
                return c
        except Exception:
            continue
    raise RuntimeError(f"could not locate ghost_cursor __init__.py; tried {candidates}")


PLUGIN = _find_plugin_init()


@pytest.fixture()
def gc(monkeypatch, tmp_path, request):
    """Load the real plugin module with an isolated HERMES_HOME and a UNIQUE
    session key per test (so the same-repo concurrency guard and the handle
    table never leak state between tests)."""
    home = tmp_path / "hermes_home"
    (home / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("GHOST_CURSOR_MODEL", CURSOR_MODEL)
    # Restore the real key the Hermes hermetic fixture scrubbed — this is the
    # opt-in real-network suite and needs it to reach cursor.
    if _REAL_CURSOR_KEY:
        monkeypatch.setenv("CURSOR_API_KEY", _REAL_CURSOR_KEY)
    spec = importlib.util.spec_from_file_location("gc_e2e", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    # The plugin uses relative intra-package imports (it loads under
    # hermes_plugins.<slug> in real Hermes), so the package must be in
    # sys.modules before exec for `from . import ...` to resolve.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod._resolve_progress_callback = lambda: None
    # unique key per test — prevents cross-test handle/concurrency collisions
    key = f"e2e-{request.node.name}"
    mod._resolve_session_key = lambda: key
    return mod


def _repo(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    return str(d)


def _create(gc, repo):
    """cursor_create_session → the minted session name (from the text ack)."""
    ack = gc._handle_cursor_create_session({"repo": repo, "model": CURSOR_MODEL})
    assert ack.startswith("session: "), f"not a create ack: {ack!r}"
    return ack.splitlines()[0].split("session: ", 1)[1].strip()


def _send(gc, session, message):
    return gc._handle_cursor_send_message({"session": session, "message": message})


def _status_word(gc, session):
    """The value of the 'status:' header line (or '' if the output is a
    prose error like an unknown-session message)."""
    out = gc._handle_cursor_status({"session": session})
    first = out.splitlines()[0]
    if first.startswith("status: "):
        return first.split("status: ", 1)[1].strip()
    return ""


def _wait_done(gc, session, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        word = _status_word(gc, session)
        if word in TERMINAL:
            return word
        time.sleep(4)
    return _status_word(gc, session)


def test_create_send_status_readonly_and_completion(gc, tmp_path):
    """create mints a named session lazily; the first send starts the run;
    polling status never cancels; the run completes and the file lands."""
    repo = _repo(tmp_path)
    name = _create(gc, repo)
    # Lazy: nothing terminal, nothing running — the handle merely exists.
    assert _status_word(gc, name) not in TERMINAL

    ack = _send(gc, name, "Create calc.py with a function add(a, b) that returns a + b.")
    assert f"sent to {name}" in ack
    assert "running in background" in ack

    # poll twice mid-run — must NOT cancel
    for _ in range(2):
        time.sleep(6)
        status = gc._handle_cursor_status({"session": name})
        assert f"session: {name}" in status
    # whether or not we caught it running, the key invariant: status never killed it
    final = _wait_done(gc, name)
    assert final == "completed", f"run did not complete: {final}"

    calc = Path(repo) / "calc.py"
    assert calc.exists(), "calc.py was not created"
    ns = {}
    exec(calc.read_text(), ns)
    assert ns["add"](2, 3) == 5

    # the persisted event history pages as text, keyed by the same name
    events = gc._handle_cursor_events({"session": name})
    assert events.startswith("events "), f"not an events page: {events[:80]!r}"
    # the session shows up in the scoped TSV listing
    listing = gc._handle_cursor_list({})
    assert name in listing
    assert listing.splitlines()[0].split("\t")[0].strip() == "session"


def test_followup_send_carries_context(gc, tmp_path):
    """Second message on the SAME named session builds on the first task."""
    repo = _repo(tmp_path)
    name = _create(gc, repo)
    _send(gc, name, "Create calc.py with add(a, b).")
    assert _wait_done(gc, name) == "completed"

    ack = _send(gc, name, "Add subtract(a, b) to calc.py in the same style.")
    assert f"sent to {name}" in ack
    assert _wait_done(gc, name) == "completed"

    calc = Path(repo) / "calc.py"
    ns = {}
    exec(calc.read_text(), ns)
    assert ns["add"](2, 3) == 5
    assert ns["subtract"](5, 2) == 3


def test_same_repo_second_run_rejected(gc, tmp_path):
    """Sending into a second session on a repo with an active run is refused
    with actionable prose naming the active session."""
    repo = _repo(tmp_path)
    first = _create(gc, repo)
    _send(gc, first, "Create a.py with a long docstring and function a().")

    second = _create(gc, repo)
    out = _send(gc, second, "Create b.py.")
    assert "already running" in out, f"second run should be refused: {out!r}"
    assert first in out  # the refusal hands back the ACTIVE session's name
    _wait_done(gc, first)


def test_bogus_handle_is_graceful(gc):
    """status/send/stop/events on an unknown handle degrade to actionable
    prose — no exception, no traceback."""
    for handler in (
        gc._handle_cursor_status,
        gc._handle_cursor_stop,
        gc._handle_cursor_events,
    ):
        out = handler({"session": "does-not-exist-xyz"})
        assert "no session named 'does-not-exist-xyz'" in out
    out = gc._handle_cursor_send_message(
        {"session": "does-not-exist-xyz", "message": "hi"})
    assert "no session named 'does-not-exist-xyz'" in out


def test_stop_cancels_running(gc, tmp_path):
    """cursor_stop on a running job cancels it gracefully and reports the
    outcome as a status-headed text block."""
    repo = _repo(tmp_path)
    name = _create(gc, repo)
    _send(gc, name, "Create big.py with five functions, each with a long docstring.")
    time.sleep(5)
    out = gc._handle_cursor_stop({"session": name})
    assert out.startswith("status: "), f"not a stop report: {out[:80]!r}"
    final = _wait_done(gc, name, timeout=30)
    assert final in TERMINAL
