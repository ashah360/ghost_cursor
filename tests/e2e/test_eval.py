"""E2E EVAL (shape 2) — LLM-as-judge on a real plugin run.

Shapes 1 & 3 assert on *facts* (tool called, file exists, no exception). This
shape evaluates *quality*: it runs a real cursor task through the v0.4 plugin
(create session → send message), captures the full context (the plain-text
tool acks, the final status text, the paged event history, and the final
files), and asks a cheap judge model: "did this go smoothly, or are there
concerns a user would notice?" — catching soft regressions (garbled diffs,
error text leaking into output, empty/confused results) that a pass/fail
assertion misses.

Requires (skips cleanly otherwise):
  - GHOST_CURSOR_E2E=1
  - CURSOR_API_KEY   (cursor)
  - OPENAI_API_KEY   (the judge model)
  - cursor-agent on PATH
  - a Hermes checkout importable

Models:
  - cursor task:  GHOST_CURSOR_TEST_MODEL   (default gpt-5.4-nano)
  - judge:        GHOST_CURSOR_JUDGE_MODEL  (default gpt-5.4-nano — cheap)
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

_CURSOR_KEY = os.environ.get("CURSOR_API_KEY")
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
CURSOR_MODEL = os.environ.get("GHOST_CURSOR_TEST_MODEL", "gpt-5.4-nano")
JUDGE_MODEL = os.environ.get("GHOST_CURSOR_JUDGE_MODEL", "gpt-5.4-nano")

_run = os.environ.get("GHOST_CURSOR_E2E") == "1"
_have_bin = shutil.which("cursor-agent") is not None

pytestmark = pytest.mark.skipif(
    not (_run and _CURSOR_KEY and _OPENAI_KEY and _have_bin),
    reason="e2e-eval opt-in: needs GHOST_CURSOR_E2E=1, CURSOR_API_KEY, OPENAI_API_KEY, cursor-agent",
)

TERMINAL = ("completed", "failed", "cancelled", "timeout")


def _find_plugin_init() -> Path:
    override = os.environ.get("GHOST_CURSOR_PLUGIN_INIT")
    if override and Path(override).is_file():
        return Path(override)
    here = Path(__file__).resolve()
    cands = []
    for n in (2, 1):
        if len(here.parents) > n:
            cands.append(here.parents[n] / "__init__.py")
    cands.append(Path("plugins/ghost_cursor/__init__.py").resolve())
    for root in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if root:
            cands.append(Path(root) / "plugins" / "ghost_cursor" / "__init__.py")
    for c in cands:
        try:
            if c.is_file() and "def cursor_create_session" in c.read_text(encoding="utf-8"):
                return c
        except Exception:
            continue
    raise RuntimeError(f"plugin __init__ not found; tried {cands}")


def _judge(context: str) -> dict:
    """Ask a cheap judge model to rate the plugin run. Returns {verdict, reason}."""
    import urllib.request
    prompt = (
        "You are a strict QA reviewer for a coding-agent plugin (ghost_cursor) that "
        "delegates edits to the Cursor agent. Below is the FULL context of one real "
        "run: the plugin's plain-text tool outputs (session ack, final status block, "
        "paged event history) and the final files.\n\n"
        "Judge whether the run went SMOOTHLY from a user's perspective. Concerns include: "
        "error/traceback text leaking into output, empty or confused results, a diff that "
        "doesn't match the request, the tool reporting completion but no real change, or "
        "auth/protocol errors. If the task was accomplished cleanly with a coherent result, "
        "it PASSES.\n\n"
        f"=== RUN CONTEXT ===\n{context[:6000]}\n=== END ===\n\n"
        'Respond with ONLY compact JSON: {"verdict":"pass"|"fail","reason":"<one sentence>"}'
    )
    body = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + _OPENAI_KEY, "Content-Type": "application/json"},
        method="POST",
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
    text = resp["choices"][0]["message"]["content"].strip()
    # tolerate code fences
    text = text.strip("`").lstrip("json").strip()
    try:
        return json.loads(text)
    except Exception:
        return {"verdict": "fail", "reason": f"judge returned unparseable: {text[:120]}"}


def test_plugin_run_is_clean_by_llm_judge(tmp_path):
    home = tmp_path / "hermes_home"
    (home / "state").mkdir(parents=True, exist_ok=True)
    os.environ["HERMES_HOME"] = str(home)
    if _CURSOR_KEY:
        os.environ["CURSOR_API_KEY"] = _CURSOR_KEY

    spec = importlib.util.spec_from_file_location("gc_eval", _find_plugin_init())
    gc = importlib.util.module_from_spec(spec)
    # Relative intra-package imports need the package registered before exec.
    sys.modules[spec.name] = gc
    spec.loader.exec_module(gc)
    gc._resolve_progress_callback = lambda: None
    gc._resolve_session_key = lambda: f"eval-{int(time.time())}"

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    create_ack = gc._handle_cursor_create_session(
        {"repo": str(repo), "model": CURSOR_MODEL})
    assert create_ack.startswith("session: "), f"no session minted: {create_ack!r}"
    name = create_ack.splitlines()[0].split("session: ", 1)[1].strip()

    send_ack = gc._handle_cursor_send_message({
        "session": name,
        "message": "Create stringutils.py with a function shout(s) that returns s "
                   "uppercased with '!' appended. Include a docstring.",
    })
    assert f"sent to {name}" in send_ack, f"send not acknowledged: {send_ack!r}"

    # wait for a terminal status header, capturing the final status text
    final = None
    deadline = time.time() + 180
    while time.time() < deadline:
        status = gc._handle_cursor_status({"session": name})
        first = status.splitlines()[0]
        if first.startswith("status: ") and first.split("status: ", 1)[1].strip() in TERMINAL:
            final = status
            break
        time.sleep(4)
    assert final is not None, "run never reached a terminal state"

    events = gc._handle_cursor_events({"session": name, "limit": 20})

    files_txt = ""
    su = repo / "stringutils.py"
    if su.exists():
        files_txt = su.read_text()

    context = (
        f"--- create ack ---\n{create_ack}\n\n"
        f"--- send ack ---\n{send_ack}\n\n"
        f"--- final status ---\n{final}\n\n"
        f"--- event history (last 20) ---\n{events}\n\n"
        f"--- final stringutils.py ---\n{files_txt}"
    )

    verdict = _judge(context)
    assert verdict.get("verdict") == "pass", \
        f"LLM judge flagged the run: {verdict.get('reason')}\n--- context ---\n{context[:1500]}"
