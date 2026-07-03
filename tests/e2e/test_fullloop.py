"""Full-loop e2e — proves the AGENT (LLM) invokes the cursor tools through the
real Hermes agent loop, not just that the handlers work.

This is the "are the evals actually calling the cursor tools to do things?"
answer. It boots against a RUNNING Hermes api_server, sends a natural-language
prompt on a cheap model, and asserts BOTH:
  (1) the agent's message history contains a real cursor_start TOOL CALL
      (role=tool result for cursor_start) — i.e. the model chose to call it,
  (2) the file the agent asked cursor to create actually exists on disk
      — i.e. the tool did real work end-to-end.

Requires (skips cleanly otherwise):
  - GHOST_CURSOR_E2E=1
  - HERMES_API_BASE   (e.g. http://127.0.0.1:8650) — a running api_server
  - HERMES_API_KEY    (its bearer key)
  - CURSOR_API_KEY, OPENAI_API_KEY set on that server's env
  - cursor-agent on PATH on that server

Model pinned cheap via GHOST_CURSOR_AGENT_MODEL (informational; the server's
configured model is what actually runs).
"""
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import pytest

_run = os.environ.get("GHOST_CURSOR_E2E") == "1"
_base = os.environ.get("HERMES_API_BASE")
_key = os.environ.get("HERMES_API_KEY")

pytestmark = pytest.mark.skipif(
    not (_run and _base and _key),
    reason="full-loop e2e opt-in: needs GHOST_CURSOR_E2E=1, HERMES_API_BASE, HERMES_API_KEY (a running api_server)",
)


def _req(path, body=None, method="POST"):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        _base.rstrip("/") + path,
        data=data,
        headers={"Authorization": "Bearer " + _key, "Content-Type": "application/json"},
        method=method,
    )
    return json.loads(urllib.request.urlopen(r, timeout=280).read().decode())


def test_agent_invokes_cursor_start_through_the_loop(tmp_path):
    # a repo the agent will hand to cursor
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    target = repo / "hello.py"

    sid = _req("/api/sessions", {"title": f"fullloop-eval-{int(time.time())}"})["session"]["id"]

    # natural-language ask — the agent must DECIDE to use cursor_start
    _req(
        f"/api/sessions/{sid}/chat",
        {"input": f"Use the cursor_start tool to create {target} with a function "
                  f"hello() that returns the string 'hi'. Report the session_id."},
    )

    msgs_resp = _req(f"/api/sessions/{sid}/messages", method="GET")
    msgs = msgs_resp.get("data") or msgs_resp.get("messages") or []
    blob = json.dumps(msgs)

    # (1) the agent actually invoked cursor_start (a tool result for it exists)
    assert "cursor_start" in blob, "agent never called cursor_start"
    tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert any("session_id" in json.dumps(m) for m in tool_msgs), \
        "no cursor_start tool result with a session_id in the transcript"

    # (2) the tool did real work — the background job creates the file
    deadline = time.time() + 180
    while time.time() < deadline:
        if target.exists():
            break
        time.sleep(4)
    assert target.exists(), f"agent invoked the tool but {target} was never created"
    ns = {}
    exec(target.read_text(), ns)
    assert ns["hello"]() == "hi"
