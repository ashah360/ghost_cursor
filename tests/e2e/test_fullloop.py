"""Full-loop e2e — proves the AGENT (LLM) invokes the cursor tools through the
real Hermes agent loop, not just that the handlers work.

This is the "are the evals actually calling the cursor tools to do things?"
answer. It boots against a RUNNING Hermes api_server, sends a natural-language
prompt on a cheap model, and asserts BOTH:
  (1) the agent's message history contains real cursor tool calls — a
      cursor_create_session result (the 'session: <name>' ack) and a
      cursor_send_message result (the 'sent to <name>' ack) — i.e. the model
      chose to call the v0.4 surface,
  (2) the file the agent asked cursor to create actually exists on disk
      — i.e. the tools did real work end-to-end.

Requires (skips cleanly otherwise):
  - GHOST_CURSOR_E2E=1
  - HERMES_API_BASE   (e.g. http://127.0.0.1:8650) — a running api_server
  - HERMES_API_KEY    (its bearer key)
  - CURSOR_API_KEY, OPENAI_API_KEY set on that server's env
  - cursor-sdk installed on that server

Model pinned cheap via GHOST_CURSOR_AGENT_MODEL (informational; the server's
configured model is what actually runs).
"""
import json
import os
import time
import urllib.request

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


def test_agent_invokes_cursor_session_tools_through_the_loop(tmp_path):
    # a repo the agent will hand to cursor
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    target = repo / "hello.py"

    sid = _req("/api/sessions", {"title": f"fullloop-eval-{int(time.time())}"})["session"]["id"]

    # natural-language ask — the agent must DECIDE to use the session tools
    _req(
        f"/api/sessions/{sid}/chat",
        {"input": f"Use the cursor_create_session and cursor_send_message tools "
                  f"to create {target} with a function hello() that returns the "
                  f"string 'hi'. Report the cursor session name you used."},
    )

    msgs_resp = _req(f"/api/sessions/{sid}/messages", method="GET")
    msgs = msgs_resp.get("data") or msgs_resp.get("messages") or []
    blob = json.dumps(msgs)

    # (1) the agent actually invoked the v0.4 tools (their acks are in the
    # transcript: 'session: <name>' from create, 'sent to <name>' from send)
    assert "cursor_send_message" in blob, "agent never called cursor_send_message"
    tool_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "tool"]
    assert any("session: " in json.dumps(m) for m in tool_msgs), \
        "no cursor_create_session ack (session: <name>) in the transcript"
    assert any("sent to " in json.dumps(m) for m in tool_msgs), \
        "no cursor_send_message ack (sent to <name>) in the transcript"

    # (2) the tools did real work — the background job creates the file
    deadline = time.time() + 180
    while time.time() < deadline:
        if target.exists():
            break
        time.sleep(4)
    assert target.exists(), f"agent invoked the tools but {target} was never created"
    ns = {}
    exec(target.read_text(), ns)
    assert ns["hello"]() == "hi"
