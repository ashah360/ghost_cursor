# ghost_cursor

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that lets your agent **delegate coding tasks to the [Cursor](https://cursor.com) agent** — and watch it work in real time.

Registers a single tool, **`cursor_edit`**, that runs `cursor-agent` inside a target repo over **ACP** (Agent Client Protocol — JSON-RPC over stdio), streams per-edit progress (reasoning + full file diffs) back through the calling agent's progress callback, and returns a structured summary of everything that changed.

Because it's an ordinary Hermes tool call inside a real session, the result **persists in the transcript and reloads for free** — and interrupts map to a native `session/cancel`.

## Why ACP instead of scraping stdout

The obvious way to drive `cursor-agent` from another program is `cursor-agent -p "<task>" --output-format stream-json` and parse its stdout. That works, but it's brittle: no structured cancellation, synthesized tool IDs, and it breaks the moment the output format shifts.

`ghost_cursor` speaks **ACP** (`cursor-agent acp`) instead — a real JSON-RPC protocol:

| | stdout scraping | ACP (this plugin) |
|---|---|---|
| Event format | freeform JSON text, inferred | typed `session/update` notifications |
| Tool IDs | synthesized | real `toolCallId` from Cursor |
| Cancellation | `kill -9` the process | native `session/cancel` |
| Robustness | breaks on format change | versioned protocol contract |

The legacy `--print` runner is kept in `runner.py` as a reference/fallback.

## What you get (v0.3 — session-handle interface)

Four explicit tools mirroring Hermes's `terminal`/`process` split. The single handle is the cursor **`session_id`**, returned by `cursor_start` and passed to the rest.

- **`cursor_start(task, repo, model?, session_id?)`** — dispatch a coding task; returns a `session_id` **immediately** and runs in the **background** (the conversation stays free). Pass a prior `session_id` to continue that Cursor session with full context (ACP `session/load`); expired → graceful fresh start (`resumed: false`). Optional `model` overrides the cursor-agent model (config fallback: `plugins.ghost_cursor.model`).
- **`cursor_send(session_id, message)`** — steer / follow up. Honest semantics: Cursor's ACP has **no true queue** — this interrupts the current prompt (`session/cancel`) and re-prompts the same session with `message` + full context. It's "interrupt + re-prompt with context", not "append to a running turn". Works mid-run or after a run settled.
- **`cursor_status(session_id)`** — **strictly read-only** progress view: status, files changed with diffs so far, latest reasoning, session_id, elapsed. Polling **never cancels** the run (tested property — it was the footgun that killed foreground runs).
- **`cursor_stop(session_id)`** — graceful `session/cancel`, SIGKILL only on hang. Returns final status + partial `files_changed`.

Cross-cutting: **live streaming** (reasoning + per-edit `file_diff`s via the agent's `tool_progress_callback`), **completion delivery** on every terminal state (success/fail/error/timeout/cancel), **same-repo concurrency guard** (a second `cursor_start` on a repo with an active handle is rejected — two agents on one tree = corruption; different repos run in parallel), **handle persistence** across turns (a JSON table under `<HERMES_HOME>/state/`), **git-diff fallback** for shell-driven edits, and a **`check_fn`** so the tools only appear when `cursor-agent` is installed.

> Migrating from v0.2? The single blocking `cursor_edit` + hidden auto-resume registry are **gone** (breaking). Start work with `cursor_start`, resume by passing the handle back — no repo+timestamp heuristic guessing which session to continue.

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
- [`cursor-agent`](https://cursor.com/cli) on `PATH`, logged in (`cursor-agent login`)
- The target repo should be a git repo (enables the diff fallback)

## Install

Drop the plugin into your Hermes plugins directory and enable it:

```bash
# 1. copy the plugin
mkdir -p ~/.hermes/plugins/ghost_cursor
cp __init__.py acp_runner.py events.py runner.py jobs.py handles.py plugin.yaml ~/.hermes/plugins/ghost_cursor/

# 2. enable it in ~/.hermes/config.yaml
#    plugins:
#      enabled:
#        - ghost_cursor

# 3. restart the gateway so the tools load
hermes gateway restart
```

Verify it registered:

```bash
# cursor_start / cursor_send / cursor_status / cursor_stop should show up
# as tools once cursor-agent is on PATH
```

## Usage

Once loaded, the agent gains a `cursor_edit` tool. In practice you just talk to your agent normally — when a task is coding work, it reaches for `cursor_edit`:

> "Add a `subtract(a, b)` function to `calc.py`"

The tool spawns `cursor-agent` in the repo, streams the edit live, and returns the diff.

Programmatic shape of the result:

```json
{
  "success": true,
  "status": "completed",
  "repo": "/path/to/repo",
  "summary": "Added subtract(a, b) to calc.py.",
  "files_changed": [
    { "path": "calc.py", "added": 4, "removed": 0, "status": "M", "diff": "--- a/calc.py\n+++ b/calc.py\n@@ ..." }
  ],
  "files_changed_count": 1,
  "live_progress": true
}
```

## Iterative / multi-turn

Reuse the `session_id` from a result to continue that Cursor session — it keeps full prior context, so follow-ups build on earlier work (matching style, remembering decisions) instead of re-deriving from scratch:

```
call 1:  cursor_edit(task="Create calc.py with add(a, b).")
         → { ..., "session_id": "b5b4dbe1-…", "resumed": false }

call 2:  cursor_edit(task="Now add subtract in the same style.",
                     session_id="b5b4dbe1-…")
         → { ..., "session_id": "b5b4dbe1-…", "resumed": true }
```

If the prior session is gone (Cursor restarted, id expired), call 2 transparently starts fresh and reports `resumed: false` — the task still runs, just without the earlier context.

> Note: this is cross-turn *resume* (continue between calls), not mid-flight steering — you can't inject a nudge into a prompt that's currently running; cancel and re-prompt (with the same `session_id`) for that.

## Interject — steer a running task mid-flight

Cursor's ACP has no true mid-prompt queue (a second prompt cancels and replaces the first), so "interject" is built as **stop + auto-resume**: when a `cursor_edit` run is interrupted, its cursor `session_id` is eagerly persisted to a small registry (keyed by the calling session + repo). The **next** `cursor_edit` in the same session/repo — with **no** explicit `session_id` — automatically continues that interrupted cursor session, folding your new instruction in with full prior context.

So the flow is: run a task → interrupt it → send a nudge → it picks up the same cursor session and keeps going. No id-threading required. Guards: auto-resume only fires for a recently interrupted run (≤10 min, cancelled/running) — a cleanly *completed* run is never auto-resumed, so an unrelated next task starts fresh. Passing `session_id` explicitly always overrides. The result reports `auto_resumed: true` when this kicked in.

Honest label: this is *interject/steer*, not seamless queuing — there's a cancel boundary, so work in flight at the moment of interruption is discarded, then continued from the nudge with context intact.

## Background mode — don't block the conversation

By default `cursor_edit` runs synchronously (best for quick edits — you see the diff in the same turn). For longer work, pass **`background: true`**: the tool dispatches a tracked job and returns immediately with a `job_id`, so **the conversation stays free** — you can keep talking to the agent without interrupting (or killing) the running cursor job.

- **`cursor_status(job_id?)`** — a **strictly read-only** progress view: current status, files touched so far with per-edit diffs, latest reasoning, `session_id`, elapsed. Polling it **never cancels** the job (that property is tested, not assumed — it was the exact footgun that killed foreground runs). Omit `job_id` for the most recent job in this session+repo.
- **Completion delivery** — when the job ends it delivers a message into the session for **every terminal state** (success, failure, cursor error, timeout, cancelled) — never a silent death. The payload carries the full result (`files_changed`, `session_id`, …) so resume/interject still work across the async boundary.
- **Auto-promote-on-overrun** — a synchronous run that exceeds a soft threshold (default 90s, `plugins.ghost_cursor.promote_after_seconds` in config.yaml, 0 disables) is detached to a background job instead of blocking — belt-and-suspenders for a misjudged sync run.
- **Same-repo concurrency guard** — a second background run against a repo that already has an active job is rejected (two agents on one working tree = corruption).

Why this matters: a synchronous tool holds the conversation turn open for the whole run, so messaging the agent mid-run triggers an interrupt that cancels the turn — and the cursor work with it. Background mode decouples the run from the turn, so "how's it going?" becomes a safe read instead of a kill.

## How live progress works (no core patch)

A registry-dispatched tool handler isn't handed the calling `AIAgent`, but Hermes installs the agent's `_touch_activity` as a thread-local activity callback right before each tool dispatch. `_resolve_progress_callback()` reads that thread-local, walks `__self__` back to the live agent, and uses its `tool_progress_callback`. Each emission is:

```
tool_progress_callback("reasoning.available", "cursor_edit", <json-envelope>, None)
```

which the api_server session-chat-stream forwards mid-turn as `event: tool.progress`. The JSON `delta` is a canonical envelope — `content` / `tool_use` / `tool_result` / `lifecycle` / `file_diff` — that a UI keys on (`tool_name == "cursor_edit"`, `source: "ghost"`) to render live tool cards and diffs.

## Files

| File | Role |
|---|---|
| `__init__.py` | Plugin entry — registers `cursor_edit`, resolves the progress callback, builds the result |
| `acp_runner.py` | ACP client — spawns `cursor-agent acp`, JSON-RPC over stdio, maps `session/update` → envelopes, native cancel |
| `events.py` | Canonical envelope builders + `session/update` → envelope mapping |
| `runner.py` | Legacy `--print` stdout runner (reference/fallback) + shared helpers |
| `plugin.yaml` | Plugin manifest |
| `test_ghost_cursor_plugin.py` | Tests |

## License

MIT — see [LICENSE](LICENSE).
