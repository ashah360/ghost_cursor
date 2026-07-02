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

## What you get

- **`cursor_edit(task, repo?, session_id?)`** — delegate a coding task; Cursor edits real files in `repo`.
- **Multi-turn resume** — every result returns a `session_id`; pass it back on the next call to **continue that Cursor session with full prior context** (refine, fix, iterate). Under the hood it uses ACP `session/load`; if the session expired it falls back to a fresh one (`resumed: false`) rather than erroring. Omit `session_id` for a one-shot.
- **Live streaming** — reasoning fragments + per-edit `file_diff`s (path / before / after / unified diff / +added / −removed) emitted as they happen, via the calling agent's `tool_progress_callback`.
- **Structured result** — `{success, status, repo, summary, files_changed:[{path, added, removed, status, diff}], files_changed_count, live_progress, session_id, resumed, ...}`.
- **Native cancel** — an interrupt sends ACP `session/cancel`, waits briefly, then hard-terminates.
- **Git-diff fallback** — for shell-driven edits the ACP stream didn't carry a diff for, diffs are recovered from `git`.
- **`check_fn`** — the tool only appears when the `cursor-agent` binary is installed.

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent)
- [`cursor-agent`](https://cursor.com/cli) on `PATH`, logged in (`cursor-agent login`)
- The target repo should be a git repo (enables the diff fallback)

## Install

Drop the plugin into your Hermes plugins directory and enable it:

```bash
# 1. copy the plugin
mkdir -p ~/.hermes/plugins/ghost_cursor
cp __init__.py acp_runner.py events.py runner.py plugin.yaml ~/.hermes/plugins/ghost_cursor/

# 2. enable it in ~/.hermes/config.yaml
#    plugins:
#      enabled:
#        - ghost_cursor

# 3. restart the gateway so the tool loads
hermes gateway restart
```

Verify it registered:

```bash
# cursor_edit should show up as a tool once cursor-agent is on PATH
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
