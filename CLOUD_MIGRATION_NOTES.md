# Cloud-machine runtime migration — Phase 0 findings

Date: 2026-07-08. Probe artifacts in `/tmp/gc-probe/` (probe scripts, logs,
raw captures). Worker used: `mocha-smoke` (pre-existing, pid 55074, running
in `/private/tmp/machine-smoke-test`).

## Verdict: GATE FAILED — the python cursor-sdk REQUIRES a bridge sidecar for cloud agents

Per the spec ("If the python sdk REQUIRES a bridge even for cloud agents:
STOP, do not build a workaround"), phases 1–4 were NOT implemented. The
findings below are the inputs for the human decision between the three
options the spec names (newer sdk / REST client / thin bridge).

### Evidence (verified, not inferred)

1. **Code path**: `cursor_sdk._client._default_client()` unconditionally
   launches a local `cursor-sdk-bridge` node subprocess (`Bridge.launch`,
   `_client.py:78`) unless `CURSOR_SDK_BRIDGE_URL` + `CURSOR_SDK_BRIDGE_TOKEN`
   point at an *externally managed* bridge. There is no bridge-less
   construction: `Client.__init__` raises
   `"Client requires a BridgeEndpoint or base_url/auth_token"` without one,
   and the transport speaks Connect-RPC (`sdk.v1.SdkAgentService` etc.) to
   the bridge only. Identical structure in the async client.
2. **Runtime confirmation**: during the live probe,
   `cursor_sdk._client._DEFAULT_BRIDGE` was `None` before `Agent.create`
   and `LAUNCHED pid=<n> alive=True url=http://127.0.0.1:<port>` immediately
   after — for a pure cloud/machine agent with an explicit `api_key`. All
   events streamed through that sidecar.
3. **No hosted bridge endpoint**: POSTs to
   `https://api.cursor.com|api2.cursor.sh|api2direct.cursor.sh/sdk.v1.SdkAgentService/ListAgents`
   all return 404 — you cannot point `Client.connect` at Cursor's servers.
4. **No newer sdk**: PyPI latest is `cursor-sdk 0.1.9` — exactly what the
   hermes venv has installed. The vendored bridge inside the wheel is
   `@cursor/sdk` **1.0.23** (`_vendor/bridge/manifest.json`). No upgrade
   available to change the verdict.

## What DID work (the cloud path itself is healthy end-to-end)

The probe (hermes venv python, `CURSOR_API_KEY` from env, no
`LocalAgentOptions`, no explicit `launch_bridge`) succeeded fully once it
targeted a healthy worker:

- `Agent.create(model=..., cloud=CloudAgentOptions(env=CloudEnvironment(type="machine", name="mocha-smoke"), repos=[CloudRepository(url=..., starting_ref="main")]), api_key=...)`
  → agent `bc-766f384b-…`; tool calls executed on this box (proof:
  `run_terminal_cmd` result `stdout: "mac\ngc-probe-ok\n"` — local hostname).
- Events stream via the standard `run.events()` path: `sdk_message`
  (status / assistant / thinking / tool_call), then `result` + `done`.
- `Agent.resume("bc-…", {"apiKey": ...})` works; `model` is `None`
  post-resume (existing rule holds); a follow-up `send` on the resumed
  handle completed (`probe-resume-ok`).
- `work_on_current_branch` is the exact python field name
  (`CloudAgentOptionsDict.work_on_current_branch`, wire
  `workOnCurrentBranch`) — verified in `types.py`.
- `claude-fable-5` (our `DEFAULT_MODEL`) is accepted on the sdk cloud path
  (run finished; earlier failures were worker routing, see below), even
  though it does not appear in the sdk's `list_models` output.
- Cloud `ListAgentMessages` is NOT implemented:
  `unimplemented: Cloud ListAgentMessages is not supported by @cursor/sdk yet`.

## Nuance that changes the "thin bridge" option: the bridge is stateless for cloud agents

Kill test (`/tmp/gc-probe/bridge_kill_test2.py`): SIGKILLed the sidecar's
actual node process mid-run.

- The local stream died immediately (`NetworkError: Bridge request failed:
  RemoteProtocolError: peer closed connection...`).
- The server-side run **kept executing and finished** — verified by
  resuming through a fresh bridge and asking the agent for the prior
  command's output: it returned `node-kill-survived`.
- A fresh default client + `Agent.resume` recovered the session completely.

So for cloud agents the bridge holds no run state; it is a disposable local
RPC proxy. The token-wedge / bridge-death / expensive-resume failure class
the spec targets does not apply the same way: a dead bridge costs one
stream re-attach (resume + observe), not a lost run. Caveat (not verified):
the bridge's ~1h internal auth-token expiry was root-caused for the OLD
local-agent path; I did not run a >1h cloud run to confirm whether an aged
bridge wedges cloud streams the same way. Cloud RPCs carry the explicit
`api_key` per request, and bridges can be recycled freely mid-run, which is
a design escape hatch the old architecture lacked.

## Bridge-free alternative verified live: REST v1 API

`POST https://api.cursor.com/v1/agents` with
`{"prompt": {...}, "model": {"id": ...}, "env": {"type": "machine", "name": "mocha-smoke"}, "repos": [{"url": ..., "startingRef": "main"}]}`
(Bearer CURSOR_API_KEY) → agent `bc-add3194a-…`, `env` echoed back, run
FINISHED on the local worker. Docs: cursor.com/docs/cloud-agent/api/endpoints.
Notes:

- Model catalogs differ by surface: REST v1 rejected `composer-2`
  (`invalid_model`) but accepted `claude-sonnet-4-5`; `GET /v0/models`
  returns a different list (includes `composer-2.5`,
  `claude-fable-5-thinking-high`, …). The sdk accepted both `composer-2`
  and `claude-fable-5`. Any REST client needs its own model validation.
- I did not find a REST streaming-events endpoint equivalent to
  `run.events()`; `GET /v0/agents/{id}/conversation` returned messages
  after the fact. Event-loop parity (digests/subscriptions ride streamed
  events) is the open question for the REST option — needs investigation
  before choosing it.

## Worker findings (input for the eventual phase 1 design)

1. **Second worker on the same repo+box never received assignments.** A
   fresh worker `mocha-probe-gc` (started in the same checkout
   `/private/tmp/machine-smoke-test` while `mocha-smoke` was live) came up
   ("Worker is now running", registration logged with correct
   `x-repository-url` and name), but agents targeting
   `env.name="mocha-probe-gc"` errored after a consistent ~35s with zero
   conversation messages, and the worker's verbose log showed no
   assignment ever arriving. Reproduced twice (two models — ruled the
   model out). Same agents targeting `mocha-smoke` succeeded instantly.
   I have not confirmed the mechanism (name-match failure vs. same-dir
   conflict vs. stale workerId reuse — the worker id `54b8bcc5-…` persisted
   across my restarts). Design implication: one-worker-per-checkout
   (`ensure_worker`) cannot assume a freshly spawned worker is routable;
   it needs a live routability check, and co-existing workers on one
   checkout are suspect.
2. **Failure signature of a routing hard-reject**: agent status
   CREATING→(sometimes RUNNING)→ERROR after ~35s, `RUN_LIFECYCLE_STATUS_ERROR`
   terminal event with no error detail, empty conversation. No fallback to
   cursor-hosted occurred (matches the spec's "hard reject, no fallback").
3. **CLI flags**: `agent worker start --help` does NOT list `--name`, but
   the parent `agent worker --help` does (`--name`, env
   `CURSOR_WORKER_NAME`) and `worker start --name <x>` is accepted in
   practice. `--worker-dir <path>` is repeatable; first value is the
   assignment identity, defaults to cwd. Version tested: `2026.07.01-777f564`.
4. Startup line to poll for: `Worker is now running` (then `Name:` /
   `Directory:` lines). Verbose logs show the registration payload
   including `x-repository-url` derived from the worker-dir's git remote.

## Captured event payload shapes (phase 4 fixtures)

`fixtures/machine_cloud_stream.jsonl` — 132 raw pre-parse event payloads
from the successful machine-routed probe (create + resume + 2 runs).
Shapes observed:

- `sdkMessage.type="tool_call"` with `message.tool_call` fields:
  `call_id`, `name`, `status` (`running`→`completed`), `args`, `result`.
  Tool names seen: **`run_terminal_cmd`** (args: `command`, `timeout`,
  `simpleCommands`, `parsingResult`; result: `success.stdout`,
  `interleavedOutput`, `executionTime`), **`edit_file`** (args stream
  `path` then `streamContent`; result: `success.diffString`, `linesAdded`,
  `linesRemoved`, `afterFullFileContent`), **`delete_file`** (result:
  `success.prevContent`, `fileSize`).
- `sdkMessage.type="assistant"` — many small content deltas per response.
- `sdkMessage.type="thinking"` and `type="status"`
  (`CREATING`/`RUNNING`/`FINISHED`/`ERROR`).
- Terminal: `result` (with `status`, `result` text, `durationMs`,
  `git.branches[].repoUrl`) then `done`.
- NOT observed in this probe (spec's TS smoke test saw them):
  `file_search`, `pr_management`. A future normalizer still needs fixtures
  for those; capture during a PR-flow run.
- Error-run captures (status-only streams ending in
  `RUN_LIFECYCLE_STATUS_ERROR`) preserved at
  `/tmp/gc-probe/raw_events_errored*.jsonl`.

## Options for the human (with the evidence each now has)

1. **Newer python sdk** — not available; 0.1.9 is PyPI latest.
2. **REST API client** — machine routing proven live; open question is
   event streaming parity for digests/subscriptions.
3. **Thin bridge** — the sidecar is stateless for cloud agents and
   recycles cheaply mid-run (kill test); the plugin would keep a bridge
   dependency but shed the state-loss failure modes that motivated the
   migration. The old bridge-lifecycle complexity (max-age, health probes,
   recycle-on-retry) could shrink to "restart on error, resume, re-observe".
