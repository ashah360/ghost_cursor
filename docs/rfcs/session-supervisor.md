# RFC: task-backed session supervisor (cloud runtime)

status: implemented (steps 1–2; see CLOUD_MIGRATION_NOTES.md §session
supervisor for the section-by-section mapping and remaining step-3
work) · branch: feat/session-supervisor · 2026-07-08

pairs with the shared session-event schema agreed with the twin team
(schematic T-055; twin's counterpart RFC: slashfi/slash#17194
`packages/atlas/atlas-docs/rfcs/interrupt-send-session-runtime.md`).

## problem

the plugin's session lifecycle is owned by in-process state: asyncio tickers
for digests, a jobs registry for completion delivery, lazy orphan
reconciliation at read time. a gateway restart kills every live run's
supervision — runs get orphaned (`failed (orphaned: plugin process restarted
mid-run)`), digests stop, completions are lost. the REST/SSE pivot removes the
local bridge, which removes the last excuse: the run itself now lives entirely
server-side (cursor cloud agent), so supervision dying with our process is a
self-inflicted wound.

twin's PR 17133 solved the same problem with a `subagent.session` task kind in
their Atlas task runtime: a reducer-owned lifecycle that survives daemon death,
with push events + a poll watchdog fallback. we adopt the shape, scaled to a
single-box plugin.

## design

### 1. supervision state is durable, owned by a supervisor loop, not the tool call

- `handles.json` grows a `supervision` record per session:
  `{ phase, current_attempt_id, attempt_n, last_seq_delivered: {subscriber: seq}, watchdog: {last_poll_ts, last_remote_status} }`
  (`current_attempt_id` is the stable id stamped on every event as `attemptId`;
  `attempt_n` is display/debug only)
- one supervisor task per live session, spawned by a **reconciler** that runs
  at plugin init and every 60s: for every handle in a non-terminal phase with
  no live supervisor task, spawn one. this is the k8s controller pattern twin
  uses — the reconciler makes restarts a non-event: gateway comes back,
  reconciler sees live handles, re-attaches supervisors, digests resume.
- supervisor = single loop per session that owns: SSE stream consumption,
  seq assignment, jsonl append, digest ticks, completion delivery, watchdog.
  everything that today lives in three modules with implicit coupling.

### 2. push with poll fallback

- primary: SSE `GET /v1/agents/{id}/runs/{runId}/stream` with `Last-Event-ID`
  resume. stream drop → reconnect with backoff from last event id.
  `error {code: stream_unavailable}` + `done` = reconnect signal, NOT run
  failure (verified live).
- fallback watchdog: if the stream is silent AND unreconnectable for
  `watchdog_interval` (default 60s), poll `GET /v1/agents/{id}/runs` for
  remote status. remote terminal + local non-terminal → settle from the GET.
  **INVARIANT (terminal precedence): the remote GET's terminal status always
  wins over a replayed stream's terminal status** — a cancelled run's replay
  emits `status: FINISHED` while the GET says CANCELLED (verified live).
  settlement never reads terminal state from replay.
- stream retention is 4 days server-side (`410 stream_expired`) — a supervisor
  re-attaching after any realistic gateway downtime always has full replay.

### 3. settlement is single-writer

only the supervisor settles a session (terminal phase + completion fan-out to
all subscribers). tool calls (send/stop/status) request transitions; the
supervisor applies them. this kills the historical race class: stop acking the
wrong run, interrupt stacking ticker loops, double completions.

### 4. ingest boundary (shared-schema rules)

- dedupe `interaction_update` twins by provider event id BEFORE seq assignment
- assign monotonic per-session `seq` post-dedupe, append to jsonl
- stamp `attemptId` on every event (mandatory — retry forensics)
- derive `lifecycle.durable_progress` supervisor-side (controller-derived from
  observed `file_diff` / irreversible completed `tool_use` events). never trust
  agent self-report. same rule as twin's server-derived variant — the deriver
  is whoever owns the event log, not the agent.

### 5. retry policy (mechanical, from derived events)

- attempt has zero `durable_progress` events → auto-retry allowed, cap 3,
  emit `lifecycle.retry_started` with the new attemptId (legacy `sdk.autoretry`
  kept as an alias during migration for existing log tooling)
- any `durable_progress` → no auto-reprompt (double-apply risk); emit
  `lifecycle.retry_suppressed`, surface failure, require explicit resume
- death-shape tagging on the terminal event: `fast_fail` (<2min,
  lifecycle-only events) vs `mid_flight` (had real events) — the recovery
  playbook differs and the caller shouldn't have to re-derive it from the log

### 6. orphans become impossible, reconciliation stays as backstop

with the reconciler, "orphaned" only means "handle whose remote agent id no
longer resolves". read-time reconciliation stays (list calls never lie) but
should fire ~never.

## portability note

implementations may store the registry differently (twin: server table;
ghost_cursor: handles.json + jsonl); the portable contract is the append-only
event envelope and lifecycle semantics (schematic T-055).

## non-goals

- no server-side registry service: handles.json + jsonl per session IS the
  registry at single-box scale. the schema is what's shared with twin, not the
  storage.
- no digest storage: digests remain derived views (status header + files
  rollup + events since subscriber's last_seq). optional
  `lifecycle.digest_sent` audit event only. `last_seq_delivered` advances
  only after successful delivery onto the completion queue; duplicate digest
  delivery is acceptable (consumers dedupe on delegation id), completion
  delivery is exactly-once per subscriber.
- interrupt-send semantics unchanged at the tool surface; internally it
  becomes a supervisor-mediated transition (cancel in-flight turn → emit
  `lifecycle.interrupt_requested`/`interrupted` → re-prompt), aligned with
  twin's RFC so the event trace reads identically on both systems.

## migration

1. land the REST/SSE client + worker manager (already on this branch)
2. supervisor module + reconciler, tickers/jobs delegate to it
3. delete the legacy per-job ticker chain + read-time-only orphan logic
4. e2e: kill -9 the plugin process mid-run, restart, assert digests resume
   and completion delivers exactly once (the demo failure from today becomes
   the regression test)
