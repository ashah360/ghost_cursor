"""SDK runner for cursor — official ``cursor-sdk`` python package transport.

Replaces the ACP transport (``acp_runner.py``, JSON-RPC over a cursor-agent
stdio child). The ACP path omitted ``enableAgentRetries``, so long turns died
on transient http/2 CANCEL drops with no recovery. The SDK's bridge
architecture fixes that natively: runs live in a ``cursor-sdk-bridge``
sidecar with per-workspace on-disk state (not in our stream connection),
``run.observe(after_offset=...)`` re-attaches to a live run after a dropped
stream, and ``Agent.resume(agent_id)`` reattaches across process restarts.

Yielded event tuples (consumed by ``events.SdkNormalizer``):

* ``("sdk.session", {...})`` — agent established (agentId, cwd, model,
  resumed). ``resumed`` is True when a persisted agent was continued via
  ``client.agents.resume``, False for a fresh create.
* ``("sdk.message", <dict>)`` — one SDKMessage from the run stream,
  converted to a plain dict (``type`` discriminator: assistant / thinking /
  tool_call / status / usage / ...). Tool_call payloads are explicitly
  unstable upstream — consumers parse them defensively.
* ``("sdk.reattached", {"offset": ..., "attempt": n})`` — the event stream
  dropped while the run stayed alive and was transparently re-attached via
  ``run.observe(after_offset=<last offset>)``. Lifecycle/log signal only:
  the user sees nothing, no synthetic messages, no re-prompt.
* ``("sdk.model_warning", {"warning": ..., "requested": ..., "using":
  ...})`` — the requested model string was unparseable and DEFAULT_MODEL
  was substituted (see :func:`translate_model`). Yielded before any run
  events so the substitution is visible in the event log.
* ``("sdk.result", {"status": ...})`` — terminal run status as reported by
  the SDK: finished | cancelled | expired. A terminal status of "error" is
  emitted as ``sdk.error`` instead (below) so the typed detail travels
  with it.
* ``("sdk.error", {"error": ..., ...})`` — hard failure. Two shapes:
  mid-run (watchdog abort, unrecoverable stream/bridge error) carries
  ``{"error": ..., "timeout": bool}``; a run that settled with terminal
  status "error" carries ``{"error": "<TypeName>: <message>", "retryable":
  bool|None, "retry_after": str|None, "run_status": "error"}`` — the typed
  ``CursorAgentError`` fields mined defensively off the run handle
  (:func:`_terminal_error_detail`), generic text when nothing was
  recoverable. Preflight failures raise :class:`SdkRunnerError` instead so
  the tool returns a clean, actionable error.

Run watchdogs keep the ACP semantics (INACTIVITY-based, not wall-clock): a
run that keeps streaming events is alive and is never aborted for total
elapsed time. The watchdog fires only after ``inactivity_timeout_s`` seconds
of SILENCE (no stream events at all — every event resets the clock). An
in-flight tool call (``tool_call`` with status "running" seen, no terminal
update yet) also counts as activity: cursor streams nothing while a long
local command runs, so the inactivity clock is suspended until the call
finishes. A separate, optional ``max_wall_s`` hard ceiling (disabled by
default) is the safety net for true runaways; the abort error names
whichever limit fired.

Transient bridge/HTTP failures around create/resume/send are retried with
bounded backoff driven by the SDK's typed errors (``is_retryable`` +
``retry_after``).

Bridge lifecycle: ONE bridge sidecar per workspace
(``CursorClient.launch_bridge(workspace=repo)``), cached and reused across
sessions on the same repo; :func:`shutdown_bridges` closes them all on
plugin unload. Bridge state root stays at the SDK default.

Threading model: Hermes tool handlers run in ordinary worker threads, so the
blocking SDK stream loop runs in a dedicated background thread and hands
events to the calling thread through a queue; ``run_sdk`` is a plain
synchronous generator like the old ``run_acp``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from .events import unified_diff_text
from .runner import (
    DEFAULT_MODEL,
    HarnessError,
    resolve_repo,
)

logger = logging.getLogger(__name__)

# Watchdog defaults. The inactivity threshold aborts a run only after this
# much SILENCE (no stream events received); streamed progress resets the
# clock, so a long run that keeps working is never killed by it. The wall
# ceiling caps TOTAL run time as a runaway safety net — 0 disables it.
DEFAULT_INACTIVITY_TIMEOUT_S = 600.0
DEFAULT_MAX_WALL_S = 0.0

# After run.cancel(), how long the consumer waits for the worker to settle
# before abandoning it (daemon thread; the bridge owns the actual run).
CANCEL_GRACE_S = 15.0
_POLL_S = 0.2

# Bounded transparent recovery: how many consecutive stream drops we bridge
# with run.observe(after_offset=...) before declaring the run failed. Any
# successfully received event resets the counter.
MAX_STREAM_REATTACHES = 5
# Linear backoff step between re-attach attempts (attempt N sleeps N*step,
# capped). Module-level so tests can zero it.
_REATTACH_BACKOFF_S = 2.0
_REATTACH_BACKOFF_CAP_S = 10.0
# Bounded retries for transient create/resume/send failures (is_retryable).
MAX_CALL_ATTEMPTS = 3

# SDK run statuses that mean the run is over.
_TERMINAL_RUN_STATUSES = ("finished", "error", "cancelled", "expired")
# tool_call stream statuses that mean the call is no longer in flight.
_TERMINAL_TOOL_STATUSES = ("completed", "error", "failed", "cancelled")

# Cap for diff text produced by the git fallback (mirrors events.MAX_DIFF_CHARS).
_FALLBACK_DIFF_CHARS = 100_000

API_KEY_ENV = "CURSOR_API_KEY"


class SdkRunnerError(HarnessError):
    """Hard SDK failure before the run started — no run happened."""


def sdk_available() -> bool:
    """True when the cursor-sdk package is importable."""
    try:
        return importlib.util.find_spec("cursor_sdk") is not None
    except Exception:
        return False


def _default_cancel_check() -> bool:
    """Poll the Hermes per-thread interrupt flag (set by AIAgent.interrupt())."""
    try:
        from tools.interrupt import is_interrupted

        return is_interrupted()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Legacy model-string translation (ACP-era slugs → SDK model selection)
# ---------------------------------------------------------------------------
# The cursor-sdk model catalog exposes BASE ids only ("claude-fable-5");
# thinking/effort/context are per-model parameters on a ModelSelection
# (verified live via Cursor.models.list(), 2026-07-03: claude-fable-5 has
# params thinking={false,true}, context={300k,1m},
# effort={low,medium,high,xhigh,max}). Two legacy string forms still reach
# us and would be rejected by the SDK with BadRequestError:
#
# * dash suffix   — "claude-fable-5-thinking-high" (the old CLI shorthand,
#   previously our DEFAULT_MODEL and possibly in user config).
# * bracket suffix — "claude-fable-5[thinking=true,context=300k,effort=high]"
#   (ACP-era handle records; resumed sessions replay these verbatim).
#
# translate_model maps both onto a base id + params raw-dict ModelSelection
# (the SDK's documented dict convenience — keeps this module importable
# without the cursor_sdk dataclasses). Unparseable forms fall back to
# DEFAULT_MODEL with a warning event rather than failing the run.

# Legacy effort levels → catalog values ("extra-high" was a display-style
# alias for what the catalog calls "xhigh").
_EFFORT_LEVELS = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "extra-high": "xhigh",
    "xhigh": "xhigh",
    "max": "max",
}

_BRACKET_MODEL_RE = re.compile(r"^(?P<base>[^\[\]]*)\[(?P<params>[^\[\]]*)\]$")
_THINKING_SUFFIX_RE = re.compile(r"^(?P<base>.+?)-thinking(?:-(?P<level>.+))?$")

# str base id, raw-dict ModelSelection, or None (= cursor's default).
ModelValue = Any


def translate_model(model: Optional[str]) -> Tuple[Optional[ModelValue], Optional[str]]:
    """Normalize a requested model string for the cursor-sdk.

    Returns ``(sdk_model, warning)``:

    * plain base ids pass through unchanged (as strings);
    * ``<base>-thinking[-<level>]`` becomes ``{"id": base, "params":
      [thinking=true, effort=<level>]}``;
    * ``<base>[k=v,...]`` (legacy handle records) becomes ``{"id": base,
      "params": [...]}``;
    * unparseable/unknown suffix forms fall back to ``DEFAULT_MODEL`` with
      a human-readable ``warning`` (the caller emits it as a warning event).
    """
    raw = str(model or "").strip()
    if not raw:
        return None, None

    def _fallback(reason: str) -> Tuple[str, str]:
        return DEFAULT_MODEL, (
            f"requested model {raw!r} {reason} — "
            f"falling back to '{DEFAULT_MODEL}'"
        )

    if "[" in raw or "]" in raw:
        match = _BRACKET_MODEL_RE.match(raw)
        base = (match.group("base").strip() if match else "")
        if not match or not base:
            return _fallback("has an unparseable bracket suffix")
        params: List[Dict[str, str]] = []
        for part in match.group("params").split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, val = part.partition("=")
            if not sep or not key.strip() or not val.strip():
                return _fallback(f"has an unparseable bracket parameter {part!r}")
            params.append({"id": key.strip(), "value": val.strip()})
        if not params:
            return base, None
        return {"id": base, "params": params}, None

    match = _THINKING_SUFFIX_RE.match(raw)
    if match:
        level = match.group("level")
        params = [{"id": "thinking", "value": "true"}]
        if level is not None:
            effort = _EFFORT_LEVELS.get(level.strip().lower())
            if effort is None:
                return _fallback("has an unrecognized '-thinking-…' suffix")
            params.append({"id": "effort", "value": effort})
        return {"id": match.group("base"), "params": params}, None

    return raw, None


def model_id_of(model: Optional[ModelValue]) -> str:
    """The base model id of a translated model value ("" when None)."""
    if isinstance(model, dict):
        return str(model.get("id") or "")
    return str(model or "")


# ---------------------------------------------------------------------------
# Bridge lifecycle — one sidecar per workspace, reused across sessions
# ---------------------------------------------------------------------------

_bridges: Dict[str, Any] = {}
# Live python Agent handles, keyed (workspace, agent_id). Reusing the SAME
# handle for follow-up sends is the SDK's canonical multi-turn flow —
# ``Agent.resume`` is for process restarts. It also matters for stability:
# resuming an agent that is still registered on the SAME live bridge makes
# the bridge re-register the id and async-dispose the previous handle
# (bridge registry source), and that disposal path can crash the bridge
# process mid-run (known upstream issue — see the bridge's
# process-error-survivors module; observed live 2026-07-03 as the follow-up
# send's stream dying with "peer closed connection" then "connection
# refused"). Guarded by _bridges_lock.
_agents: Dict[Tuple[str, str], Any] = {}
_bridges_lock = threading.Lock()

# The bridge's documented opt-in band-aid for the disposal crash above:
# log uncaughtException/unhandledRejection and keep serving instead of
# dying. A possibly-degraded bridge beats a dead one here — runs live in
# the bridge process, and our streams re-attach transparently.
_BRIDGE_SURVIVE_ENV = "CURSOR_SDK_BRIDGE_SURVIVE_UNCAUGHT"


def _bridge_alive(client: Any) -> bool:
    """Best-effort liveness probe for a cached bridge client."""
    try:
        ping = getattr(client, "ping", None)
        if callable(ping):
            ping()
        return True
    except Exception:
        return False


def get_bridge(workspace: str) -> Any:
    """The (cached, health-checked) bridge client for ``workspace``.

    Launches ``CursorClient.launch_bridge(workspace=...)`` on first use and
    reuses the client for every later session on the same repo. A cached
    bridge that stopped answering (crashed sidecar) is closed and
    relaunched — its cached agent handles are dropped with it, so the next
    run resumes from the bridge's on-disk state instead of sending into a
    dead process forever. Tests monkeypatch this function with a fake
    client factory.
    """
    key = str(workspace)
    with _bridges_lock:
        client = _bridges.get(key)
    if client is not None:
        if _bridge_alive(client):
            return client
        logger.warning(
            "cursor-sdk bridge for %s stopped answering — relaunching", key
        )
        with _bridges_lock:
            if _bridges.get(key) is client:
                del _bridges[key]
                _drop_agents_for_workspace_locked(key)
        _close_client(client)

    # The bridge process inherits our env (cursor_sdk builds the subprocess
    # env from os.environ), so the opt-in must be set process-wide.
    os.environ.setdefault(_BRIDGE_SURVIVE_ENV, "1")
    from cursor_sdk import CursorClient

    client = CursorClient.launch_bridge(workspace=key)
    with _bridges_lock:
        # Two racing launches: keep the first, close the loser.
        existing = _bridges.get(key)
        if existing is not None:
            _close_client(client)
            return existing
        _bridges[key] = client
    return client


def _drop_agents_for_workspace_locked(workspace: str) -> None:
    """Drop cached agent handles for ``workspace``. Caller holds _bridges_lock."""
    for cache_key in [k for k in _agents if k[0] == workspace]:
        del _agents[cache_key]


def get_cached_agent(workspace: str, agent_id: str) -> Optional[Any]:
    """The live Agent handle for (workspace, agent_id), if this process has one."""
    with _bridges_lock:
        return _agents.get((str(workspace), str(agent_id)))


def cache_agent(workspace: str, agent_id: str, agent: Any) -> None:
    """Remember a live Agent handle for follow-up sends (see _agents)."""
    if not agent_id:
        return
    with _bridges_lock:
        _agents[(str(workspace), str(agent_id))] = agent


def _close_client(client: Any) -> None:
    try:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:
        logger.debug("bridge client close failed", exc_info=True)


def shutdown_bridges() -> None:
    """Close every cached bridge client (plugin unload / process exit)."""
    with _bridges_lock:
        clients = list(_bridges.values())
        _bridges.clear()
        _agents.clear()
    for client in clients:
        _close_client(client)


# ---------------------------------------------------------------------------
# Defensive conversion of SDK objects to plain dicts
# ---------------------------------------------------------------------------

def _to_plain(obj: Any, _depth: int = 0) -> Any:
    """A best-effort plain-data view of an SDK object.

    Tool_call payload schemas are explicitly unstable upstream, so nothing
    here assumes shape: dataclasses, mappings, sequences, and plain-attribute
    objects all reduce to dict/list/scalar; anything exotic degrades to
    ``str(obj)`` instead of raising.
    """
    if _depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {
                f.name: _to_plain(getattr(obj, f.name, None), _depth + 1)
                for f in dataclasses.fields(obj)
            }
        if isinstance(obj, dict):
            return {str(k): _to_plain(v, _depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_plain(v, _depth + 1) for v in obj]
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict) and attrs:
            return {
                str(k): _to_plain(v, _depth + 1)
                for k, v in attrs.items()
                if not str(k).startswith("_")
            }
    except Exception:
        pass
    return str(obj)


def _message_dict(event: Any) -> Optional[Dict[str, Any]]:
    """The SDKMessage dict carried by a RunStreamEvent, or None.

    The installed SDK puts it at ``event.sdk_message``; older/newer builds
    are probed defensively.
    """
    for attr in ("sdk_message", "message", "data"):
        candidate = getattr(event, attr, None)
        if candidate is None:
            continue
        plain = _to_plain(candidate)
        if isinstance(plain, dict) and plain.get("type"):
            return plain
    return None


def _run_status(run: Any) -> str:
    try:
        return str(getattr(run, "status", "") or "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Terminal-error detail mining (run settled with status "error")
# ---------------------------------------------------------------------------

def _error_fields(obj: Any, _depth: int = 0) -> Optional[Dict[str, Any]]:
    """The ``{"error", "retryable", "retry_after"}`` view of an error-ish
    object, or None.

    Error-ish = an Exception instance, or anything carrying a non-empty
    ``message`` string (the ``CursorAgentError`` shape). Non-error carriers
    (e.g. a RunResult) are probed one level for a nested ``error``
    attribute. Everything is getattr-guarded — the run-handle surface is
    not trusted here.
    """
    if obj is None or isinstance(obj, str) or _depth > 2:
        return None
    try:
        message = getattr(obj, "message", None)
        if isinstance(obj, BaseException) or (isinstance(message, str) and message):
            text = message if isinstance(message, str) and message else str(obj)
            retryable = getattr(obj, "is_retryable", None)
            retry_after = getattr(obj, "retry_after", None)
            return {
                "error": f"{type(obj).__name__}: {text}",
                "retryable": bool(retryable) if retryable is not None else None,
                "retry_after": str(retry_after) if retry_after else None,
            }
        nested = getattr(obj, "error", None)
    except Exception:
        return None
    if nested is not obj:
        return _error_fields(nested, _depth + 1)
    return None


def _terminal_error_detail(run: Any) -> Optional[Dict[str, Any]]:
    """Typed error detail mined off a run that settled with status "error".

    The SDK's Run/RunResult carries a ``CursorAgentError`` (message,
    is_retryable, retry_after) somewhere on the handle, but WHERE is not
    stable across builds — so this probes ``run.error``, ``run.result``,
    and a final ``run.wait()`` (which either returns the RunResult or
    raises the typed error itself), all defensively. None when nothing
    error-shaped was recoverable.
    """
    candidates: List[Any] = []
    for attr in ("error", "result"):
        try:
            candidates.append(getattr(run, attr, None))
        except Exception:
            pass
    try:
        wait = getattr(run, "wait", None)
        if callable(wait):
            candidates.append(wait())
    except Exception as exc:
        candidates.append(exc)
    for candidate in candidates:
        detail = _error_fields(candidate)
        if detail is not None:
            return detail
    return None


# ---------------------------------------------------------------------------
# Bounded retries for transient bridge/HTTP failures
# ---------------------------------------------------------------------------

def _retry_delay_s(exc: Exception, attempt: int) -> float:
    """Backoff for a retryable error: server-supplied retry_after when it
    parses as seconds, else exponential."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        try:
            return max(0.0, float(str(retry_after)))
        except (TypeError, ValueError):
            pass  # HTTP-date form — fall through to exponential
    return float(2 ** attempt)


def _call_with_retries(what: str, fn: Callable[[], Any]) -> Any:
    """Call ``fn``, transparently retrying bounded times on errors the SDK
    marks retryable (``is_retryable``). Non-retryable errors raise through."""
    for attempt in range(MAX_CALL_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:
            last = attempt == MAX_CALL_ATTEMPTS - 1
            if last or not bool(getattr(exc, "is_retryable", False)):
                raise
            delay = _retry_delay_s(exc, attempt)
            logger.warning(
                "cursor-sdk %s failed with retryable %s: %s — retrying in %.1fs",
                what, type(exc).__name__, exc, delay,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Worker (runs on a dedicated background thread)
# ---------------------------------------------------------------------------

class _SdkWorker:
    def __init__(
        self,
        task: str,
        workdir: Path,
        out_q: "queue.Queue[Tuple[str, Dict[str, Any]]]",
        cancel_requested: threading.Event,
        agent_id: Optional[str] = None,
        model: Optional[ModelValue] = None,
    ) -> None:
        self._task = task
        self._workdir = workdir
        self._out_q = out_q
        self._cancel_requested = cancel_requested
        # Prior agent to continue via Agent.resume (None = fresh create).
        self._resume_agent_id = agent_id
        # Already translated (run_sdk calls translate_model): a base-id
        # string, a raw-dict ModelSelection, or None.
        self._model = model or None
        self._model_id = model_id_of(model)

        # "cancel" | "timeout" — written by the consumer thread before it
        # sets cancel_requested (single write, read after the event fires).
        self.abort_reason: Optional[str] = None
        # Human-readable detail naming WHICH watchdog fired.
        self.abort_detail: Optional[str] = None
        # Monotonic timestamp of the last stream event received. Written by
        # the worker thread, read by the consumer's inactivity watchdog — a
        # plain float is fine, attribute writes are atomic under the GIL.
        self.last_activity_monotonic: float = time.monotonic()
        # call_ids with a tool_call seen (status "running") but no terminal
        # update yet. A pending tool call means cursor is legitimately BUSY
        # even though no events stream while it runs, so the inactivity
        # watchdog is suspended while this is non-empty. Mutated only on the
        # worker thread; the consumer only reads truthiness (GIL-safe).
        self._pending_tool_calls: set = set()

        self._run: Optional[Any] = None
        self._settled = False

    # -- plumbing ----------------------------------------------------------

    def _put(self, key: str, obj: Dict[str, Any]) -> None:
        self._out_q.put((key, obj))

    def has_pending_tool_call(self) -> bool:
        """True while any tool call has started but not yet finished."""
        return bool(self._pending_tool_calls)

    def _track_tool_call(self, message: Dict[str, Any]) -> None:
        if str(message.get("type") or "") != "tool_call":
            return
        call_id = str(message.get("call_id") or "")
        if not call_id:
            return
        if str(message.get("status") or "") in _TERMINAL_TOOL_STATUSES:
            self._pending_tool_calls.discard(call_id)
        else:
            self._pending_tool_calls.add(call_id)

    def _timeout_error(self) -> Dict[str, Any]:
        detail = self.abort_detail or "watchdog abort"
        return {"error": f"cursor run timed out: {detail}", "timeout": True}

    # -- agent establishment -------------------------------------------------

    def _establish_agent(self, client: Any) -> Tuple[Any, bool, bool]:
        """The agent for this send: cached handle, resume, or fresh create.

        Returns ``(agent, resumed, from_cache)``.

        Multi-turn priority order (each step is a live-bridge lesson):

        1. **Reuse the live Agent handle** from a previous run in this
           process (the SDK's canonical follow-up flow: just call
           ``agent.send`` again). Critically, this AVOIDS re-resuming an
           agent that is still registered on the same live bridge — that
           re-registration async-disposes the previous handle inside the
           bridge, and the local-agent disposal path can crash the bridge
           process mid-run (known upstream issue, documented in the
           bridge's process-error-survivors module; observed live
           2026-07-03: follow-up run's stream died with "peer closed
           connection", then every re-attach got "connection refused").
           Only reused when the requested model matches (or none was
           requested) — an explicit model switch goes through resume.
        2. **``client.agents.resume``** (process restart / no live handle).
           The resume MUST re-supply ``model``: a resumed handle carries no
           model ("agent.model is None on resume unless you pass model
           again" — SDK docs, verified in the bridge source), and a local
           agent whose handle has no model rejects every send with the
           non-retryable "Local SDK agents require an explicit model"
           error. That broke ALL follow-up sends before the handle cache
           existed (e2e test_followup_send_carries_context).
        3. **Fresh create** when the resume fails (expired/unknown agent)
           so the task still runs, just without prior context — the
           ``sdk.session`` event's ``resumed`` field reports what happened.
        """
        if self._resume_agent_id:
            cached = get_cached_agent(str(self._workdir), self._resume_agent_id)
            if cached is not None:
                cached_model = str(
                    getattr(getattr(cached, "model", None), "id", "") or ""
                )
                # Base-id comparison: params-only differences reuse the live
                # handle too (a resume for a param tweak is not worth the
                # bridge-disposal crash risk documented on _agents).
                if not self._model_id or self._model_id == cached_model:
                    return cached, True, True
            try:
                agent = _call_with_retries(
                    "agents.resume",
                    lambda: client.agents.resume(
                        self._resume_agent_id,
                        {"model": self._model or DEFAULT_MODEL},
                    ),
                )
                return agent, True, False
            except Exception as exc:
                logger.warning(
                    "cursor-sdk resume of agent %s failed (%s: %s) — "
                    "falling back to a fresh agent",
                    self._resume_agent_id, type(exc).__name__, exc,
                )
        agent = _call_with_retries(
            "agents.create",
            # Raw-dict options (documented SDK convenience) keep this path
            # importable without the cursor_sdk dataclasses (offline tests).
            lambda: client.agents.create(
                model=self._model or DEFAULT_MODEL,
                local={"cwd": str(self._workdir)},
            ),
        )
        return agent, False, False

    # -- cancellation --------------------------------------------------------

    def _cancel_watcher(self) -> None:
        while not self._cancel_requested.is_set():
            if self._settled:
                return
            time.sleep(_POLL_S)
        if self._settled:
            return
        run = self._run
        if run is None:
            return
        try:
            if _run_status(run) not in _TERMINAL_RUN_STATUSES:
                run.cancel()
        except Exception:
            logger.debug("run.cancel() failed", exc_info=True)

    # -- stream consumption ----------------------------------------------------

    def _consume_stream(self, run: Any) -> None:
        """Drain the run's event stream, transparently re-attaching on drops.

        A dropped stream while the run is still alive is bridged with
        ``run.observe(after_offset=<last offset>)`` — bounded attempts, the
        counter resets on any successfully received event. The user sees
        nothing; a ``sdk.reattached`` tuple flows to the JSONL log only.
        """
        stream: Iterator[Any] = iter(run.events())
        last_offset: Optional[str] = None
        reattaches = 0
        while True:
            try:
                event = next(stream)
            except StopIteration:
                return
            except Exception as exc:
                if self._cancel_requested.is_set():
                    return  # cancel racing the stream teardown — settled below
                if _run_status(run) in _TERMINAL_RUN_STATUSES:
                    return  # run is over; the stream just died reporting it
                logger.warning(
                    "cursor-sdk event stream dropped (%s: %s) — re-attaching "
                    "via observe(after_offset=%r)",
                    type(exc).__name__, exc, last_offset,
                )
                # A raised generator is spent (next() would yield a bogus
                # StopIteration), so keep retrying observe() itself here
                # until it hands back a live stream or the budget runs out.
                while True:
                    reattaches += 1
                    if reattaches > MAX_STREAM_REATTACHES:
                        raise
                    time.sleep(
                        min(_REATTACH_BACKOFF_S * reattaches, _REATTACH_BACKOFF_CAP_S)
                    )
                    if self._cancel_requested.is_set():
                        return
                    try:
                        stream = iter(run.observe(after_offset=last_offset))
                        break
                    except Exception:
                        logger.debug("observe re-attach failed", exc_info=True)
                self._put(
                    "sdk.reattached",
                    {"offset": last_offset, "attempt": reattaches},
                )
                continue

            reattaches = 0
            self.last_activity_monotonic = time.monotonic()
            offset = getattr(event, "offset", None)
            if offset is not None:
                last_offset = str(offset)
            message = _message_dict(event)
            if message is not None:
                self._track_tool_call(message)
                self._put("sdk.message", message)

    # -- main flow -----------------------------------------------------------

    def run(self) -> None:
        watcher = threading.Thread(
            target=self._cancel_watcher, name="ghost-cursor-sdk-cancel", daemon=True
        )
        try:
            try:
                client = get_bridge(str(self._workdir))
            except Exception as exc:
                self._put(
                    "sdk.fatal",
                    {
                        "error": (
                            "failed to launch the cursor-sdk bridge for "
                            f"{self._workdir} ({type(exc).__name__}: {exc})"
                        )
                    },
                )
                return

            try:
                agent, resumed, from_cache = self._establish_agent(client)
            except Exception as exc:
                self._put(
                    "sdk.fatal",
                    {
                        "error": (
                            f"cursor-sdk agent create failed "
                            f"({type(exc).__name__}: {exc}). Check "
                            f"{API_KEY_ENV} and the configured model."
                        )
                    },
                )
                return

            agent_id = str(getattr(agent, "agent_id", "") or "")
            if not from_cache:
                cache_agent(str(self._workdir), agent_id, agent)
            model_sel = getattr(agent, "model", None)
            model_id = str(
                getattr(model_sel, "id", None) or self._model_id or DEFAULT_MODEL
            )
            self._put(
                "sdk.session",
                {
                    "agentId": agent_id,
                    "cwd": str(self._workdir),
                    "model": model_id,
                    "resumed": resumed,
                },
            )

            try:
                run = _call_with_retries(
                    "agent.send", lambda: agent.send(self._task)
                )
            except Exception as exc:
                self._put(
                    "sdk.error",
                    {"error": f"cursor-sdk send failed: {type(exc).__name__}: {exc}"},
                )
                return
            self._run = run
            watcher.start()

            try:
                self._consume_stream(run)
                status = _run_status(run) or "finished"
                self._settled = True
                if self.abort_reason == "timeout":
                    self._put("sdk.error", self._timeout_error())
                elif status == "error":
                    # Terminal "error": surface the typed detail instead of
                    # a bare status so downstream renders more than
                    # "cursor run ended with status: error".
                    detail = _terminal_error_detail(run) or {}
                    self._put(
                        "sdk.error",
                        {
                            "error": detail.get("error")
                            or f"cursor run ended with status: {status}",
                            "retryable": detail.get("retryable"),
                            "retry_after": detail.get("retry_after"),
                            "run_status": status,
                        },
                    )
                else:
                    self._put("sdk.result", {"status": status})
            except Exception as exc:
                self._settled = True
                if self.abort_reason == "timeout":
                    self._put("sdk.error", self._timeout_error())
                elif self.abort_reason == "cancel":
                    self._put("sdk.result", {"status": "cancelled"})
                else:
                    self._put(
                        "sdk.error",
                        {
                            "error": (
                                "cursor-sdk stream failed mid-run: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        },
                    )
        except Exception as exc:  # belt-and-braces: never strand the consumer
            logger.exception("cursor-sdk worker crashed")
            self._put("sdk.error", {"error": f"cursor-sdk worker crashed: {exc}"})
        finally:
            self._settled = True
            self._put("__done__", {})


# ---------------------------------------------------------------------------
# Synchronous generator facade (what the orchestration consumes)
# ---------------------------------------------------------------------------

def run_sdk(
    task: str,
    repo: str,
    inactivity_timeout_s: Optional[float] = None,
    max_wall_s: Optional[float] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    agent_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Run cursor on ``task`` inside ``repo`` via the cursor-sdk, yielding events.

    Yields ``("sdk.session"|"sdk.message"|"sdk.reattached"|"sdk.result"|
    "sdk.error", obj)`` tuples (see module docstring). Polls ``cancel_check``
    (default: the Hermes per-thread interrupt flag) and the run watchdogs
    between events; every trigger fires a native ``run.cancel()``.

    Watchdog semantics (inactivity-based, NOT wall-clock — unchanged from
    the ACP transport):

    * ``inactivity_timeout_s`` — abort only after this many seconds with NO
      stream events received. Streamed activity resets the clock; a PENDING
      tool call suspends it (cursor emits nothing while a long local command
      runs, but it is busy, not hung). Default
      ``DEFAULT_INACTIVITY_TIMEOUT_S``; 0 disables.
    * ``max_wall_s`` — optional hard ceiling on TOTAL run time. Default
      ``DEFAULT_MAX_WALL_S`` (0 = disabled).

    ``agent_id`` continues a persisted cursor agent via ``Agent.resume``
    (multi-turn / across restarts). If the resume fails, the run falls back
    to a fresh agent — the ``sdk.session`` event's ``resumed`` field reports
    what actually happened.

    ``model`` accepts base ids AND the legacy ACP-era string forms
    ("<id>-thinking-<level>" dash suffixes, "<id>[k=v,...]" bracket-suffix
    handle records) — :func:`translate_model` maps them onto id + params
    before anything reaches the SDK. An unparseable form falls back to
    ``DEFAULT_MODEL`` and yields a ``("sdk.model_warning", {...})`` event
    first so the substitution is visible in the run's event log.

    Raises:
        HarnessError: empty task / bad repo (preflight).
        SdkRunnerError: cursor-sdk not importable or CURSOR_API_KEY missing —
            actionable message, no run.
    """
    if not str(task).strip():
        raise HarnessError("empty task")
    workdir = resolve_repo(repo)
    if not sdk_available():
        raise SdkRunnerError(
            "the cursor-sdk package is not installed — "
            "`pip install cursor-sdk` (requires python >= 3.10)"
        )
    if not os.environ.get(API_KEY_ENV):
        raise SdkRunnerError(
            f"{API_KEY_ENV} is not set — create an API key at "
            "https://cursor.com/dashboard (API Keys) and export it, e.g. "
            f"`export {API_KEY_ENV}=your-key`"
        )
    if cancel_check is None:
        cancel_check = _default_cancel_check

    inactivity_s = float(
        inactivity_timeout_s
        if inactivity_timeout_s is not None
        else DEFAULT_INACTIVITY_TIMEOUT_S
    )
    wall_s = float(max_wall_s if max_wall_s is not None else DEFAULT_MAX_WALL_S)

    model_value, model_warning = translate_model(model)
    if model_warning:
        logger.warning("ghost_cursor model translation: %s", model_warning)

    out_q: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
    cancel_requested = threading.Event()
    worker = _SdkWorker(
        task=str(task),
        workdir=workdir,
        out_q=out_q,
        cancel_requested=cancel_requested,
        agent_id=(str(agent_id).strip() or None) if agent_id else None,
        model=model_value,
    )
    thread = threading.Thread(
        target=worker.run, name="ghost-cursor-sdk", daemon=True
    )
    thread.start()

    if model_warning:
        yield (
            "sdk.model_warning",
            {
                "warning": model_warning,
                "requested": str(model or "").strip(),
                "using": model_id_of(model_value) or DEFAULT_MODEL,
            },
        )

    started = time.monotonic()
    fatal: Optional[str] = None
    try:
        while True:
            if not cancel_requested.is_set():
                now = time.monotonic()
                if (
                    inactivity_s > 0
                    and now - worker.last_activity_monotonic >= inactivity_s
                    # An in-flight tool call IS activity: cursor streams no
                    # events while a long local command runs, but it is
                    # busy, not hung. Only true silence — no events AND no
                    # pending tool call — times out.
                    and not worker.has_pending_tool_call()
                ):
                    worker.abort_reason = "timeout"
                    worker.abort_detail = f"no activity for {int(inactivity_s)}s"
                    cancel_requested.set()
                elif wall_s > 0 and now - started >= wall_s:
                    worker.abort_reason = "timeout"
                    worker.abort_detail = f"exceeded max wall time ({int(wall_s)}s)"
                    cancel_requested.set()
                elif cancel_check():
                    worker.abort_reason = "cancel"
                    cancel_requested.set()
            try:
                key, obj = out_q.get(timeout=_POLL_S)
            except queue.Empty:
                if not thread.is_alive() and out_q.empty():
                    break  # producer died without its sentinel — defensive
                continue
            if key == "__done__":
                break
            if key == "sdk.fatal":
                fatal = str(obj.get("error") or "cursor-sdk failure")
                continue  # drain to the sentinel, then raise below
            yield key, obj
    finally:
        # Consumer abandoned us (exception/GeneratorExit) or normal exit:
        # make sure the run is cancelled and the worker unwinds.
        cancel_requested.set()
        thread.join(timeout=CANCEL_GRACE_S)

    if fatal is not None:
        raise SdkRunnerError(fatal)


# ---------------------------------------------------------------------------
# Git fallback for files_changed (when the stream carried no diffs)
# ---------------------------------------------------------------------------

def _git(args: List[str], cwd: Path) -> str:
    """Run git, returning stdout ("" on any failure). Never raises."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
        return proc.stdout or ""
    except Exception:
        return ""


def git_status_snapshot(workdir: Path | str) -> str:
    """Raw ``git status --porcelain`` text, captured before the run."""
    return _git(["status", "--porcelain"], Path(workdir))


def git_fallback_diffs(
    workdir: Path | str, before_status: str
) -> List[Dict[str, Any]]:
    """Diff entries for files whose git status changed during the run.

    Used when cursor made edits through paths that emitted no parseable
    diff content in the stream (tool_call payloads are unstable upstream).
    Compares ``git status --porcelain`` against the pre-run snapshot so
    pre-existing dirty files aren't misattributed; a pre-existing-dirty file
    that cursor edits *further* is the known blind spot of this fallback.

    Returns:
        Dicts with the :func:`events.file_diff` keyword shape:
        ``{path, before, after, diff, added, removed, status}``.
    """
    workdir = Path(workdir)
    after_status = _git(["status", "--porcelain"], workdir)
    if not after_status:
        return []
    before_lines = set((before_status or "").splitlines())
    entries: List[Dict[str, Any]] = []
    for line in after_status.splitlines():
        if not line or line in before_lines:
            continue
        xy, rel = line[:2], line[3:].strip()
        if "->" in rel:  # rename: take the new side
            rel = rel.split("->", 1)[1].strip()
        if rel.startswith('"') and rel.endswith('"'):
            try:
                rel = json.loads(rel)  # git quotes exotic paths C-style
            except ValueError:
                rel = rel.strip('"')
        abs_path = workdir / rel
        untracked = xy == "??"
        deleted = "D" in xy

        before_text = "" if untracked else _git(["show", f"HEAD:{rel}"], workdir)
        after_text = ""
        if not deleted:
            try:
                after_text = abs_path.read_text("utf-8")
            except Exception:
                continue  # binary/unreadable — skip rather than mislead

        diff_text, added, removed = unified_diff_text(before_text, after_text, str(abs_path))
        if not diff_text:
            continue
        entries.append(
            {
                "path": str(abs_path),
                "before": before_text,
                "after": after_text,
                "diff": diff_text[:_FALLBACK_DIFF_CHARS],
                "added": added,
                "removed": removed,
                "status": "A" if untracked or "A" in xy else ("D" if deleted else "M"),
            }
        )
    return entries
