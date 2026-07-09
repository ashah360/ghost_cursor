"""Subscribable progress digests for running cursor sessions.

While a run is active, a timer per (cursor session, subscribing Hermes
session) periodically builds a compact progress digest (status header +
the events since the previous tick) and pushes it to the hermes agent
loop over the SAME channel run completions use:
``tools.process_registry.process_registry.completion_queue``.

Multi-subscriber model
----------------------
Subscriptions are hermes_session ← cursor_session, persisted on the
session's handle entry as ``subscribers: {hermes_session_key:
interval_s}`` (``handles.py``) so they survive gateway restarts.
``cursor_send_message`` AUTO-SUBSCRIBES the dispatching Hermes session at
the effective ``update_interval_s`` (explicit param, else that session's
persisted subscriber interval, else the 180s default) — whoever prompts
is always watching. ``cursor_subscribe`` subscribes/retunes the CALLING
Hermes session only (one subscription per hermes session per cursor
session — resubscribing retunes, never duplicates; ``interval_s=0``
removes only the caller's). Each subscriber runs its OWN ticker at its
own interval with its own last_seq / digest counter / emit-floor state,
and each receives its own copy of every event — duplicate events ACROSS
Hermes sessions are by design. The event's ``session_key`` field routes
each copy to its subscriber's session.

Completion-queue consumer findings (hermes core, read 2026-07-03)
-----------------------------------------------------------------
How the hermes side consumes ``completion_queue`` events, and why the
digest event is shaped the way it is:

* Every consumer differentiates event kinds by the ``"type"`` field ONLY:
  ``process_registry.drain_notifications`` / ``format_process_notification``
  (CLI process_loop), ``gateway/run.py``'s ``_drain_gateway_watch_events``
  (owns ``watch_match``/``watch_disabled``) + ``_async_delegation_watcher``
  (owns ``type=="async_delegation"``, requeues everything else), and
  ``tui_gateway/server.py``'s ``_notification_poller_loop``.
* ``type="async_delegation"`` is the ONLY kind that is (a) delivered
  promptly in every environment while the agent is idle AND (b) rendered
  as a generic self-contained notification. An invented type would be
  requeued forever by the gateway idle watcher and mis-rendered by
  ``format_process_notification``'s fallthrough (unknown type = process
  completion → bogus "Background process unknown exited" text). So the
  digest rides ``type="async_delegation"``, same as the terminal
  completion event in ``jobs._push_completion_event``.
* NO consumer settles, deregisters, or finishes anything on receipt — the
  queue is a pure notification rail; job lifecycle lives entirely on the
  producer side (this plugin). A digest therefore cannot be "mistaken for
  a terminal completion" by the core: the properties we must guarantee are
  our own — the digest must never mark the job finished, never suppress
  jobs.py's completion enqueue, and never land on the queue AFTER the
  run's completion event (guarded by the job lock, see ``_Ticker._tick``).
* The TUI poller dedups async-delegation events by ``(delegation_id,
  type)`` (``_notification_event_dedup_key``). Each digest therefore
  carries a UNIQUE ``delegation_id``:
  ``{session}#progress-{n}@{subscriber_suffix(session_key)}`` — the
  counter keeps successive digests apart within one subscription, and
  the short subscriber-key hash keeps per-subscriber copies apart from
  each other (counters are per subscriber, so two subscribers both emit
  an n=3) and from the real completion (whose delegation_id is the
  plain session name for the dispatcher, suffixed for other
  subscribers — see ``jobs._push_completion_event``).

Interval enforcement + single-chain guarantee (issue #10)
---------------------------------------------------------
``interval_s`` is a HARD FLOOR between digest deliveries, enforced at
emit time and not merely by timer scheduling: ``_deliver`` drops any
digest attempted less than the current interval after the previously
enqueued one (per (session, subscriber), ``_last_emit``, monotonic
clock). The floor outlives individual tickers — interrupt-and-reprompt
keeps it, exactly like the digest numbering. Independently, only ONE
timer chain per (session, subscriber) can stay alive: registration in
``_tickers`` is swapped atomically (``start_for_job``), a tick fired by
a ticker that is no longer the registered one cancels itself instead of
delivering or re-arming (``_is_registered``), and arming always
supersedes a pending timer so a chain cannot fork. Re-prompting
therefore cannot stack digest loops, and even a stale chain that fires
once before noticing is muted by the floor.

Delivery-order guarantee
------------------------
``jobs.CursorJobRegistry._finalize`` flips the job to a terminal status
under ``job._lock`` and cancels this module's ticker; the tick enqueues
its digest under the SAME lock only after re-checking ``status ==
"running"``. Either the tick wins the lock while the run is still live
(digest enqueued strictly before the completion event, which _finalize
enqueues after releasing the lock) or finalize wins (the tick sees a
terminal status and drops the digest). A digest can never arrive after
the completion event.
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import eventlog as _eventlog
from . import handles as _handles
from . import render as _render

logger = logging.getLogger(__name__)

# Default digest cadence when cursor_send_message doesn't say otherwise
# (spec: 180s; 0 disables).
DEFAULT_UPDATE_INTERVAL_S = 180.0

# Validation bounds for intervals supplied at the tool boundary (issue
# #14): positive requests below the minimum clamp UP (a sub-15s cadence
# is digest spam — the emit-time floor above already drops deliveries
# inside the interval, this keeps the CONTRACT honest too), and requests
# above the maximum clamp DOWN (a >24h "subscription" is a typo, not a
# cadence). 0 stays the documented unsubscribe value and negatives are
# rejected outright — see validate_interval.
MIN_UPDATE_INTERVAL_S = 15.0
MAX_UPDATE_INTERVAL_S = 24 * 3600.0

_RUNNING = "running"

_lock = threading.Lock()
# (session name, subscriber hermes session_key) -> live ticker. One active
# run per session, one ticker per subscriber of that run.
_tickers: Dict[Tuple[str, str], "_Ticker"] = {}
# (session name, subscriber key) -> digests delivered so far. Deliberately
# OUTLIVES any one ticker: interrupt-and-reprompt carries the subscription
# to the new run and the numbering continues.
_counters: Dict[Tuple[str, str], int] = {}
# (session name, subscriber key) -> time.monotonic() of the last digest
# actually enqueued. The hard interval floor (issue #10): delivery is
# suppressed while less than the current interval has elapsed since the
# subscriber's previous digest, no matter which timer chain fired the
# tick. Like _counters it outlives any one ticker, so
# interrupt-and-reprompt cannot reset the floor either.
_last_emit: Dict[Tuple[str, str], float] = {}


def subscriber_suffix(session_key: str) -> str:
    """A short stable hash of a subscriber's hermes session_key.

    Suffixed onto delegation_ids so per-subscriber copies of the same
    digest/completion stay distinct under the TUI's (delegation_id, type)
    dedup while the ``{session}#progress-{n}`` scheme stays readable.
    """
    return hashlib.sha1(str(session_key or "").encode("utf-8")).hexdigest()[:8]


def _floor_slack(interval_s: float) -> float:
    """Slack subtracted from the floor comparison so scheduler jitter on
    an exactly-on-time tick never drops a legitimate digest. Small in
    absolute terms and capped relative to the interval so short (test)
    intervals keep a meaningful floor."""
    return min(0.02, interval_s * 0.1)


def validate_interval(value: Any, param: str = "interval_s") -> Tuple[float, Optional[str]]:
    """Validate/clamp a subscription interval supplied at a tool boundary.

    The shared contract for ``cursor_subscribe.interval_s`` and
    ``cursor_send_message.update_interval_s`` (issue #14):

    * non-numeric / NaN → ``ValueError`` with a user-facing message
    * negative → ``ValueError`` (NOT silently treated as unsubscribe)
    * 0 → unsubscribe, accepted as-is
    * 0 < value < :data:`MIN_UPDATE_INTERVAL_S` → clamped UP
    * value > :data:`MAX_UPDATE_INTERVAL_S` → clamped DOWN

    Returns ``(effective_interval, note)`` where ``note`` is the clamp
    sentence for the tool ack, or ``None`` when the value was accepted
    unchanged. ``param`` names the offending parameter in messages.
    """
    try:
        interval = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{param} must be a number — seconds between digests "
            "(0 unsubscribes)."
        ) from None
    if math.isnan(interval):
        raise ValueError(
            f"{param} must be a number — seconds between digests "
            "(0 unsubscribes)."
        )
    if interval < 0:
        raise ValueError(f"{param} must be >= 0 (0 unsubscribes).")
    if interval == 0:
        return 0.0, None
    if interval < MIN_UPDATE_INTERVAL_S:
        return MIN_UPDATE_INTERVAL_S, (
            f"{param} clamped to the "
            f"{_render.dur_compact(MIN_UPDATE_INTERVAL_S)} minimum."
        )
    if interval > MAX_UPDATE_INTERVAL_S:
        return MAX_UPDATE_INTERVAL_S, (
            f"{param} clamped to the "
            f"{_render.dur_compact(MAX_UPDATE_INTERVAL_S)} maximum."
        )
    return interval, None


def resolve_interval(
    entry: Optional[Dict[str, Any]],
    explicit: Optional[float],
    session_key: str = "",
) -> float:
    """The effective subscription interval for the DISPATCHING session.

    Explicit param (0 included) wins; otherwise that Hermes session's
    persisted subscription on the handle entry (``subscribers`` map, with
    the legacy scalar migrated by ``handles.subscribers_of``); otherwise
    the 180s default. Never raises.
    """
    if explicit is not None:
        try:
            return max(float(explicit), 0.0)
        except (TypeError, ValueError):
            return DEFAULT_UPDATE_INTERVAL_S
    persisted = _handles.subscribers_of(entry).get(str(session_key or ""))
    if persisted is not None:
        return persisted
    return DEFAULT_UPDATE_INTERVAL_S


class _Ticker:
    """One subscriber's digest timer for one run. Daemon ``threading.Timer``
    chain, re-armed after each tick while the run stays running."""

    def __init__(self, job: Any, sub_key: str, interval_s: float) -> None:
        self.job = job
        self.name = job.session_name or job.job_id
        # The subscribing hermes session (event routing key; "" = CLI).
        self.sub_key = str(sub_key or "")
        # Registration/state key: per (cursor session, subscriber).
        self.key = (self.name, self.sub_key)
        self.interval_s = float(interval_s)
        # Events already in the session log when the subscription started;
        # the first digest covers only what happened after it.
        self.last_seq = self._log_total()
        self._timer: Optional[threading.Timer] = None
        self._cancelled = False
        self._tlock = threading.Lock()

    # -- log helpers -----------------------------------------------------

    def _log_key(self) -> str:
        return self.job.session_name or self.job.cursor_session_id or self.job.job_id

    def _log_total(self) -> int:
        stats = _eventlog.stats(self._log_key())
        return int((stats or {}).get("total_events") or 0)

    # -- timer control ---------------------------------------------------

    def start(self) -> None:
        with self._tlock:
            self._arm_locked()

    def _arm_locked(self) -> None:
        if self._cancelled or self.interval_s <= 0:
            return
        # A pending timer is superseded, never left running alongside the
        # new one — arming is a reschedule, so a chain can never fork into
        # two parallel timer chains (issue #10).
        if self._timer is not None:
            self._timer.cancel()
        timer = threading.Timer(self.interval_s, self._tick)
        # The tick identifies which timer fired it (see _tick: only the
        # CURRENT chain may re-arm, so a concurrent set_interval reschedule
        # can never fork a second parallel timer chain).
        timer.args = (timer,)
        timer.daemon = True
        timer.name = f"ghost-cursor-progress-{self.name}"
        self._timer = timer
        timer.start()

    def set_interval(self, interval_s: float) -> None:
        """Change the cadence mid-run.

        0 unsubscribes (pending timer cancelled). A SHORTER interval
        reschedules the pending timer immediately (next tick = now + new
        interval); a longer one takes effect on the next re-arm.
        """
        with self._tlock:
            old = self.interval_s
            self.interval_s = float(interval_s)
            if self.interval_s <= 0:
                if self._timer is not None:
                    self._timer.cancel()
                    self._timer = None
            elif self.interval_s < old and self._timer is not None:
                self._timer.cancel()
                self._arm_locked()

    def cancel(self) -> None:
        with self._tlock:
            self._cancelled = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    # -- the tick ----------------------------------------------------------

    def _is_registered(self) -> bool:
        """Whether this ticker is still its subscriber's live one.

        A re-prompt swaps the registration to the new run's ticker; a
        chain that fires afterwards is stale and must tear itself down
        instead of delivering or re-arming (issue #10: interrupts must
        not stack digest loops)."""
        with _lock:
            return _tickers.get(self.key) is self

    def _tick(self, fired: threading.Timer) -> None:
        try:
            if not self._is_registered():
                self.cancel()
                return
            if self.job.status == _RUNNING:
                self._deliver()
        except Exception:
            # A digest failure must never crash the runner/timer chain.
            logger.exception(
                "ghost_cursor progress digest for %s failed", self.name
            )
        finally:
            with self._tlock:
                # Re-arm only when this timer is still the live chain — a
                # set_interval() that ran during the tick already armed a
                # replacement, and re-arming here too would fork the chain.
                if (
                    not self._cancelled
                    and self._timer is fired
                    and self.job.status == _RUNNING
                ):
                    self._arm_locked()

    def _deliver(self) -> None:
        job = self.job
        # Hard interval floor (issue #10): regardless of what fired this
        # tick — the healthy chain, a stale duplicate, a mid-run retune —
        # a digest inside interval_s of the previously ENQUEUED one is
        # dropped. last_seq is deliberately untouched so the skipped
        # events roll into the next digest instead of being lost.
        now = time.monotonic()
        with _lock:
            last = _last_emit.get(self.key)
        if last is not None and now - last < self.interval_s - _floor_slack(
            self.interval_s
        ):
            return
        total = self._log_total()
        new_count = max(total - self.last_seq, 0)
        events: List[Dict[str, Any]] = []
        if new_count:
            page = _eventlog.read_events(
                self._log_key(), offset=-1, limit=_render.DIGEST_MAX_EVENTS
            )
            events = [
                e for e in ((page or {}).get("events") or [])
                if int(e.get("seq") or 0) >= self.last_seq
            ]
        self.last_seq = total

        snap = job.snapshot()
        with _lock:
            n = _counters.get(self.key, 0) + 1
        text = _render.digest_text(
            name=self.name,
            n=n,
            status=str(snap.get("status") or _RUNNING),
            elapsed_s=snap.get("elapsed_s"),
            last_activity_s=snap.get("last_activity_s"),
            files=snap.get("files_changed_so_far") or [],
            pending_tool=str(snap.get("pending_tool") or ""),
            pending_tool_s=snap.get("pending_tool_s"),
            pending_tools=snap.get("pending_tools") or [],
            plan=snap.get("plan") or [],
            events=events,
            new_count=new_count,
            # The CURRENT interval at tick time (not the dispatch-time
            # value): a mid-run cursor_subscribe retune shows in the very
            # next digest.
            next_update_s=self.interval_s,
        )

        try:
            from tools.process_registry import process_registry
        except Exception as exc:  # pragma: no cover — core import failure
            logger.error(
                "ghost_cursor progress digest for %s: process_registry "
                "import failed: %s", self.name, exc,
            )
            return

        evt = {
            # See the module docstring: "async_delegation" is the only
            # event type every completion_queue consumer delivers promptly
            # as a generic notification; kinds are differentiated by the
            # "type" field alone, and receipt settles nothing core-side.
            "type": "async_delegation",
            # UNIQUE per digest AND per subscriber — the TUI dedups on
            # (delegation_id, type), counters are per subscriber (two
            # subscribers both emit an n=3), and the real completion uses
            # the plain session name (dispatcher) / suffixed name (other
            # subscribers).
            "delegation_id": (
                f"{self.name}#progress-{n}@{subscriber_suffix(self.sub_key)}"
            ),
            # Routes this copy to the SUBSCRIBING hermes session, not the
            # dispatching one — the multi-subscriber fix.
            "session_key": self.sub_key,
            "goal": (
                f"cursor progress update {n} for session '{self.name}' "
                "(run still active — NOT the final result; it arrives "
                "separately on completion)"
            ),
            "context": None,
            "toolsets": None,
            "role": "cursor",
            "model": job.model or job.requested_model or "cursor",
            "status": _RUNNING,
            "summary": text,
            "error": None,
            "api_calls": 0,
            "duration_seconds": round(time.time() - job.created_at, 2),
            "dispatched_at": job.created_at,
            "completed_at": time.time(),
            # Structured markers for programmatic consumers (the core
            # formatters ignore unknown keys).
            "cursor_progress_update": n,
            "cursor_job_id": job.job_id,
            "cursor_session_id": job.cursor_session_id,
        }

        # Terminal-race guard: the enqueue and finalize's status flip share
        # the job lock, so a digest is either strictly before the completion
        # event on the queue or dropped — never after it.
        try:
            with job._lock:
                if job.status != _RUNNING:
                    return
                process_registry.completion_queue.put(evt)
        except Exception as exc:
            # Log and continue — never crash the runner (spec requirement).
            logger.error(
                "ghost_cursor progress digest for %s: enqueue failed: %s",
                self.name, exc,
            )
            return
        with _lock:
            _counters[self.key] = n
            _last_emit[self.key] = time.monotonic()
        # Durable delivery-ack cursor (RFC: last_seq_delivered advances
        # only after the successful enqueue) — a supervisor re-attaching
        # after a restart resumes this subscriber's digests from here.
        _handles.advance_delivery_cursor(self.name, self.sub_key, self.last_seq)


# ---------------------------------------------------------------------------
# Public API (called from __init__.py tool handlers and jobs.py)
# ---------------------------------------------------------------------------

def start_for_job(job: Any, subscribers: Dict[str, float]) -> None:
    """Start (or restart) the digest timers for a freshly-armed run.

    Called by the dispatch path once the running handle has been handed
    back, with the session's persisted ``subscribers`` map — one ticker
    per subscriber with a positive interval. Any stale tickers for the
    session (previous run's, including subscribers no longer in the map)
    are cancelled either way.
    """
    name = job.session_name or job.job_id
    fresh = {
        (name, str(sub_key or "")): _Ticker(job, sub_key, interval)
        for sub_key, interval in (subscribers or {}).items()
        if float(interval) > 0
    }
    # Swap the registrations in ONE lock hold: there is never a window in
    # which two tickers are (or believe they are) live for one subscriber
    # — the moment the new one is visible the old one is stale, and a
    # stale chain tears itself down on its next fire (_is_registered).
    with _lock:
        old = [
            _tickers.pop(key)
            for key in [k for k in _tickers if k[0] == name]
        ]
        _tickers.update(fresh)
    for ticker in old:
        ticker.cancel()
    for ticker in fresh.values():
        ticker.start()


def cancel_for_job(job: Any) -> None:
    """Cancel every pending timer at terminal state (called by _finalize)."""
    name = job.session_name or job.job_id
    with _lock:
        victims = [
            _tickers.pop(key)
            for key in [
                k for k, t in _tickers.items()
                if k[0] == name and t.job is job
            ]
        ]
    for ticker in victims:
        ticker.cancel()


def subscribe(
    name: str, session_key: str, interval_s: float, job: Any = None
) -> None:
    """Set ONE Hermes session's subscription: persist it and retune (or
    start, or cancel) that subscriber's live ticker.

    Works whether or not a run is active — with no live run (``job`` is
    None) the persisted value simply seeds the next run's timers. With a
    live run, a subscriber that had no ticker yet (e.g. a Hermes session
    other than the dispatching one, subscribing mid-run) gets one started
    against the running job. Other subscribers are never touched.
    """
    interval = max(float(interval_s), 0.0)
    sub_key = str(session_key or "")
    _handles.set_subscriber(name, sub_key, interval)
    key = (name, sub_key)
    created: Optional[_Ticker] = None
    with _lock:
        ticker = _tickers.get(key)
        if interval <= 0:
            if ticker is not None:
                _tickers.pop(key, None)
        elif ticker is None and job is not None and job.status == _RUNNING:
            created = _Ticker(job, sub_key, interval)
            _tickers[key] = created
    if created is not None:
        created.start()
        return
    if ticker is None:
        return
    if interval <= 0:
        ticker.cancel()
    else:
        ticker.set_interval(interval)


def _reset_for_tests() -> None:
    """Cancel every live ticker and forget digest numbering (test isolation)."""
    with _lock:
        tickers = list(_tickers.values())
        _tickers.clear()
        _counters.clear()
        _last_emit.clear()
    for ticker in tickers:
        ticker.cancel()
