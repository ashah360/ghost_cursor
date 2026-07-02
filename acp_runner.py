"""ACP runner for cursor-agent (Agent Client Protocol, JSON-RPC over stdio).

Replaces the stdout-scraping ``--print --output-format stream-json`` harness
(kept in ``runner.py`` as the importable legacy/reference implementation) with
a structured protocol client. Spawns ``cursor-agent --trust acp`` inside the
target repo, performs the ACP handshake (``initialize`` → ``session/new`` —
or ``session/load`` when resuming a prior session — → ``session/prompt``),
and yields the streaming ``session/update`` notifications that cursor emits
while it works.

Yielded event tuples (consumed by ``events.AcpNormalizer``):

* ``("acp.session", {...})`` — session established (sessionId, cwd, model,
  resumed). ``resumed`` is True when an existing session was continued via
  ``session/load`` (see below), False for a fresh ``session/new``.
* ``("acp.update", <update>)`` — one raw ``session/update`` payload, i.e. the
  object under ``params.update`` (``sessionUpdate`` discriminator field).
* ``("acp.result", {"stopReason": ...})`` — the ``session/prompt`` response.
  ``stopReason`` is ``"end_turn"`` on success, ``"cancelled"`` after a
  ``session/cancel`` (verified live).
* ``("acp.error", {"error": ..., "timeout": bool})`` — mid-run hard failure
  (timeout, connection loss). Preflight/handshake failures raise
  :class:`AcpError` instead so the tool returns a clean, actionable error.

Cancellation is native: the consumer-side loop polls ``cancel_check`` (by
default the Hermes per-thread interrupt flag) and the overall deadline; on
either trigger it sends the ``session/cancel`` notification, waits up to
``CANCEL_GRACE_S`` for the prompt to resolve (observed: it resolves
immediately with ``stopReason: "cancelled"``), then SIGKILLs the process
group as a last resort.

Client-side ACP methods cursor may call back are answered minimally:
``fs/read_text_file`` / ``fs/write_text_file`` are served from disk and
``session/request_permission`` auto-approves (the tool runs with ``--trust``
semantics, matching the old ``--force --trust`` flags — in practice cursor
never asks under ``--trust``). Anything else gets a JSON-RPC
method-not-found error, which cursor handles by using its own tooling.

Threading model: Hermes tool handlers run in ordinary worker threads, so the
asyncio client runs in a dedicated background thread and hands events to the
calling thread through a queue; ``run_acp`` is a plain synchronous generator
like the legacy ``run_harness``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from plugins.ghost_cursor.events import unified_diff_text
from plugins.ghost_cursor.runner import (
    CURSOR_AGENT_BIN,
    DEFAULT_TIMEOUT_S,
    TERM_GRACE_S,
    HarnessError,
    cursor_agent_available,  # noqa: F401  (re-exported for the tool gate)
    resolve_repo,
    subprocess_env,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
INIT_TIMEOUT_S = 30.0
SESSION_NEW_TIMEOUT_S = 30.0
# session/load replays the prior session's history as session/update
# notifications before resolving (observed live, 2026-07-02), so a long
# prior session takes longer to load than a fresh session/new.
SESSION_LOAD_TIMEOUT_S = 60.0
# After session/cancel, how long to wait for the prompt to resolve before
# SIGKILLing the process group. Observed resolve time is ~0s; the grace only
# matters when cursor is genuinely hung.
CANCEL_GRACE_S = 15.0
_POLL_S = 0.2
_READ_CHUNK = 65536

# Cap for diff text produced by the git fallback (mirrors events.MAX_DIFF_CHARS).
_FALLBACK_DIFF_CHARS = 100_000


class AcpError(HarnessError):
    """Hard ACP failure before/at handshake — no run happened."""


class _ConnectionClosed(RuntimeError):
    """cursor-agent's stdout hit EOF while requests were pending."""


def _default_cancel_check() -> bool:
    """Poll the Hermes per-thread interrupt flag (set by AIAgent.interrupt())."""
    try:
        from tools.interrupt import is_interrupted

        return is_interrupted()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Async JSON-RPC client (runs on a dedicated event-loop thread)
# ---------------------------------------------------------------------------

class _AcpClient:
    def __init__(
        self,
        task: str,
        workdir: Path,
        out_q: "queue.Queue[Tuple[str, Dict[str, Any]]]",
        cancel_requested: threading.Event,
        timeout: float,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self._task = task
        self._workdir = workdir
        self._out_q = out_q
        self._cancel_requested = cancel_requested
        self._timeout = timeout
        # Prior cursor session to continue via session/load (None = fresh).
        self._resume_session_id = session_id
        # Model override. INSTRUMENTED (2026-07-02, cursor-agent
        # 2026.07.01-777f564): passing `model`/`modelId` in session/new
        # params is silently ignored, and `session/set_model` rejects plain
        # model ids ("Invalid model value") — but the global `--model` flag
        # on the `cursor-agent ... acp` invocation IS honored (session/new
        # reports it as models.currentModelId). So the override rides the
        # argv. Caveat: cursor-agent persists the flag as its new default
        # for later flag-less runs (upstream behavior).
        self._model = (str(model).strip() or None) if model else None
        # "cancel" | "timeout" — written by the consumer thread before it sets
        # cancel_requested (single write, read after the event fires).
        self.abort_reason: Optional[str] = None

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._next_id = 0
        self._pending: Dict[int, "asyncio.Future[Dict[str, Any]]"] = {}
        self._session_id: Optional[str] = None
        self._prompt_settled = False
        self._eof = False

    # -- plumbing ----------------------------------------------------------

    def _put(self, key: str, obj: Dict[str, Any]) -> None:
        self._out_q.put((key, obj))

    async def _send(self, obj: Dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _notify(self, method: str, params: Dict[str, Any]) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(
        self, method: str, params: Dict[str, Any], timeout: Optional[float]
    ) -> Dict[str, Any]:
        self._next_id += 1
        rid = self._next_id
        fut: "asyncio.Future[Dict[str, Any]]" = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        # EOF cleanup and this registration run on the same loop, so this
        # check is race-free: either _eof is already set, or the pending
        # future above will be failed by the read loop's EOF handler.
        if self._eof:
            self._pending.pop(rid, None)
            raise _ConnectionClosed("cursor-agent closed the ACP connection")
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout)

    # -- child process -----------------------------------------------------

    async def _spawn(self) -> None:
        argv: List[str] = [CURSOR_AGENT_BIN, "--trust"]
        if self._model:
            argv += ["--model", self._model]
        argv.append("acp")
        self._proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self._workdir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=subprocess_env(),
            # Own process group so the kill switch reaches forked helpers.
            start_new_session=True,
        )

    def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass

    async def _shutdown(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), TERM_GRACE_S)
        except (asyncio.TimeoutError, Exception):
            self._kill()
            try:
                await proc.wait()
            except Exception:
                pass

    # -- inbound dispatch ----------------------------------------------------

    async def _read_loop(self) -> None:
        # Manual line buffering: session/update lines carry full-file diff
        # payloads that overflow StreamReader.readline()'s limit (which
        # poisons the stream); read() has no such limit.
        assert self._proc is not None and self._proc.stdout is not None
        buf = b""
        while True:
            chunk = await self._proc.stdout.read(_READ_CHUNK)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                await self._handle_line(line)
        if buf.strip():
            await self._handle_line(buf)
        # EOF: fail everything still pending so awaiting requests unwind.
        self._eof = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(_ConnectionClosed("cursor-agent closed the ACP connection"))
        self._pending.clear()

    async def _handle_line(self, line: bytes) -> None:
        raw = line.decode("utf-8", "replace").strip()
        if not raw:
            return
        try:
            obj = json.loads(raw)
        except ValueError:
            return  # non-JSON noise on stdout — skip
        if not isinstance(obj, dict):
            return
        try:
            await self._dispatch(obj)
        except Exception:
            logger.exception("ACP dispatch failed for: %.200s", raw)

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        while True:
            chunk = await self._proc.stderr.read(_READ_CHUNK)
            if not chunk:
                break
            text = chunk.decode("utf-8", "replace").strip()
            if text:
                logger.debug("cursor-agent stderr: %s", text)

    async def _dispatch(self, obj: Dict[str, Any]) -> None:
        if "id" in obj and ("result" in obj or "error" in obj):
            fut = self._pending.pop(obj["id"], None)
            if fut is not None and not fut.done():
                if "error" in obj:
                    err = obj["error"] or {}
                    fut.set_exception(
                        RuntimeError(
                            f"ACP error {err.get('code')}: {err.get('message')}"
                        )
                    )
                else:
                    result = obj.get("result")
                    fut.set_result(result if isinstance(result, dict) else {})
        elif "id" in obj and "method" in obj:
            await self._handle_agent_request(obj)
        elif obj.get("method") == "session/update":
            params = obj.get("params") or {}
            update = params.get("update")
            if isinstance(update, dict):
                self._put("acp.update", update)
        # other notifications: ignore

    async def _handle_agent_request(self, obj: Dict[str, Any]) -> None:
        """Answer agent→client requests (fs/*, permission). Minimal set —
        cursor under ``--trust`` normally uses its own tooling and never asks."""
        method = str(obj.get("method") or "")
        params = obj.get("params") or {}
        result: Optional[Dict[str, Any]] = None
        error: Optional[Dict[str, Any]] = None
        try:
            if method == "session/request_permission":
                # Auto-approve: the tool's contract is --force/--trust semantics.
                options = [o for o in (params.get("options") or []) if isinstance(o, dict)]
                chosen = next(
                    (o for o in options if "allow" in str(o.get("kind", "")).lower()),
                    options[0] if options else None,
                )
                result = {
                    "outcome": {
                        "outcome": "selected",
                        "optionId": (chosen or {}).get("optionId", "allow"),
                    }
                }
            elif method == "fs/read_text_file":
                content = Path(str(params.get("path") or "")).read_text("utf-8")
                result = {"content": content}
            elif method == "fs/write_text_file":
                Path(str(params.get("path") or "")).write_text(
                    str(params.get("content") or ""), "utf-8"
                )
                result = {}
            else:
                error = {"code": -32601, "message": f"method not supported: {method}"}
        except Exception as exc:
            error = {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}
        resp: Dict[str, Any] = {"jsonrpc": "2.0", "id": obj.get("id")}
        if error is not None:
            resp["error"] = error
        else:
            resp["result"] = result
        await self._send(resp)

    # -- session establishment -------------------------------------------------

    async def _establish_session(self) -> Tuple[Dict[str, Any], bool]:
        """Create or resume the cursor session.

        When a resume id was provided, try ``session/load`` first (cursor
        advertises ``loadSession: true``). Observed live (2026-07-02): load
        REPLAYS the prior session's history as ``session/update``
        notifications before resolving — those flow through the normal
        update path — and resolves with the same shape as ``session/new``
        minus ``sessionId``. If load fails (expired/unknown id → JSON-RPC
        ``Invalid params``), fall back to a fresh ``session/new`` so the
        task still runs, just without prior context.

        Returns:
            ``(result, resumed)`` — the session/new-or-load result dict and
            whether the prior session was actually resumed. Raises on
            failure of the final ``session/new`` attempt (no session at all).
        """
        if self._resume_session_id:
            try:
                sess = await self._request(
                    "session/load",
                    {
                        "sessionId": self._resume_session_id,
                        "cwd": str(self._workdir),
                        "mcpServers": [],
                    },
                    timeout=SESSION_LOAD_TIMEOUT_S,
                )
                self._session_id = self._resume_session_id
                return sess, True
            except Exception as exc:
                logger.warning(
                    "ACP session/load for %s failed (%s: %s) — "
                    "falling back to a fresh session/new",
                    self._resume_session_id,
                    type(exc).__name__,
                    exc,
                )
        sess = await self._request(
            "session/new",
            {"cwd": str(self._workdir), "mcpServers": []},
            timeout=SESSION_NEW_TIMEOUT_S,
        )
        self._session_id = str(sess.get("sessionId") or "")
        return sess, False

    # -- cancellation --------------------------------------------------------

    async def _cancel_watcher(self) -> None:
        while not self._cancel_requested.is_set():
            await asyncio.sleep(_POLL_S)
        if self._prompt_settled:
            return
        if self._session_id is not None:
            try:
                await self._notify("session/cancel", {"sessionId": self._session_id})
            except Exception:
                pass
        # Grace: the prompt normally resolves ~immediately with
        # stopReason "cancelled". Kill only if cursor is truly hung.
        grace = float(CANCEL_GRACE_S)
        waited = 0.0
        while waited < grace:
            if self._prompt_settled:
                return
            await asyncio.sleep(_POLL_S)
            waited += _POLL_S
        self._kill()

    # -- main flow -----------------------------------------------------------

    async def run(self) -> None:
        try:
            try:
                await self._spawn()
            except FileNotFoundError as exc:
                self._put("acp.fatal", {"error": f"cursor-agent not found on PATH: {exc}"})
                return

            reader = asyncio.ensure_future(self._read_loop())
            stderr_drain = asyncio.ensure_future(self._drain_stderr())
            watcher = asyncio.ensure_future(self._cancel_watcher())
            try:
                try:
                    await self._request(
                        "initialize",
                        {
                            "protocolVersion": PROTOCOL_VERSION,
                            "clientCapabilities": {},
                        },
                        timeout=INIT_TIMEOUT_S,
                    )
                except Exception as exc:
                    self._put(
                        "acp.fatal",
                        {
                            "error": (
                                "ACP initialize handshake with cursor-agent failed "
                                f"({type(exc).__name__}: {exc}). The installed "
                                "cursor-agent may not support the `acp` subcommand — "
                                "run `cursor-agent update` and retry."
                            )
                        },
                    )
                    return

                try:
                    sess, resumed = await self._establish_session()
                except Exception as exc:
                    self._put(
                        "acp.fatal",
                        {
                            "error": (
                                f"ACP session/new failed ({type(exc).__name__}: {exc}). "
                                "Check cursor-agent auth (`cursor-agent login`)."
                            )
                        },
                    )
                    return
                models = sess.get("models") if isinstance(sess.get("models"), dict) else {}
                self._put(
                    "acp.session",
                    {
                        "sessionId": self._session_id,
                        "cwd": str(self._workdir),
                        "model": models.get("currentModelId"),
                        "resumed": resumed,
                    },
                )

                try:
                    # No per-request timeout: the consumer loop owns the
                    # overall deadline and triggers cancel/kill on expiry.
                    result = await self._request(
                        "session/prompt",
                        {
                            "sessionId": self._session_id,
                            "prompt": [{"type": "text", "text": self._task}],
                        },
                        timeout=None,
                    )
                    self._prompt_settled = True
                    if self.abort_reason == "timeout":
                        self._put(
                            "acp.error",
                            {
                                "error": f"ACP run timed out after {int(self._timeout)}s",
                                "timeout": True,
                            },
                        )
                    else:
                        self._put("acp.result", {"stopReason": result.get("stopReason")})
                except Exception as exc:
                    self._prompt_settled = True
                    if self.abort_reason == "timeout":
                        self._put(
                            "acp.error",
                            {
                                "error": f"ACP run timed out after {int(self._timeout)}s",
                                "timeout": True,
                            },
                        )
                    elif self.abort_reason == "cancel":
                        # Connection died before the cancelled prompt resolved.
                        self._put("acp.result", {"stopReason": "cancelled"})
                    else:
                        self._put(
                            "acp.error",
                            {"error": f"ACP connection failed mid-run: {exc}"},
                        )
            finally:
                self._prompt_settled = True
                watcher.cancel()
                await self._shutdown()
                reader.cancel()
                stderr_drain.cancel()
        except Exception as exc:  # belt-and-braces: never strand the consumer
            logger.exception("ACP client crashed")
            self._put("acp.error", {"error": f"ACP client crashed: {exc}"})
        finally:
            self._put("__done__", {})


# ---------------------------------------------------------------------------
# Synchronous generator facade (what cursor_edit consumes)
# ---------------------------------------------------------------------------

def run_acp(
    task: str,
    repo: str,
    timeout: float = DEFAULT_TIMEOUT_S,
    cancel_check: Optional[Callable[[], bool]] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Run cursor-agent on ``task`` inside ``repo`` over ACP, yielding events.

    Yields ``("acp.session"|"acp.update"|"acp.result"|"acp.error", obj)``
    tuples (see module docstring). Polls ``cancel_check`` (default: the
    Hermes per-thread interrupt flag) and the overall ``timeout`` between
    events; both trigger a native ``session/cancel``.

    ``session_id`` continues a prior cursor session via ``session/load``
    (multi-turn). If the load fails (expired/unknown id), the run falls back
    to a fresh ``session/new`` — the ``acp.session`` event's ``resumed``
    field reports what actually happened.

    ``model`` overrides the cursor-agent model for this run. Instrumented:
    the ACP session/new params do NOT accept a model, so the override is
    passed as ``--model`` on the cursor-agent invocation (see _AcpClient);
    the session's actual model comes back on the ``acp.session`` event.

    Raises:
        HarnessError: empty task / bad repo (preflight).
        AcpError: the ACP handshake failed — actionable message, no run.
    """
    if not str(task).strip():
        raise HarnessError("empty task")
    workdir = resolve_repo(repo)
    if cancel_check is None:
        cancel_check = _default_cancel_check

    out_q: "queue.Queue[Tuple[str, Dict[str, Any]]]" = queue.Queue()
    cancel_requested = threading.Event()
    client = _AcpClient(
        task=str(task),
        workdir=workdir,
        out_q=out_q,
        cancel_requested=cancel_requested,
        timeout=float(timeout),
        session_id=(str(session_id).strip() or None) if session_id else None,
        model=model,
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(client.run()),
        name="ghost-cursor-acp",
        daemon=True,
    )
    thread.start()

    deadline = time.monotonic() + float(timeout)
    fatal: Optional[str] = None
    try:
        while True:
            if not cancel_requested.is_set():
                if time.monotonic() >= deadline:
                    client.abort_reason = "timeout"
                    cancel_requested.set()
                elif cancel_check():
                    client.abort_reason = "cancel"
                    cancel_requested.set()
            try:
                key, obj = out_q.get(timeout=_POLL_S)
            except queue.Empty:
                if not thread.is_alive() and out_q.empty():
                    break  # producer died without its sentinel — defensive
                continue
            if key == "__done__":
                break
            if key == "acp.fatal":
                fatal = str(obj.get("error") or "ACP failure")
                continue  # drain to the sentinel, then raise below
            yield key, obj
    finally:
        # Consumer abandoned us (exception/GeneratorExit) or normal exit:
        # make sure the producer unwinds and the child dies.
        cancel_requested.set()
        thread.join(timeout=CANCEL_GRACE_S + TERM_GRACE_S)

    if fatal is not None:
        raise AcpError(fatal)


# ---------------------------------------------------------------------------
# Git fallback for files_changed (when the ACP stream carried no diffs)
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

    Used only when cursor made edits through paths that emitted no ACP diff
    content (e.g. shell commands). Compares ``git status --porcelain`` against
    the pre-run snapshot so pre-existing dirty files aren't misattributed;
    a pre-existing-dirty file that cursor edits *further* is the known blind
    spot of this fallback.

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
