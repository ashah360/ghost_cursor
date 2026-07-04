"""Synchronous runner for cursor-agent (Cursor CLI, headless stream-json mode).

Spawns ``cursor-agent -p <task> --output-format stream-json`` inside a target
repo and yields each parsed stream event as ``(event_key, obj)`` tuples, where
``event_key`` is ``"{type}.{subtype}"`` when a subtype is present (e.g.
``"tool_call.completed"``, ``"system.init"``) and just ``"{type}"`` otherwise.

Adapted from the Threshold bridge harness (``app/bridge/harness.py``), but
synchronous: Hermes tool handlers run in ordinary (worker) threads, so a
blocking readline loop with a watchdog timer replaces the asyncio variant.

Safety posture:

* The repo must be an existing directory — never run in a made-up path.
* A hard overall timeout guards the documented ``-p`` hang cases. On timeout
  a watchdog thread SIGTERMs the process (SIGKILL after a grace period) and a
  synthetic ``harness.error`` event is yielded so the tool returns a clean
  partial-result error instead of hanging the agent turn.
* Non-JSON stdout lines (merged stderr noise, partial writes) are skipped.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

# Base model id only — the cursor-sdk model catalog has no combined
# "<id>-thinking-<level>" slugs (those were ACP-era CLI shorthand; the SDK
# rejects them with BadRequestError). Thinking/effort ride as ModelSelection
# params instead — see sdk_runner.translate_model.
DEFAULT_MODEL = "claude-fable-5"
DEFAULT_TIMEOUT_S = 600.0
# SIGTERM -> SIGKILL grace, mirrors the Threshold harness.
TERM_GRACE_S = 10.0

CURSOR_AGENT_BIN = "cursor-agent"


class HarnessError(RuntimeError):
    """Pre-flight failure: bad repo, empty task, missing cursor-agent binary."""


def event_key(obj: Dict[str, Any]) -> str:
    """Compute the dispatch key for a raw stream-json event."""
    etype = str(obj.get("type") or "")
    subtype = obj.get("subtype")
    return f"{etype}.{subtype}" if subtype else etype


def resolve_repo(repo: str) -> Path:
    """Resolve and validate the working repo. Must be an existing directory."""
    path = Path(str(repo)).expanduser().resolve()
    if not path.is_dir():
        raise HarnessError(f"repo is not an existing directory: {repo}")
    return path


def subprocess_env() -> Dict[str, str]:
    """Process env with ~/.local/bin on PATH (where cursor-agent installs)."""
    env = dict(os.environ)
    local_bin = str(Path.home() / ".local" / "bin")
    path = env.get("PATH", "")
    if local_bin not in path.split(":"):
        env["PATH"] = f"{local_bin}:{path}" if path else local_bin
    return env


def cursor_agent_available() -> bool:
    """True when the cursor-agent binary is resolvable on the (patched) PATH."""
    import shutil

    return shutil.which(CURSOR_AGENT_BIN, path=subprocess_env().get("PATH")) is not None


def _signal_group(proc: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group (it runs in its own session).

    cursor-agent forks helpers that inherit the stdout pipe; signalling only
    the direct child would leave grandchildren holding the pipe open and the
    readline loop blocked until they exit on their own.
    """
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except Exception:
            pass


def _terminate(proc: subprocess.Popen) -> None:
    """SIGTERM the process group, escalating to SIGKILL after TERM_GRACE_S."""
    if proc.poll() is None:
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(TERM_GRACE_S)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            proc.wait()


def run_harness(
    task: str,
    repo: str,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Run cursor-agent on ``task`` inside ``repo``, yielding parsed events.

    Yields:
        ``(event_key, obj)`` per JSON line of the stream-json output. On
        timeout or a non-zero exit a synthetic ``("harness.error", {...})``
        event is yielded (with ``timeout: True`` for the timeout case).

    Raises:
        HarnessError: if the task is empty, the repo is not an existing
            directory, or the cursor-agent binary is missing.
    """
    if not str(task).strip():
        raise HarnessError("empty task")
    workdir = resolve_repo(repo)

    try:
        proc = subprocess.Popen(
            [
                CURSOR_AGENT_BIN,
                "-p",
                str(task),
                "--model",
                str(model),
                "--output-format",
                "stream-json",
                "--force",
                "--trust",
            ],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            # Merge stderr into stdout: non-JSON lines are skipped below, and
            # a separate pipe would need its own drain thread to avoid deadlock.
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=subprocess_env(),
            # Own session/process group so the timeout kill reaches forked
            # helpers that inherited the stdout pipe (see _signal_group).
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise HarnessError(f"cursor-agent not found on PATH: {exc}") from exc

    # Watchdog: readline() blocks indefinitely on a silently-hung child, so
    # deadline enforcement lives in a timer that terminates the process —
    # readline then unblocks at EOF and the loop exits.
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        _terminate(proc)

    watchdog = threading.Timer(float(timeout), _on_timeout)
    watchdog.daemon = True
    watchdog.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            raw = line.decode("utf-8", "replace").strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except ValueError:
                continue  # non-JSON noise (merged stderr, banners) — skip
            if not isinstance(obj, dict):
                continue
            yield event_key(obj), obj

        rc = proc.wait()
        if timed_out.is_set():
            yield (
                "harness.error",
                {"error": f"harness timed out after {int(timeout)}s", "timeout": True},
            )
        elif rc != 0:
            yield (
                "harness.error",
                {"error": f"cursor-agent exited with code {rc}", "exit_code": rc},
            )
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            _terminate(proc)
