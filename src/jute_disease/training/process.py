"""Run one fit in a child process so its time limit is enforced from outside.

A fit that hangs inside native code cannot be interrupted from within, so the
queue launches every fit as a child and kills the whole process tree when the
limit passes. Output goes to a log file, never to memory.
"""

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

TAIL_LINES = 20
SIGTERM_EXIT_STATUS = 128 + signal.SIGTERM


def _exit_on_signal(signum: int, frame: object) -> None:
    raise SystemExit(SIGTERM_EXIT_STATUS)


def exit_on_sigterm() -> None:
    """Turn SIGTERM into ``SystemExit`` so a waiting ``run_child`` kills its child.

    Fit children run in their own sessions, which a group signal to the queue
    cannot reach. Exiting unwinds through ``run_child``, which kills the child
    tree before the process goes away. Call from the main thread only.
    """
    signal.signal(signal.SIGTERM, _exit_on_signal)


@dataclass(frozen=True)
class ChildResult:
    """How a child ended; ``tail`` is the end of its log for failure reasons."""

    returncode: int
    timed_out: bool
    seconds: float
    tail: str


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _tail(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-TAIL_LINES:])


def run_child(
    argv: Sequence[str], timeout: float | None, log_path: Path | str
) -> ChildResult:
    """Run ``argv`` to completion, or kill its process tree after ``timeout``.

    Raises:
        KeyboardInterrupt: the wait was interrupted; the child tree is killed
            first so no orphan keeps using the GPU.
    """
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    timed_out = False
    with log.open("wb") as stream:
        group: dict[str, object] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            list(argv), stdout=stream, stderr=subprocess.STDOUT, **group
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(process)
        except BaseException:
            _kill_tree(process)
            raise
    return ChildResult(
        process.returncode, timed_out, time.monotonic() - started, _tail(log)
    )
