"""Run a Linux command with a deadline and process-group cleanup.

This process owns the run-chain flock (inherited as fd 9 from lab_run.sh) and
keeps it until every member of the command's process group is gone. The
command is launched with ``close_fds``, so no child holds the lock itself.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

GRACE_SECONDS = 5.0
KILL_SETTLE_SECONDS = 2.0
POLL_SECONDS = 0.05

_received: list[int] = []


def _write_error(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def _remember_signal(signum: int, frame: object) -> None:
    _received.append(signum)


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _group_alive(process: subprocess.Popen[bytes]) -> bool:
    """Whether any process is still in the command's group.

    The leader is reaped first, because a zombie leader still counts as a member.
    """
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until_gone(process: subprocess.Popen[bytes], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while _group_alive(process):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


def _stop_group(process: subprocess.Popen[bytes]) -> None:
    """End the whole group: SIGTERM, a grace period, then SIGKILL."""
    _signal_group(process, signal.SIGTERM)
    if not _wait_until_gone(process, GRACE_SECONDS):
        _signal_group(process, signal.SIGKILL)
        _wait_until_gone(process, KILL_SETTLE_SECONDS)
    process.wait()


def _shell_status(returncode: int) -> int:
    if returncode < 0:
        return 128 - returncode
    return returncode


def run(command: list[str], timeout_seconds: int) -> int:
    _received.clear()
    previous = {
        signum: signal.signal(signum, _remember_signal)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        return _run(command, timeout_seconds)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run(command: list[str], timeout_seconds: int) -> int:
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except FileNotFoundError as error:
        _write_error(f"{command[0]}: {error.strerror}")
        return 127
    except PermissionError as error:
        _write_error(f"{command[0]}: {error.strerror}")
        return 126

    deadline = time.monotonic() + timeout_seconds
    while True:
        returncode = process.poll()
        if returncode is not None:
            break
        if _received:
            _stop_group(process)
            return 128 + _received[0]
        if time.monotonic() >= deadline:
            _stop_group(process)
            return 124
        time.sleep(POLL_SECONDS)

    # The command is done; anything it left behind goes before the lock does.
    if _group_alive(process):
        _stop_group(process)
    return _shell_status(returncode)


def main(arguments: list[str]) -> int:
    if len(arguments) < 2:
        _write_error("Usage: lab_process.py SECONDS COMMAND...")
        return 2

    try:
        timeout_seconds = int(arguments[0])
    except ValueError:
        _write_error("Timeout must be a positive integer")
        return 2
    if timeout_seconds < 1:
        _write_error("Timeout must be a positive integer")
        return 2

    return run(arguments[1:], timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
