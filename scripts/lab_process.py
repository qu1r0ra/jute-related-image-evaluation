"""Run a Linux command with a deadline and process-group cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

GRACE_SECONDS = 5.0


def _write_error(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _shell_status(returncode: int) -> int:
    if returncode < 0:
        return 128 - returncode
    return returncode


def run(command: list[str], timeout_seconds: int) -> int:
    try:
        process = subprocess.Popen(command, start_new_session=True)
    except FileNotFoundError as error:
        _write_error(f"{command[0]}: {error.strerror}")
        return 127
    except PermissionError as error:
        _write_error(f"{command[0]}: {error.strerror}")
        return 126

    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGTERM)
        time.sleep(GRACE_SECONDS)
        _signal_group(process, signal.SIGKILL)
        process.wait()
        return 124

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
