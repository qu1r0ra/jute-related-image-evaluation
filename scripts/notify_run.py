"""Environment-only ntfy notifications around an argv command (no shell)."""

import argparse
import math
import os
import subprocess
import sys
import threading
import time
import urllib.request


def heartbeat_interval() -> float:
    """Return the configured heartbeat interval in seconds."""
    raw_interval = os.environ.get("NTFY_HEARTBEAT_SEC", "3600")
    try:
        interval = float(raw_interval)
    except ValueError:
        raise ValueError("NTFY_HEARTBEAT_SEC must be a nonnegative number") from None
    if not math.isfinite(interval) or interval < 0 or interval > threading.TIMEOUT_MAX:
        raise ValueError("NTFY_HEARTBEAT_SEC must be a nonnegative number")
    return interval


def _format_duration(seconds: float) -> str:
    """Format elapsed runtime as a compact duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_remainder = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds_remainder}s"
    return f"{minutes}m {seconds_remainder}s"


def notify(message: str) -> None:
    """Send one authenticated notification without exposing the token in argv."""
    topic = os.environ.get("NTFY_TOPIC", "")
    token = os.environ.get("NTFY_TOKEN", "")
    if not topic or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in topic
    ):
        raise ValueError("NTFY_TOPIC must be a nonempty topic name")
    if not token or "\n" in token or "\r" in token:
        raise ValueError("NTFY_TOKEN must be a nonempty single-line token")
    request = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=message.encode(),
        headers={"Authorization": f"Bearer {token}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        response.read()


def main() -> int:
    """Run the connectivity check or notify around a child process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.test:
        notify("Jute notification connectivity test")
        return 0
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Supply a command or --test")
    interval = heartbeat_interval()
    notify("Jute run started")
    stopped = threading.Event()
    started_at = time.monotonic()

    def heartbeat(process: subprocess.Popen[bytes]) -> None:
        while not stopped.wait(interval):
            if stopped.is_set() or process.poll() is not None:
                return
            try:
                elapsed = _format_duration(time.monotonic() - started_at)
                notify(f"Jute run heartbeat (elapsed: {elapsed})")
            except Exception:
                sys.stderr.write("Heartbeat notification failed\n")

    worker: threading.Thread | None = None
    try:
        try:
            process = subprocess.Popen(command)
        except OSError:
            sys.stderr.write("Run command could not be started\n")
            return_code = 127
        else:
            if interval > 0:
                worker = threading.Thread(
                    target=heartbeat, args=(process,), daemon=True
                )
                worker.start()
            return_code = process.wait()
    finally:
        stopped.set()
        if worker is not None:
            worker.join()
    try:
        notify("Jute run succeeded" if return_code == 0 else "Jute run failed")
    except Exception:
        sys.stderr.write("Completion notification failed\n")
    return return_code if return_code >= 0 else 128 - return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyError, ValueError, OSError:
        sys.stderr.write("Notification setup or delivery failed\n")
        raise SystemExit(2) from None
