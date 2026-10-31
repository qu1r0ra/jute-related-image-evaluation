"""Environment-only ntfy notifications around an argv command (no shell)."""

import argparse
import os
import subprocess
import sys
import threading
import urllib.request


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
    notify("Jute run started")
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(3600):
            try:
                notify("Jute run heartbeat")
            except Exception:
                sys.stderr.write("Heartbeat notification failed\n")

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        try:
            result = subprocess.run(command, check=False)
            return_code = result.returncode
        except OSError:
            sys.stderr.write("Run command could not be started\n")
            return_code = 127
    finally:
        stopped.set()
        worker.join(timeout=16)
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
