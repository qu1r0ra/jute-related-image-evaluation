"""Run one queue invocation and record a hash inventory of its run store."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

TRANSFER_COMMAND = Path(__file__).resolve().with_name("verify_transfer.py")


def _shell_status(returncode: int) -> int:
    """Return a conventional shell status for a child process result."""
    return 128 - returncode if returncode < 0 else returncode


def main() -> int:
    """Run a supplied queue command and snapshot its artifacts after it exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply a queue command after --")

    try:
        args.transfer_root.mkdir(parents=True, exist_ok=True)
        args.manifest_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        sys.stderr.write("Queue batch directories could not be prepared\n")
        return 2

    try:
        result = subprocess.run(command, check=False)
        queue_status = _shell_status(result.returncode)
    except OSError:
        sys.stderr.write("Queue command could not be started\n")
        queue_status = 127

    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    manifest = args.manifest_dir / f"queue-{batch_id}.json"
    try:
        transfer = subprocess.run(
            [
                sys.executable,
                str(TRANSFER_COMMAND),
                "record",
                str(args.transfer_root),
                str(manifest),
            ],
            check=False,
        )
    except OSError:
        sys.stderr.write("Queue batch transfer inventory could not be recorded\n")
        return queue_status or 2

    transfer_status = _shell_status(transfer.returncode)
    return queue_status or transfer_status


if __name__ == "__main__":
    raise SystemExit(main())
