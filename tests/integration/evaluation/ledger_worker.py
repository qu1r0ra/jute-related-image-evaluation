"""Subprocess worker: one ``evaluate`` command with a rendezvous at the ledger write.

Run as ``python ledger_worker.py <name> <rendezvous dir> <cli arguments...>``.
Two workers started together each pause where they write the project ledger
until the other has either reached that point or found the ledger lock held.
A check made before the lock lets both through; a check made under the lock
lets only the first one write. The outcome is left in ``<name>.result.json``.
"""

import json
import sys
import time
from pathlib import Path

from jute_disease import cli
from jute_disease.evaluation import freeze_record
from jute_disease.training import FileLock, FitLockedError

LEDGER = "test-reads.jsonl"
WAIT_SECONDS = 20.0


def _peer_arrived(rendezvous: Path, name: str) -> bool:
    return any(
        marker.name.split("-", 1)[1] != name
        for pattern in ("ready-*", "contended-*")
        for marker in rendezvous.glob(pattern)
    )


def main() -> int:
    name, rendezvous, *arguments = sys.argv[1:]
    marks = Path(rendezvous)
    appended = freeze_record._append_durably
    acquired = FileLock.acquire

    def staged(path: Path, text: str) -> None:
        if path.name == LEDGER:
            (marks / f"ready-{name}").touch()
            deadline = time.monotonic() + WAIT_SECONDS
            while time.monotonic() < deadline and not _peer_arrived(marks, name):
                time.sleep(0.02)
        appended(path, text)

    def contended(self: FileLock) -> None:
        try:
            acquired(self)
        except FitLockedError:
            if self.path.name == f"{LEDGER}.lock":
                (marks / f"contended-{name}").touch()
            raise

    freeze_record._append_durably = staged
    FileLock.acquire = contended
    outcome: dict[str, object] = {"exit": 0, "message": ""}
    try:
        cli.main(arguments)
    except SystemExit as stop:
        outcome = {"exit": 1, "message": str(stop)}
    (marks / f"{name}.result.json").write_text(json.dumps(outcome), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
