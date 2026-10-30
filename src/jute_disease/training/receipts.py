"""Authoritative local run states and completion receipts.

Every fit owns ``<root>/<fit id>/``. Work happens in numbered ``attempt-N``
directories; ``state.json`` records the lifecycle and ``receipt.json`` is written
last, so a run counts as complete only when its receipt parses, carries the
expected identity, and every recorded artifact still has its recorded bytes.
Anything else is a partial attempt: it is kept for disclosure and never reused.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"
RECEIPT_FILE = "receipt.json"
SUPERSEDED_DIR = "superseded"


class RunState(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed-out"
    INTERRUPTED = "interrupted"
    OMITTED = "omitted"


class Resume(StrEnum):
    """What to do with a fit given what is on disk."""

    SKIP = "skip"
    RUN_FRESH = "run-fresh"


def write_json_atomic(path: Path | str, payload: Any) -> None:
    """Write ``payload`` as JSON so readers see the old file or the new, never half.

    Raises:
        TypeError: ``payload`` is not JSON serializable (nothing is written).
    """
    path = Path(path)
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    # Imported on first use: the data package itself imports training.logger.
    from jute_disease.data import file_sha256

    return file_sha256(path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


@dataclass
class ActiveRun:
    """A started attempt; ``complete`` or ``fail`` ends it."""

    store: "RunStore"
    fit_id: str
    identity: str
    attempt: int

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir(self.fit_id)

    @property
    def attempt_dir(self) -> Path:
        return self.run_dir / f"attempt-{self.attempt}"

    def complete(self, results: dict[str, Any], artifacts: dict[str, Path]) -> None:
        """Hash the artifacts, write the receipt, then mark the run completed."""
        recorded: dict[str, dict[str, Any]] = {}
        for name, path in artifacts.items():
            path = Path(path).resolve()
            try:
                recorded_path = path.relative_to(self.run_dir.resolve()).as_posix()
            except ValueError:
                # Shared preparation artifacts live in the configured cache.
                recorded_path = str(path)
            recorded[name] = {
                "path": recorded_path,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        write_json_atomic(
            self.run_dir / RECEIPT_FILE,
            {
                "fit_id": self.fit_id,
                "identity": self.identity,
                "attempt": self.attempt,
                "completed_at": _now(),
                "results": results,
                "artifacts": recorded,
            },
        )
        self.store.write_state(self, RunState.COMPLETED)

    def fail(self, state: RunState, reason: str) -> None:
        """Record a non-complete end: failed, timed-out or interrupted."""
        self.store.write_state(self, state, reason)


class RunStore:
    """All run directories under one root."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def run_dir(self, fit_id: str) -> Path:
        return self.root.joinpath(*fit_id.split("/"))

    def write_state(
        self, run: ActiveRun, state: RunState, reason: str | None = None
    ) -> None:
        write_json_atomic(
            self.run_dir(run.fit_id) / STATE_FILE,
            {
                "state": state.value,
                "identity": run.identity,
                "attempt": run.attempt,
                "reason": reason,
                "updated_at": _now(),
            },
        )

    def _state_payload(self, fit_id: str) -> dict[str, Any] | None:
        return _read_json(self.run_dir(fit_id) / STATE_FILE)

    def _valid_receipt(self, fit_id: str) -> dict[str, Any] | None:
        directory = self.run_dir(fit_id)
        receipt = _read_json(directory / RECEIPT_FILE)
        if receipt is None or "identity" not in receipt:
            return None
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        for entry in artifacts.values():
            path = directory / str(entry.get("path", ""))
            if not path.is_file() or _sha256(path) != entry.get("sha256"):
                return None
        return receipt

    def receipt(self, fit_id: str, missing_ok: bool = False) -> dict[str, Any] | None:
        """Return the verified receipt.

        Raises:
            FileNotFoundError: there is no valid receipt and ``missing_ok`` is false.
        """
        receipt = self._valid_receipt(fit_id)
        if receipt is None and not missing_ok:
            raise FileNotFoundError(f"No valid receipt for {fit_id}")
        return receipt

    def state(self, fit_id: str) -> RunState:
        payload = self._state_payload(fit_id)
        if payload is None:
            return (
                RunState.INTERRUPTED
                if self.run_dir(fit_id).exists()
                else (RunState.PLANNED)
            )
        try:
            state = RunState(payload["state"])
        except (KeyError, ValueError):
            return RunState.INTERRUPTED
        if state is RunState.COMPLETED and self._valid_receipt(fit_id) is None:
            return RunState.INTERRUPTED
        return state

    def reason(self, fit_id: str) -> str:
        payload = self._state_payload(fit_id) or {}
        return str(payload.get("reason") or "")

    def resolve(self, fit_id: str, identity: str) -> Resume:
        """Skip only a complete run whose identity matches exactly."""
        receipt = self._valid_receipt(fit_id)
        if receipt is not None and receipt["identity"] == identity:
            return Resume.SKIP
        return Resume.RUN_FRESH

    def settle(self, fit_id: str) -> RunState:
        """Mark a run left in ``running`` as interrupted.

        Call only when no process owns the run (the queue is serial, so the
        previous attempt's process is gone before the next one starts).
        """
        payload = self._state_payload(fit_id)
        if payload is not None and payload.get("state") == RunState.RUNNING:
            run = ActiveRun(
                self, fit_id, str(payload.get("identity")), int(payload["attempt"])
            )
            self.write_state(
                run, RunState.INTERRUPTED, "process ended without a result"
            )
        return self.state(fit_id)

    def _attempts(self, fit_id: str) -> list[int]:
        directory = self.run_dir(fit_id)
        if not directory.exists():
            return []
        return sorted(
            int(child.name.removeprefix("attempt-"))
            for child in directory.glob("attempt-*")
            if child.name.removeprefix("attempt-").isdigit()
        )

    def _supersede(self, fit_id: str) -> None:
        directory = self.run_dir(fit_id)
        receipt = directory / RECEIPT_FILE
        if not receipt.exists():
            return
        target = directory / SUPERSEDED_DIR
        target.mkdir(exist_ok=True)
        number = len(list(target.glob("*.json")))
        os.replace(receipt, target / f"receipt-{number}.json")

    def start(self, fit_id: str, identity: str) -> ActiveRun:
        """Open a new attempt; any earlier receipt is retired, never reused."""
        self.settle(fit_id)
        self._supersede(fit_id)
        attempts = self._attempts(fit_id)
        run = ActiveRun(self, fit_id, identity, (attempts[-1] + 1) if attempts else 1)
        run.attempt_dir.mkdir(parents=True)
        self.write_state(run, RunState.RUNNING)
        return run

    def omit(self, fit_id: str, identity: str, reason: str) -> None:
        """Record that a planned fit was deliberately not run, and why."""
        attempts = self._attempts(fit_id)
        run = ActiveRun(self, fit_id, identity, attempts[-1] if attempts else 0)
        self.write_state(run, RunState.OMITTED, reason)
