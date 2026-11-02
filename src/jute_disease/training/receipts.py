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
ATTEMPT_FILE = "attempt.json"
RECEIPT_FORMAT_VERSION = 2
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


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


@dataclass
class ActiveRun:
    """A started attempt; ``complete`` or ``fail`` ends it."""

    store: RunStore
    fit_id: str
    identity: str
    attempt: int

    @property
    def run_dir(self) -> Path:
        return self.store.run_dir(self.fit_id)

    @property
    def attempt_dir(self) -> Path:
        return self.run_dir / f"attempt-{self.attempt}"

    def complete(
        self,
        results: dict[str, Any],
        artifacts: dict[str, Path],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Hash artifacts, record this attempt, then write its completion receipt."""
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
        self.store.finish_attempt(self, RunState.COMPLETED, metadata=metadata)
        write_json_atomic(
            self.run_dir / RECEIPT_FILE,
            {
                "format_version": RECEIPT_FORMAT_VERSION,
                "fit_id": self.fit_id,
                "identity": self.identity,
                "attempt": self.attempt,
                "completed_at": _now(),
                "results": results,
                "artifacts": recorded,
                "metadata": metadata or {},
                "attempts": self.store.attempts(self.fit_id, identity=self.identity),
            },
        )
        self.store.write_state(self, RunState.COMPLETED)

    def fail(
        self,
        state: RunState,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a non-complete end: failed, timed-out or interrupted."""
        self.store.finish_attempt(self, state, reason, metadata=metadata)
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
        return read_json(self.run_dir(fit_id) / STATE_FILE)

    def _valid_receipt(self, fit_id: str) -> dict[str, Any] | None:
        directory = self.run_dir(fit_id)
        receipt = read_json(directory / RECEIPT_FILE)
        if (
            receipt is None
            or receipt.get("format_version") != RECEIPT_FORMAT_VERSION
            or not isinstance(receipt.get("identity"), str)
            or receipt.get("fit_id") != fit_id
        ):
            return None
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, dict):
            return None
        for entry in artifacts.values():
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("path"), str)
                or not isinstance(entry.get("sha256"), str)
            ):
                return None
            path = directory / entry["path"]
            if not path.is_file() or _sha256(path) != entry.get("sha256"):
                return None
        attempts = receipt.get("attempts")
        terminal_states = {
            RunState.COMPLETED.value,
            RunState.FAILED.value,
            RunState.TIMED_OUT.value,
            RunState.INTERRUPTED.value,
            RunState.OMITTED.value,
        }
        if (
            not isinstance(attempts, list)
            or not attempts
            or not all(isinstance(attempt, dict) for attempt in attempts)
        ):
            return None
        attempt_numbers = [attempt.get("attempt") for attempt in attempts]
        if any(type(number) is not int for number in attempt_numbers):
            return None
        if (
            type(receipt.get("attempt")) is not int
            or attempts[-1].get("state") != RunState.COMPLETED.value
            or receipt.get("attempt") != attempt_numbers[-1]
            or attempt_numbers
            != list(range(attempt_numbers[0], attempt_numbers[0] + len(attempts)))
            or any(
                attempt.get("identity") != receipt["identity"]
                or attempt.get("state") not in terminal_states
                or not isinstance(attempt.get("metadata"), dict)
                for attempt in attempts
            )
        ):
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
        except KeyError, ValueError:
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
        if (
            receipt is not None
            and receipt["identity"] == identity
            and self.state(fit_id) is RunState.COMPLETED
        ):
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
            runtime = read_json(run.attempt_dir / "runtime.json")
            attempt = read_json(run.attempt_dir / ATTEMPT_FILE) or {}
            original_metadata = attempt.get("metadata")
            metadata = (
                dict(original_metadata) if isinstance(original_metadata, dict) else {}
            )
            if isinstance(runtime, dict):
                metadata["runtime"] = runtime
            run.fail(
                RunState.INTERRUPTED,
                "process ended without a result",
                metadata=metadata,
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

    def attempts(
        self, fit_id: str, *, identity: str | None = None
    ) -> list[dict[str, Any]]:
        """Return durable attempts in order, optionally for one run identity."""
        directory = self.run_dir(fit_id)
        records: list[dict[str, Any]] = []
        for number in self._attempts(fit_id):
            record = read_json(directory / f"attempt-{number}" / ATTEMPT_FILE)
            if record is not None and (
                identity is None or record.get("identity") == identity
            ):
                records.append(record)
        return records

    def finish_attempt(
        self,
        run: ActiveRun,
        state: RunState,
        reason: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one attempt's terminal outcome without replacing its log."""
        record = read_json(run.attempt_dir / ATTEMPT_FILE) or {}
        current_metadata = record.get("metadata", {})
        if not isinstance(current_metadata, dict):
            current_metadata = {}
        write_json_atomic(
            run.attempt_dir / ATTEMPT_FILE,
            {
                **record,
                "attempt": run.attempt,
                "identity": run.identity,
                "state": state.value,
                "reason": reason,
                "finished_at": _now(),
                "metadata": {**current_metadata, **(metadata or {})},
            },
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

    def start(
        self,
        fit_id: str,
        identity: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ActiveRun:
        """Open a new attempt; any earlier receipt is retired, never reused."""
        self.settle(fit_id)
        self._supersede(fit_id)
        attempts = self._attempts(fit_id)
        run = ActiveRun(self, fit_id, identity, (attempts[-1] + 1) if attempts else 1)
        run.attempt_dir.mkdir(parents=True)
        write_json_atomic(
            run.attempt_dir / ATTEMPT_FILE,
            {
                "attempt": run.attempt,
                "identity": identity,
                "state": RunState.RUNNING.value,
                "started_at": _now(),
                "metadata": metadata or {},
            },
        )
        self.write_state(run, RunState.RUNNING)
        return run

    def omit(self, fit_id: str, identity: str, reason: str) -> None:
        """Record that a planned fit was deliberately not run, and why."""
        attempts = self._attempts(fit_id)
        run = ActiveRun(self, fit_id, identity, attempts[-1] if attempts else 0)
        self.write_state(run, RunState.OMITTED, reason)
