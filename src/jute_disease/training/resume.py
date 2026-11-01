"""Mid-fit resume: which checkpoint a restarted CNN fit may continue from.

A CNN fit keeps its checkpoints in one run-level directory, shared by its
attempts. The parent process records there which run identity owns them
(``owner.json``); the child records after every epoch the hash of the full
training state it just saved (``last.json``). A restarted attempt continues from
``last.ckpt`` only when the owner identity equals the identity it is about to run
and the file still has the recorded bytes. Anything else is refused: the old
checkpoints are moved aside, never reused, and the fit starts fresh.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jute_disease.training.receipts import SUPERSEDED_DIR, read_json, write_json_atomic

CHECKPOINT_DIR = "checkpoints"
LAST_CHECKPOINT = "last.ckpt"
OWNER_FILE = "owner.json"
PROGRESS_FILE = "last.json"
RESUME_FILE = "resume.json"


@dataclass(frozen=True)
class ResumePlan:
    """What the next attempt may start from, and why not when it may not.

    ``resumed_from`` names the attempt, epochs and bytes of the state continued;
    ``refusal`` is set when checkpoints existed but were not compatible.
    """

    checkpoint: Path | None = None
    resumed_from: dict[str, Any] | None = None
    refusal: str | None = None


def _sha256(path: Path) -> str:
    from jute_disease.data import file_sha256

    return file_sha256(path)


def _move_aside(directory: Path) -> None:
    target = directory.parent / SUPERSEDED_DIR
    target.mkdir(exist_ok=True)
    number = len(list(target.glob(f"{directory.name}-*")))
    os.replace(directory, target / f"{directory.name}-{number}")


def record_progress(directory: Path | str, attempt: str, epochs_completed: int) -> None:
    """Record the saved training state after an epoch (called by the child)."""
    directory = Path(directory)
    last = directory / LAST_CHECKPOINT
    if last.is_file():
        write_json_atomic(
            directory / PROGRESS_FILE,
            {
                "sha256": _sha256(last),
                "epochs_completed": epochs_completed,
                "attempt": attempt,
            },
        )


def verified_progress(directory: Path | str) -> dict[str, Any] | None:
    """The recorded progress when ``last.ckpt`` still has the recorded bytes."""
    directory = Path(directory)
    progress = read_json(directory / PROGRESS_FILE)
    last = directory / LAST_CHECKPOINT
    if (
        progress is None
        or not last.is_file()
        or _sha256(last) != progress.get("sha256")
    ):
        return None
    return {
        "attempt": progress["attempt"],
        "epochs_completed": progress["epochs_completed"],
        "checkpoint_sha256": progress["sha256"],
    }


def prepare_checkpoints(
    directory: Path | str, identity: str, attempt: str
) -> ResumePlan:
    """Decide whether the attempt resumes, and claim the directory for ``identity``.

    Called by the parent after the attempt is opened and before the child starts.
    """
    directory = Path(directory)
    plan = ResumePlan()
    if directory.exists() and any(
        child.name != OWNER_FILE for child in directory.iterdir()
    ):
        owner = read_json(directory / OWNER_FILE) or {}
        progress = verified_progress(directory)
        if owner.get("identity") != identity:
            plan = ResumePlan(
                refusal="checkpoints belong to a different run identity"
                if owner
                else "checkpoints have no recorded owner"
            )
        elif owner.get("finished"):
            plan = ResumePlan(refusal="the earlier training had already finished")
        elif progress is None:
            plan = ResumePlan(refusal="no complete checkpoint with its recorded bytes")
        else:
            plan = ResumePlan(
                checkpoint=directory / LAST_CHECKPOINT, resumed_from=progress
            )
        if plan.checkpoint is None:
            _move_aside(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        directory / OWNER_FILE, {"identity": identity, "attempt": attempt}
    )
    return plan


def mark_finished(directory: Path | str, identity: str, attempt: str) -> None:
    """Record that training ended, so its checkpoints are never resumed."""
    write_json_atomic(
        Path(directory) / OWNER_FILE,
        {"identity": identity, "attempt": attempt, "finished": True},
    )
