"""Which checkpoint a restarted CNN fit may continue from, and when it must not."""

import json
from pathlib import Path

import pytest

from jute_disease.training import (
    mark_finished,
    prepare_checkpoints,
    record_progress,
    verified_progress,
)

IDENTITY = "identity-a"


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    return tmp_path / "run" / "checkpoints"


def _train(directory: Path, epochs: int = 2, attempt: str = "attempt-1") -> None:
    """What a child leaves behind after ``epochs`` epochs."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "last.ckpt").write_bytes(f"state-{epochs}".encode())
    record_progress(directory, attempt, epochs)


def _aside(directory: Path) -> list[str]:
    return sorted(p.name for p in (directory.parent / "superseded").glob("*"))


def test_a_first_attempt_starts_fresh_and_claims_the_directory(
    directory: Path,
) -> None:
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-1")
    assert (plan.checkpoint, plan.resumed_from, plan.refusal) == (None, None, None)
    owner = json.loads((directory / "owner.json").read_text("utf-8"))
    assert owner == {"identity": IDENTITY, "attempt": "attempt-1"}


def test_a_killed_attempt_resumes_from_its_verified_checkpoint(
    directory: Path,
) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    _train(directory)
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-2")
    assert plan.checkpoint == directory / "last.ckpt"
    assert plan.refusal is None
    assert plan.resumed_from is not None
    assert plan.resumed_from["attempt"] == "attempt-1"
    assert plan.resumed_from["epochs_completed"] == 2
    assert plan.resumed_from["checkpoint_sha256"]
    assert _aside(directory) == []


def test_a_directory_with_only_an_owner_is_a_fresh_start(directory: Path) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-2")
    assert (plan.checkpoint, plan.refusal) == (None, None)
    assert _aside(directory) == []


def test_a_different_identity_is_refused_and_moved_aside(directory: Path) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    _train(directory)
    plan = prepare_checkpoints(directory, "identity-b", "attempt-2")
    assert plan.checkpoint is None
    assert plan.refusal == "checkpoints belong to a different run identity"
    assert _aside(directory) == ["checkpoints-0"]
    assert not (directory / "last.ckpt").exists()
    owner = json.loads((directory / "owner.json").read_text("utf-8"))
    assert owner["identity"] == "identity-b"


def test_checkpoints_without_an_owner_are_refused(directory: Path) -> None:
    _train(directory)
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-1")
    assert plan.checkpoint is None
    assert plan.refusal == "checkpoints have no recorded owner"
    assert _aside(directory) == ["checkpoints-0"]


def test_a_checkpoint_whose_bytes_changed_is_refused(directory: Path) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    _train(directory)
    (directory / "last.ckpt").write_bytes(b"truncated")
    assert verified_progress(directory) is None
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-2")
    assert plan.checkpoint is None
    assert plan.refusal == "no complete checkpoint with its recorded bytes"
    assert _aside(directory) == ["checkpoints-0"]


def test_a_checkpoint_with_no_progress_record_is_refused(directory: Path) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    (directory / "last.ckpt").write_bytes(b"state")
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-2")
    assert plan.checkpoint is None
    assert plan.refusal == "no complete checkpoint with its recorded bytes"


def test_finished_training_is_never_resumed(directory: Path) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    _train(directory)
    mark_finished(directory, IDENTITY, "attempt-1")
    plan = prepare_checkpoints(directory, IDENTITY, "attempt-2")
    assert plan.checkpoint is None
    assert plan.refusal == "the earlier training had already finished"
    assert _aside(directory) == ["checkpoints-0"]


def test_each_refusal_keeps_its_own_aside_copy(directory: Path) -> None:
    prepare_checkpoints(directory, IDENTITY, "attempt-1")
    _train(directory, 1)
    prepare_checkpoints(directory, "identity-b", "attempt-2")
    _train(directory, 2, "attempt-2")
    prepare_checkpoints(directory, "identity-c", "attempt-3")
    assert _aside(directory) == ["checkpoints-0", "checkpoints-1"]


def test_progress_is_not_recorded_before_a_checkpoint_exists(
    directory: Path,
) -> None:
    directory.mkdir(parents=True)
    record_progress(directory, "attempt-1", 1)
    assert not (directory / "last.json").exists()
