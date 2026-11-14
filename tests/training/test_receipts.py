"""Atomic run states, compatible resume and partial-output rejection."""

import json
from pathlib import Path

import pytest

from jute_disease.data import file_sha256
from jute_disease.training import (
    CompletedFitError,
    FitLockedError,
    Resume,
    RunState,
    RunStore,
    write_json_atomic,
)
from jute_disease.training.receipts import RECEIPT_FORMAT_VERSION


def _complete(store: RunStore, fit_id: str, identity: str) -> Path:
    run = store.start(fit_id, identity)
    artifact = run.attempt_dir / "model.bin"
    artifact.write_bytes(b"weights")
    run.complete({"metric": 1.0}, artifacts={"model": artifact})
    return artifact


def test_atomic_write_leaves_no_partial_file(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_failed_atomic_write_keeps_previous_content(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    write_json_atomic(target, {"a": 1})
    with pytest.raises(TypeError):
        write_json_atomic(target, {"a": object()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_a_new_run_is_planned_then_running_then_completed(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    assert store.state("cnn/a") is RunState.PLANNED
    run = store.start("cnn/a", "id-1")
    assert store.state("cnn/a") is RunState.RUNNING
    artifact = run.attempt_dir / "model.bin"
    artifact.write_bytes(b"weights")
    run.complete(
        {"metric": 1.0},
        artifacts={"model": artifact},
        metadata={"official": False},
    )
    assert store.state("cnn/a") is RunState.COMPLETED
    receipt = store.receipt("cnn/a")
    assert receipt["format_version"] == RECEIPT_FORMAT_VERSION
    assert receipt["identity"] == "id-1"
    assert receipt["metadata"] == {"official": False}
    assert [attempt["attempt"] for attempt in receipt["attempts"]] == [1]
    assert [attempt["identity"] for attempt in receipt["attempts"]] == ["id-1"]
    assert [attempt["state"] for attempt in receipt["attempts"]] == ["completed"]
    assert receipt["attempts"][0]["metadata"] == {"official": False}
    assert receipt["artifacts"]["model"]["sha256"] == file_sha256(artifact)


def test_completed_matching_run_is_skipped(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    assert store.resolve("cnn/a", "id-1") is Resume.SKIP


def test_matching_receipt_with_noncompleted_state_is_not_reused(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    write_json_atomic(
        tmp_path / "cnn" / "a" / "state.json",
        {"state": "failed", "identity": "id-1", "attempt": 1},
    )
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH


def test_incompatible_identity_is_never_reused(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    assert store.resolve("cnn/a", "id-2") is Resume.RUN_FRESH
    store.start("cnn/a", "id-2")
    assert store.state("cnn/a") is RunState.RUNNING
    assert store.receipt("cnn/a", missing_ok=True) is None
    superseded = list((tmp_path / "cnn" / "a").glob("superseded/*.json"))
    assert len(superseded) == 1


def test_modified_artifact_invalidates_the_receipt(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    artifact = _complete(store, "cnn/a", "id-1")
    artifact.write_bytes(b"tampered")
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH


def test_missing_artifact_invalidates_the_receipt(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    artifact = _complete(store, "cnn/a", "id-1")
    artifact.unlink()
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH


def test_interrupted_run_is_partial_and_restarts_in_a_new_attempt(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    first = store.start("cnn/a", "id-1")
    (first.attempt_dir / "partial.ckpt").write_bytes(b"half")
    # The process died: state still says running and no receipt exists.
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH
    second = store.start("cnn/a", "id-1")
    assert second.attempt_dir != first.attempt_dir
    assert (first.attempt_dir / "partial.ckpt").exists()
    assert store.state("cnn/a") is RunState.RUNNING
    assert [attempt["state"] for attempt in store.attempts("cnn/a")] == [
        "interrupted",
        "running",
    ]


def test_stale_running_state_is_reported_interrupted(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.start("cnn/a", "id-1")
    assert store.settle("cnn/a") is RunState.INTERRUPTED
    assert store.state("cnn/a") is RunState.INTERRUPTED


def test_settling_an_interrupted_run_preserves_runtime_observations(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    run = store.start("cnn/a", "id-1", metadata={"official": True})
    runtime = {"torch": "2.6.0", "cuda_driver": "570.0"}
    write_json_atomic(run.attempt_dir / "runtime.json", runtime)

    store.settle("cnn/a")

    attempts = store.attempts("cnn/a")
    assert attempts[0]["state"] == "interrupted"
    assert attempts[0]["metadata"] == {"official": True, "runtime": runtime}


def test_half_written_receipt_is_incomplete(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    receipt = tmp_path / "cnn" / "a" / "receipt.json"
    receipt.write_text('{"identity": "id-1", "artifa', encoding="utf-8")
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH
    assert store.receipt("cnn/a", missing_ok=True) is None


def test_failure_states_are_preserved_with_a_reason(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run = store.start("classical/a", "id-1")
    run.fail(RunState.TIMED_OUT, "exceeded 1800 s")
    assert store.state("classical/a") is RunState.TIMED_OUT
    assert store.reason("classical/a") == "exceeded 1800 s"
    assert store.resolve("classical/a", "id-1") is Resume.RUN_FRESH


def test_complete_state_without_receipt_is_not_complete(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run = store.start("cnn/a", "id-1")
    write_json_atomic(
        tmp_path / "cnn" / "a" / "state.json",
        {"state": "completed", "identity": "id-1", "attempt": run.attempt},
    )
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH


def test_omitted_runs_record_why(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.omit("cnn/b", "id-9", "does not fit before the finalization reservation")
    assert store.state("cnn/b") is RunState.OMITTED
    assert "finalization" in store.reason("cnn/b")


def test_omit_refuses_to_overwrite_a_completed_fit(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    with pytest.raises(CompletedFitError, match="cnn/a"):
        store.omit("cnn/a", "", "upstream did not complete")
    assert store.state("cnn/a") is RunState.COMPLETED
    assert store.resolve("cnn/a", "id-1") is Resume.SKIP


def test_omit_may_replace_a_fit_whose_receipt_is_no_longer_valid(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1").write_bytes(b"culled")
    store.omit("cnn/a", "", "inputs unavailable")
    assert store.state("cnn/a") is RunState.OMITTED


def test_a_second_writer_to_a_fit_is_refused_until_the_first_leaves(
    tmp_path: Path,
) -> None:
    first, second = RunStore(tmp_path), RunStore(tmp_path)
    with first.exclusive("cnn/a"):
        with first.exclusive("cnn/a"):  # re-entering through the holder is allowed
            pass
        with pytest.raises(FitLockedError, match="cnn/a"):
            second.start("cnn/a", "id-1")
        with pytest.raises(FitLockedError):
            second.omit("cnn/a", "", "upstream did not complete")
        with second.exclusive("cnn/b"):  # other fits are unaffected
            pass
    assert second.state("cnn/a") is RunState.PLANNED
    second.start("cnn/a", "id-1")
    assert second.state("cnn/a") is RunState.RUNNING


def test_receipt_from_an_older_format_is_not_reusable(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    receipt_path = tmp_path / "cnn" / "a" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["format_version"] = RECEIPT_FORMAT_VERSION - 1
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert store.receipt("cnn/a", missing_ok=True) is None
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH


def test_receipt_with_incomplete_attempt_history_is_not_reusable(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    _complete(store, "cnn/a", "id-1")
    receipt_path = tmp_path / "cnn" / "a" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["attempts"] = []
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    assert store.receipt("cnn/a", missing_ok=True) is None
    assert store.resolve("cnn/a", "id-1") is Resume.RUN_FRESH
