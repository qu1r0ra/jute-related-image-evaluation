"""The run record binds each fit's receipt, weights and predictions by digest."""

import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.evaluation import (
    FreezeRecordError,
    authorize_test_read,
    receipt_manifest_digest,
    training_description,
    training_receipt_digest,
    write_freeze_record,
    write_prediction_receipt,
    write_run_record,
)
from jute_disease.training import RunStore

from .conftest import audit_manifest_of

MANIFEST = "b" * 64
FIT = "cnn/mobilenet_v2_direct/pool-A/view-1/near-duplicate/split-0/seed-42"
ANALYSES = [
    "background-probe",
    "border-probe",
    "embeddings-test-pass",
    "gradcam-test-pass",
    "leaf-only",
    "scene-cells",
]


@pytest.fixture
def freeze(
    tmp_path: Path, target: Path, root: Path, policy_dir: Path, mask_audit_report: Path
) -> Path:
    spec = {
        "fit_ids": [FIT, f"{FIT}-leaf"],
        "analysis_ids": ANALYSES,
        "selections": {"letterbox": "letterbox"},
        "letterbox_decision": {"choice": "letterbox"},
        "diagnostic": {"pool": "A", "view": 1, "policy": "near-duplicate"},
        "border_probe": {
            "start": {"width_px": 32, "leaf_cutoff": 0.05},
            "final": {"width_px": 32, "leaf_cutoff": 0.05},
        },
        "budget": {},
    }
    outcomes = {
        "partition": "val",
        "letterbox_comparison": {"letterbox": 0.9, "resize": 0.8},
        "selections": {"letterbox": {"selected": "letterbox"}},
    }
    write_freeze_record(
        spec,
        outcomes,
        tmp_path / "freeze.json",
        root,
        target_dir=target,
        policy_dir=policy_dir,
        mask_audit_report=mask_audit_report,
        mask_audit_manifest=audit_manifest_of(mask_audit_report),
    )
    return tmp_path / "freeze.json"


def _issue(freeze: Path, runs: Path, fit_id: str = FIT) -> Path:
    """Write the receipt ``evaluate predict-test`` writes for a fit's predictions."""
    store = RunStore(runs)
    description, kind, artifact = training_description(store.receipt(fit_id), fit_id)
    run_dir = store.run_dir(fit_id)
    write_prediction_receipt(
        run_dir,
        record_hash=json.loads(freeze.read_text(encoding="utf-8"))["hash"],
        fit_id=fit_id,
        training_receipt_sha256=training_receipt_digest(run_dir),
        artifact_kind=kind,
        artifact_sha256=artifact["sha256"],
        fit_manifest_sha256=receipt_manifest_digest(description),
        predictions_sha256=hashlib.sha256(
            (runs / "predictions.json").read_bytes()
        ).hexdigest(),
    )
    return run_dir


def _complete_deep(store: RunStore, *, seed: int = 42) -> Path:
    active = store.start(FIT, "synthetic identity")
    checkpoint = active.attempt_dir / "checkpoint.ckpt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    active.complete(
        {
            "fit": {
                "fit_id": FIT,
                "system": "mobilenet_v2_direct",
                "seed": seed,
                "pool": "A",
                "view": 1,
                "policy": "near-duplicate",
                "split": 0,
                "seed_index": 0,
            },
            "metrics": {"val_macro_f1": 0.6},
            "lineage": {"manifests": {"train": MANIFEST}},
        },
        {"checkpoint": checkpoint},
        metadata={"assignment_table": {"sha256": "a" * 64}},
    )
    return checkpoint


def _run(fit_id: str = FIT, seed: int = 42, **extra: object) -> dict[str, object]:
    return {
        "fit_id": fit_id,
        "system": "mobilenet_v2_direct",
        "seed": seed,
        "predictions": "predictions.json",
        **extra,
    }


def test_run_record_binds_checkpoint_validation_and_assignment_receipt(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    checkpoint = _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    _issue(freeze, runs)

    record = write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")

    entry = record["body"]["fits"][FIT]
    assert entry["validation_macro_f1"] == 0.6
    assert (
        entry["checkpoint_sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert entry["assignment_table_sha256"] == "a" * 64
    assert entry["seed_index"] == 0
    receipt = (runs / FIT / "receipt.json").read_bytes()
    assert entry["training_receipt_sha256"] == hashlib.sha256(receipt).hexdigest()
    assert entry["lineage_fit_id"] == FIT

    with pytest.raises(FreezeRecordError, match="receipt seed"):
        write_run_record(
            freeze, [_run(seed=7)], runs, tmp_path / "mismatched-run-record.json"
        )


def test_a_record_written_outside_the_run_store_still_finds_the_receipts(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    _issue(freeze, runs)
    elsewhere = tmp_path / "somewhere" / "else" / "run-record.json"
    elsewhere.parent.mkdir(parents=True)

    record = write_run_record(freeze, [_run()], runs, elsewhere)

    assert record["body"]["fits"][FIT]["checkpoint_sha256"]
    assert json.loads(elsewhere.read_text(encoding="utf-8")) == record


def test_a_classical_receipt_binds_its_model(tmp_path: Path, freeze: Path) -> None:
    runs = tmp_path / "runs"
    active = RunStore(runs).start(FIT, "synthetic identity")
    model = active.attempt_dir / "model.joblib"
    model.write_bytes(b"synthetic model")
    active.complete(
        {
            "fit": FIT,
            "seed": 42,
            "pool": "A",
            "view": 1,
            "policy": "near-duplicate",
            "split": 0,
            "metrics": {"val_macro_f1": 0.4},
            "manifest": MANIFEST,
        },
        {"model": model},
    )
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    _issue(freeze, runs)

    record = write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")

    entry = record["body"]["fits"][FIT]
    assert entry["model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert entry["model"] == f"{FIT}/attempt-1/model.joblib"
    assert "checkpoint_sha256" not in entry
    assert entry["validation_macro_f1"] == 0.4
    assert entry["pool"] == "A"


def test_a_probe_variant_takes_the_lineage_of_the_fit_it_was_made_from(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    checkpoint = _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    _issue(freeze, runs)

    record = write_run_record(
        freeze,
        [_run(), _run(f"{FIT}-leaf", variant="leaf_only", of=FIT)],
        runs,
        tmp_path / "run-record.json",
    )

    leaf = record["body"]["fits"][f"{FIT}-leaf"]
    assert leaf["lineage_fit_id"] == FIT
    assert (
        leaf["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert (
        leaf["training_receipt_sha256"]
        == (record["body"]["fits"][FIT]["training_receipt_sha256"])
    )


def test_frozen_fits_missing_from_the_run_record_are_surfaced(
    tmp_path: Path, freeze: Path, root: Path, target: Path
) -> None:
    runs = tmp_path / "runs"
    _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    _issue(freeze, runs)
    run_record = tmp_path / "run-record.json"
    write_run_record(freeze, [_run()], runs, run_record)

    authorization = authorize_test_read(freeze, run_record, root, target_dir=target)

    assert [run.fit_id for run in authorization.runs] == [FIT]
    assert authorization.missing_fit_ids == (f"{FIT}-leaf",)


def test_a_fit_without_a_verified_receipt_is_refused(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    (runs).mkdir()
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    with pytest.raises(FreezeRecordError, match="no verified training receipt"):
        write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")

    checkpoint = _complete_deep(RunStore(runs))
    checkpoint.write_bytes(b"swapped after the receipt")
    with pytest.raises(FreezeRecordError, match="no verified training receipt"):
        write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")
    assert not (tmp_path / "run-record.json").exists()


def test_handwritten_predictions_without_a_gate_receipt_are_refused(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")

    with pytest.raises(FreezeRecordError, match="no prediction receipt"):
        write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")
    assert not (tmp_path / "run-record.json").exists()


def test_predictions_swapped_after_the_gate_are_refused(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    _issue(freeze, runs)
    (runs / "predictions.json").write_text('{"predictions":["x"]}', encoding="utf-8")

    with pytest.raises(FreezeRecordError, match="not the ones predict-test wrote"):
        write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")


def test_a_prediction_receipt_edited_after_the_gate_is_refused(
    tmp_path: Path, freeze: Path
) -> None:
    runs = tmp_path / "runs"
    _complete_deep(RunStore(runs))
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")
    run_dir = _issue(freeze, runs)
    receipt = next(run_dir.glob("predictions.*.receipt.json"))
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["body"]["training_receipt_sha256"] = "c" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FreezeRecordError, match="does not verify"):
        write_run_record(freeze, [_run()], runs, tmp_path / "run-record.json")
