"""The run record binds each fit's receipt, weights and predictions by digest."""

import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.evaluation import (
    FreezeRecordError,
    write_freeze_record,
    write_run_record,
)
from jute_disease.training import RunStore

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
    )
    return tmp_path / "freeze.json"


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
        },
        {"model": model},
    )
    (runs / "predictions.json").write_text('{"predictions":[]}', encoding="utf-8")

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
