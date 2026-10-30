"""Receipt-gated fixture evaluation uses manifests and labels, never pixels."""

import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.data import Manifest, ManifestEntry, write_manifest
from jute_disease.evaluation import evaluate_fixture_runs


def _target_manifest() -> Manifest:
    entries = tuple(
        sorted(
            (
                ManifestEntry(
                    image_id=f"{partition}-{label}",
                    path=f"unused/{partition}-{label}.jpg",
                    label=label,
                    partition=partition,
                    group=f"group-{partition}-{label}",
                    source="synthetic",
                    width=1,
                    height=1,
                )
                for partition in ("train", "val", "test")
                for label in ("a", "b")
            ),
            key=lambda entry: entry.image_id,
        )
    )
    return Manifest(
        seed=1,
        fractions=(("test", 1 / 3), ("train", 1 / 3), ("val", 1 / 3)),
        entries=entries,
        excluded=(),
        merged=(),
        quarantined=(),
        constraints=(),
        dependencies=(("fixture", "synthetic"),),
    )


def _write_run(
    root: Path,
    *,
    fit_id: str,
    identity: str,
    config_digest: str,
    manifest_digest: str,
    artifact_path: str = "predictions.json",
    image_ids: tuple[str, ...] = (),
    predictions: tuple[str, ...] = (),
    partition: str = "val",
    artifact_bytes: bytes | None = None,
) -> Path:
    receipt_path = root / fit_id / "receipt.json"
    receipt_path.parent.mkdir(parents=True)
    prediction_path = receipt_path.parent / artifact_path
    raw = artifact_bytes
    if raw is None:
        raw = json.dumps(
            {"image_ids": list(image_ids), "predictions": list(predictions)}
        ).encode()
        prediction_path.write_bytes(raw)
    receipt_path.write_text(
        json.dumps(
            {
                "fit_id": fit_id,
                "identity": identity,
                "results": {
                    "seed": int(fit_id.rsplit("-", 1)[1]),
                    "config_digest": config_digest,
                    "manifests": {"target": manifest_digest},
                    "partition": partition,
                },
                "artifacts": {
                    "predictions": {
                        "path": artifact_path,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return receipt_path


def test_missing_target_identity_is_rejected_before_prediction_read(
    tmp_path: Path,
) -> None:
    manifest = _target_manifest()
    target_dir = tmp_path / "target"
    write_manifest(manifest, target_dir)
    receipt = _write_run(
        tmp_path / "runs",
        fit_id="fixture-seed-1",
        identity="expected-run-1",
        config_digest="c" * 64,
        manifest_digest=manifest.digest(),
    )

    with pytest.raises(ValueError, match="target freeze identity is required"):
        evaluate_fixture_runs(
            target_dir=target_dir,
            target_freeze_identity=None,
            config_digest="c" * 64,
            expected_run_identities={"fixture-seed-1": "expected-run-1"},
            run_receipts=[receipt],
            output_dir=tmp_path / "out",
        )


def test_mismatched_target_identity_is_rejected_before_prediction_read(
    tmp_path: Path,
) -> None:
    manifest = _target_manifest()
    target_dir = tmp_path / "target"
    write_manifest(manifest, target_dir)
    receipt = _write_run(
        tmp_path / "runs",
        fit_id="fixture-seed-1",
        identity="expected-run-1",
        config_digest="c" * 64,
        manifest_digest=manifest.digest(),
    )

    with pytest.raises(ValueError, match="target freeze identity does not match"):
        evaluate_fixture_runs(
            target_dir=target_dir,
            target_freeze_identity="0" * 64,
            config_digest="c" * 64,
            expected_run_identities={"fixture-seed-1": "expected-run-1"},
            run_receipts=[receipt],
            output_dir=tmp_path / "out",
        )


def test_incompatible_target_receipt_is_rejected_before_prediction_read(
    tmp_path: Path,
) -> None:
    manifest = _target_manifest()
    target_dir = tmp_path / "target"
    write_manifest(manifest, target_dir)
    freeze_receipt_path = target_dir / "receipt.json"
    freeze_receipt = json.loads(freeze_receipt_path.read_text(encoding="utf-8"))
    freeze_receipt["partition_digest"] = "0" * 64
    freeze_receipt_path.write_text(json.dumps(freeze_receipt), encoding="utf-8")
    receipt = _write_run(
        tmp_path / "runs",
        fit_id="fixture-seed-1",
        identity="expected-run-1",
        config_digest="c" * 64,
        manifest_digest=manifest.digest(),
    )
    (receipt.parent / "predictions.json").unlink()

    with pytest.raises(ValueError, match="target freeze receipt is incompatible"):
        evaluate_fixture_runs(
            target_dir=target_dir,
            target_freeze_identity=manifest.digest(),
            config_digest="c" * 64,
            expected_run_identities={"fixture-seed-1": "expected-run-1"},
            run_receipts=[receipt],
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize(
    ("receipt_change", "message"),
    [
        ({"identity": "unexpected"}, "run identity does not match"),
        ({"config_digest": "d" * 64}, "run configuration does not match"),
        ({"manifest_digest": "e" * 64}, "target manifest does not match"),
        ({"partition": "test"}, "is not for the val partition"),
    ],
)
def test_incompatible_run_receipt_is_rejected_before_prediction_read(
    tmp_path: Path, receipt_change: dict[str, str], message: str
) -> None:
    manifest = _target_manifest()
    target_dir = tmp_path / "target"
    write_manifest(manifest, target_dir)
    run = {
        "identity": "expected-run-1",
        "config_digest": "c" * 64,
        "manifest_digest": manifest.digest(),
        "partition": "val",
    }
    run.update(receipt_change)
    receipt = _write_run(
        tmp_path / "runs",
        fit_id="fixture-seed-1",
        **run,
        artifact_bytes=None,
    )
    # Make a missing file deliberate: compatibility errors must win before any
    # attempt to read prediction bytes.
    (receipt.parent / "predictions.json").unlink()

    with pytest.raises(ValueError, match=message):
        evaluate_fixture_runs(
            target_dir=target_dir,
            target_freeze_identity=manifest.digest(),
            config_digest="c" * 64,
            expected_run_identities={"fixture-seed-1": "expected-run-1"},
            run_receipts=[receipt],
            output_dir=tmp_path / "out",
        )


def test_matching_fixture_receipts_save_deterministic_val_outputs(
    tmp_path: Path,
) -> None:
    manifest = _target_manifest()
    target_dir = tmp_path / "target"
    write_manifest(manifest, target_dir)
    image_ids = ("val-a", "val-b")
    first = _write_run(
        tmp_path / "runs",
        fit_id="fixture-seed-1",
        identity="run-identity-1",
        config_digest="c" * 64,
        manifest_digest=manifest.digest(),
        image_ids=image_ids,
        predictions=("a", "b"),
    )
    second = _write_run(
        tmp_path / "runs",
        fit_id="fixture-seed-2",
        identity="run-identity-2",
        config_digest="c" * 64,
        manifest_digest=manifest.digest(),
        image_ids=image_ids,
        predictions=("b", "b"),
    )
    expected = {
        "fixture-seed-1": "run-identity-1",
        "fixture-seed-2": "run-identity-2",
    }

    first_output = evaluate_fixture_runs(
        target_dir=target_dir,
        target_freeze_identity=manifest.digest(),
        config_digest="c" * 64,
        expected_run_identities=expected,
        run_receipts=[first, second],
        output_dir=tmp_path / "out-first",
    )
    second_output = evaluate_fixture_runs(
        target_dir=target_dir,
        target_freeze_identity=manifest.digest(),
        config_digest="c" * 64,
        expected_run_identities=expected,
        run_receipts=[second, first],
        output_dir=tmp_path / "out-second",
    )

    assert first_output.predictions_path.read_bytes() == (
        second_output.predictions_path.read_bytes()
    )
    assert (
        first_output.metrics_path.read_bytes()
        == second_output.metrics_path.read_bytes()
    )
    saved_predictions = json.loads(first_output.predictions_path.read_text())
    assert saved_predictions["image_ids"] == list(image_ids)
    assert saved_predictions["predictions"] == {"1": ["a", "b"], "2": ["b", "b"]}
    assert "test-a" not in saved_predictions["image_ids"]
    saved_metrics = json.loads(first_output.metrics_path.read_text())
    assert saved_metrics["metrics"]["overall"]["1"]["n"] == 2
    assert saved_metrics["metrics"]["overall"]["1"]["accuracy"] == 1.0
    assert saved_metrics["metrics"]["overall"]["2"]["accuracy"] == 0.5
