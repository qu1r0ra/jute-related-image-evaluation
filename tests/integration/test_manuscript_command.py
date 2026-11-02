"""Manuscript fixture exports remain tied to synthetic frozen evidence."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from jute_disease.models.ml import classical_configurations
from jute_disease.reporting import export_manuscript, validate_figure_sources


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_synthetic_receipt(runs_root: Path, fit_id: str, identity: str) -> None:
    from jute_disease.training import RunStore

    runtime = {
        "cuda_available": True,
        "device": {"name": "NVIDIA A100 synthetic fixture"},
    }
    active = RunStore(runs_root).start(fit_id, identity)
    result_artifact = active.attempt_dir / "synthetic-result.json"
    result_artifact.write_text("{}\n", encoding="utf-8")
    metrics_artifact = active.attempt_dir / "metrics" / "version_0" / "metrics.csv"
    metrics_artifact.parent.mkdir(parents=True)
    metrics_artifact.write_text(
        "epoch,step,val_loss,val_acc,train_loss\n1,2,0.4,0.8,0.3\n",
        encoding="utf-8",
    )
    config_bytes = b"synthetic training configuration\n"
    active.complete(
        {
            "fit": {
                "kind": "target",
                "stage": "jute",
                "upstream": None,
                "system": "mobilenet_v2_direct",
            },
            "stack": runtime,
        },
        {"result": result_artifact, "metrics": metrics_artifact},
        metadata={
            "runtime": runtime,
            "configuration": {
                "path": "synthetic-benchmark.json",
                "sha256": hashlib.sha256(config_bytes).hexdigest(),
            },
        },
    )


def _write_evaluation_bundles(
    root: Path, configuration_sha256: str, manifest_digest: str
) -> tuple[Path, Path]:
    evaluation_dir = root / "evaluation"
    run_receipt_path = evaluation_dir / "run-receipts" / "seed-42.json"
    run_receipt_path.parent.mkdir(parents=True)
    run_identity = "synthetic-evaluation-run"
    run_receipt = {
        "fit_id": "evaluation/seed-42",
        "identity": run_identity,
        "results": {
            "config_digest": configuration_sha256,
            "manifests": {"target": manifest_digest},
            "partition": "val",
            "seed": 42,
        },
    }
    run_receipt_path.write_text(json.dumps(run_receipt), encoding="utf-8")
    metrics = {
        "config_digest": configuration_sha256,
        "manifest_digest": manifest_digest,
        "metrics": {
            "overall": {
                "summary": {
                    "accuracy": {"mean": 0.75, "n": 2},
                    "macro_f1": {"mean": 0.7, "sd": 0.05, "n": 2},
                }
            },
            "confusion": [[3, 1], [0, 4]],
        },
    }
    predictions = {
        "config_digest": configuration_sha256,
        "manifest_digest": manifest_digest,
        "run_identities": {"evaluation/seed-42": run_identity},
    }
    metrics_path = evaluation_dir / "metrics.json"
    predictions_path = evaluation_dir / "predictions.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    evaluation_receipt = {
        "schema_version": 1,
        "kind": "evaluation",
        "configuration_sha256": configuration_sha256,
        "manifest_digest": manifest_digest,
        "partition": "val",
        "artifacts": {
            "metrics": {
                "path": "metrics.json",
                "sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            },
            "predictions": {
                "path": "predictions.json",
                "sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            },
        },
        "run_receipts": [
            {
                "fit_id": "evaluation/seed-42",
                "identity": run_identity,
                "path": "run-receipts/seed-42.json",
                "sha256": hashlib.sha256(run_receipt_path.read_bytes()).hexdigest(),
            }
        ],
    }
    evaluation_receipt_path = evaluation_dir / "receipt.json"
    evaluation_receipt_path.write_text(json.dumps(evaluation_receipt), encoding="utf-8")

    statistics_dir = root / "statistics"
    statistics_dir.mkdir()
    statistics = {
        "config_digest": configuration_sha256,
        "manifest_digest": manifest_digest,
        "evaluation_receipt_sha256": hashlib.sha256(
            evaluation_receipt_path.read_bytes()
        ).hexdigest(),
        "statistics": {
            "macro_f1": {
                "route_a": {"estimate": 0.71, "lower": 0.65, "upper": 0.78},
                "resamples": 500,
            }
        },
    }
    statistics_path = statistics_dir / "bootstrap.json"
    statistics_path.write_text(json.dumps(statistics), encoding="utf-8")
    statistics_receipt = {
        "schema_version": 1,
        "kind": "statistics",
        "configuration_sha256": configuration_sha256,
        "manifest_digest": manifest_digest,
        "evaluation_receipt_sha256": statistics["evaluation_receipt_sha256"],
        "artifact": {
            "path": "bootstrap.json",
            "sha256": hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
        },
    }
    (statistics_dir / "receipt.json").write_text(
        json.dumps(statistics_receipt), encoding="utf-8"
    )
    return evaluation_dir, statistics_dir


def test_fixture_freezes_exports_rebuilds_and_checks_macros(tmp_path: Path) -> None:
    output = tmp_path / "manuscript-fixture"
    result = _run("manuscript", "fixture", "--output", str(output))

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["check"]["pass"] is True
    assert report["check"]["checked_macros"] > 0
    assert report["rebuild"]["verified_images"] > 0

    export_dir = output / "export"
    metadata = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
    macros = json.loads((export_dir / "macros.json").read_text(encoding="utf-8"))
    audit = json.loads((export_dir / "claim-audit.json").read_text(encoding="utf-8"))
    kit = json.loads((export_dir / "rebuild-kit.json").read_text(encoding="utf-8"))
    assert metadata["command"] == "jute-disease manuscript fixture"
    assert metadata["configuration"]["purpose"] == "synthetic fixture only"
    assert metadata["artifacts"]["manifest"]["manifest_digest"]
    assert metadata["outputs"]["source-metadata.json"]["sha256"]
    assert kit["entries"]
    assert all(
        {
            "source",
            "url",
            "version",
            "path",
            "sha256",
            "family_id",
            "split",
            "label",
        }
        <= entry.keys()
        for entry in kit["entries"]
    )
    assert len(macros) == 1 and report["export"]["export_id"] in macros
    assert (
        len(macros[report["export"]["export_id"]]) == report["check"]["checked_macros"]
    )
    for family in ("gnb", "knn", "lr", "rf", "svm"):
        assert macros[report["export"]["export_id"]][
            f"classical_tuning_{family}_configurations"
        ] == len(classical_configurations(family))
    assert audit["status"] == "pending"
    assert audit["owner_spot_check"]["status"] == "pending"
    assert audit["owner_signoff"]["status"] == "pending"
    assert audit["claim_lists"]["may_report"]
    assert audit["claim_lists"]["may_not_claim"]
    assert audit["claim_list_url"].endswith("#issuecomment-6545452441")
    assert audit["claim_audit_decision_url"].endswith("#issuecomment-6572818142")
    assert audit["reviewer"]["fresh_context"] is False
    pending_audit_check = _run("manuscript", "audit-check", str(export_dir))
    assert pending_audit_check.returncode != 0
    assert "claim audit is pending" in pending_audit_check.stderr

    audit["status"] = "complete"
    audit["reviewer"] = {
        "name": "synthetic reviewer",
        "provider": "synthetic provider",
        "fresh_context": True,
        "reviewed_at": "synthetic fixture",
        "record": "synthetic fixture",
    }
    audit["manuscript_sha256"] = "0" * 64
    (export_dir / "claim-audit.json").write_text(json.dumps(audit), encoding="utf-8")
    manuscript = tmp_path / "synthetic-manuscript.tex"
    manuscript.write_text("synthetic fixture\n", encoding="utf-8")
    stale_audit_check = _run(
        "manuscript",
        "audit-check",
        str(export_dir),
        "--manuscript",
        str(manuscript),
    )
    assert stale_audit_check.returncode != 0
    assert "SHA-256 does not match" in stale_audit_check.stderr
    audit["manuscript_sha256"] = []
    (export_dir / "claim-audit.json").write_text(json.dumps(audit), encoding="utf-8")
    malformed_audit_check = _run(
        "manuscript",
        "audit-check",
        str(export_dir),
        "--manuscript",
        str(manuscript),
    )
    assert malformed_audit_check.returncode != 0
    assert "must record the reviewed manuscript SHA-256" in malformed_audit_check.stderr
    assert "Traceback" not in malformed_audit_check.stderr

    runs_root = tmp_path / "synthetic-runs"
    fit_id = "cnn/mobilenet_v2_direct/seed-42"
    _write_synthetic_receipt(runs_root, fit_id, "synthetic-receipt-identity")
    config_value = json.loads(
        (output / "fixture-export-config.json").read_text(encoding="utf-8")
    )
    config_sha256 = hashlib.sha256(
        (json.dumps(config_value, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    evaluation_artifacts, statistics_artifacts = _write_evaluation_bundles(
        tmp_path,
        config_sha256,
        metadata["artifacts"]["manifest"]["manifest_digest"],
    )
    evaluation_receipt_path = evaluation_artifacts / "receipt.json"
    run_receipt_path = evaluation_artifacts / "run-receipts" / "seed-42.json"
    original_evaluation_receipt = evaluation_receipt_path.read_bytes()
    original_run_receipt = run_receipt_path.read_bytes()
    evaluation_receipt = json.loads(original_evaluation_receipt)
    run_receipt = json.loads(original_run_receipt)
    evaluation_receipt["partition"] = "test"
    run_receipt["results"]["partition"] = "test"
    run_receipt_path.write_text(json.dumps(run_receipt), encoding="utf-8")
    evaluation_receipt["run_receipts"][0]["sha256"] = hashlib.sha256(
        run_receipt_path.read_bytes()
    ).hexdigest()
    evaluation_receipt_path.write_text(json.dumps(evaluation_receipt), encoding="utf-8")
    try:
        export_manuscript(
            output / "freeze" / "target",
            output / "fixture-export-config.json",
            tmp_path / "test-partition-export",
            source_metadata=json.loads(
                (export_dir / "source-metadata.json").read_text(encoding="utf-8")
            ),
            evaluation_artifacts_path=evaluation_artifacts,
        )
    except ValueError as error:
        assert "does not match the export configuration" in str(error)
    else:
        raise AssertionError("test-partition evaluation reports should be rejected")
    evaluation_receipt_path.write_bytes(original_evaluation_receipt)
    run_receipt_path.write_bytes(original_run_receipt)
    evaluation_metrics = tmp_path / "synthetic-evaluation-summary.csv"
    evaluation_metrics.write_text(
        "Experiment,epoch,step,val_loss,val_acc,train_loss\n"
        "cnn/mobilenet_v2_direct/seed-42,1,2,0.4,0.8,0.3\n",
        encoding="utf-8",
    )
    held_out_metrics = tmp_path / "held-out-metrics.csv"
    held_out_metrics.write_text(
        "Experiment,val_loss,test_f1\ncnn/fixture,0.4,0.7\n", encoding="utf-8"
    )
    try:
        export_manuscript(
            output / "freeze" / "target",
            output / "fixture-export-config.json",
            tmp_path / "held-out-export",
            source_metadata=json.loads(
                (export_dir / "source-metadata.json").read_text(encoding="utf-8")
            ),
            evaluation_metrics_path=held_out_metrics,
        )
    except ValueError as error:
        assert "unsupported or held-out columns" in str(error)
    else:
        raise AssertionError("held-out evaluation metrics should be rejected")
    unverified_runs = tmp_path / "unverified-runs"
    unverified_runs.mkdir()
    try:
        export_manuscript(
            output / "freeze" / "target",
            output / "fixture-export-config.json",
            tmp_path / "unverified-metrics-export",
            source_metadata=json.loads(
                (export_dir / "source-metadata.json").read_text(encoding="utf-8")
            ),
            training_runs_dir=unverified_runs,
            evaluation_metrics_path=evaluation_metrics,
        )
    except ValueError as error:
        assert "no completed verified receipt" in str(error)
    else:
        raise AssertionError("unverified evaluation metrics should be rejected")
    cost_export_dir = tmp_path / "cost-export"
    cost_export = _run(
        "manuscript",
        "export",
        "--freeze-dir",
        str(output / "freeze" / "target"),
        "--config",
        str(output / "fixture-export-config.json"),
        "--output",
        str(cost_export_dir),
        "--source-metadata",
        str(export_dir / "source-metadata.json"),
        "--training-runs",
        str(runs_root),
        "--evaluation-metrics",
        str(evaluation_metrics),
        "--evaluation-artifacts",
        str(evaluation_artifacts),
        "--statistics-artifacts",
        str(statistics_artifacts),
    )
    assert cost_export.returncode == 0, cost_export.stderr
    cost_report = json.loads(cost_export.stdout)
    cost_values = cost_report["macros"][cost_report["export_id"]]
    assert cost_values["compute_core_fit_count"] == 1
    assert cost_values["compute_core_attempt_count"] == 1
    assert cost_values["compute_core_a100_gpu_hours"] >= 0
    assert cost_values["classical_tuning_gnb_configurations"] == 9
    assert cost_values["evaluation_cnn_mobilenet_v2_direct_seed_42_val_loss"] == 0.4
    assert cost_values["evaluation_cnn_mobilenet_v2_direct_seed_42_val_acc"] == 0.8
    assert cost_values["evaluation_overall_summary_macro_f1_mean"] == 0.7
    assert cost_values["evaluation_confusion_item_1_item_1"] == 4
    assert cost_values["statistics_macro_f1_route_a_lower"] == 0.65
    assert cost_values["statistics_macro_f1_resamples"] == 500
    cost_metadata = json.loads(
        (cost_export_dir / "export.json").read_text(encoding="utf-8")
    )
    assert cost_metadata["evaluation_metrics"]["path"] == str(
        evaluation_metrics.resolve()
    )
    assert cost_metadata["evaluation_metrics"]["sha256"]
    assert cost_metadata["evaluation_metrics"]["fits"][0]["receipt_sha256"]
    assert cost_metadata["evaluation_metrics"]["fits"][0]["configuration_sha256"]
    assert cost_metadata["evaluation_metrics"]["fits"][0]["metrics_sha256"]
    assert "evaluation-metrics.csv" in cost_metadata["outputs"]
    assert "evaluation/metrics.json" in cost_metadata["outputs"]
    assert "statistics/report.json" in cost_metadata["outputs"]
    assert cost_metadata["evaluation_artifacts"]["run_receipts"][0]["sha256"]
    assert cost_metadata["statistics_artifacts"]["evaluation_receipt_sha256"]
    cost_check = _run("manuscript", "check", str(cost_export_dir))
    assert cost_check.returncode == 0, cost_check.stderr
    evaluation_metrics_path = evaluation_artifacts / "metrics.json"
    original_evaluation_metrics = evaluation_metrics_path.read_bytes()
    altered_evaluation_metrics = json.loads(original_evaluation_metrics)
    altered_evaluation_metrics["metrics"]["overall"]["summary"]["macro_f1"]["mean"] = (
        0.71
    )
    evaluation_metrics_path.write_text(
        json.dumps(altered_evaluation_metrics), encoding="utf-8"
    )
    altered_evaluation_check = _run("manuscript", "check", str(cost_export_dir))
    assert altered_evaluation_check.returncode != 0
    assert "does not match its receipt SHA-256" in altered_evaluation_check.stderr
    evaluation_metrics_path.write_bytes(original_evaluation_metrics)
    evaluation_metrics.write_text(
        evaluation_metrics.read_text(encoding="utf-8").replace("0.4", "0.5"),
        encoding="utf-8",
    )
    changed_metrics_check = _run("manuscript", "check", str(cost_export_dir))
    assert changed_metrics_check.returncode != 0
    assert "does not match its verified run" in changed_metrics_check.stderr

    alternate_runs = tmp_path / "alternate-synthetic-runs"
    _write_synthetic_receipt(alternate_runs, fit_id, "alternate-receipt-identity")
    cost_metadata = json.loads(
        (cost_export_dir / "export.json").read_text(encoding="utf-8")
    )
    alternate_export = export_manuscript(
        output / "freeze" / "target",
        output / "fixture-export-config.json",
        tmp_path / "alternate-cost-export",
        source_metadata=json.loads(
            (export_dir / "source-metadata.json").read_text(encoding="utf-8")
        ),
        training_runs_dir=alternate_runs,
        command=cost_metadata["command"],
    )
    assert alternate_export["export_id"] != cost_report["export_id"]
    alternate_check = _run(
        "manuscript", "check", str(tmp_path / "alternate-cost-export")
    )
    assert alternate_check.returncode == 0, alternate_check.stderr

    rebuild_script = Path(__file__).parents[2] / "scripts" / "rebuild_splits.py"
    kit_entries = kit["entries"]
    synthetic_test_entry = next(
        entry for entry in kit_entries if entry["split"] == "test"
    )
    test_image_path = output / "images" / synthetic_test_entry["path"]
    test_image_path.unlink()
    rebuilt = subprocess.run(
        [
            sys.executable,
            str(rebuild_script),
            "--kit-manifest",
            str(export_dir / "rebuild-kit.json"),
            "--image-root",
            str(output / "images"),
            "--output",
            str(tmp_path / "script-rebuilt-splits.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rebuilt.returncode == 0, rebuilt.stderr
    assert not test_image_path.exists()
    rebuilt_report = json.loads(rebuilt.stdout)
    assert rebuilt_report["split_digest"] == report["rebuild"]["split_digest"]
    assert rebuilt_report["skipped_test_images"] == sum(
        entry["split"] == "test" for entry in kit_entries
    )

    malformed_kit = tmp_path / "malformed-rebuild-kit.json"
    malformed_kit.write_text(
        json.dumps(
            {
                "version": 1,
                "seed": 42,
                "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
                "entries": [None],
            }
        ),
        encoding="utf-8",
    )
    malformed_rebuild = subprocess.run(
        [
            sys.executable,
            str(rebuild_script),
            "--kit-manifest",
            str(malformed_kit),
            "--image-root",
            str(output / "images"),
            "--output",
            str(tmp_path / "malformed-rebuilt-splits.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert malformed_rebuild.returncode != 0
    assert "entry 0 is malformed" in malformed_rebuild.stderr
    assert "Traceback" not in malformed_rebuild.stderr

    macros[report["export"]["export_id"]][
        next(iter(macros[report["export"]["export_id"]]))
    ] += 1
    (export_dir / "macros.json").write_text(json.dumps(macros), encoding="utf-8")
    mismatch = _run("manuscript", "check", str(export_dir))
    assert mismatch.returncode != 0
    assert "macro JSON values" in mismatch.stderr


def test_figure_sources_require_allowed_license_and_attribution() -> None:
    validate_figure_sources(
        [
            {
                "source": "Example",
                "url": "https://example.invalid/image",
                "license": "CC BY",
                "attribution": "Author",
            }
        ]
    )
    try:
        validate_figure_sources(
            [
                {
                    "source": "Example",
                    "url": "https://example.invalid/image",
                    "license": "All rights reserved",
                    "attribution": "Author",
                }
            ]
        )
    except ValueError as error:
        assert "unsupported license" in str(error)
    else:
        raise AssertionError("unsupported figure license should be rejected")
    try:
        validate_figure_sources(
            [
                {
                    "source": "Example",
                    "url": "https://example.invalid/image",
                    "license": ["CC BY"],
                    "attribution": "Author",
                }
            ]
        )
    except ValueError as error:
        assert "unsupported license" in str(error)
    else:
        raise AssertionError("unhashable figure license should be rejected")
