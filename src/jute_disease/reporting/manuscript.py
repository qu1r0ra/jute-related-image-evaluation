"""Reproducible manuscript exports and their evidence checks."""

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any

from jute_disease.data import (
    PARTITIONS,
    load_manifest,
    rebuild_partition_assignments,
    validate_partition_fractions,
)
from jute_disease.data.manifest import load_complete_manifest

CLAIM_LIST_URL = (
    "https://github.com/qu1r0ra/jute-related-image-evaluation/issues/108"
    "#issuecomment-6545452441"
)
CLAIM_AUDIT_DECISION_URL = (
    "https://github.com/qu1r0ra/jute-related-image-evaluation/issues/108"
    "#issuecomment-6572818142"
)
_CLAIM_LISTS = {
    "may_report": (
        "The grouping rule, its threshold and the leakage risk that remains.",
        "Macro-F1 differences between policies for this pool and this rule, "
        "with hierarchical bootstrap intervals.",
        "Route rankings, conditional on MobileNetV2, its fixed source "
        "checkpoints and this pool.",
        "Grad-CAM as a prespecified qualitative diagnostic.",
        "Descriptive t-SNE/UMAP plots at fixed, recorded settings.",
        "Background probes, bounded to how much label information the "
        "background carries in this pool (see #106).",
    ),
    "may_not_claim": (
        "Leakage-free or zero leakage. The manuscript says 'we did not identify' "
        "instead.",
        "Plant- or session-independent splits.",
        "Field, deployment or new-acquisition generalization.",
        "Rankings that extend beyond MobileNetV2.",
        '"The model attends to the lesion."',
        "Meaning from t-SNE/UMAP cluster sizes or distances, or their use "
        "as evidence of performance or leakage.",
        "Family-disjoint scores presented as the true field estimate.",
    ),
}
ALLOWED_FIGURE_LICENSES = {"CC0", "CC BY", "CC BY-SA"}
_BASE_EXPORT_OUTPUTS = (
    "configuration.json",
    "source-metadata.json",
    "macros.json",
    "macros.tex",
    "support.csv",
    "support.png",
    "figure-sources.json",
    "compute-report.json",
    "compute.csv",
    "classical-tuning-budget.csv",
    "rebuild-kit.json",
)
_EXPORT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _label_key(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def _support_values(manifest: Any) -> dict[str, int | float]:
    support = manifest.support()
    return {
        f"{partition}_{_label_key(label)}_n": support[partition].get(label, 0)
        for partition in PARTITIONS
        for label in manifest.labels
    }


def _macro_tex(export_id: str, values: dict[str, int | float]) -> str:
    rows = [
        f"\\expandafter\\def\\csname juteExport@{export_id}@{key}\\endcsname"
        f"{{{format(value, 'g') if isinstance(value, float) else value}}}"
        for key, value in sorted(values.items())
    ]
    return "\n".join(rows) + "\n"


def _classical_tuning_budget() -> dict[str, int]:
    from jute_disease.models.ml import classical_configurations

    return {
        family: len(classical_configurations(family))
        for family in ("gnb", "knn", "lr", "rf", "svm")
    }


def _classical_tuning_values() -> dict[str, int | float]:
    return {
        f"classical_tuning_{family}_configurations": count
        for family, count in _classical_tuning_budget().items()
    }


def _fit_stage(fit: dict[str, Any], fit_id: str) -> str | None:
    kind = fit.get("kind")
    if kind == "source":
        stage = fit.get("stage")
        if isinstance(stage, str) and stage.strip():
            return _label_key(f"source_{stage}")
        raise ValueError(f"training receipt {fit_id} has no source stage")
    if kind == "target":
        upstream = fit.get("upstream")
        if upstream is None:
            return "core"
        parts = str(upstream).split("/")
        if len(parts) >= 2 and parts[0] == "source":
            return _label_key(parts[1])
        raise ValueError(f"training receipt {fit_id} has an unknown source stage")
    return None


def _parse_timestamp(value: object, fit_id: str, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"training receipt {fit_id} is missing {field}")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"training receipt {fit_id} has an invalid {field} timestamp"
        ) from error
    if timestamp.tzinfo is None:
        raise ValueError(f"training receipt {fit_id} has a timezone-free {field}")
    return timestamp


def _parse_audit_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"claim audit requires a {field} timestamp")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"claim audit has an invalid {field} timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"claim audit {field} timestamp must include a timezone")
    return timestamp


def _compute_report(runs_dir: Path) -> dict[str, Any]:
    """Measure A100 training time from verified completion receipts."""
    from jute_disease.training import RunStore

    runs_dir = Path(runs_dir).resolve()
    if not runs_dir.is_dir():
        raise ValueError(f"training receipts directory does not exist: {runs_dir}")
    store = RunStore(runs_dir)
    stages: dict[str, dict[str, float | int]] = {}
    phases: dict[str, dict[str, float | int]] = {}
    receipt_records: list[dict[str, str]] = []
    for receipt_path in sorted(runs_dir.rglob("receipt.json")):
        fit_id = receipt_path.parent.relative_to(runs_dir).as_posix()
        receipt = store.receipt(fit_id, missing_ok=True)
        if receipt is None:
            raise ValueError(f"training receipt is incomplete or invalid: {fit_id}")
        results = receipt.get("results", {})
        fit = results.get("fit", {}) if isinstance(results, dict) else {}
        if not isinstance(fit, dict):
            raise ValueError(f"training receipt {fit_id} has no fit metadata")
        stage = _fit_stage(fit, fit_id)
        if stage is None:
            continue
        phase_name = "search" if fit.get("phase") == "tune" else "training"
        receipt_records.append({"fit_id": fit_id, "sha256": _sha256_file(receipt_path)})
        summary = stages.setdefault(
            stage, {"fit_count": 0, "attempt_count": 0, "seconds": 0.0}
        )
        phase_summary = phases.setdefault(
            phase_name, {"fit_count": 0, "attempt_count": 0, "seconds": 0.0}
        )
        summary["fit_count"] += 1
        phase_summary["fit_count"] += 1
        for attempt in receipt["attempts"]:
            attempt_metadata = attempt.get("metadata", {})
            runtime = (
                attempt_metadata.get("runtime")
                if isinstance(attempt_metadata, dict)
                else None
            )
            device = runtime.get("device") if isinstance(runtime, dict) else None
            device_name = device.get("name") if isinstance(device, dict) else None
            if (
                not isinstance(device_name, str)
                or "A100" not in device_name
                or runtime.get("cuda_available") is not True
            ):
                raise ValueError(
                    f"training receipt {fit_id} attempt {attempt.get('attempt')} "
                    "does not record an A100 GPU"
                )
            started = _parse_timestamp(attempt.get("started_at"), fit_id, "started_at")
            finished = _parse_timestamp(
                attempt.get("finished_at"), fit_id, "finished_at"
            )
            elapsed = (finished - started).total_seconds()
            if elapsed < 0:
                raise ValueError(f"training receipt {fit_id} has negative elapsed time")
            summary["seconds"] += elapsed
            summary["attempt_count"] += 1
            phase_summary["seconds"] += elapsed
            phase_summary["attempt_count"] += 1
    if not receipt_records:
        raise ValueError("no verified A100 training receipts were found")
    stage_rows = [
        {
            "stage": stage,
            "fit_count": int(values["fit_count"]),
            "attempt_count": int(values["attempt_count"]),
            "a100_gpu_hours": round(float(values["seconds"]) / 3600, 6),
        }
        for stage, values in sorted(stages.items())
    ]
    phase_rows = [
        {
            "phase": phase,
            "fit_count": int(values["fit_count"]),
            "attempt_count": int(values["attempt_count"]),
            "a100_gpu_hours": round(float(values["seconds"]) / 3600, 6),
        }
        for phase, values in sorted(phases.items())
    ]
    return {
        "status": "measured",
        "runs_root": str(runs_dir),
        "receipts": receipt_records,
        "stages": stage_rows,
        "phases": phase_rows,
    }


def compute_training_costs(runs_dir: Path | str) -> dict[str, Any]:
    """Report receipt-bound GPU time, search attempts and classical search costs."""
    from jute_disease.training import RunStore

    root = Path(runs_dir).resolve()
    report = _compute_report(root)
    store = RunStore(root)
    classical_fits = 0
    classical_attempts = 0
    classical_seconds = 0.0
    classical_receipts: list[dict[str, str]] = []
    for receipt_path in sorted(root.rglob("receipt.json")):
        fit_id = receipt_path.parent.relative_to(root).as_posix()
        receipt = store.receipt(fit_id, missing_ok=True)
        if receipt is None:
            raise ValueError(f"training receipt is incomplete or invalid: {fit_id}")
        result = receipt.get("results", {})
        fit = result.get("fit", {}) if isinstance(result, dict) else {}
        if (
            not isinstance(fit, dict)
            or fit.get("kind") != "classical"
            or fit.get("phase") != "tune"
        ):
            continue
        classical_fits += 1
        classical_receipts.append(
            {"fit_id": fit_id, "sha256": _sha256_file(receipt_path)}
        )
        for attempt in receipt["attempts"]:
            started = _parse_timestamp(attempt.get("started_at"), fit_id, "started_at")
            finished = _parse_timestamp(
                attempt.get("finished_at"), fit_id, "finished_at"
            )
            elapsed = (finished - started).total_seconds()
            if elapsed < 0:
                raise ValueError(f"training receipt {fit_id} has negative elapsed time")
            classical_seconds += elapsed
            classical_attempts += 1
    report["classical_search"] = {
        "fit_count": classical_fits,
        "attempt_count": classical_attempts,
        "attempt_seconds": round(classical_seconds, 3),
        "configurations_per_fit": _classical_tuning_budget(),
        "receipts": classical_receipts,
        "source": "verified completed training receipts",
    }
    return report


def _pending_compute_report() -> dict[str, Any]:
    return {"status": "pending", "runs_root": None, "receipts": [], "stages": []}


def _compute_values(report: dict[str, Any]) -> dict[str, int | float]:
    values: dict[str, int | float] = {}
    for row in report["stages"]:
        stage = row["stage"]
        values[f"compute_{stage}_fit_count"] = row["fit_count"]
        values[f"compute_{stage}_attempt_count"] = row["attempt_count"]
        values[f"compute_{stage}_a100_gpu_hours"] = row["a100_gpu_hours"]
    return values


def _compute_csv(report: dict[str, Any]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=("stage", "fit_count", "attempt_count", "a100_gpu_hours"),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in report["stages"]:
        writer.writerow(row)
    return stream.getvalue()


def _classical_tuning_csv() -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=("family", "configurations"), lineterminator="\n"
    )
    writer.writeheader()
    for family, configurations in _classical_tuning_budget().items():
        writer.writerow({"family": family, "configurations": configurations})
    return stream.getvalue()


def _evaluation_metrics(
    path: Path | None,
    runs_dir: Path | None,
) -> tuple[dict[str, int | float], dict[str, Any] | None, bytes | None]:
    """Verify an aggregate against metrics artifacts covered by run receipts."""
    if path is None:
        return {}, None, None
    path = Path(path).resolve()
    raw = path.read_bytes()
    try:
        reader = csv.DictReader(StringIO(raw.decode("utf-8-sig"), newline=""))
        columns = reader.fieldnames
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("evaluation metrics must be a UTF-8 CSV") from error
    if (
        not columns
        or len(columns) != len(set(columns))
        or "Experiment" not in columns
        or len(columns) < 2
    ):
        raise ValueError("evaluation metrics require unique columns and Experiment")
    if not rows:
        raise ValueError("evaluation metrics aggregate contains no experiments")
    metric_columns = [column for column in columns if column != "Experiment"]
    if not any(column.startswith("val_") for column in metric_columns):
        raise ValueError("evaluation metrics must include validation metrics")
    invalid_columns = [
        column
        for column in metric_columns
        if not (column.startswith(("val_", "train_")) or column in {"epoch", "step"})
    ]
    if invalid_columns:
        raise ValueError(
            "evaluation metrics contain unsupported or held-out columns: "
            + ", ".join(invalid_columns)
        )
    values: dict[str, int | float] = {}
    experiments: set[str] = set()
    experiment_keys: dict[str, str] = {}
    experiment_order: list[str] = []
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"evaluation metrics row {row_number} has extra fields")
        if any(row.get(column) is None for column in columns):
            raise ValueError(f"evaluation metrics row {row_number} is incomplete")
        experiment = row.get("Experiment")
        if not isinstance(experiment, str) or not experiment.strip():
            raise ValueError(f"evaluation metrics row {row_number} has no Experiment")
        if experiment in experiments:
            raise ValueError(f"evaluation metrics repeat Experiment {experiment!r}")
        experiments.add(experiment)
        experiment_order.append(experiment)
        experiment_key = _label_key(experiment)
        if not experiment_key:
            raise ValueError(f"evaluation metrics Experiment {experiment!r} is invalid")
        if experiment_key in experiment_keys:
            raise ValueError(
                "evaluation metrics Experiment names produce the same macro key: "
                f"{experiment_keys[experiment_key]!r} and {experiment!r}"
            )
        experiment_keys[experiment_key] = experiment
        validation_value_seen = False
        for column in metric_columns:
            raw_value = row.get(column)
            if raw_value is None or not raw_value.strip():
                continue
            try:
                number = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"evaluation metrics row {row_number} column {column!r} "
                    "is not numeric"
                ) from error
            if not math.isfinite(number):
                raise ValueError(
                    f"evaluation metrics row {row_number} column {column!r} "
                    "is not finite"
                )
            if column.startswith("val_"):
                validation_value_seen = True
            metric_key = _label_key(column)
            key = f"evaluation_{experiment_key}_{metric_key}"
            if key in values:
                raise ValueError(f"evaluation metrics produce duplicate macro {key}")
            values[key] = number
        if not validation_value_seen:
            raise ValueError(
                f"evaluation metrics row {row_number} has no validation values"
            )

    if runs_dir is None:
        raise ValueError(
            "evaluation metrics require a training runs directory with "
            "verified receipts"
        )
    runs_root = Path(runs_dir).resolve()
    if not runs_root.is_dir():
        raise ValueError(f"training receipts directory does not exist: {runs_root}")
    from jute_disease.reporting import aggregate_metric_rows, latest_metrics_path
    from jute_disease.training import RunState, RunStore

    store = RunStore(runs_root)
    fit_records: list[dict[str, str]] = []
    for fit_id in experiment_order:
        fit_path = PurePosixPath(fit_id)
        if (
            fit_path.is_absolute()
            or fit_path.as_posix() != fit_id
            or "\\" in fit_id
            or ":" in fit_id
            or any(part in {"", ".", ".."} for part in fit_path.parts)
        ):
            raise ValueError(f"evaluation metrics Experiment {fit_id!r} is unsafe")
        receipt = store.receipt(fit_id, missing_ok=True)
        if receipt is None or store.state(fit_id) is not RunState.COMPLETED:
            raise ValueError(
                f"evaluation metrics Experiment {fit_id!r} has no completed "
                "verified receipt"
            )
        receipt_metadata = receipt.get("metadata")
        configuration = (
            receipt_metadata.get("configuration")
            if isinstance(receipt_metadata, dict)
            else None
        )
        if (
            not isinstance(configuration, dict)
            or not isinstance(configuration.get("path"), str)
            or not configuration["path"].strip()
            or not isinstance(configuration.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", configuration["sha256"])
        ):
            raise ValueError(
                f"evaluation metrics Experiment {fit_id!r} receipt has no "
                "configuration lineage"
            )
        run_dir = store.run_dir(fit_id).resolve()
        metrics_path = latest_metrics_path(run_dir)
        receipt_metrics_path = (
            run_dir
            / f"attempt-{receipt['attempt']}"
            / "metrics"
            / "version_0"
            / "metrics.csv"
        ).resolve()
        if metrics_path is None or metrics_path.resolve() != receipt_metrics_path:
            raise ValueError(
                f"evaluation metrics Experiment {fit_id!r} latest metrics are not "
                "from its completed receipt"
            )
        relative_metrics_path = metrics_path.resolve().relative_to(run_dir).as_posix()
        metric_sha256 = _sha256_file(metrics_path)
        artifacts = receipt.get("artifacts", {})
        if not isinstance(artifacts, dict) or not any(
            isinstance(artifact, dict)
            and artifact.get("path") == relative_metrics_path
            and artifact.get("sha256") == metric_sha256
            for artifact in artifacts.values()
        ):
            raise ValueError(
                f"evaluation metrics Experiment {fit_id!r} metrics file is not "
                "receipt-bound"
            )
        fit_records.append(
            {
                "fit_id": fit_id,
                "run_identity": receipt["identity"],
                "receipt_sha256": _sha256_file(run_dir / "receipt.json"),
                "configuration_path": configuration["path"],
                "configuration_sha256": configuration["sha256"],
                "metrics_path": relative_metrics_path,
                "metrics_sha256": metric_sha256,
            }
        )

    expected_rows = aggregate_metric_rows(experiment_order, runs_root)
    if len(expected_rows) != len(rows):
        raise ValueError("evaluation metrics rows do not match verified run receipts")
    expected_columns = ["Experiment"]
    for expected_row in expected_rows:
        expected_columns.extend(
            column for column in expected_row if column not in expected_columns
        )
    if columns != expected_columns:
        raise ValueError(
            "evaluation metrics columns do not match verified run artifacts"
        )
    for row_number, (actual, expected) in enumerate(
        zip(rows, expected_rows, strict=True), start=2
    ):
        if actual["Experiment"] != expected.get("Experiment"):
            raise ValueError(
                f"evaluation metrics row {row_number} does not match its verified run"
            )
        for column in expected_columns[1:]:
            raw_value = actual[column]
            expected_value = expected.get(column)
            if expected_value is None:
                if raw_value.strip():
                    raise ValueError(
                        f"evaluation metrics row {row_number} column {column!r} "
                        "does not match its verified run"
                    )
                continue
            expected_number = float(expected_value)
            if math.isnan(expected_number):
                if raw_value.strip():
                    raise ValueError(
                        f"evaluation metrics row {row_number} column {column!r} "
                        "does not match its verified run"
                    )
                continue
            try:
                actual_number = float(raw_value)
            except ValueError as error:
                raise ValueError(
                    f"evaluation metrics row {row_number} column {column!r} "
                    "does not match its verified run"
                ) from error
            if actual_number != expected_number:
                raise ValueError(
                    f"evaluation metrics row {row_number} column {column!r} "
                    "does not match its verified run"
                )
    return (
        values,
        {
            "path": str(path),
            "sha256": _sha256_bytes(raw),
            "rows": len(rows),
            "runs_root": str(runs_root),
            "fits": fit_records,
        },
        raw,
    )


def _read_json_object(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload, raw


def _verified_bundle_artifact(
    root: Path, record: object, description: str
) -> tuple[Path, dict[str, Any], bytes]:
    if not isinstance(record, dict):
        raise ValueError(f"{description} receipt entry is malformed")
    relative_name = record.get("path")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(relative_name, str)
        or not relative_name.strip()
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError(f"{description} receipt entry requires path and SHA-256")
    relative = PurePosixPath(relative_name)
    if (
        relative.is_absolute()
        or "\\" in relative_name
        or ":" in relative_name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{description} receipt path is unsafe")
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{description} receipt path escapes its bundle")
    payload, raw = _read_json_object(path, description)
    if _sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"{description} does not match its receipt SHA-256")
    return path, payload, raw


def _numeric_macros(prefix: str, payload: object) -> dict[str, int | float]:
    """Flatten all finite numeric leaves beneath an authoritative report field."""
    values: dict[str, int | float] = {}

    def visit(value: object, parts: list[str]) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, int):
            number: int | float = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("authoritative report contains a non-finite number")
            number = value
        elif isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ValueError("authoritative report object keys must be strings")
                visit(value[key], [*parts, key])
            return
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, [*parts, f"item_{index}"])
            return
        else:
            return
        segments = [_label_key(part) or "field" for part in [prefix, *parts]]
        key = "_".join(segments)
        if key in values:
            raise ValueError(f"authoritative report produces duplicate macro {key}")
        values[key] = number

    visit(payload, [])
    if not values:
        raise ValueError(f"authoritative {prefix} report contains no numeric values")
    return values


def _evaluation_artifacts(
    bundle_path: Path | None,
    configuration_sha256: str,
    manifest_digest: str,
) -> tuple[dict[str, int | float], dict[str, Any] | None, dict[str, bytes]]:
    """Import receipt-bound evaluation JSON without opening partition images."""
    if bundle_path is None:
        return {}, None, {}
    root = Path(bundle_path).resolve()
    if not root.is_dir():
        raise ValueError(f"evaluation artifact bundle does not exist: {root}")
    receipt_path = root / "receipt.json"
    receipt, receipt_bytes = _read_json_object(receipt_path, "evaluation receipt")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("kind") != "evaluation"
        or receipt.get("configuration_sha256") != configuration_sha256
        or receipt.get("manifest_digest") != manifest_digest
        or receipt.get("partition") != "val"
    ):
        raise ValueError("evaluation receipt does not match the export configuration")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("evaluation receipt has no artifact records")
    metrics_path, metrics_report, metrics_bytes = _verified_bundle_artifact(
        root, artifacts.get("metrics"), "evaluation metrics report"
    )
    predictions_path, predictions_report, predictions_bytes = _verified_bundle_artifact(
        root, artifacts.get("predictions"), "evaluation predictions report"
    )
    for description, report in (
        ("evaluation metrics", metrics_report),
        ("evaluation predictions", predictions_report),
    ):
        if (
            report.get("config_digest") != configuration_sha256
            or report.get("manifest_digest") != manifest_digest
        ):
            raise ValueError(f"{description} lineage does not match this export")
    metrics = metrics_report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("evaluation metrics report has no metrics object")

    run_records = receipt.get("run_receipts")
    if not isinstance(run_records, list) or not run_records:
        raise ValueError("evaluation receipt requires run receipt records")
    run_identities: dict[str, str] = {}
    verified_runs: list[dict[str, str]] = []
    for index, record in enumerate(run_records):
        if not isinstance(record, dict):
            raise ValueError(f"evaluation run receipt {index} is malformed")
        fit_id, identity = record.get("fit_id"), record.get("identity")
        if (
            not isinstance(fit_id, str)
            or not fit_id.strip()
            or not isinstance(identity, str)
            or not identity.strip()
            or fit_id in run_identities
        ):
            raise ValueError(f"evaluation run receipt {index} has invalid identity")
        _, run_receipt, _ = _verified_bundle_artifact(
            root, record, f"evaluation run receipt {fit_id}"
        )
        results = run_receipt.get("results")
        if (
            run_receipt.get("fit_id") != fit_id
            or run_receipt.get("identity") != identity
            or not isinstance(results, dict)
            or results.get("config_digest") != configuration_sha256
            or not isinstance(results.get("manifests"), dict)
            or results["manifests"].get("target") != manifest_digest
            or results.get("partition") != receipt["partition"]
        ):
            raise ValueError(f"evaluation run receipt {fit_id} has mismatched lineage")
        run_identities[fit_id] = identity
        verified_runs.append(
            {
                "fit_id": fit_id,
                "identity": identity,
                "path": record["path"],
                "sha256": record["sha256"],
            }
        )
    if predictions_report.get("run_identities") != run_identities:
        raise ValueError("evaluation predictions do not match verified run receipts")

    values = _numeric_macros("evaluation", metrics)
    outputs = {
        "evaluation/metrics.json": metrics_bytes,
        "evaluation/predictions.json": predictions_bytes,
        "evaluation/receipt.json": receipt_bytes,
    }
    lineage = {
        "path": str(root),
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "configuration_sha256": configuration_sha256,
        "manifest_digest": manifest_digest,
        "partition": receipt["partition"],
        "artifacts": {
            "metrics": {
                "path": str(metrics_path.relative_to(root).as_posix()),
                "sha256": _sha256_bytes(metrics_bytes),
            },
            "predictions": {
                "path": str(predictions_path.relative_to(root).as_posix()),
                "sha256": _sha256_bytes(predictions_bytes),
            },
        },
        "run_receipts": verified_runs,
        "outputs": sorted(outputs),
    }
    return values, lineage, outputs


def _statistics_artifacts(
    bundle_path: Path | None,
    configuration_sha256: str,
    manifest_digest: str,
    evaluation_lineage: dict[str, Any] | None,
) -> tuple[dict[str, int | float], dict[str, Any] | None, dict[str, bytes]]:
    """Import receipt-bound statistics summaries tied to one evaluation report."""
    if bundle_path is None:
        return {}, None, {}
    if evaluation_lineage is None:
        raise ValueError("statistics artifacts require verified evaluation artifacts")
    root = Path(bundle_path).resolve()
    if not root.is_dir():
        raise ValueError(f"statistics artifact bundle does not exist: {root}")
    receipt_path = root / "receipt.json"
    receipt, receipt_bytes = _read_json_object(receipt_path, "statistics receipt")
    artifact_record = receipt.get("artifact")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("kind") != "statistics"
        or receipt.get("configuration_sha256") != configuration_sha256
        or receipt.get("manifest_digest") != manifest_digest
        or receipt.get("evaluation_receipt_sha256")
        != evaluation_lineage["receipt_sha256"]
    ):
        raise ValueError(
            "statistics receipt does not match verified evaluation lineage"
        )
    report_path, report, report_bytes = _verified_bundle_artifact(
        root, artifact_record, "statistics report"
    )
    if (
        report.get("config_digest") != configuration_sha256
        or report.get("manifest_digest") != manifest_digest
        or report.get("evaluation_receipt_sha256")
        != evaluation_lineage["receipt_sha256"]
    ):
        raise ValueError("statistics report lineage does not match verified evaluation")
    statistics = report.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("statistics report has no statistics object")
    values = _numeric_macros("statistics", statistics)
    outputs = {
        "statistics/report.json": report_bytes,
        "statistics/receipt.json": receipt_bytes,
    }
    lineage = {
        "path": str(root),
        "receipt_sha256": _sha256_bytes(receipt_bytes),
        "configuration_sha256": configuration_sha256,
        "manifest_digest": manifest_digest,
        "evaluation_receipt_sha256": evaluation_lineage["receipt_sha256"],
        "artifact": {
            "path": report_path.relative_to(root).as_posix(),
            "sha256": _sha256_bytes(report_bytes),
        },
        "outputs": sorted(outputs),
    }
    return values, lineage, outputs


def validate_figure_sources(sources: list[dict[str, Any]]) -> None:
    if not isinstance(sources, list):
        raise ValueError("figure sources must be a JSON array")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"figure source {index} must be an object")
        license_name = source.get("license")
        if (
            not isinstance(license_name, str)
            or license_name not in ALLOWED_FIGURE_LICENSES
        ):
            raise ValueError(
                f"figure source {index} has unsupported license; allowed: "
                f"{', '.join(sorted(ALLOWED_FIGURE_LICENSES))}"
            )
        for field in ("source", "url", "attribution"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"figure source {index} requires {field}")
        if "path" in source and not isinstance(source["path"], str):
            raise ValueError(f"figure source {index} path must be a string")


def _write_figure(path: Path, manifest: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = manifest.labels
    support = manifest.support()
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(max(8, len(labels) * 1.35), 5.5))
    for index, partition in enumerate(PARTITIONS):
        values = [support[partition].get(label, 0) for label in labels]
        axis.bar(x + (index - 1) * width, values, width, label=partition)
        for offset, value in zip(x + (index - 1) * width, values, strict=True):
            axis.annotate(
                str(value),
                (offset, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_yticks([])
    axis.set_ylabel("Images")
    axis.set_title("Frozen image support by partition and class")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=160, metadata={"Software": "jute-disease"})
    plt.close(figure)


def export_manuscript(
    freeze_dir: Path,
    config_path: Path,
    output_dir: Path,
    *,
    source_metadata: dict[str, dict[str, str]],
    training_runs_dir: Path | None = None,
    evaluation_metrics_path: Path | None = None,
    evaluation_artifacts_path: Path | None = None,
    statistics_artifacts_path: Path | None = None,
    figure_sources_path: Path | None = None,
    export_id: str | None = None,
    command: str = "jute-disease manuscript export",
) -> dict[str, Any]:
    """Write a table, figure, macro files, rebuild kit and audit template.

    The source freeze must already contain a matching manifest and receipt. An
    existing non-empty output directory is never overwritten.
    """
    freeze_dir = Path(freeze_dir)
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    manifest = load_manifest(freeze_dir)
    complete = load_complete_manifest(freeze_dir)
    receipt_path = freeze_dir / "receipt.json"
    manifest_path = freeze_dir / "manifest.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("export configuration must be a JSON object")
    if not isinstance(source_metadata, dict):
        raise ValueError("source metadata must be a JSON object")
    expected_sources = {entry.source for entry in complete.entries}
    if set(source_metadata) != expected_sources:
        raise ValueError(
            "source metadata must contain exactly the source names in the manifest"
        )
    for source, details in source_metadata.items():
        if not isinstance(details, dict) or any(
            not isinstance(details.get(field), str) or not details[field].strip()
            for field in ("url", "version")
        ):
            raise ValueError(f"source metadata for {source!r} requires URL and version")
    config_bytes = _canonical_json(config) + b"\n"
    if figure_sources_path is None:
        figure_sources: list[dict[str, Any]] = []
    else:
        figure_sources = json.loads(
            Path(figure_sources_path).read_text(encoding="utf-8")
        )
        if not isinstance(figure_sources, list):
            raise ValueError("figure source credits must be a JSON array")
    validate_figure_sources(figure_sources)

    evaluation_values, evaluation_source, evaluation_bytes = _evaluation_metrics(
        evaluation_metrics_path, training_runs_dir
    )
    configuration_sha256 = _sha256_bytes(config_bytes)
    (
        evaluation_artifact_values,
        evaluation_artifact_source,
        evaluation_artifact_outputs,
    ) = _evaluation_artifacts(
        evaluation_artifacts_path, configuration_sha256, manifest.digest()
    )
    (
        statistics_values,
        statistics_source,
        statistics_outputs,
    ) = _statistics_artifacts(
        statistics_artifacts_path,
        configuration_sha256,
        manifest.digest(),
        evaluation_artifact_source,
    )

    compute_report = (
        _pending_compute_report()
        if training_runs_dir is None
        else _compute_report(training_runs_dir)
    )
    identity_compute_report = {
        "status": compute_report["status"],
        "receipts": compute_report["receipts"],
        "stages": compute_report["stages"],
    }

    identity = {
        "manifest_digest": manifest.digest(),
        "configuration_sha256": _sha256_bytes(config_bytes),
        "source_metadata": source_metadata,
        "figure_sources": figure_sources,
        "evaluation_metrics": (
            None
            if evaluation_source is None
            else {
                "sha256": evaluation_source["sha256"],
                "fits": evaluation_source["fits"],
            }
        ),
        "evaluation_artifacts": (
            None
            if evaluation_artifact_source is None
            else {
                "receipt_sha256": evaluation_artifact_source["receipt_sha256"],
                "artifacts": evaluation_artifact_source["artifacts"],
                "run_receipts": evaluation_artifact_source["run_receipts"],
            }
        ),
        "statistics_artifacts": (
            None
            if statistics_source is None
            else {
                "receipt_sha256": statistics_source["receipt_sha256"],
                "artifact": statistics_source["artifact"],
            }
        ),
        "compute_report": identity_compute_report,
        "command": command,
    }
    selected_id = export_id or f"exp-{_sha256_bytes(_canonical_json(identity))[:12]}"
    if not _EXPORT_ID.fullmatch(selected_id):
        raise ValueError(
            "export ID may contain letters, digits, dot, underscore and dash"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(
            f"Refusing to overwrite non-empty export directory: {output_dir}"
        )

    value_groups = (
        _support_values(complete),
        _classical_tuning_values(),
        _compute_values(compute_report),
        evaluation_values,
        evaluation_artifact_values,
        statistics_values,
    )
    values: dict[str, int | float] = {}
    for group in value_groups:
        overlap = values.keys() & group.keys()
        if overlap:
            raise ValueError(
                "manuscript artifacts produce duplicate macro keys: "
                + ", ".join(sorted(overlap))
            )
        values.update(group)
    macros = {selected_id: values}
    (output_dir / "configuration.json").write_bytes(config_bytes)
    _write_json(output_dir / "source-metadata.json", source_metadata)
    _write_json(output_dir / "macros.json", macros)
    (output_dir / "macros.tex").write_text(
        _macro_tex(selected_id, values), encoding="utf-8", newline="\n"
    )
    with (output_dir / "support.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("partition", "label", "count"))
        writer.writeheader()
        support = complete.support()
        for partition in PARTITIONS:
            for label in complete.labels:
                writer.writerow(
                    {
                        "partition": partition,
                        "label": label,
                        "count": support[partition].get(label, 0),
                    }
                )
    _write_figure(output_dir / "support.png", complete)
    _write_json(output_dir / "figure-sources.json", figure_sources)
    _write_json(output_dir / "compute-report.json", compute_report)
    (output_dir / "compute.csv").write_text(
        _compute_csv(compute_report), encoding="utf-8", newline="\n"
    )
    (output_dir / "classical-tuning-budget.csv").write_text(
        _classical_tuning_csv(), encoding="utf-8", newline="\n"
    )
    if evaluation_bytes is not None:
        (output_dir / "evaluation-metrics.csv").write_bytes(evaluation_bytes)
    for relative_name, content in {
        **evaluation_artifact_outputs,
        **statistics_outputs,
    }.items():
        artifact_path = output_dir / relative_name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)

    kit_entries = [
        {
            "source": entry.source,
            "url": source_metadata[entry.source]["url"],
            "version": source_metadata[entry.source]["version"],
            "path": entry.path,
            "sha256": entry.image_id,
            "family_id": entry.group,
            "split": entry.partition,
            "label": entry.label,
            "image_id": entry.image_id,
        }
        for entry in complete.entries
    ]
    kit = {
        "version": 1,
        "seed": manifest.seed,
        "fractions": dict(manifest.fractions),
        "entries": kit_entries,
    }
    _write_json(output_dir / "rebuild-kit.json", kit)

    manifest_ref_text = Path(
        os.path.relpath(manifest_path.resolve(), output_dir.resolve())
    ).as_posix()
    receipt_ref_text = Path(
        os.path.relpath(receipt_path.resolve(), output_dir.resolve())
    ).as_posix()

    output_names = _BASE_EXPORT_OUTPUTS + (
        ("evaluation-metrics.csv",) if evaluation_bytes is not None else ()
    )
    output_names += tuple(sorted(evaluation_artifact_outputs))
    output_names += tuple(sorted(statistics_outputs))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "export_id": selected_id,
        "command": command,
        "configuration": config,
        "source_metadata": source_metadata,
        "configuration_file": "configuration.json",
        "configuration_sha256": _sha256_file(output_dir / "configuration.json"),
        "artifacts": {
            "manifest": {
                "path": manifest_ref_text,
                "sha256": _sha256_file(manifest_path),
                "manifest_digest": manifest.digest(),
            },
            "receipt": {
                "path": receipt_ref_text,
                "sha256": _sha256_file(receipt_path),
            },
        },
        "evaluation_metrics": evaluation_source,
        "evaluation_artifacts": evaluation_artifact_source,
        "statistics_artifacts": statistics_source,
        "outputs": {
            name: {"path": name, "sha256": _sha256_file(output_dir / name)}
            for name in output_names
        },
    }
    _write_json(output_dir / "export.json", metadata)
    audit = {
        "issue": 108,
        "claim_list_url": CLAIM_LIST_URL,
        "claim_audit_decision_url": CLAIM_AUDIT_DECISION_URL,
        "status": "pending",
        "auditor_requirements": (
            "Fresh context; reviewer provider differs from the writing agent; "
            "review only the manuscript, generated macros, #108 claim lists and "
            "authoritative artifacts. Record the manuscript SHA-256, one row "
            "for every flagged claim (or an explicit no-flags result), and every "
            "unflagged numeric statement (or an explicit no-statements result)."
        ),
        "writer_provider": "Codex",
        "reviewer": {
            "name": "",
            "provider": "",
            "fresh_context": False,
            "reviewed_at": "",
            "record": "",
        },
        "manuscript_sha256": "",
        "review_scope": [
            "manuscript",
            "generated macros",
            "#108 claim lists",
            "authoritative artifacts",
        ],
        "claim_inventory_complete": False,
        "numeric_scan_complete": False,
        "claim_lists": {name: list(items) for name, items in _CLAIM_LISTS.items()},
        "macro_rows": [
            {"key": key, "value": value, "status": "pending", "evidence": ""}
            for key, value in sorted(values.items())
        ],
        "claim_rows": [],
        "claim_row_template": {
            "claim": "",
            "source_location": "",
            "may_report_basis": "",
            "may_not_claim_review": "",
            "status": "pending",
            "evidence": "",
        },
        "unflagged_numeric_statements": [],
        "numeric_statement_template": {
            "statement": "",
            "source_location": "",
            "evidence": "",
            "status": "pending",
        },
        "blocking_rule": "Any fail or unsupported row blocks submission; no waiver.",
        "owner_spot_check": {
            "status": "pending",
            "fraction": 0.1,
            "selection_method": "random",
            "selection_record": "",
            "checked_rows": [],
        },
        "owner_signoff": {"status": "pending", "signed_by": "", "signed_at": ""},
        "re_audit_rule": (
            "Re-audit changed numbers and claims after every manuscript change."
        ),
        "allowed_claims_reference": CLAIM_LIST_URL,
        "prohibited_claims_reference": CLAIM_LIST_URL,
    }
    _write_json(output_dir / "claim-audit.json", audit)
    return {"export_id": selected_id, "output_dir": str(output_dir), "macros": macros}


def rebuild_kit_splits(
    kit_path: Path, image_root: Path, output_path: Path
) -> dict[str, Any]:
    """Verify source bytes and reproduce the kit's deterministic split map."""
    kit = json.loads(Path(kit_path).read_text(encoding="utf-8"))
    if not isinstance(kit, dict):
        raise ValueError("rebuild kit must be a JSON object")
    entries = kit.get("entries")
    if (
        type(kit.get("version")) is not int
        or kit["version"] != 1
        or type(kit.get("seed")) is not int
        or not isinstance(kit.get("fractions"), dict)
        or not isinstance(entries, list)
        or not entries
    ):
        raise ValueError("unsupported or malformed rebuild kit")
    fractions = kit["fractions"]
    validate_partition_fractions(fractions)
    required_fields = (
        "source",
        "url",
        "version",
        "path",
        "sha256",
        "family_id",
        "split",
        "label",
        "image_id",
    )
    image_root = Path(image_root).resolve()
    verified_images = 0
    skipped_test_images = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or any(
            not isinstance(entry.get(field), str) or not entry[field].strip()
            for field in required_fields
        ):
            raise ValueError(f"rebuild kit entry {index} is malformed")
        relative = PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or "\\" in entry["path"]
            or ":" in entry["path"]
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"rebuild kit entry {index} has an unsafe source path")
        if entry["split"] not in PARTITIONS:
            raise ValueError(f"rebuild kit entry {index} has an unknown split")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
            or entry["image_id"] != entry["sha256"]
        ):
            raise ValueError(f"rebuild kit entry {index} has an invalid image hash")
        if entry["split"] == "test":
            skipped_test_images += 1
        else:
            path = (image_root / Path(*relative.parts)).resolve()
            if not path.is_relative_to(image_root):
                raise ValueError(f"rebuild kit entry {index} escapes the image root")
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"source hash mismatch for {entry['path']}")
            verified_images += 1
    assignments = rebuild_partition_assignments(entries, kit["seed"], kit["fractions"])
    expected = {entry["image_id"]: entry["split"] for entry in entries}
    if assignments != expected:
        mismatched = next(
            key for key in sorted(expected) if assignments.get(key) != expected[key]
        )
        raise ValueError(f"rebuilt split differs for image {mismatched}")
    report = {
        "version": 1,
        "source_manifest": str(Path(kit_path)),
        "verified_images": verified_images,
        "skipped_test_images": skipped_test_images,
        "assignments": dict(sorted(assignments.items())),
        "split_digest": _sha256_bytes(_canonical_json(sorted(assignments.items()))),
    }
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def validate_export(output_dir: Path) -> dict[str, Any]:
    """Check macros, table, rebuild kit and outputs against frozen artifacts."""
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "export.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("export metadata must be a JSON object")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("export artifacts metadata must be a JSON object")
    manifest_artifact = artifacts.get("manifest")
    receipt_artifact = artifacts.get("receipt")
    if not isinstance(manifest_artifact, dict) or not isinstance(
        receipt_artifact, dict
    ):
        raise ValueError("export manifest and receipt metadata must be objects")
    manifest_ref = manifest_artifact.get("path")
    receipt_ref = receipt_artifact.get("path")
    if not isinstance(manifest_ref, str) or not isinstance(receipt_ref, str):
        raise ValueError("export manifest and receipt paths must be strings")
    manifest_path = (output_dir / manifest_ref).resolve()
    receipt_path = (output_dir / receipt_ref).resolve()
    if _sha256_file(manifest_path) != manifest_artifact.get("sha256"):
        raise ValueError("authoritative manifest bytes changed after export")
    if _sha256_file(receipt_path) != receipt_artifact.get("sha256"):
        raise ValueError("authoritative receipt bytes changed after export")
    manifest = load_manifest(manifest_path.parent)
    complete = load_complete_manifest(manifest_path.parent)
    if manifest.digest() != manifest_artifact.get("manifest_digest"):
        raise ValueError("export lineage manifest digest mismatch")
    configuration_sha256 = metadata.get("configuration_sha256")
    if not isinstance(configuration_sha256, str):
        raise ValueError("export configuration SHA-256 is malformed")
    if metadata.get("configuration_file") != "configuration.json":
        raise ValueError("export configuration file reference is malformed")
    if _sha256_file(output_dir / "configuration.json") != configuration_sha256:
        raise ValueError("export configuration changed after export")
    configuration_snapshot = metadata.get("configuration")
    if not isinstance(configuration_snapshot, dict):
        raise ValueError("export configuration snapshot must be a JSON object")
    configuration_file = json.loads(
        (output_dir / "configuration.json").read_text(encoding="utf-8")
    )
    if configuration_file != configuration_snapshot:
        raise ValueError("export configuration snapshot differs from its file")

    evaluation_values: dict[str, int | float] = {}
    evaluation_source = metadata.get("evaluation_metrics")
    if evaluation_source is None:
        if (output_dir / "evaluation-metrics.csv").exists():
            raise ValueError("unrecorded evaluation metrics table is present")
    else:
        if not isinstance(evaluation_source, dict) or not isinstance(
            evaluation_source.get("path"), str
        ):
            raise ValueError("evaluation metrics lineage is malformed")
        if not isinstance(evaluation_source.get("runs_root"), str):
            raise ValueError("evaluation metrics receipt lineage is malformed")
        evaluation_values, current_source, current_bytes = _evaluation_metrics(
            Path(evaluation_source["path"]), Path(evaluation_source["runs_root"])
        )
        if current_source != evaluation_source:
            raise ValueError("authoritative evaluation metrics changed after export")
        if (output_dir / "evaluation-metrics.csv").read_bytes() != current_bytes:
            raise ValueError("exported evaluation metrics differ from source artifact")

    evaluation_artifact_values: dict[str, int | float] = {}
    evaluation_artifact_source = metadata.get("evaluation_artifacts")
    evaluation_artifact_outputs: dict[str, bytes] = {}
    if evaluation_artifact_source is not None:
        if not isinstance(evaluation_artifact_source, dict) or not isinstance(
            evaluation_artifact_source.get("path"), str
        ):
            raise ValueError("evaluation artifact lineage is malformed")
        (
            evaluation_artifact_values,
            current_evaluation_source,
            evaluation_artifact_outputs,
        ) = _evaluation_artifacts(
            Path(evaluation_artifact_source["path"]),
            configuration_sha256,
            manifest.digest(),
        )
        if current_evaluation_source != evaluation_artifact_source:
            raise ValueError("authoritative evaluation artifacts changed after export")
        for name, content in evaluation_artifact_outputs.items():
            if (output_dir / name).read_bytes() != content:
                raise ValueError(f"exported evaluation artifact differs: {name}")

    statistics_values: dict[str, int | float] = {}
    statistics_source = metadata.get("statistics_artifacts")
    statistics_outputs: dict[str, bytes] = {}
    if statistics_source is not None:
        if not isinstance(statistics_source, dict) or not isinstance(
            statistics_source.get("path"), str
        ):
            raise ValueError("statistics artifact lineage is malformed")
        statistics_values, current_statistics_source, statistics_outputs = (
            _statistics_artifacts(
                Path(statistics_source["path"]),
                configuration_sha256,
                manifest.digest(),
                evaluation_artifact_source,
            )
        )
        if current_statistics_source != statistics_source:
            raise ValueError("authoritative statistics artifacts changed after export")
        for name, content in statistics_outputs.items():
            if (output_dir / name).read_bytes() != content:
                raise ValueError(f"exported statistics artifact differs: {name}")

    export_id = metadata.get("export_id")
    if not isinstance(export_id, str) or not _EXPORT_ID.fullmatch(export_id):
        raise ValueError("export ID metadata is malformed")
    compute_report = json.loads(
        (output_dir / "compute-report.json").read_text(encoding="utf-8")
    )
    if not isinstance(compute_report, dict):
        raise ValueError("compute report must be a JSON object")
    if compute_report.get("status") == "measured":
        runs_root = compute_report.get("runs_root")
        if not isinstance(runs_root, str) or not runs_root:
            raise ValueError("measured compute report has no receipts directory")
        if _compute_report(Path(runs_root)) != compute_report:
            raise ValueError(
                "compute report differs from authoritative training receipts"
            )
    elif compute_report != _pending_compute_report():
        raise ValueError("compute report is incomplete or malformed")
    if (output_dir / "compute.csv").read_text(encoding="utf-8") != _compute_csv(
        compute_report
    ):
        raise ValueError("compute table does not match training receipt data")
    if (output_dir / "classical-tuning-budget.csv").read_text(
        encoding="utf-8"
    ) != _classical_tuning_csv():
        raise ValueError("classical tuning budget differs from the accepted grid")
    expected_values: dict[str, int | float] = {}
    for group in (
        _support_values(complete),
        _classical_tuning_values(),
        _compute_values(compute_report),
        evaluation_values,
        evaluation_artifact_values,
        statistics_values,
    ):
        overlap = expected_values.keys() & group.keys()
        if overlap:
            raise ValueError(
                "authoritative artifacts produce duplicate macro keys: "
                + ", ".join(sorted(overlap))
            )
        expected_values.update(group)
    macros = json.loads((output_dir / "macros.json").read_text(encoding="utf-8"))
    if macros != {export_id: expected_values}:
        raise ValueError("macro JSON values do not match the authoritative manifest")
    if (output_dir / "macros.tex").read_text(encoding="utf-8") != _macro_tex(
        export_id, expected_values
    ):
        raise ValueError("TeX macros do not match the authoritative manifest")

    expected_csv = ["partition,label,count"]
    support = complete.support()
    expected_csv.extend(
        f"{partition},{label},{support[partition].get(label, 0)}"
        for partition in PARTITIONS
        for label in complete.labels
    )
    csv_text = (
        (output_dir / "support.csv").read_text(encoding="utf-8").replace("\r\n", "\n")
    )
    if csv_text != "\n".join(expected_csv) + "\n":
        raise ValueError(
            "exported support table does not match the authoritative manifest"
        )
    kit = json.loads((output_dir / "rebuild-kit.json").read_text(encoding="utf-8"))
    source_metadata = json.loads(
        (output_dir / "source-metadata.json").read_text(encoding="utf-8")
    )
    if source_metadata != metadata.get("source_metadata"):
        raise ValueError("export source metadata differs from its recorded lineage")
    expected_sources = {entry.source for entry in complete.entries}
    if (
        not isinstance(source_metadata, dict)
        or set(source_metadata) != expected_sources
    ):
        raise ValueError(
            "export source metadata does not match the authoritative manifest"
        )
    expected_kit = [
        {
            "source": entry.source,
            "url": source_metadata[entry.source]["url"],
            "version": source_metadata[entry.source]["version"],
            "path": entry.path,
            "sha256": entry.image_id,
            "family_id": entry.group,
            "split": entry.partition,
            "label": entry.label,
            "image_id": entry.image_id,
        }
        for entry in complete.entries
    ]
    if kit.get("entries") != expected_kit:
        raise ValueError("rebuild kit entries do not match the authoritative manifest")
    validate_figure_sources(
        json.loads((output_dir / "figure-sources.json").read_text(encoding="utf-8"))
    )
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict) or any(
        not isinstance(name, str)
        or not isinstance(record, dict)
        or record.get("path") != name
        or not isinstance(record.get("sha256"), str)
        for name, record in outputs.items()
    ):
        raise ValueError("export output metadata is malformed or has a mismatched path")
    expected_output_names = set(_BASE_EXPORT_OUTPUTS)
    if evaluation_source is not None:
        expected_output_names.add("evaluation-metrics.csv")
    expected_output_names.update(evaluation_artifact_outputs)
    expected_output_names.update(statistics_outputs)
    if set(outputs) != expected_output_names:
        raise ValueError("export output inventory does not match generated artifacts")
    for name, record in outputs.items():
        if _sha256_file(output_dir / name) != record["sha256"]:
            raise ValueError(f"export output changed after export: {name}")
    return {
        "export_id": export_id,
        "checked_macros": len(expected_values),
        "pass": True,
    }


def validate_claim_audit(
    output_dir: Path, manuscript_path: Path | None = None
) -> dict[str, Any]:
    """Block finalization until the independent review and owner gates pass."""
    export_check = validate_export(output_dir)
    audit = json.loads(
        (Path(output_dir) / "claim-audit.json").read_text(encoding="utf-8")
    )
    if not isinstance(audit, dict):
        raise ValueError("claim audit must be a JSON object")
    if audit.get("status") != "complete":
        raise ValueError("claim audit is pending")
    if audit.get("claim_list_url") != CLAIM_LIST_URL:
        raise ValueError("claim audit must reference the accepted #108 claim lists")
    if audit.get("claim_audit_decision_url") != CLAIM_AUDIT_DECISION_URL:
        raise ValueError("claim audit must reference the accepted #108 audit decision")
    expected_claim_lists = {name: list(items) for name, items in _CLAIM_LISTS.items()}
    if audit.get("claim_lists") != expected_claim_lists:
        raise ValueError("claim audit claim lists differ from the accepted #108 lists")
    reviewer = audit.get("reviewer")
    if (
        not isinstance(reviewer, dict)
        or any(
            not isinstance(reviewer.get(field), str) or not reviewer[field].strip()
            for field in ("name", "provider", "reviewed_at", "record")
        )
        or reviewer.get("fresh_context") is not True
        or not isinstance(audit.get("writer_provider"), str)
        or reviewer["provider"] == audit.get("writer_provider")
    ):
        raise ValueError(
            "claim audit requires a different-provider fresh-context reviewer"
        )
    reviewed_at = _parse_audit_timestamp(reviewer["reviewed_at"], "reviewed_at")
    manuscript_sha256 = audit.get("manuscript_sha256")
    if not isinstance(manuscript_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", manuscript_sha256
    ):
        raise ValueError("claim audit must record the reviewed manuscript SHA-256")
    if manuscript_path is None:
        raise ValueError("claim audit check requires the current manuscript path")
    if _sha256_file(Path(manuscript_path)) != manuscript_sha256:
        raise ValueError("claim audit manuscript SHA-256 does not match the manuscript")
    review_scope = audit.get("review_scope")
    if (
        not isinstance(review_scope, list)
        or not all(isinstance(item, str) for item in review_scope)
        or not {
            "manuscript",
            "generated macros",
            "#108 claim lists",
            "authoritative artifacts",
        }.issubset(review_scope)
    ):
        raise ValueError("claim audit review scope is incomplete")
    macro_rows = audit.get("macro_rows")
    expected_macros = json.loads(
        (Path(output_dir) / "macros.json").read_text(encoding="utf-8")
    )[export_check["export_id"]]
    if not isinstance(macro_rows, list) or any(
        not isinstance(row, dict)
        or not isinstance(row.get("key"), str)
        or not row["key"]
        for row in macro_rows
    ):
        raise ValueError("claim audit must contain one row for every generated macro")
    rows = macro_rows
    row_values = {row.get("key"): row.get("value") for row in rows}
    if row_values != expected_macros or len(row_values) != len(rows):
        raise ValueError("claim audit must contain one row for every generated macro")
    if any(
        not isinstance(row.get("evidence"), str) or not row["evidence"].strip()
        for row in rows
    ):
        raise ValueError("claim audit macro rows require artifact evidence")
    claim_rows = audit.get("claim_rows")
    if not isinstance(claim_rows, list) or any(
        not isinstance(row, dict)
        or not all(
            isinstance(row.get(field), str) and row[field].strip()
            for field in (
                "claim",
                "source_location",
                "may_report_basis",
                "may_not_claim_review",
                "evidence",
            )
        )
        for row in claim_rows
    ):
        raise ValueError("claim audit contains an invalid claim row")
    if not claim_rows:
        raise ValueError(
            "claim audit must record flagged claims or an explicit no-flags result"
        )
    unflagged_rows = audit.get("unflagged_numeric_statements")
    if not isinstance(unflagged_rows, list) or any(
        not isinstance(row, dict)
        or not all(
            isinstance(row.get(field), str) and row[field].strip()
            for field in ("statement", "source_location", "evidence")
        )
        for row in unflagged_rows
    ):
        raise ValueError("numeric scan contains an invalid statement row")
    if not unflagged_rows:
        raise ValueError(
            "numeric scan must record unflagged statements or an explicit "
            "no-statements result"
        )
    statuses = [row.get("status") for row in [*rows, *claim_rows, *unflagged_rows]]
    if any(status in {"fail", "unsupported"} for status in statuses):
        raise ValueError("failed or unsupported claim rows block submission")
    if any(status != "pass" for status in statuses):
        raise ValueError("claim audit contains rows that have not passed")
    if audit.get("claim_inventory_complete") is not True:
        raise ValueError("claim inventory is incomplete")
    if audit.get("numeric_scan_complete") is not True:
        raise ValueError("unflagged numeric statement scan is incomplete")
    spot_check = audit.get("owner_spot_check", {})
    if (
        not isinstance(spot_check, dict)
        or spot_check.get("status") != "pass"
        or spot_check.get("fraction") != 0.1
        or spot_check.get("selection_method") != "random"
        or not spot_check.get("selection_record")
    ):
        raise ValueError("owner 10% spot check is incomplete")
    row_ids = [
        *(f"macro:{row['key']}" for row in rows),
        *(f"claim:{index}" for index, _ in enumerate(claim_rows, start=1)),
        *(f"numeric:{index}" for index, _ in enumerate(unflagged_rows, start=1)),
    ]
    checked_rows = spot_check.get("checked_rows")
    required_spot_checks = (len(row_ids) + 9) // 10
    if (
        not isinstance(checked_rows, list)
        or not all(isinstance(row_id, str) for row_id in checked_rows)
        or len(checked_rows) != required_spot_checks
        or len(set(checked_rows)) != len(checked_rows)
        or any(row_id not in row_ids for row_id in checked_rows)
    ):
        raise ValueError("owner must record a random 10% sample of passing rows")
    signoff = audit.get("owner_signoff", {})
    if (
        not isinstance(signoff, dict)
        or signoff.get("status") != "signed"
        or not isinstance(signoff.get("signed_by"), str)
        or not signoff["signed_by"].strip()
        or not isinstance(signoff.get("signed_at"), str)
        or not signoff["signed_at"].strip()
    ):
        raise ValueError("owner sign-off is incomplete")
    signed_at = _parse_audit_timestamp(signoff["signed_at"], "signed_at")
    if signed_at <= reviewed_at:
        raise ValueError("owner sign-off must occur after the independent review")
    return {
        "export_id": export_check["export_id"],
        "checked_macros": export_check["checked_macros"],
        "checked_claims": len(claim_rows),
        "checked_numeric_statements": len(unflagged_rows),
        "owner_spot_checks": len(checked_rows),
        "pass": True,
    }


def run_manuscript_fixture(output_dir: Path) -> dict[str, Any]:
    """Exercise synthetic data, freeze, export, rebuild and mechanical checking."""
    from jute_disease.config import CURATION_RULES_PATH
    from jute_disease.data import make_synthetic_data

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(
            f"Refusing to overwrite non-empty fixture directory: {output_dir}"
        )
    image_root = output_dir / "images"
    make_synthetic_data(image_root, per_class=20, seed=42)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    rules_path = output_dir / "fixture-rules.json"
    reviews_path = output_dir / "fixture-reviews.json"
    _write_json(rules_path, rules)
    _write_json(reviews_path, {"version": 1, "decisions": []})
    freeze_dir = output_dir / "freeze" / "target"
    freeze_command = [
        sys.executable,
        "-m",
        "jute_disease.cli",
        "freeze",
        "target",
        "--image-root",
        str(image_root),
        "--rules",
        str(rules_path),
        "--reviews",
        str(reviews_path),
        "--output",
        str(output_dir / "freeze"),
        "--seed",
        "42",
        "--workers",
        "2",
    ]
    frozen = subprocess.run(
        freeze_command,
        capture_output=True,
        text=True,
        check=False,
    )
    if frozen.returncode:
        raise ValueError(f"fixture freeze failed: {frozen.stderr.strip()}")
    config = {
        "purpose": "synthetic fixture only",
        "freeze_command": "jute-disease freeze target",
        "seed": 42,
        "fractions": dict(load_manifest(freeze_dir).fractions),
        "rules_sha256": _sha256_file(rules_path),
        "reviews_sha256": _sha256_file(reviews_path),
    }
    config_path = output_dir / "fixture-export-config.json"
    _write_json(config_path, config)
    frozen_manifest = load_complete_manifest(freeze_dir)
    source_metadata = {
        source: {"url": "synthetic://local-fixture", "version": "synthetic-1"}
        for source in {entry.source for entry in frozen_manifest.entries}
    }
    export_dir = output_dir / "export"
    export = export_manuscript(
        freeze_dir,
        config_path,
        export_dir,
        source_metadata=source_metadata,
        command="jute-disease manuscript fixture",
    )
    rebuild = rebuild_kit_splits(
        export_dir / "rebuild-kit.json",
        image_root,
        output_dir / "rebuilt-splits.json",
    )
    checked = validate_export(export_dir)
    expected_verified = sum(
        entry.partition != "test" for entry in frozen_manifest.entries
    )
    expected_skipped = sum(
        entry.partition == "test" for entry in frozen_manifest.entries
    )
    if (
        rebuild["verified_images"] != expected_verified
        or rebuild["skipped_test_images"] != expected_skipped
    ):
        raise ValueError("fixture rebuild did not respect the non-test read boundary")
    return {"export": export, "rebuild": rebuild, "check": checked}
