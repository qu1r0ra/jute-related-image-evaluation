"""Receipt-gated evaluation over small, label-only fixture predictions.

This module validates the frozen target identity and completed run receipts
before it opens any prediction artifacts. It intentionally reads manifest
metadata only; it has no image, checkpoint, training, or test-scoring path.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jute_disease.data import load_manifest, receipt_for
from jute_disease.evaluation.scores import (
    Evaluation,
    PredictionSet,
    evaluate_predictions,
)


class FixtureEvaluationError(ValueError):
    """The freeze or run receipts do not authorize this fixture evaluation."""


@dataclass(frozen=True)
class FixtureEvaluationOutput:
    """Paths and calculated scores from a deterministic fixture evaluation."""

    predictions_path: Path
    metrics_path: Path
    evaluation: Evaluation


@dataclass(frozen=True)
class _RunRecord:
    fit_id: str
    identity: str
    seed: int
    prediction_path: Path
    prediction_sha256: str
    prediction_bytes: int


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise FixtureEvaluationError(f"Cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise FixtureEvaluationError(f"{description} must contain a JSON object")
    return payload


def _require_digest(value: str | None, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FixtureEvaluationError(f"{name} is required as a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise FixtureEvaluationError(f"{name} must be a SHA-256 digest") from error
    return value


def _run_records(
    run_receipts: Sequence[Path | str],
    expected_run_identities: Mapping[str, str],
    config_digest: str,
    manifest_digest: str,
) -> tuple[_RunRecord, ...]:
    if not expected_run_identities:
        raise FixtureEvaluationError("at least one expected run identity is required")

    records: list[_RunRecord] = []
    seen: set[str] = set()
    for receipt_arg in run_receipts:
        receipt_path = Path(receipt_arg)
        receipt = _read_json(receipt_path, "run receipt")
        fit_id = receipt.get("fit_id")
        identity = receipt.get("identity")
        if not isinstance(fit_id, str) or not fit_id or fit_id in seen:
            raise FixtureEvaluationError("run receipts need unique fit identities")
        seen.add(fit_id)
        expected_identity = expected_run_identities.get(fit_id)
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(expected_identity, str)
            or not expected_identity
            or expected_identity != identity
        ):
            raise FixtureEvaluationError(f"run identity does not match for {fit_id}")

        results = receipt.get("results")
        if not isinstance(results, dict):
            raise FixtureEvaluationError(f"run receipt for {fit_id} has no results")
        if results.get("config_digest") != config_digest:
            raise FixtureEvaluationError(
                f"run configuration does not match for {fit_id}"
            )
        manifests = results.get("manifests")
        if (
            not isinstance(manifests, dict)
            or manifests.get("target") != manifest_digest
        ):
            raise FixtureEvaluationError(f"target manifest does not match for {fit_id}")
        if results.get("partition") != "val":
            raise FixtureEvaluationError(
                f"fixture run {fit_id} is not for the val partition"
            )
        seed = results.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise FixtureEvaluationError(f"run receipt for {fit_id} has no valid seed")

        artifacts = receipt.get("artifacts")
        prediction = (
            artifacts.get("predictions") if isinstance(artifacts, dict) else None
        )
        if not isinstance(prediction, dict):
            raise FixtureEvaluationError(
                f"run receipt for {fit_id} has no predictions artifact"
            )
        relative_path = prediction.get("path")
        sha256 = prediction.get("sha256")
        byte_count = prediction.get("bytes")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
        ):
            raise FixtureEvaluationError(
                f"run receipt for {fit_id} has an invalid predictions artifact"
            )
        try:
            int(sha256, 16)
        except ValueError as error:
            raise FixtureEvaluationError(
                f"run receipt for {fit_id} has an invalid predictions digest"
            ) from error
        records.append(
            _RunRecord(
                fit_id=fit_id,
                identity=identity,
                seed=seed,
                prediction_path=receipt_path.parent / relative_path,
                prediction_sha256=sha256,
                prediction_bytes=byte_count,
            )
        )

    if seen != set(expected_run_identities):
        raise FixtureEvaluationError(
            "run receipts do not match the expected run identities"
        )
    if len({record.seed for record in records}) != len(records):
        raise FixtureEvaluationError("run receipts must have unique seeds")
    return tuple(sorted(records, key=lambda item: item.seed))


def _load_predictions(
    records: Sequence[_RunRecord], image_ids: tuple[str, ...], classes: tuple[str, ...]
) -> dict[int, tuple[str, ...]]:
    predictions: dict[int, tuple[str, ...]] = {}
    for record in records:
        try:
            raw = record.prediction_path.read_bytes()
        except OSError as error:
            raise FixtureEvaluationError(
                f"Cannot read predictions for {record.fit_id}"
            ) from error
        if len(raw) != record.prediction_bytes or hashlib.sha256(raw).hexdigest() != (
            record.prediction_sha256
        ):
            raise FixtureEvaluationError(
                f"Predictions artifact does not match receipt for {record.fit_id}"
            )
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as error:
            raise FixtureEvaluationError(
                f"Predictions artifact for {record.fit_id} is invalid JSON"
            ) from error
        artifact_image_ids = (
            payload.get("image_ids") if isinstance(payload, dict) else None
        )
        if (
            not isinstance(artifact_image_ids, list)
            or tuple(artifact_image_ids) != image_ids
        ):
            raise FixtureEvaluationError(
                f"Predictions for {record.fit_id} do not match frozen val image ids"
            )
        labels = payload.get("predictions")
        if not isinstance(labels, list) or len(labels) != len(image_ids):
            raise FixtureEvaluationError(
                f"Predictions for {record.fit_id} must have one label per val image"
            )
        if any(label not in classes for label in labels):
            raise FixtureEvaluationError(
                f"Predictions for {record.fit_id} contain an unknown class"
            )
        predictions[record.seed] = tuple(labels)
    return predictions


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def evaluate_fixture_runs(
    *,
    target_dir: Path | str,
    target_freeze_identity: str | None,
    config_digest: str,
    expected_run_identities: Mapping[str, str],
    run_receipts: Sequence[Path | str],
    output_dir: Path | str,
) -> FixtureEvaluationOutput:
    """Validate receipts and score synthetic validation labels deterministically.

    ``target_freeze_identity`` is the frozen manifest SHA-256. Every run receipt
    must match its expected identity, the same configuration digest, and that
    target manifest. All receipts are checked before any prediction file opens.
    Only ``val`` entries from the manifest are used; their image paths are never
    read. This API is intended for fixtures until the #720 integration arrives.
    """
    target_identity = _require_digest(target_freeze_identity, "target freeze identity")
    config_identity = _require_digest(config_digest, "configuration digest")
    manifest = load_manifest(Path(target_dir))
    if manifest.digest() != target_identity:
        raise FixtureEvaluationError("target freeze identity does not match manifest")
    freeze_receipt = _read_json(Path(target_dir) / "receipt.json", "freeze receipt")
    if freeze_receipt != receipt_for(manifest):
        raise FixtureEvaluationError(
            "target freeze receipt is incompatible with manifest"
        )

    records = _run_records(
        run_receipts,
        expected_run_identities,
        config_identity,
        target_identity,
    )

    validation_entries = tuple(
        sorted(manifest.partition("val"), key=lambda entry: entry.image_id)
    )
    image_ids = tuple(entry.image_id for entry in validation_entries)
    classes = tuple(manifest.labels)
    predictions = _load_predictions(records, image_ids, classes)
    prediction_set = PredictionSet(
        classes=classes,
        image_ids=image_ids,
        truth=tuple(entry.label for entry in validation_entries),
        groups=tuple(entry.group for entry in validation_entries),
        sources=tuple(entry.source for entry in validation_entries),
        predictions=predictions,
    )
    evaluation = evaluate_predictions(prediction_set)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "predictions.json"
    metrics_path = output / "metrics.json"
    _write_json(
        predictions_path,
        {
            "config_digest": config_identity,
            "manifest_digest": target_identity,
            "classes": list(classes),
            "image_ids": list(image_ids),
            "truth": list(prediction_set.truth),
            "predictions": {
                str(seed): list(prediction_set.predictions[seed])
                for seed in prediction_set.seeds
            },
            "run_identities": {record.fit_id: record.identity for record in records},
        },
    )
    _write_json(
        metrics_path,
        {
            "config_digest": config_identity,
            "manifest_digest": target_identity,
            "metrics": asdict(evaluation),
        },
    )
    return FixtureEvaluationOutput(predictions_path, metrics_path, evaluation)
