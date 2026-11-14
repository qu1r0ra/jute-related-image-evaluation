"""Gated test predictions: the one way a fit's protected-test labels are produced.

``predict_test`` authorises against the frozen record, logs the read before any
test image opens, and writes the predictions with their SHA-256. Every fit,
deep or classical, is read through the same letterbox and mask pipeline over
the complete manifest, so a prediction cannot differ from the data the frozen
record binds.
"""

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from jute_disease.data import (
    DataModule,
    Manifest,
    ManifestDataset,
    classical_pipeline,
    load_manifest,
    load_policy_complete_manifest,
    load_policy_manifest,
)
from jute_disease.data.manifest import load_complete_manifest
from jute_disease.evaluation.freeze_record import (
    FreezeRecordError,
    TestReadRefusedError,
    bound_path,
    default_reads_log_path,
    log_test_read,
    receipt_manifest_digest,
    refuse_conflicting_repeat,
    training_description,
    training_receipt_digest,
    verify_freeze_record,
    write_prediction_receipt,
)
from jute_disease.models.dl import Classifier, TimmBackbone, read_checkpoint_state
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    BaseFeatureExtractor,
    RawPixelFeatureExtractor,
    extract_features,
)
from jute_disease.training import RunStore

PREDICTIONS_FILE = "predictions.json"
OPERATION = "predict_test"


def _cell(description: Mapping[str, Any]) -> dict[str, Any]:
    """The pool, view, policy and draw a pooled fit declares in its receipt."""
    cell = {
        "pool": description.get("pool"),
        "view": description.get("view"),
        "policy": description.get("policy"),
        "draw": description.get("split"),
        "image_size": (description.get("recipe") or {}).get("image_size"),
    }
    if any(value is None for value in cell.values()):
        raise TestReadRefusedError(
            "test read refused: the training receipt does not name its policy cell"
        )
    return cell


def _bound_policy_dir(
    body: Mapping[str, Any], root: Path | str, policy_dir: Path | str | None
) -> Path:
    bound = bound_path(body["policy_binding"]["path"], root)
    if policy_dir is not None and Path(policy_dir).resolve() != bound.resolve():
        raise TestReadRefusedError(
            "policy dir is not the one bound by the freeze record"
        )
    return bound


def _require_frozen_cell(body: Mapping[str, Any], cell: Mapping[str, Any]) -> None:
    matrix = body["policy_binding"]["matrix"]
    for field, key in (
        ("pools", "pool"),
        ("views", "view"),
        ("policies", "policy"),
        ("draws", "draw"),
    ):
        if cell[key] not in matrix[field]:
            raise TestReadRefusedError(
                f"test read refused: {key} {cell[key]!r} is outside the frozen matrix"
            )


def _check_fit_manifest(
    body: Mapping[str, Any],
    description: Mapping[str, Any],
    fit_id: str,
    target_dir: Path,
    root: Path | str,
    policy_dir: Path | str | None,
) -> str:
    """Require the fit to have trained on the draw the freeze record binds.

    Returns the digest of that train manifest. A fit trained on another draw,
    another policy artifact or a manifest from before a re-freeze records a
    different digest and is refused before any test image opens.
    """
    recorded = receipt_manifest_digest(description)
    if not description.get("pool"):
        if policy_dir is not None:
            raise TestReadRefusedError("this fit has no pool; do not pass a policy dir")
        expected = load_manifest(target_dir).digest()
    else:
        cell = _cell(description)
        _require_frozen_cell(body, cell)
        expected = load_policy_manifest(
            _bound_policy_dir(body, root, policy_dir),
            pool=cell["pool"],
            view=int(cell["view"]),
            policy=str(cell["policy"]),
            draw=int(cell["draw"]),
            image_size=int(cell["image_size"]),
        ).digest()
    if recorded != expected:
        raise TestReadRefusedError(
            f"test read refused: {fit_id} was not trained on the manifest the "
            "freeze record binds"
        )
    return recorded


def _complete_manifest(
    body: Mapping[str, Any],
    description: Mapping[str, Any],
    target_dir: Path,
    root: Path | str,
    policy_dir: Path | str | None,
) -> Manifest:
    """The complete manifest of the data this fit trained on, from the record."""
    if not description.get("pool"):
        return load_complete_manifest(target_dir)
    cell = _cell(description)
    return load_policy_complete_manifest(
        _bound_policy_dir(body, root, policy_dir),
        pool=cell["pool"],
        view=int(cell["view"]),
        policy=str(cell["policy"]),
        draw=int(cell["draw"]),
        image_size=int(cell["image_size"]),
    )


def _extractor(features: str, image_size: int) -> BaseFeatureExtractor:
    if features not in FEATURE_EXTRACTORS:
        raise FreezeRecordError(f"unknown feature set {features!r}")
    if features == "raw":
        return RawPixelFeatureExtractor(image_size)
    return FEATURE_EXTRACTORS[features]()


def _deep_labels(
    manifest: Manifest,
    image_root: Path,
    image_size: int,
    model_name: str,
    checkpoint: Path,
) -> tuple[list[str], list[str]]:
    module = DataModule(
        frozen_manifest=manifest,
        image_root=image_root,
        allow_test=True,
        num_workers=0,
        pin_memory=False,
        image_size=image_size,
    )
    module.setup("predict")
    dataset = module.jute_predict
    assert isinstance(dataset, ManifestDataset)
    model = Classifier(TimmBackbone(model_name, pretrained=False), len(manifest.labels))
    model.load_state_dict(read_checkpoint_state(checkpoint), strict=True)
    model.eval()
    indices: list[int] = []
    with torch.no_grad():
        for batch, _ in module.predict_dataloader():
            indices.extend(model(batch).argmax(dim=1).tolist())
    return list(dataset.image_ids), [manifest.labels[index] for index in indices]


def _classical_labels(
    manifest: Manifest,
    image_root: Path,
    image_size: int,
    features: str,
    model_path: Path,
) -> tuple[list[str], list[str]]:
    import joblib

    dataset = ManifestDataset(
        manifest,
        "test",
        image_root,
        transform=classical_pipeline(image_size, include_mask=True),
    )
    x, _ = extract_features(dataset, _extractor(features, image_size), cache=False)
    predicted = joblib.load(model_path).predict(x)
    return list(dataset.image_ids), [manifest.labels[int(i)] for i in predicted]


def _write_predictions(path: Path, image_ids: list[str], labels: list[str]) -> str:
    ordered = sorted(zip(image_ids, labels, strict=True))
    payload = {
        "image_ids": [image_id for image_id, _ in ordered],
        "predictions": [label for _, label in ordered],
    }
    data = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise FreezeRecordError(f"predictions already exist and differ: {path}")
    handle, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)
    path.with_name(f"{path.name}.sha256").write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    return digest


def predict_test(
    freeze_record: Path | str | None,
    fit_id: str,
    run_store: Path | str,
    *,
    root: Path | str,
    image_root: Path | str,
    target_dir: Path | str,
    policy_dir: Path | str | None = None,
    pool_roots: Mapping[str, Path | str] | None = None,
    reads_log: Path | str | None = None,
) -> dict[str, Any]:
    """Write one fit's protected-test predictions after the gate allows the read.

    The record must list ``fit_id``, the run store must hold its verified
    receipt, and the read is logged before any test image is opened. A fit with
    a pool is read through the policy manifest the record binds; a pool-free fit
    through the frozen target's complete manifest.

    Raises:
        TestReadRefusedError: the gate refused the read.
        FreezeRecordError: the receipt or its artifacts cannot be used.
    """
    record = verify_freeze_record(freeze_record, root, target_dir)
    body = record["body"]
    if fit_id not in body["fit_ids"]:
        raise TestReadRefusedError(
            f"test read refused: {fit_id} is not in the freeze record"
        )
    base = Path(run_store).resolve()
    store = RunStore(base)
    try:
        receipt = store.receipt(fit_id, missing_ok=True)
    except (OSError, ValueError) as error:
        raise TestReadRefusedError(
            f"training receipt of {fit_id} does not verify: {error}"
        ) from error
    if receipt is None:
        raise TestReadRefusedError(
            f"test read refused: no verified training receipt for {fit_id}"
        )
    description, kind, artifact = training_description(receipt, fit_id)
    run_dir = store.run_dir(fit_id)
    artifact_path = (run_dir / str(artifact["path"])).resolve()
    if not artifact_path.is_relative_to(base) or not artifact_path.is_file():
        raise FreezeRecordError(f"{kind} of {fit_id} is missing or escapes the store")
    fit_manifest = _check_fit_manifest(
        body, description, fit_id, Path(target_dir), root, policy_dir
    )
    image_size = int(description["recipe"]["image_size"])
    manifest = _complete_manifest(body, description, Path(target_dir), root, policy_dir)
    pool = description.get("pool") or ""
    roots = dict(pool_roots or {})
    pool_root = Path(roots.get(pool, image_root))

    receipt_digest = training_receipt_digest(run_dir)
    details = {
        "training_receipt_sha256": receipt_digest,
        "artifact_kind": kind,
        "artifact_sha256": str(artifact["sha256"]),
    }
    ledger = default_reads_log_path(root)
    extra_logs = [Path(reads_log)] if reads_log is not None else []
    refuse_conflicting_repeat(
        [ledger, *extra_logs], record["hash"], fit_id, OPERATION, details
    )
    # The read is on disk before the first test image opens. The project ledger
    # is always written; a custom log is an additional copy, never a replacement.
    log_test_read(
        ledger,
        record["hash"],
        [fit_id],
        operation=OPERATION,
        additional_logs=extra_logs,
        details=details,
    )
    if kind == "checkpoint":
        image_ids, labels = _deep_labels(
            manifest,
            pool_root,
            image_size,
            str(description["model"]),
            artifact_path,
        )
    else:
        image_ids, labels = _classical_labels(
            manifest,
            pool_root,
            image_size,
            str(description["features"]),
            artifact_path,
        )
    path = run_dir / PREDICTIONS_FILE
    digest = _write_predictions(path, image_ids, labels)
    prediction_receipt = write_prediction_receipt(
        run_dir,
        record_hash=record["hash"],
        fit_id=fit_id,
        training_receipt_sha256=receipt_digest,
        artifact_kind=kind,
        artifact_sha256=str(artifact["sha256"]),
        fit_manifest_sha256=fit_manifest,
        predictions_sha256=digest,
    )
    return {
        "fit_id": fit_id,
        "predictions": path.relative_to(base).as_posix(),
        "predictions_sha256": digest,
        "prediction_receipt_sha256": prediction_receipt,
        "images": len(image_ids),
        "freeze_record_hash": record["hash"],
        "post_hoc": bool(body["post_hoc"]),
    }
