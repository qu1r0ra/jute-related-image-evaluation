"""Gated test predictions: the one way a fit's protected-test labels are produced.

``predict_test`` authorises against the frozen record, logs the read before any
test image opens, and writes the predictions with their SHA-256. Every fit,
deep or classical, is read through the same letterbox and mask pipeline over
the complete manifest, so a prediction cannot differ from the data the frozen
record binds.
"""

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset, Subset

from jute_disease.config import BATCH_SIZE
from jute_disease.data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ColorThresholdBackend,
    DataModule,
    ImageWithMask,
    Manifest,
    ManifestDataset,
    PreprocessingSpec,
    classical_pipeline,
    create_pipeline,
    load_manifest,
    load_policy_complete_manifest,
    load_policy_manifest,
)
from jute_disease.data.manifest import load_complete_manifest
from jute_disease.evaluation.breakdowns import LEAF_SAMPLE_SIZE, stratified_sample
from jute_disease.evaluation.diagnostics import letterbox_leaf_mask
from jute_disease.evaluation.freeze_record import (
    FreezeRecordError,
    TestReadRefusedError,
    bound_path,
    default_reads_log_path,
    log_test_read,
    receipt_manifest_digest,
    training_description,
    training_receipt_digest,
    verify_freeze_record,
    write_leaf_prediction_receipt,
    write_prediction_receipt,
)
from jute_disease.models.dl import Classifier, TimmBackbone, read_checkpoint_state
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    PAD_CUE_FEATURES,
    BaseFeatureExtractor,
    PadCueFeatureExtractor,
    RawPixelFeatureExtractor,
    extract_features,
)
from jute_disease.training import RunStore

PREDICTIONS_FILE = "predictions.json"
OPERATION = "predict_test"
LEAF_OPERATION = "predict_leaf_only"


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


def _recorded_image_cache(description: Mapping[str, Any]) -> object:
    """The image-cache identity a fit's receipt records, however the engine files it."""
    manifests = (description.get("lineage") or {}).get("manifests") or {}
    return description.get("image_cache") or manifests.get("image_cache")


def _preprocessing_of(
    description: Mapping[str, Any], image_size: int
) -> PreprocessingSpec | None:
    """How the fit's pixels were prepared, so test pixels are prepared the same way.

    ``None`` means the fit read the original images. A fit that read the image
    cache saw every image downsized to the cache's shortest side, so inference
    applies the same downsize in memory; it never needs a cache of the test
    images. A receipt that records a cache without saying how the pixels were
    prepared, or a spec this code cannot reproduce, is refused before the read
    is logged.
    """
    recorded = description.get("preprocessing")
    cached = _recorded_image_cache(description)
    if recorded is None:
        if cached:
            raise TestReadRefusedError(
                "test read refused: the fit read an image cache but its receipt "
                "records no preprocessing"
            )
        return None
    source = recorded.get("source") if isinstance(recorded, Mapping) else None
    if source == "original":
        if cached:
            raise TestReadRefusedError(
                "test read refused: the receipt records original pixels and an "
                "image cache"
            )
        return None
    if source != "cache" or not cached:
        raise TestReadRefusedError(
            "test read refused: the receipt's preprocessing record is inconsistent "
            "with its image cache"
        )
    try:
        spec = PreprocessingSpec(**recorded["spec"])
    except (KeyError, TypeError) as error:
        raise TestReadRefusedError(
            f"test read refused: unreadable preprocessing record: {error}"
        ) from error
    if spec.digest() != recorded.get("spec_digest"):
        raise TestReadRefusedError(
            "test read refused: the preprocessing record does not match its digest"
        )
    if spec != PreprocessingSpec(size=image_size):
        raise TestReadRefusedError(
            "test read refused: the fit's preprocessing cannot be reproduced "
            "at inference"
        )
    return spec


def _extractor(features: str, image_size: int) -> BaseFeatureExtractor:
    if features == PAD_CUE_FEATURES:
        return PadCueFeatureExtractor()
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
    weights: bytes,
    preprocessing: PreprocessingSpec | None,
    variant: str = "",
) -> tuple[list[str], list[str]]:
    module = DataModule(
        frozen_manifest=manifest,
        image_root=image_root,
        allow_test=True,
        num_workers=0,
        pin_memory=False,
        image_size=image_size,
        downsize_original=preprocessing is not None,
        variant=variant,
    )
    module.setup("predict")
    dataset = module.jute_predict
    assert isinstance(dataset, ManifestDataset)
    model = _classifier(model_name, len(manifest.labels), weights)
    indices: list[int] = []
    with torch.no_grad():
        for batch, _ in module.predict_dataloader():
            indices.extend(model(batch).argmax(dim=1).tolist())
    return list(dataset.image_ids), [manifest.labels[index] for index in indices]


def _classifier(model_name: str, classes: int, weights: bytes) -> Classifier:
    """The fit's network, loaded from the snapshot of its verified checkpoint."""
    model = Classifier(TimmBackbone(model_name, pretrained=False), classes)
    model.load_state_dict(read_checkpoint_state(io.BytesIO(weights)), strict=True)
    model.eval()
    return model


def _classical_labels(
    manifest: Manifest,
    image_root: Path,
    image_size: int,
    features: str,
    weights: bytes,
    preprocessing: PreprocessingSpec | None,
    variant: str = "",
) -> tuple[list[str], list[str]]:
    import joblib

    dataset = ManifestDataset(
        manifest,
        "test",
        image_root,
        transform=classical_pipeline(image_size, include_mask=True, variant=variant),
        downsize_spec=preprocessing,
    )
    x, _ = extract_features(dataset, _extractor(features, image_size), cache=False)
    predicted = joblib.load(io.BytesIO(weights)).predict(x)
    return list(dataset.image_ids), [manifest.labels[int(i)] for i in predicted]


def _snapshot(path: Path, kind: str, fit_id: str, expected_sha256: str) -> bytes:
    """Read the artifact once and require those bytes to be the receipt's.

    Inference loads its model from this snapshot, never from the file again, so
    weights swapped on disk after the check cannot produce predictions filed
    under the receipt's digest.

    Raises:
        TestReadRefusedError: the bytes are not the ones the receipt records.
    """
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise TestReadRefusedError(
            f"test read refused: the {kind} of {fit_id} changed after its "
            "receipt was verified"
        )
    return data


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


@dataclass(frozen=True)
class _Source:
    """A fit the gate has authorised to read the protected test partition."""

    base: Path
    run_dir: Path
    description: Mapping[str, Any]
    kind: str
    weights: bytes
    artifact_sha256: str
    receipt_digest: str
    fit_manifest: str
    manifest: Manifest
    preprocessing: PreprocessingSpec | None
    image_size: int
    pool_root: Path
    details: dict[str, str]


def _authorise(
    record: Mapping[str, Any],
    fit_id: str,
    run_store: Path | str,
    *,
    root: Path | str,
    image_root: Path | str,
    target_dir: Path | str,
    policy_dir: Path | str | None,
    pool_roots: Mapping[str, Path | str] | None,
) -> _Source:
    """Everything a test read needs from ``fit_id``, checked before any image opens."""
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
    preprocessing = _preprocessing_of(description, image_size)
    manifest = _complete_manifest(body, description, Path(target_dir), root, policy_dir)
    pool = description.get("pool") or ""
    pool_root = Path(dict(pool_roots or {}).get(pool, image_root))
    receipt_digest = training_receipt_digest(run_dir)
    artifact_sha256 = str(artifact["sha256"])
    return _Source(
        base=base,
        run_dir=run_dir,
        description=description,
        kind=kind,
        weights=_snapshot(artifact_path, kind, fit_id, artifact_sha256),
        artifact_sha256=artifact_sha256,
        receipt_digest=receipt_digest,
        fit_manifest=fit_manifest,
        manifest=manifest,
        preprocessing=preprocessing,
        image_size=image_size,
        pool_root=pool_root,
        details={
            "training_receipt_sha256": receipt_digest,
            "artifact_kind": kind,
            "artifact_sha256": artifact_sha256,
        },
    )


class LeafOnlyTransform:
    """Letterbox an image, then replace everything the leaf mask calls background.

    The mask is the audited colour-threshold backend's, computed on the image as the
    fit would read it and resized onto the letterbox canvas. Background and
    letterbox padding take the pad fill, so the fit sees the leaf and nothing else.
    """

    deterministic = True

    def __init__(self, size: int, *, is_dl: bool) -> None:
        self.size = size
        self.is_dl = is_dl
        self._backend = ColorThresholdBackend()
        self._pipeline = create_pipeline(size, False, False, include_mask=True)
        self._fill = tuple(round(value * 255) for value in IMAGENET_MEAN)
        self.description = (
            f"leaf-only;letterbox={size};dl={is_dl};"
            f"mask={self._backend.model}@{self._backend.version}"
        )

    def record(self, fit_preprocessing: object) -> dict[str, Any]:
        """The preprocessing a leaf-only receipt binds: the fit's own plus the mask."""
        return {
            "fit": fit_preprocessing,
            "leaf_only": {
                "mask_model": self._backend.model,
                "mask_version": self._backend.version,
                "fill": list(self._fill),
                "letterbox": self.size,
            },
        }

    def __call__(self, image: Any) -> torch.Tensor | np.ndarray | ImageWithMask:
        rgb = PILImage.fromarray(np.asarray(image)).convert("RGB")
        mask = self._backend.predict(rgb).mask
        letterboxed = self._pipeline(rgb)
        assert isinstance(letterboxed, ImageWithMask)
        leaf = letterbox_leaf_mask(mask, letterboxed.valid_mask)
        pixels = np.array(letterboxed.pixels)
        pixels[~leaf] = self._fill
        if not self.is_dl:
            return ImageWithMask(pixels, letterboxed.valid_mask)
        return F.normalize(
            F.to_tensor(PILImage.fromarray(pixels)), IMAGENET_MEAN, IMAGENET_STD
        )


def _deep_sample_labels(source: _Source, sample: Dataset[Any]) -> list[str]:
    model = _classifier(
        str(source.description["model"]), len(source.manifest.labels), source.weights
    )
    indices: list[int] = []
    with torch.no_grad():
        for batch, _ in DataLoader(
            sample, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        ):
            indices.extend(model(batch).argmax(dim=1).tolist())
    return [source.manifest.labels[index] for index in indices]


def _classical_sample_labels(source: _Source, sample: Dataset[Any]) -> list[str]:
    import joblib

    x, _ = extract_features(
        sample,
        _extractor(str(source.description["features"]), source.image_size),
        cache=False,
    )
    predicted = joblib.load(io.BytesIO(source.weights)).predict(x)
    return [source.manifest.labels[int(i)] for i in predicted]


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
    source = _authorise(
        record,
        fit_id,
        run_store,
        root=root,
        image_root=image_root,
        target_dir=target_dir,
        policy_dir=policy_dir,
        pool_roots=pool_roots,
    )
    # The read is on disk before the first test image opens. The project ledger
    # is always written; a custom log is an additional copy, never a replacement.
    # A repeat from a different fit state is refused under the same lock that
    # appends the row, so two readers cannot both pass the check.
    log_test_read(
        default_reads_log_path(root),
        record["hash"],
        [fit_id],
        operation=OPERATION,
        additional_logs=[Path(reads_log)] if reads_log is not None else [],
        details=source.details,
        refuse_conflict=True,
    )
    variant = str(source.description.get("variant") or "")
    if source.kind == "checkpoint":
        image_ids, labels = _deep_labels(
            source.manifest,
            source.pool_root,
            source.image_size,
            str(source.description["model"]),
            source.weights,
            source.preprocessing,
            variant,
        )
    else:
        image_ids, labels = _classical_labels(
            source.manifest,
            source.pool_root,
            source.image_size,
            str(source.description["features"]),
            source.weights,
            source.preprocessing,
            variant,
        )
    path = source.run_dir / PREDICTIONS_FILE
    digest = _write_predictions(path, image_ids, labels)
    prediction_receipt = write_prediction_receipt(
        source.run_dir,
        record_hash=record["hash"],
        fit_id=fit_id,
        training_receipt_sha256=source.receipt_digest,
        artifact_kind=source.kind,
        artifact_sha256=source.artifact_sha256,
        fit_manifest_sha256=source.fit_manifest,
        predictions_sha256=digest,
        variant=variant,
    )
    return {
        "fit_id": fit_id,
        "predictions": path.relative_to(source.base).as_posix(),
        "predictions_sha256": digest,
        "prediction_receipt_sha256": prediction_receipt,
        "images": len(image_ids),
        "freeze_record_hash": record["hash"],
        "post_hoc": bool(record["body"]["post_hoc"]),
    }


def predict_leaf_only(
    freeze_record: Path | str | None,
    fit_id: str,
    of: str,
    run_store: Path | str,
    *,
    root: Path | str,
    image_root: Path | str,
    target_dir: Path | str,
    policy_dir: Path | str | None = None,
    pool_roots: Mapping[str, Path | str] | None = None,
    reads_log: Path | str | None = None,
    sample_size: int = LEAF_SAMPLE_SIZE,
    seed: int = 42,
) -> dict[str, Any]:
    """Write a leaf-only run's test predictions and the receipt that vouches for them.

    ``of`` names the full fit whose weights predict; ``fit_id`` is the leaf-only
    run's own ID, which the freeze record must also list. The images are the
    stratified sample the evaluator expects, each read through the full fit's
    preprocessing with everything the colour-threshold mask calls background
    replaced by the letterbox fill. Like ``predict_test`` the read is logged before
    any test image opens, and the weights are the bytes the receipt verified.

    Raises:
        TestReadRefusedError: the gate refused the read.
        FreezeRecordError: the receipt or its artifacts cannot be used.
    """
    record = verify_freeze_record(freeze_record, root, target_dir)
    if fit_id == of:
        raise TestReadRefusedError(
            f"test read refused: the leaf-only run {fit_id} must differ from the "
            "fit it is made of"
        )
    if fit_id not in record["body"]["fit_ids"]:
        raise TestReadRefusedError(
            f"test read refused: {fit_id} is not in the freeze record"
        )
    source = _authorise(
        record,
        of,
        run_store,
        root=root,
        image_root=image_root,
        target_dir=target_dir,
        policy_dir=policy_dir,
        pool_roots=pool_roots,
    )
    test = source.manifest.partition("test")
    sample = stratified_sample(
        [e.image_id for e in test], [e.label for e in test], sample_size, seed
    )
    transform = LeafOnlyTransform(source.image_size, is_dl=source.kind == "checkpoint")
    log_test_read(
        default_reads_log_path(root),
        record["hash"],
        [fit_id],
        operation=LEAF_OPERATION,
        additional_logs=[Path(reads_log)] if reads_log is not None else [],
        details={
            **source.details,
            "of": of,
            "sample_size": len(sample),
            "seed": seed,
        },
        refuse_conflict=True,
    )
    dataset = ManifestDataset(
        source.manifest,
        "test",
        source.pool_root,
        transform=transform,
        downsize_spec=source.preprocessing,
    )
    position = {image_id: i for i, image_id in enumerate(dataset.image_ids)}
    chosen = Subset(dataset, [position[image_id] for image_id in sample])
    if source.kind == "checkpoint":
        labels = _deep_sample_labels(source, chosen)
    else:
        labels = _classical_sample_labels(source, chosen)
    run_dir = RunStore(source.base).run_dir(fit_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PREDICTIONS_FILE
    digest = _write_predictions(path, list(sample), labels)
    receipt = write_leaf_prediction_receipt(
        run_dir,
        record_hash=record["hash"],
        fit_id=fit_id,
        of=of,
        source_training_receipt_sha256=source.receipt_digest,
        source_artifact_kind=source.kind,
        source_artifact_sha256=source.artifact_sha256,
        preprocessing=transform.record(source.description.get("preprocessing")),
        sample_ids=sorted(sample),
        predictions_sha256=digest,
    )
    return {
        "fit_id": fit_id,
        "of": of,
        "predictions": path.relative_to(source.base).as_posix(),
        "predictions_sha256": digest,
        "leaf_receipt_sha256": receipt,
        "images": len(sample),
        "freeze_record_hash": record["hash"],
        "post_hoc": bool(record["body"]["post_hoc"]),
    }
