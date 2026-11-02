"""The classical fits: deterministic features, train-only scaling, explicit weights.

Features come from the same frozen train and validation partitions the CNN fits
read, extracted once per feature set into an explicit cache. A fit scales with
statistics from its training rows only, applies balanced sample weights where the
estimator supports them and records where it does not, and never calibrates or
sweeps. The test partition is never opened.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from jute_disease.data import (
    Manifest,
    ManifestDataset,
    classical_pipeline,
    file_sha256,
    load_manifest,
)
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    BaseFeatureExtractor,
    RawPixelFeatureExtractor,
    build_classical_classifier,
    extract_features,
    feature_cache_key,
    select_classical_configuration,
)
from jute_disease.training.benchmark import FitSpec, Recipe
from jute_disease.training.inputs import DataInputs, target_manifest_for_fit
from jute_disease.training.results import FitResult

FIT_PARTITIONS = ("train", "val")
MODEL_FILE = "model.joblib"
WEIGHTING = "balanced"


def _extractor(features: str, recipe: Recipe) -> BaseFeatureExtractor:
    if features not in FEATURE_EXTRACTORS:
        raise ValueError(
            f"Unknown features {features!r}; use one of {sorted(FEATURE_EXTRACTORS)}"
        )
    if features == "raw":
        return RawPixelFeatureExtractor(recipe.image_size)
    return FEATURE_EXTRACTORS[features]()


def _matrix(
    manifest: Manifest,
    partition: str,
    features: str,
    recipe: Recipe,
    data: DataInputs,
    cache_dir: Path,
    require_cache: bool = False,
    pool: str = "",
    training_view: str = "raw-training",
    resize: str = "letterbox",
    mask_checkpoint: Path | None = None,
    fixture_masks: bool = False,
    border_frame_width_px: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = ManifestDataset(
        manifest,
        partition,
        data.target_root(pool) if pool else data.image_root,
        transform=classical_pipeline(
            recipe.image_size,
            include_mask=True,
            resize=resize,
            training_view=training_view,
            mask_checkpoint=mask_checkpoint,
            fixture_masks=fixture_masks,
            border_frame_width_px=border_frame_width_px,
        ),
    )
    extractor = _extractor(features, recipe)
    x, y = extract_features(
        dataset, extractor, cache_dir=cache_dir, require_cache=require_cache
    )
    info = {
        "key": feature_cache_key(dataset.identity, extractor, len(dataset)),
        "rows": int(x.shape[0]),
        "dimensions": int(x.shape[1]),
        "x_sha256": hashlib.sha256(np.ascontiguousarray(x)).hexdigest(),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
    }
    return x, y, info


def prepare_features(
    fit: FitSpec, recipe: Recipe, data: DataInputs, cache_dir: Path | str
) -> FitResult:
    """Extract every declared feature set for train and validation, once.

    The matrices are cached under ``cache_dir`` and pinned in the receipt by
    digest, so later fits can be checked against exactly these features.
    """
    started = time.perf_counter()
    cache = Path(cache_dir)
    manifest = load_manifest(data.target_dir)
    features: dict[str, dict[str, Any]] = {}
    for name in FEATURE_EXTRACTORS:
        features[name] = {
            partition: _matrix(manifest, partition, name, recipe, data, cache)[2]
            for partition in FIT_PARTITIONS
        }
    artifacts = {
        f"{name}-{partition}-{suffix}": cache / f"{entry['key']}{extension}"
        for name, partitions in features.items()
        for partition, entry in partitions.items()
        for suffix, extension in (("x", "_X.npy"), ("y", "_y.npy"), ("meta", ".json"))
    }
    return FitResult(
        {
            "fit": fit.fit_id,
            "recipe": {"image_size": recipe.image_size},
            "manifest": manifest.digest(),
            "features": features,
            "seconds": time.perf_counter() - started,
            "test_accessed": False,
        },
        artifacts,
    )


def fit_classical(
    fit: FitSpec,
    recipe: Recipe,
    data: DataInputs,
    attempt_dir: Path | str,
    cache_dir: Path | str,
    *,
    require_cache: bool = False,
    selected_parameters: dict[str, object] | None = None,
    training_view: str = "raw-training",
    resize: str = "letterbox",
    mask_checkpoint: Path | None = None,
    fixture_masks: bool = False,
    border_frame_width_px: int = 0,
) -> FitResult:
    """Fit one estimator on one feature set and describe exactly what happened.

    Raises:
        ValueError: the estimator or feature set is not declared.
    """
    from jute_disease.models.ml import ML_CLASSIFIERS

    if fit.estimator not in ML_CLASSIFIERS:
        raise ValueError(
            f"Unknown estimator {fit.estimator!r}; use one of {sorted(ML_CLASSIFIERS)}"
        )
    _extractor(fit.features, recipe)
    attempt = Path(attempt_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    manifest = target_manifest_for_fit(fit, recipe, data)
    x_train, y_train, train_info = _matrix(
        manifest,
        "train",
        fit.features,
        recipe,
        data,
        cache,
        require_cache,
        pool=fit.pool,
        training_view=training_view,
        resize=resize,
        mask_checkpoint=mask_checkpoint,
        fixture_masks=fixture_masks,
        border_frame_width_px=border_frame_width_px,
    )
    x_val, y_val, val_info = _matrix(
        manifest,
        "val",
        fit.features,
        recipe,
        data,
        cache,
        require_cache,
        pool=fit.pool,
        training_view=training_view,
        resize=resize,
        mask_checkpoint=mask_checkpoint,
        fixture_masks=fixture_masks,
        border_frame_width_px=border_frame_width_px,
    )

    if fit.phase == "tune":
        selection = select_classical_configuration(
            fit.estimator,
            fit.features,
            fit.seed,
            x_train,
            y_train,
            x_val,
            y_val,
        )
        path = attempt / "tuning.json"
        path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return FitResult(
            {
                "fit": fit.fit_id,
                "phase": fit.phase,
                "estimator": fit.estimator,
                "features": fit.features,
                "pool": fit.pool,
                "selected_parameters": selection["selected"],
                "selected_reserve_macro_f1": selection["selected_macro_f1"],
                "configurations": selection["configurations"],
                "reserve": train_info | {"validation": val_info},
                "scaling": "train-only",
                "test_accessed": False,
            },
            {"tuning": path},
        )

    model = build_classical_classifier(
        fit.estimator,
        fit.seed,
        fit.features,
        selected_parameters,
        dimensions=x_train.shape[1],
    )
    weights = compute_sample_weight(WEIGHTING, y_train)
    started = time.perf_counter()
    model.fit(x_train, y_train, sample_weight=weights)
    fit_seconds = time.perf_counter() - started
    predicted = model.predict(x_val)

    path = attempt / MODEL_FILE
    joblib.dump(model.model, path)
    return FitResult(
        {
            "fit": fit.fit_id,
            "estimator": fit.estimator,
            "features": fit.features,
            "seed": fit.seed,
            "pool": fit.pool,
            "view": fit.view,
            "policy": fit.policy,
            "split": fit.split,
            "manifest": manifest.digest(),
            "rows": {"train": train_info["rows"], "val": val_info["rows"]},
            "feature_cache": {"train": train_info, "val": val_info},
            "scaling": "train-only",
            "weighting": {"requested": WEIGHTING, "status": model.weighting},
            "calibration": False,
            "sweep": selected_parameters is not None,
            "selected_parameters": selected_parameters,
            "metrics": {
                "val_accuracy": float(accuracy_score(y_val, predicted)),
                "val_macro_f1": float(
                    f1_score(y_val, predicted, average="macro", zero_division=0)
                ),
            },
            "predictions_sha256": hashlib.sha256(
                np.ascontiguousarray(predicted.astype(np.int64))
            ).hexdigest(),
            "fit_seconds": fit_seconds,
            "model": {"path": MODEL_FILE, "sha256": file_sha256(path)},
            "test_accessed": False,
        },
        {"model": path},
    )


def fit_cue(
    fit: FitSpec,
    recipe: Recipe,
    data: DataInputs,
    attempt_dir: Path | str,
    *,
    cue: str,
) -> FitResult:
    """Fit a metadata-only border cue from training rows and score validation."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if cue not in {"pad-fraction", "aspect-ratio"}:
        raise ValueError(f"unsupported metadata cue {cue!r}")
    manifest = target_manifest_for_fit(fit, recipe, data)

    def matrix(partition: str) -> tuple[np.ndarray, np.ndarray]:
        entries = manifest.partition(partition)
        values = [
            entry.pad_fraction if cue == "pad-fraction" else entry.width / entry.height
            for entry in entries
        ]
        if any(value is None for value in values):
            raise ValueError(f"{cue} requires recorded image dimensions")
        labels = {label: index for index, label in enumerate(manifest.labels)}
        return (
            np.asarray(values, dtype=np.float64).reshape(-1, 1),
            np.asarray([labels[entry.label] for entry in entries], dtype=np.int64),
        )

    x_train, y_train = matrix("train")
    x_val, y_val = matrix("val")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=fit.seed
        ),
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_val)
    attempt = Path(attempt_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    path = attempt / MODEL_FILE
    joblib.dump(model, path)
    return FitResult(
        {
            "fit": fit.fit_id,
            "cue": cue,
            "seed": fit.seed,
            "pool": fit.pool,
            "view": fit.view,
            "policy": fit.policy,
            "split": fit.split,
            "manifest": manifest.digest(),
            "rows": {"train": int(len(y_train)), "val": int(len(y_val))},
            "metrics": {
                "val_accuracy": float(accuracy_score(y_val, predicted)),
                "val_macro_f1": float(
                    f1_score(y_val, predicted, average="macro", zero_division=0)
                ),
            },
            "selected_parameters": None,
            "test_accessed": False,
        },
        {"model": path},
    )
