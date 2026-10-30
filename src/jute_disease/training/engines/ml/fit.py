"""The classical fits: deterministic features, train-only scaling, explicit weights.

Features come from the same frozen train and validation partitions the CNN fits
read, extracted once per feature set into an explicit cache. A fit scales with
statistics from its training rows only, applies balanced sample weights where the
estimator supports them and records where it does not, and never calibrates or
sweeps. The test partition is never opened.
"""

import hashlib
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight

from jute_disease.data import (
    ManifestDataset,
    classical_pipeline,
    file_sha256,
    load_manifest,
)
from jute_disease.data.manifest import Manifest
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    ML_CLASSIFIERS,
    BaseFeatureExtractor,
    RawPixelFeatureExtractor,
    extract_features,
)
from jute_disease.models.ml.features import feature_cache_key
from jute_disease.training.benchmark import FitSpec, Recipe
from jute_disease.training.inputs import DataInputs
from jute_disease.training.results import FitResult

PARTITIONS = ("train", "val")
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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = ManifestDataset(
        manifest,
        partition,
        data.image_root,
        transform=classical_pipeline(recipe.image_size),
    )
    extractor = _extractor(features, recipe)
    x, y = extract_features(dataset, extractor, cache_dir=cache_dir)
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
            for partition in PARTITIONS
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
        {},
    )


def fit_classical(
    fit: FitSpec,
    recipe: Recipe,
    data: DataInputs,
    attempt_dir: Path | str,
    cache_dir: Path | str,
) -> FitResult:
    """Fit one estimator on one feature set and describe exactly what happened.

    Raises:
        ValueError: the estimator or feature set is not declared.
    """
    if fit.estimator not in ML_CLASSIFIERS:
        raise ValueError(
            f"Unknown estimator {fit.estimator!r}; use one of {sorted(ML_CLASSIFIERS)}"
        )
    _extractor(fit.features, recipe)
    attempt = Path(attempt_dir)
    attempt.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir)
    manifest = load_manifest(data.target_dir)
    x_train, y_train, train_info = _matrix(
        manifest, "train", fit.features, recipe, data, cache
    )
    x_val, y_val, val_info = _matrix(manifest, "val", fit.features, recipe, data, cache)

    model = ML_CLASSIFIERS[fit.estimator](random_state=fit.seed)
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
            "manifest": manifest.digest(),
            "rows": {"train": train_info["rows"], "val": val_info["rows"]},
            "feature_cache": {"train": train_info, "val": val_info},
            "scaling": "train-only",
            "weighting": {"requested": WEIGHTING, "status": model.weighting},
            "calibration": False,
            "sweep": False,
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
