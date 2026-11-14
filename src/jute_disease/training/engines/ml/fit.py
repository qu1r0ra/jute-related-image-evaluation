"""The classical fits: deterministic features, train-only scaling, explicit weights.

Features come from the same frozen train and validation partitions the CNN fits
read, extracted once per policy draw and feature set into an explicit cache that
every classical fit requires. A fit scales with statistics from its training rows
only, applies balanced sample weights where the estimator supports them and
records where it does not, and never calibrates or sweeps. The test partition is
never opened.
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
    Manifest,
    ManifestDataset,
    classical_pipeline,
    file_sha256,
)
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    PAD_CUE_FEATURES,
    BaseFeatureExtractor,
    PadCueFeatureExtractor,
    RawPixelFeatureExtractor,
    build_classical_classifier,
    extract_features,
    feature_cache_key,
    select_classical_configuration,
)
from jute_disease.training.benchmark import FitSpec, Recipe
from jute_disease.training.inputs import (
    DataInputs,
    ImageCache,
    open_image_cache,
    preprocessing_record,
    target_manifest_for_fit,
)
from jute_disease.training.receipts import write_json_atomic
from jute_disease.training.results import FitResult

FIT_PARTITIONS = ("train", "val")
MODEL_FILE = "model.joblib"
WEIGHTING = "balanced"


def _extractor(features: str, recipe: Recipe) -> BaseFeatureExtractor:
    if features == PAD_CUE_FEATURES:
        return PadCueFeatureExtractor()
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
    images: ImageCache | None = None,
    variant: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float]:
    """One feature matrix, its pinned description and the seconds it took."""
    dataset = ManifestDataset(
        manifest,
        partition,
        data.target_root(pool) if pool else data.image_root,
        transform=classical_pipeline(
            recipe.image_size, include_mask=True, variant=variant
        ),
        cache_root=None if images is None else images.root,
        spec=None if images is None else images.spec,
    )
    extractor = _extractor(features, recipe)
    started = time.perf_counter()
    x, y = extract_features(
        dataset, extractor, cache_dir=cache_dir, require_cache=require_cache
    )
    seconds = time.perf_counter() - started
    info = {
        "key": feature_cache_key(dataset.identity, extractor, len(dataset)),
        "rows": int(x.shape[0]),
        "dimensions": int(x.shape[1]),
        "x_sha256": hashlib.sha256(np.ascontiguousarray(x)).hexdigest(),
        "y_sha256": hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
    }
    return x, y, info, seconds


def prepare_features(
    fit: FitSpec, recipe: Recipe, data: DataInputs, cache_dir: Path | str
) -> FitResult:
    """Extract every declared feature set for the fit's draw, once.

    ``fit`` names one policy draw (or a pool's reserve), whose train and
    validation images are extracted. The matrices are cached under ``cache_dir``
    and pinned in the receipt by digest, so the classical fits of that draw can
    be checked against exactly these features. The seconds spent extracting are
    recorded, so the compute budget counts them.

    Raises:
        ImageCacheError: an image cache is configured and does not verify.
    """
    started = time.perf_counter()
    cache = Path(cache_dir)
    manifest = target_manifest_for_fit(fit, recipe, data)
    pool = fit.pool
    images = open_image_cache(
        data,
        recipe,
        manifest,
        data.target_root(pool) if pool else data.image_root,
        FIT_PARTITIONS,
    )
    features: dict[str, dict[str, Any]] = {name: {} for name in FEATURE_EXTRACTORS}
    extraction_seconds = 0.0
    for name in FEATURE_EXTRACTORS:
        for partition in FIT_PARTITIONS:
            _, _, info, seconds = _matrix(
                manifest, partition, name, recipe, data, cache, pool=pool, images=images
            )
            features[name][partition] = info
            extraction_seconds += seconds
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
            "image_cache": None if images is None else images.identity,
            "features": features,
            "extraction_seconds": extraction_seconds,
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
    images = open_image_cache(
        data,
        recipe,
        manifest,
        data.target_root(fit.pool) if fit.pool else data.image_root,
        FIT_PARTITIONS,
    )
    x_train, y_train, train_info, _ = _matrix(
        manifest,
        "train",
        fit.features,
        recipe,
        data,
        cache,
        require_cache,
        pool=fit.pool,
        images=images,
        variant=fit.variant,
    )
    x_val, y_val, val_info, _ = _matrix(
        manifest,
        "val",
        fit.features,
        recipe,
        data,
        cache,
        require_cache,
        pool=fit.pool,
        images=images,
        variant=fit.variant,
    )
    image_cache = None if images is None else images.identity

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
        write_json_atomic(path, selection)
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
                "image_cache": image_cache,
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
            "variant": fit.variant,
            "recipe": {"image_size": recipe.image_size},
            "manifest": manifest.digest(),
            "rows": {"train": train_info["rows"], "val": val_info["rows"]},
            "feature_cache": {"train": train_info, "val": val_info},
            "image_cache": image_cache,
            "preprocessing": preprocessing_record(images),
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
