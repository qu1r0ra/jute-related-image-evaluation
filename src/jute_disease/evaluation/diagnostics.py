"""Reproducible, descriptive embedding and CAM diagnostics.

This module contains the numerical kernels and the shared authorization/export
helpers used by the gated evaluator. It never assigns scientific meaning to
embedding distances or CAM locations.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn

from jute_disease.data import POLICIES
from jute_disease.evaluation.freeze_record import (
    Authorization,
    FreezeRecordError,
    RunEntry,
)

EMBEDDING_ANALYSIS_ID = "embeddings-test-pass"
GRADCAM_ANALYSIS_ID = "gradcam-test-pass"
EMBEDDING_SEED = 0
PCA_COMPONENTS = 50
TSNE_PERPLEXITIES = (30, 50)
UMAP_NEIGHBORS = (15, 50)
CAM_SAMPLE_PER_CLASS = 8
CAM_SAMPLE_SEED = 0
OUTPUT_MANIFEST = "output-manifest.json"
# The three sanity-check constants below (sample size, cutoff, defined share) are
# prespecified heuristic thresholds, not the output of a significance test. A
# verdict describes the sampled images only and does not generalise past them.
SANITY_SAMPLE_SIZE = 60
SANITY_SEED = 0
SANITY_MAX_MEDIAN_CORRELATION = 0.5
SANITY_MIN_DEFINED_SHARE = 0.5
SANITY_REFIT_STEPS = 100
SANITY_REFIT_LR = 0.1
SANITY_PERMUTATION_ATTEMPTS = 100


def diagnostic_configuration() -> dict[str, Any]:
    """Return the frozen settings used by the diagnostic kernels."""
    return {
        "embeddings": {
            "pca_components": PCA_COMPONENTS,
            "pca_solver": "full",
            "pca_whiten": False,
            "tsne_perplexities": list(TSNE_PERPLEXITIES),
            "umap_n_neighbors": list(UMAP_NEIGHBORS),
            "seed": EMBEDDING_SEED,
            "coordinates": 2,
            "feature_distance": "euclidean",
            "umap_n_epochs": 500,
            "umap_min_dist": 0.1,
            "interpretation": "supplementary descriptive plots only",
        },
        "gradcam": {
            "sample_per_class_correct": CAM_SAMPLE_PER_CLASS,
            "sample_per_class_errors": CAM_SAMPLE_PER_CLASS,
            "sample_seed": CAM_SAMPLE_SEED,
            "target_layer": "MobileNetV2 conv_head (last convolutional block)",
            "normalization": "ImageNet mean/std",
            "attention_score": (
                "positive CAM mass in leaf mask minus leaf area fraction"
            ),
            "padding": "exclude letterbox padding from mass and area",
            "sanity_checks": ["parameter-randomization", "label-randomization"],
        },
    }


def require_analysis(
    authorization: Authorization, analysis_id: str, fit_ids: Sequence[str]
) -> None:
    """Require a listed analysis ID and authorized fit IDs before test reads."""
    if analysis_id not in authorization.body.get("analysis_ids", ()):
        raise FreezeRecordError(
            f"test read refused: analysis {analysis_id!r} is not listed "
            "in the freeze record"
        )
    authorized = {run.fit_id for run in authorization.runs}
    missing = sorted(set(fit_ids) - authorized)
    if missing:
        raise FreezeRecordError(
            f"test read refused: diagnostic fits are not authorized: {missing}"
        )


def select_median_validation_run(runs: Sequence[RunEntry]) -> RunEntry:
    """Select the validation-median run, breaking equal scores by lower seed."""
    if not runs:
        raise ValueError("at least one authorized run is required")
    if any(run.validation_macro_f1 is None for run in runs):
        raise ValueError(
            "every candidate run needs a receipt-bound validation macro-F1"
        )
    median_score = sorted(float(run.validation_macro_f1) for run in runs)[
        len(runs) // 2
    ]
    median_runs = [
        run for run in runs if float(run.validation_macro_f1) == median_score
    ]
    return min(median_runs, key=lambda run: run.seed)


def validate_primary_policy_runs(
    assignment_table_path: Path | str, runs: Sequence[RunEntry]
) -> dict[str, Any]:
    """Bind one primary-split ImageNet policy's five seeds to #137 assignments."""
    if len(runs) != 5:
        raise FreezeRecordError(
            "diagnostics require all five authorized seeds for one policy"
        )
    if any(
        run.system != "mobilenet_v2_direct"
        or run.variant != "full"
        or run.split != 0
        or run.policy not in POLICIES
        or run.seed_index is None
        or run.pool is None
        or run.view is None
        or run.policy is None
        for run in runs
    ):
        raise FreezeRecordError(
            "diagnostics require full-image ImageNet runs for primary split 0"
        )
    if (
        len({run.seed for run in runs}) != 5
        or {run.seed_index for run in runs} != set(range(5))
        or len({(run.pool, run.view, run.policy) for run in runs}) != 1
    ):
        raise FreezeRecordError(
            "diagnostics runs must cover five unique seeds for one pool, "
            "view and policy"
        )

    path = Path(assignment_table_path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if any(run.assignment_table_sha256 != digest for run in runs):
        raise FreezeRecordError(
            "assignment table digest does not match every authorized training receipt"
        )
    try:
        rows = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FreezeRecordError("assignment table is not valid JSON") from error
    if not isinstance(rows, list) or not rows:
        raise FreezeRecordError("assignment table must contain its frozen rows")
    assignments: dict[tuple[int, int], int] = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or any(
                type(row.get(key)) is not int for key in ("split", "seed", "checkpoint")
            )
            or row["checkpoint"] != (row["seed"] + row["split"]) % 3
        ):
            raise FreezeRecordError("assignment table fails the #137 rotation rule")
        key = (row["split"], row["seed"])
        if key in assignments:
            raise FreezeRecordError("assignment table has duplicate split/seed rows")
        assignments[key] = row["checkpoint"]
    selected_assignments = {}
    for run in runs:
        key = (0, int(run.seed_index))
        if key not in assignments:
            raise FreezeRecordError(f"assignment table is missing row {key}")
        selected_assignments[str(run.seed)] = assignments[key]
    pool, view, policy = next(iter({(run.pool, run.view, run.policy) for run in runs}))
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "primary_split": 0,
        "pool": pool,
        "view": view,
        "policy": policy,
        "source_checkpoint_by_seed": selected_assignments,
    }


def embedding_coordinates(
    features: np.ndarray, image_ids: Sequence[str]
) -> dict[str, Any]:
    """Compute frozen-feature PCA, t-SNE and UMAP coordinates deterministically.

    Each requested setting is fitted to the complete, ID-aligned input matrix.
    The t-SNE perplexities require at least 51 test images, and PCA requires at
    least 50 feature dimensions.
    """
    values = np.asarray(features, dtype=np.float64)
    ids = tuple(image_ids)
    if values.ndim != 2 or values.shape[0] != len(ids):
        raise ValueError("features must be a 2D matrix aligned to image_ids")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("image_ids must be non-empty and unique")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values")
    if values.shape[0] <= max(TSNE_PERPLEXITIES):
        raise ValueError("t-SNE perplexity 50 requires at least 51 test images")
    if values.shape[1] < PCA_COMPONENTS:
        raise ValueError("PCA to 50 dimensions requires at least 50 feature columns")

    pca = PCA(
        n_components=PCA_COMPONENTS,
        svd_solver="full",
        whiten=False,
        copy=True,
        tol=0.0,
    )
    reduced = pca.fit_transform(values)
    coordinates: dict[str, list[list[float]]] = {}
    for perplexity in TSNE_PERPLEXITIES:
        embedded = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=1000,
            random_state=EMBEDDING_SEED,
            method="barnes_hut",
            metric="euclidean",
            angle=0.5,
            early_exaggeration=12.0,
            n_iter_without_progress=300,
            min_grad_norm=1e-7,
            n_jobs=1,
        ).fit_transform(reduced)
        coordinates[f"tsne_perplexity_{perplexity}"] = embedded.tolist()

    try:
        sklearn_version = version("scikit-learn")
        umap_version = version("umap-learn")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "scikit-learn and umap-learn are required for embedding diagnostics"
        ) from error
    import umap

    for neighbors in UMAP_NEIGHBORS:
        if neighbors >= values.shape[0]:
            raise ValueError(
                f"UMAP n_neighbors {neighbors} requires more than {neighbors} images"
            )
        embedded = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            random_state=EMBEDDING_SEED,
            transform_seed=EMBEDDING_SEED,
            n_jobs=1,
            metric="euclidean",
            n_epochs=500,
            min_dist=0.1,
            spread=1.0,
            learning_rate=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
        ).fit_transform(reduced)
        coordinates[f"umap_n_neighbors_{neighbors}"] = embedded.tolist()

    return {
        "image_ids": list(ids),
        "pca": {
            "components": PCA_COMPONENTS,
            "solver": "full",
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        },
        "coordinates": coordinates,
        "settings": {
            **diagnostic_configuration()["embeddings"],
            "pca": {
                "n_components": PCA_COMPONENTS,
                "svd_solver": "full",
                "whiten": False,
                "copy": True,
                "tol": 0.0,
            },
            "tsne": {
                "n_components": 2,
                "perplexities": list(TSNE_PERPLEXITIES),
                "init": "pca",
                "learning_rate": "auto",
                "max_iter": 1000,
                "method": "barnes_hut",
                "metric": "euclidean",
                "angle": 0.5,
                "early_exaggeration": 12.0,
                "n_iter_without_progress": 300,
                "min_grad_norm": 1e-7,
                "n_jobs": 1,
                "random_state": EMBEDDING_SEED,
            },
            "umap": {
                "n_components": 2,
                "n_neighbors": list(UMAP_NEIGHBORS),
                "metric": "euclidean",
                "n_jobs": 1,
                "random_state": EMBEDDING_SEED,
                "transform_seed": EMBEDDING_SEED,
                "n_epochs": 500,
                "min_dist": 0.1,
                "spread": 1.0,
                "learning_rate": 1.0,
                "repulsion_strength": 1.0,
                "negative_sample_rate": 5,
                "version": umap_version,
            },
            "software_versions": {
                "scikit-learn": sklearn_version,
                "umap-learn": umap_version,
            },
        },
        "claim_boundary": "Do not interpret cluster size or inter-point distance.",
    }


def target_classes_for_image(true_class: int, predicted_class: int) -> tuple[int, ...]:
    """Return the true target and, on an error, the predicted target as well."""
    if true_class < 0 or predicted_class < 0:
        raise ValueError("class indices must be non-negative")
    return (
        (true_class,)
        if true_class == predicted_class
        else (true_class, predicted_class)
    )


def penultimate_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract pooled pre-classifier features from a timm-backed classifier."""
    extractor = getattr(model, "feature_extractor", None)
    backbone = getattr(extractor, "backbone", None)
    if backbone is None:
        raise ValueError("model must expose feature_extractor.backbone")
    if hasattr(backbone, "forward_features") and hasattr(backbone, "forward_head"):
        features = backbone.forward_features(images)
        return backbone.forward_head(features, pre_logits=True)
    return extractor(images)


def last_mobilenetv2_conv(model: nn.Module) -> nn.Module:
    """Return the last convolutional block used by the MobileNetV2 route."""
    backbone = getattr(getattr(model, "feature_extractor", None), "backbone", None)
    layer = getattr(backbone, "conv_head", None)
    if not isinstance(layer, nn.Module):
        raise ValueError(
            "MobileNetV2 checkpoint has no conv_head last convolution block"
        )
    return layer


def gradcam_maps(
    model: nn.Module,
    images: torch.Tensor,
    target_classes: Sequence[int],
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Compute ReLU Grad-CAM maps for the specified class on each input image."""
    if images.ndim != 4 or len(target_classes) != images.shape[0]:
        raise ValueError("images must be NCHW and target_classes must align")
    layer = target_layer or last_mobilenetv2_conv(model)
    activations: list[torch.Tensor] = []

    def capture(
        _: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        activations.append(output)

    handle = layer.register_forward_hook(capture)
    maps: list[np.ndarray] = []
    try:
        was_training = model.training
        model.eval()
        try:
            for image, target in zip(images, target_classes, strict=True):
                if target < 0:
                    raise ValueError("target class indices must be non-negative")
                model.zero_grad(set_to_none=True)
                activations.clear()
                logits = model(image.unsqueeze(0))
                if logits.ndim != 2 or target >= logits.shape[1]:
                    raise ValueError("target class index is outside model outputs")
                if not activations:
                    raise RuntimeError("target layer did not run during model forward")
                activation = activations[-1]
                gradient = torch.autograd.grad(logits[0, target], activation)[0]
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
                cam = torch.nn.functional.interpolate(
                    cam,
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                maps.append(cam[0, 0].detach().cpu().numpy())
        finally:
            model.train(was_training)
    finally:
        handle.remove()
    return np.stack(maps, axis=0)


def select_gradcam_sample(
    image_ids: Sequence[str],
    truth: Sequence[str],
    predictions: Sequence[str],
    classes: Sequence[str],
    *,
    policy_predictions: Mapping[str, Mapping[str, str]] | None = None,
    per_class: int = CAM_SAMPLE_PER_CLASS,
    seed: int = CAM_SAMPLE_SEED,
) -> dict[str, Any]:
    """Select correct/error examples per class and retain every disagreement."""
    ids, labels, predicted = tuple(image_ids), tuple(truth), tuple(predictions)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or not (len(ids) == len(labels) == len(predicted))
    ):
        raise ValueError("image_ids, truth and predictions must be aligned and unique")
    if per_class < 0:
        raise ValueError("per_class must be non-negative")
    rng = np.random.default_rng(seed)
    selected: dict[str, set[str]] = {
        "correct": set(),
        "error": set(),
        "disagreement": set(),
    }
    for label in classes:
        class_indices = [index for index, value in enumerate(labels) if value == label]
        for outcome, predicate in (
            ("correct", lambda index: predicted[index] == labels[index]),
            ("error", lambda index: predicted[index] != labels[index]),
        ):
            candidates = [index for index in class_indices if predicate(index)]
            take = min(per_class, len(candidates))
            if take:
                chosen = rng.choice(np.asarray(candidates), size=take, replace=False)
                selected[outcome].update(ids[int(index)] for index in chosen)

    disagreements = _policy_disagreements(ids, policy_predictions)
    selected["disagreement"].update(disagreements)
    kept = {image_id for values in selected.values() for image_id in values}
    return {
        "seed": seed,
        "per_class": per_class,
        "classes": list(classes),
        "selected": {
            key: [image_id for image_id in ids if image_id in values]
            for key, values in selected.items()
        },
        "policy_disagreement_count": len(disagreements),
        # Computed from the policy predictions, never asserted.
        "all_policy_disagreements_included": disagreements <= kept,
    }


def _policy_disagreements(
    ids: Sequence[str], policy_predictions: Mapping[str, Mapping[str, str]] | None
) -> set[str]:
    """Image IDs on which the three split policies do not all predict one class."""
    if not policy_predictions:
        return set()
    if set(policy_predictions) != set(POLICIES):
        raise ValueError("policy predictions must cover all three split policies")
    if any(not isinstance(values, Mapping) for values in policy_predictions.values()):
        raise ValueError("each policy must map image IDs to predictions")
    missing = [
        image_id
        for values in policy_predictions.values()
        for image_id in ids
        if image_id not in values
    ]
    if missing:
        raise ValueError(f"policy predictions miss test images: {missing[:5]}")
    return {
        image_id
        for image_id in ids
        if len({values[image_id] for values in policy_predictions.values()}) > 1
    }


def leaf_attention_score(
    cam: np.ndarray, leaf_mask: np.ndarray, content_mask: np.ndarray
) -> float | None:
    """Return CAM mass enrichment over leaf area; ``None`` means all-zero CAM."""
    heatmap = np.asarray(cam, dtype=np.float64)
    leaf = np.asarray(leaf_mask, dtype=bool)
    content = np.asarray(content_mask, dtype=bool)
    if (
        heatmap.ndim != 2
        or heatmap.shape != leaf.shape
        or heatmap.shape != content.shape
    ):
        raise ValueError("CAM, leaf mask and content mask must be aligned 2D arrays")
    if not np.isfinite(heatmap).all():
        raise ValueError("CAM must contain finite values")
    mass = np.maximum(heatmap, 0.0) * content
    total = float(mass.sum())
    content_area = int(content.sum())
    if content_area == 0:
        raise ValueError("content mask cannot be empty")
    if total == 0:
        return None
    leaf_content = leaf & content
    return float(mass[leaf_content].sum() / total - leaf_content.sum() / content_area)


def letterbox_leaf_mask(mask: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Resize an original-image binary mask onto the ADR 0002 letterbox canvas."""
    source = np.asarray(mask, dtype=bool)
    content = np.asarray(content_mask, dtype=bool)
    if source.ndim != 2 or content.ndim != 2 or not content.any():
        raise ValueError("leaf and content masks must be non-empty 2D arrays")
    ys, xs = np.nonzero(content)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    resized = (
        np.asarray(
            Image.fromarray(source.astype(np.uint8) * 255).resize(
                (right - left, bottom - top), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    result = np.zeros_like(content)
    result[top:bottom, left:right] = resized
    return result


def parameter_randomization_correlations(
    reference: np.ndarray, randomized: Mapping[str, np.ndarray]
) -> dict[str, list[float | None]]:
    """Record absolute Spearman correlation with the reference at each stage.

    A constant reference map has no rank order, so its correlation is ``None``.
    A constant randomized map against a varying reference is evidence that the map
    changed, so it scores 0. The sign is dropped so an inverted map is as
    correlated as an identical one.
    """
    baseline = np.asarray(reference, dtype=np.float64)
    if baseline.ndim != 3:
        raise ValueError("reference CAMs must have shape (images, height, width)")
    result: dict[str, list[float | None]] = {}
    for stage, candidate in randomized.items():
        values = np.asarray(candidate, dtype=np.float64)
        if values.shape != baseline.shape:
            raise ValueError(
                f"randomized CAMs for {stage} do not match reference shape"
            )
        correlations: list[float | None] = []
        for before, after in zip(baseline, values, strict=True):
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise ValueError("CAM maps must contain finite values")
            if np.ptp(before) == 0:
                correlations.append(None)
                continue
            if np.ptp(after) == 0:
                correlations.append(0.0)
                continue
            correlation = abs(float(spearmanr(before.ravel(), after.ravel()).statistic))
            correlations.append(correlation if np.isfinite(correlation) else None)
        result[stage] = correlations
    return result


def randomization_stages(model: nn.Module) -> list[tuple[str, list[nn.Module]]]:
    """Cascade order for parameter randomization: classifier first, stem last."""
    stages: list[tuple[str, list[nn.Module]]] = []
    head = getattr(model, "classifier", None)
    if isinstance(head, nn.Module):
        stages.append(("classifier", [head]))
    backbone = getattr(getattr(model, "feature_extractor", None), "backbone", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("model has no backbone to randomize")
    for name, child in reversed(list(backbone.named_children())):
        if isinstance(child, nn.Sequential):
            stages.extend(
                (f"backbone.{name}.{index}", [block])
                for index, block in reversed(list(enumerate(child)))
            )
        elif any(True for _ in child.parameters()):
            stages.append((f"backbone.{name}", [child]))
    return stages


def _median_defined(values: Sequence[float | None]) -> tuple[float | None, float]:
    defined = [value for value in values if value is not None]
    share = len(defined) / len(values) if values else 0.0
    return (float(np.median(defined)) if defined else None), share


def _sanity_outcome(
    correlations: Sequence[float | None], evidence: dict[str, Any]
) -> dict[str, Any]:
    """Apply the pass rule: maps must change once the explanation is broken."""
    median, share = _median_defined(correlations)
    passed = (
        median is not None
        and share >= SANITY_MIN_DEFINED_SHARE
        and median < SANITY_MAX_MEDIAN_CORRELATION
    )
    return {
        "status": "pass" if passed else "fail",
        "evidence": {
            **evidence,
            "median_correlation": median,
            "defined_share": share,
        },
    }


def _reset_to_default_initialisation(module: nn.Module) -> None:
    """Re-initialise every parameterised layer with its own framework default."""
    for layer in module.modules():
        if next(layer.parameters(recurse=False), None) is None:
            continue
        reset = getattr(layer, "reset_parameters", None)
        if not callable(reset):
            raise ValueError(
                f"cannot re-initialise {type(layer).__name__}: no reset_parameters"
            )
        reset()


def _permuted_labels(labels: Sequence[int], seed: int) -> list[int]:
    """A seeded permutation of the true labels that differs from them."""
    values = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    for _ in range(SANITY_PERMUTATION_ATTEMPTS):
        candidate = rng.permutation(values)
        if (candidate != values).any():
            return [int(value) for value in candidate]
    raise ValueError("could not permute the labels into a different assignment")


def _head_refit_correlations(
    model: nn.Module,
    images: torch.Tensor,
    features: torch.Tensor,
    labels: Sequence[int],
    targets: Sequence[int],
    reference: np.ndarray,
    seed: int,
) -> list[float | None]:
    """Refit only the classifier on ``labels``; score its maps against the original."""
    refit = copy.deepcopy(model).eval()
    head = refit.classifier
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head.reset_parameters()
    optimizer = torch.optim.Adam(head.parameters(), lr=SANITY_REFIT_LR)
    target_labels = torch.as_tensor(list(labels), dtype=torch.long)
    for _ in range(SANITY_REFIT_STEPS):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(features), target_labels)
        loss.backward()
        optimizer.step()
    maps = gradcam_maps(refit, images, targets)
    return parameter_randomization_correlations(reference, {"head-refit": maps})[
        "head-refit"
    ]


def sanity_check_results(
    model: nn.Module,
    images: torch.Tensor,
    targets: Sequence[int],
    true_labels: Sequence[int],
    *,
    checkpoint_sha256: str,
    image_ids: Sequence[str],
    seed: int = SANITY_SEED,
) -> dict[str, Any]:
    """Compute the Grad-CAM sanity checks, with a pass rule fixed here.

    Parameter randomization (Adebayo et al.) re-initialises the classifier and then
    each backbone stage from the top down, every layer with its own default
    initialisation drawn from a seeded generator, and compares each stage's maps
    with the original maps. It passes when the median absolute Spearman
    correlation at the final stage is below the cutoff and enough correlations are
    defined. A constant randomized map counts as changed; a constant reference map
    is undefined.

    The label-randomization variant refits only the classifier on a seeded
    permutation of ``true_labels`` over the checkpoint's own penultimate features.
    It is not the data-randomization retraining of the original method. A control
    refit on the true labels, with the same seed, steps and sample, shows how far
    the maps move without any label randomization. On the synthetic fixtures the
    two cannot be told apart, so this check is descriptive and never opens or
    closes the leaf-score gate.

    The three thresholds (sample size, correlation cutoff, defined share) are
    prespecified heuristics, not a significance test, and apply only to the
    sampled images. Every random draw is seeded, so identical inputs give
    identical results and the global random state is left as it was.
    """
    if len(targets) != images.shape[0] or len(true_labels) != len(targets):
        raise ValueError("images, targets and true labels must align")
    if len(checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in checkpoint_sha256
    ):
        raise ValueError("sanity checks need the selected checkpoint SHA-256")
    if len(set(true_labels)) < 2:
        raise ValueError("label randomization needs at least two classes in the sample")
    reference = gradcam_maps(model, images, targets)

    randomized = copy.deepcopy(model).eval()
    stage_maps: dict[str, np.ndarray] = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for name, modules in randomization_stages(randomized):
            for module in modules:
                _reset_to_default_initialisation(module)
            stage_maps[name] = gradcam_maps(randomized, images, targets)
    by_stage = parameter_randomization_correlations(reference, stage_maps)
    final_stage = next(reversed(by_stage))
    stage_median = {
        name: _median_defined(values)[0] for name, values in by_stage.items()
    }

    with torch.no_grad():
        features = penultimate_features(model, images)
    permuted = _head_refit_correlations(
        model,
        images,
        features,
        _permuted_labels(true_labels, seed),
        targets,
        reference,
        seed,
    )
    control = _head_refit_correlations(
        model, images, features, true_labels, targets, reference, seed
    )
    permuted_median, permuted_share = _median_defined(permuted)
    control_median, control_share = _median_defined(control)

    common = {
        "checkpoint_sha256": checkpoint_sha256,
        "sample_image_ids": list(image_ids),
        "seed": seed,
    }
    return {
        "method": (
            "Adebayo parameter randomization, plus a head-refit label-randomization "
            "variant (not Adebayo data-randomization retraining)"
        ),
        "pass_rule": {
            "fixed_in_code": True,
            "statistic": "median absolute Spearman correlation",
            "max_median_correlation": SANITY_MAX_MEDIAN_CORRELATION,
            "min_defined_share": SANITY_MIN_DEFINED_SHARE,
            "sample_size": SANITY_SAMPLE_SIZE,
            "prespecified_heuristic": True,
            "significance_test": False,
            "applies_to": "the sampled images only",
            "parameter_randomization_stage": final_stage,
            "parameter_randomization_initialisation": (
                "each layer's default reset_parameters, seeded"
            ),
            "label_randomization_variant": "classifier refit on permuted labels",
            "label_randomization_gates_leaf_score": False,
        },
        "parameter_randomization": _sanity_outcome(
            by_stage[final_stage],
            {
                **common,
                "stage_median_correlation": stage_median,
                "final_stage": final_stage,
            },
        ),
        "label_randomization": {
            "status": "descriptive",
            "evidence": {
                **common,
                "refit_steps": SANITY_REFIT_STEPS,
                "refit_learning_rate": SANITY_REFIT_LR,
                "median_correlation": permuted_median,
                "defined_share": permuted_share,
                "permuted_correlations": permuted,
                "control": "same seed, steps and sample, refit on true labels",
                "control_median_correlation": control_median,
                "control_defined_share": control_share,
                "control_correlations": control,
                "gates_leaf_score": False,
                "reason": (
                    "on synthetic fixtures a head refit on the true labels moved the "
                    "maps as much as a refit on permuted labels, so this variant "
                    "cannot separate label dependence from refit movement; it is "
                    "reported, not gated"
                ),
            },
        },
    }


def aggregate_leaf_scores(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate defined per-image scores by class and scene using median/IQR."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    undefined: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        class_name = str(row["class"])
        scene = str(row.get("scene", "unknown"))
        group = (class_name, scene)
        score = row.get("score")
        if score is None:
            undefined[group] += 1
        else:
            value = float(score)
            if not np.isfinite(value):
                raise ValueError("defined CAM scores must be finite")
            groups[group].append(value)
    aggregates: list[dict[str, Any]] = []
    for group in sorted(set(groups) | set(undefined)):
        values = np.asarray(groups[group], dtype=np.float64)
        aggregates.append(
            {
                "class": group[0],
                "scene": group[1],
                "n_defined": int(values.size),
                "undefined_all_zero_cam": undefined[group],
                "median": float(np.median(values)) if values.size else None,
                "q1": float(np.quantile(values, 0.25)) if values.size else None,
                "q3": float(np.quantile(values, 0.75)) if values.size else None,
                "iqr": (
                    float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                    if values.size
                    else None
                ),
            }
        )
    by_class: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class"])].append(row)
        by_scene[str(row.get("scene", "unknown"))].append(row)

    def summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = np.asarray(
            [float(row["score"]) for row in items if row.get("score") is not None]
        )
        return {
            "n_defined": int(values.size),
            "undefined_all_zero_cam": sum(row.get("score") is None for row in items),
            "median": float(np.median(values)) if values.size else None,
            "q1": float(np.quantile(values, 0.25)) if values.size else None,
            "q3": float(np.quantile(values, 0.75)) if values.size else None,
        }

    return {
        "class_scene": aggregates,
        "by_class": {key: summary(value) for key, value in sorted(by_class.items())},
        "by_scene": {key: summary(value) for key, value in sorted(by_scene.items())},
    }


def leaf_score_gate(
    *, parameter_randomization: bool, mask_audit_passed: bool
) -> dict[str, Any]:
    """Report whether the protocol permits a leaf-score summary.

    The head-refit label-randomization variant is descriptive and does not gate.
    """
    allowed = parameter_randomization and mask_audit_passed
    return {
        "reported": allowed,
        "checks": {
            "parameter_randomization": parameter_randomization,
            "adr_0002_mask_audit": mask_audit_passed,
        },
        "reason": None
        if allowed
        else "withheld until parameter randomization and the mask audit pass",
    }


def write_diagnostics_export(
    output: Path | str,
    payload: Mapping[str, Any],
    *,
    generating_command: Sequence[str],
    configuration: Mapping[str, Any],
    source_artifacts: Sequence[Path | str],
    files: Mapping[str, bytes] | None = None,
) -> Path:
    """Write a provenance-bound export atomically, with a SHA-256 for every file.

    Everything is written into a sibling staging directory, listed with its digests
    in ``output-manifest.json``, flushed, and only then renamed to ``output``. A
    crash before the rename leaves no directory at ``output``, so a complete-looking
    export is always complete. ``files`` adds already-rendered files (figures) by
    relative path.
    """
    directory = Path(output)
    if directory.exists():
        raise ValueError(f"refusing to replace {directory}")
    sources = []
    for source in source_artifacts:
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"source artifact is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append({"path": str(path), "sha256": digest})
    export = {
        "schema": 1,
        "generating_command": list(generating_command),
        "configuration": dict(configuration),
        "source_artifacts": sources,
        "diagnostics": dict(payload),
    }
    contents: dict[str, bytes] = {
        "diagnostics.json": (
            json.dumps(export, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    }
    for feature_set, result in payload.get("embeddings", {}).items():
        ids = result["image_ids"]
        for name, coordinates in result["coordinates"].items():
            buffer = io.StringIO(newline="")
            writer = csv.writer(buffer)
            writer.writerow(("image_id", "x", "y"))
            writer.writerows(
                (image_id, *coordinate)
                for image_id, coordinate in zip(ids, coordinates, strict=True)
            )
            contents[f"{feature_set}-{name}.csv"] = buffer.getvalue().encode("utf-8")
    for relative, data in (files or {}).items():
        if relative in contents or relative == OUTPUT_MANIFEST:
            raise ValueError(f"export file name collides: {relative}")
        contents[relative] = data
    manifest = {
        "schema": 1,
        "files": {
            relative: hashlib.sha256(data).hexdigest()
            for relative, data in sorted(contents.items())
        },
    }
    contents[OUTPUT_MANIFEST] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.parent / f".{directory.name}.partial-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        for relative, data in contents.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(stage, directory)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return directory / "diagnostics.json"
