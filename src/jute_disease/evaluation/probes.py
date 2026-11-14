"""Background-only and border-information probes.

Both probes ask whether a model can classify from pixels that carry no leaf
symptoms. They report bounded statements, never absence: a probe that does not
beat its permutation null reports "no detectable ... at this setting".
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from jute_disease.data.probe_pixels import (
    FRAME_WIDTH_PX,
    frame_mask,
    frame_only_image,
)
from jute_disease.evaluation.resampling import Interval, resample_metrics
from jute_disease.evaluation.scores import PredictionSet, metric_values
from jute_disease.training.benchmark import PROBE_SYSTEM, Benchmark, is_probe_base

__all__ = ["FRAME_WIDTH_PX", "PROBE_SYSTEM", "frame_mask", "frame_only_image"]

LEAF_CUTOFF = 0.05
PERMUTATIONS = 9_999
DETECTION_ALPHA = 0.05
# Prespecified judgment, not an estimate: below this share of test images in
# exchangeable strata the null moves too few labels for a miss to mean anything,
# so the report says "not informative" instead of "no detectable information".
EXCHANGEABLE_MIN_SHARE = 0.5
NOT_INFORMATIVE = "not informative at this exchangeable share"
MIN_RETAINED_PER_CLASS = 50
NO_BORDER = "no detectable border information at this frame width"
NO_BACKGROUND = "no detectable background information at this setting"
HEADLINE_LIMIT = (
    "The mask audit rates whole-leaf mask quality; leaf share in a 32 px strip is "
    "more sensitive to mask edge errors."
)


@dataclass(frozen=True)
class ProbeFit:
    fit_id: str
    variant: str
    base_fit_id: str


def probe_fits(benchmark: Benchmark, variant: str) -> tuple[ProbeFit, ...]:
    """Plan the background-only or border fits from the declared matrix.

    Classical models and the MobileNetV2 ImageNet route, the primary split, every
    pool and policy, and all five matrix seeds.
    """
    if variant not in ("background", "border"):
        raise ValueError("variant must be 'background' or 'border'")
    return tuple(
        ProbeFit(f"probe/{variant}/{fit.fit_id}", variant, fit.fit_id)
        for fit in benchmark.fits()
        if is_probe_base(fit, benchmark)
    )


def pad_cue_fits(benchmark: Benchmark) -> tuple[ProbeFit, ...]:
    """Pad-fraction and aspect-ratio-only baseline fits, one per matrix cell.

    Classical only: the inputs are two scalars, so a MobileNetV2 adds nothing.
    """
    return tuple(
        ProbeFit(fit.fit_id, "pad_cue", fit.fit_id) for fit in benchmark.pad_cue_fits()
    )


def planned_probe_fit_ids(benchmark: Benchmark) -> list[str]:
    return sorted(
        {
            fit.fit_id
            for fit in (
                *probe_fits(benchmark, "background"),
                *probe_fits(benchmark, "border"),
                *pad_cue_fits(benchmark),
            )
        }
    )


def background_only_image(
    pixels: np.ndarray, leaf_mask: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Mask the leaf and the letterbox padding, keeping only background pixels."""
    out = np.zeros_like(pixels)
    keep = np.asarray(valid_mask, dtype=bool) & ~np.asarray(leaf_mask, dtype=bool)
    out[keep] = pixels[keep]
    return out


def frame_leaf_share(
    leaf_mask: np.ndarray, valid_mask: np.ndarray, width: int
) -> float:
    """Share of the frame's pixels that the automatic leaf mask marks as leaf."""
    frame = frame_mask(valid_mask, width)
    return float(np.asarray(leaf_mask, dtype=bool)[frame].mean())


def subset(system: PredictionSet, image_ids: Sequence[str]) -> PredictionSet:
    """Restrict a prediction set to the given images, keeping its order."""
    wanted = set(image_ids)
    keep = [i for i, name in enumerate(system.image_ids) if name in wanted]
    if not keep:
        raise ValueError("subset has no images")
    return PredictionSet(
        system.classes,
        tuple(system.image_ids[i] for i in keep),
        tuple(system.truth[i] for i in keep),
        tuple(system.groups[i] for i in keep),
        tuple(system.sources[i] for i in keep),
        {s: tuple(p[i] for i in keep) for s, p in system.predictions.items()},
    )


def _family_strata(system: PredictionSet) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group families by size: member indices and sorted label multisets."""
    truth = system.truth_index()
    members: dict[str, list[int]] = {}
    for image, family in enumerate(system.groups):
        members.setdefault(family, []).append(image)
    by_size: dict[int, list[list[int]]] = {}
    for family in sorted(members):
        by_size.setdefault(len(members[family]), []).append(members[family])
    strata = []
    for size in sorted(by_size):
        index = np.array(by_size[size], dtype=np.int64).reshape(-1, size)
        strata.append((index, np.sort(truth[index], axis=1)))
    return strata


@dataclass(frozen=True)
class Exchangeable:
    """Images and families in strata whose label multisets can differ.

    A stratum of same-size families is exchangeable when it holds at least two
    distinct label multisets. Families in any other stratum keep their labels in
    every draw, so they add the same amount to the observed score and to the
    null: the test stays valid but loses power in proportion to what they hold.
    """

    images: int
    families: int
    total_images: int
    total_families: int


def exchangeable_share(strata: list[tuple[np.ndarray, np.ndarray]]) -> Exchangeable:
    images = families = total_images = total_families = 0
    for index, labels in strata:
        total_images += index.size
        total_families += index.shape[0]
        if len(np.unique(labels, axis=0)) >= 2:
            images += index.size
            families += index.shape[0]
    return Exchangeable(images, families, total_images, total_families)


def family_permutation(
    system: PredictionSet,
    rng: np.random.Generator,
    strata: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    """One null draw of true labels, permuted at family level.

    Each family's label multiset travels as a unit: it is handed to another
    family of the same size, so families with mixed labels keep their mix and
    the class counts of the test set never change. Which member of the receiving
    family carries which label is drawn at random, since the pixels of a family
    are not ordered. Returns one class index per image.
    """
    permuted = np.zeros(len(system.image_ids), dtype=np.int64)
    for index, labels in strata if strata is not None else _family_strata(system):
        moved = labels[rng.permutation(len(labels))]
        order = np.argsort(rng.random(moved.shape), axis=1)
        permuted[index] = np.take_along_axis(moved, order, axis=1)
    return permuted


def _macro_f1(truth: np.ndarray, predicted: list[np.ndarray], k: int) -> float:
    scores = []
    for pred in predicted:
        confusion = np.bincount(truth * k + pred, minlength=k * k).reshape(k, k)
        scores.append(float(metric_values(confusion, "macro_f1")))
    return float(np.mean(scores))


@dataclass(frozen=True)
class PermutationTest:
    """Observed macro-F1, the permutation null and the one-sided p-value."""

    observed: float
    null: np.ndarray
    p_value: float
    exchangeable: Exchangeable


def permutation_test(
    system: PredictionSet, permutations: int = PERMUTATIONS, seed: int = 42
) -> PermutationTest:
    """Family-level permutation test of macro-F1, with no refits.

    The p-value is ``(b + 1) / (permutations + 1)``, where ``b`` counts null
    draws at least as large as the observed macro-F1, so it is never zero and is
    valid for the finite number of draws. Predictions are not changed.
    """
    k = len(system.classes)
    predicted = [system.prediction_index(s) for s in system.seeds]
    observed = _macro_f1(system.truth_index(), predicted, k)
    strata = _family_strata(system)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations)
    for i in range(permutations):
        null[i] = _macro_f1(family_permutation(system, rng, strata), predicted, k)
    exceed = int(np.count_nonzero(null >= observed - 1e-12))
    return PermutationTest(
        observed, null, (exceed + 1) / (permutations + 1), exchangeable_share(strata)
    )


def permutation_null(
    system: PredictionSet, permutations: int = PERMUTATIONS, seed: int = 42
) -> np.ndarray:
    """Macro-F1 under family-level permuted labels, with no refits."""
    return permutation_test(system, permutations, seed).null


@dataclass(frozen=True)
class ProbeResult:
    """Macro-F1 and accuracy against the chance baselines.

    ``detected`` follows the permutation p-value alone; the bootstrap intervals
    are reported for the reader and do not decide. ``p_value_standard_error`` is
    the Monte Carlo error of the p-value from the finite number of draws. The
    exchangeable fields say how much of the test set the null could permute: a
    miss below ``EXCHANGEABLE_MIN_SHARE`` is reported as not informative.
    """

    n: int
    families: int
    macro_f1: Interval
    accuracy: Interval
    majority_accuracy: float
    null_p95: float
    p_value: float
    p_value_standard_error: float
    permutations: int
    exchangeable_images: int
    exchangeable_image_share: float
    exchangeable_families: int
    exchangeable_family_share: float
    detected: bool
    statement: str


def probe_result(
    system: PredictionSet,
    detected_statement: str,
    none_statement: str,
    *,
    resamples: int = 10_000,
    permutations: int = PERMUTATIONS,
    seed: int = 42,
) -> ProbeResult:
    """Apply the rule: detected when the permutation p-value is below 0.05."""
    wrapped = {"probe": {0: system}}
    macro = resample_metrics(wrapped, "macro_f1", resamples, seed).interval("probe")
    accuracy = resample_metrics(wrapped, "accuracy", resamples, seed).interval("probe")
    counts = np.bincount(system.truth_index(), minlength=len(system.classes))
    test = permutation_test(system, permutations, seed)
    detected = test.p_value < DETECTION_ALPHA
    share = test.exchangeable
    image_share = share.images / share.total_images
    if detected:
        statement = detected_statement
    elif image_share < EXCHANGEABLE_MIN_SHARE:
        statement = NOT_INFORMATIVE
    else:
        statement = none_statement
    return ProbeResult(
        n=len(system.image_ids),
        families=share.total_families,
        macro_f1=macro,
        accuracy=accuracy,
        majority_accuracy=float(counts.max() / counts.sum()),
        null_p95=float(np.percentile(test.null, 95)),
        p_value=test.p_value,
        p_value_standard_error=float(
            np.sqrt(test.p_value * (1 - test.p_value) / permutations)
        ),
        permutations=permutations,
        exchangeable_images=share.images,
        exchangeable_image_share=image_share,
        exchangeable_families=share.families,
        exchangeable_family_share=share.families / share.total_families,
        detected=detected,
        statement=statement,
    )


def background_probe(system: PredictionSet, **kwargs: Any) -> ProbeResult:
    return probe_result(
        system,
        "background-only pixels carry label information",
        NO_BACKGROUND,
        **kwargs,
    )


@dataclass(frozen=True)
class BorderReport:
    unrestricted: ProbeResult
    headline: ProbeResult | None
    headline_withheld: str | None
    retained_images: int
    retained_class_counts: Mapping[str, int]
    retained_scene_counts: Mapping[str, int]
    leaf_cutoff: float
    width_px: int
    pad_cue: ProbeResult | None
    limit: str


def border_probe(
    system: PredictionSet,
    leaf_share: Mapping[str, float],
    scenes: Mapping[str, str],
    *,
    mask_audit_passed: bool,
    leaf_cutoff: float = LEAF_CUTOFF,
    width_px: int = FRAME_WIDTH_PX,
    pad_cue: PredictionSet | None = None,
    **kwargs: Any,
) -> BorderReport:
    """Border result with the symptom-survival check.

    The headline is the result on images with at most ``leaf_cutoff`` leaf share
    in the frame. It is withheld when any class keeps fewer than 50 test images or
    when the mask audit failed; the unrestricted result then stands alone.
    """
    missing = [i for i in system.image_ids if i not in leaf_share]
    if missing:
        raise ValueError(f"no frame leaf share for {len(missing)} test images")

    def run(item: PredictionSet) -> ProbeResult:
        return probe_result(
            item,
            f"border information detected at {width_px} px",
            NO_BORDER,
            **kwargs,
        )

    unrestricted = run(system)
    keep = [i for i in system.image_ids if leaf_share[i] <= leaf_cutoff]
    class_counts = dict.fromkeys(system.classes, 0)
    scene_counts: dict[str, int] = {}
    truth = dict(zip(system.image_ids, system.truth, strict=True))
    for image_id in keep:
        class_counts[truth[image_id]] += 1
        tag = scenes.get(image_id, "unknown")
        scene_counts[tag] = scene_counts.get(tag, 0) + 1
    withheld = None
    if not mask_audit_passed:
        withheld = "the ADR 0002 mask audit failed; only the unrestricted result stands"
    elif min(class_counts.values()) < MIN_RETAINED_PER_CLASS:
        withheld = (
            f"a class has fewer than {MIN_RETAINED_PER_CLASS} retained test images; "
            "only the unrestricted result stands"
        )
    headline = run(subset(system, keep)) if withheld is None else None
    return BorderReport(
        unrestricted=unrestricted,
        headline=headline,
        headline_withheld=withheld,
        retained_images=len(keep),
        retained_class_counts=class_counts,
        retained_scene_counts=dict(sorted(scene_counts.items())),
        leaf_cutoff=leaf_cutoff,
        width_px=width_px,
        pad_cue=None if pad_cue is None else run(pad_cue),
        limit=HEADLINE_LIMIT,
    )
