"""Background-only and border-information probes.

Both probes ask whether a model can classify from pixels that carry no leaf
symptoms. They report bounded statements, never absence: a probe that does not
beat its permutation null reports "no detectable ... at this setting".
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from jute_disease.evaluation.resampling import Interval, resample_metrics
from jute_disease.evaluation.scores import PredictionSet, metric_values
from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec

PROBE_SYSTEM = "mobilenet_v2_direct"
BORDER_SEEDS = 3
FRAME_WIDTH_PX = 32
LEAF_CUTOFF = 0.05
PERMUTATIONS = 1000
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


def _probe_base(fit: FitSpec, benchmark: Benchmark, seeds: int | None) -> bool:
    if fit.kind is FitKind.CLASSICAL:
        pass
    elif fit.kind is FitKind.TARGET and fit.system == PROBE_SYSTEM:
        pass
    else:
        return False
    if fit.phase == "tune":
        return True
    if fit.split != benchmark.primary_split:
        return False
    return seeds is None or fit.seed_index < seeds


def probe_fits(benchmark: Benchmark, variant: str) -> tuple[ProbeFit, ...]:
    """Plan the background-only or border fits from the declared matrix.

    Classical models and the MobileNetV2 ImageNet route, the primary split, every
    pool and policy, and the five matrix seeds (three for the border probe).
    """
    if variant not in ("background", "border"):
        raise ValueError("variant must be 'background' or 'border'")
    seeds = BORDER_SEEDS if variant == "border" else None
    return tuple(
        ProbeFit(f"probe/{variant}/{fit.fit_id}", variant, fit.fit_id)
        for fit in benchmark.fits()
        if _probe_base(fit, benchmark, seeds)
    )


def pad_cue_fits(benchmark: Benchmark) -> tuple[ProbeFit, ...]:
    """Pad-fraction and aspect-ratio-only baseline fits, one per matrix cell."""
    fits = []
    for pool in benchmark.pools:
        for view in benchmark.views:
            for policy in benchmark.policies:
                for seed in benchmark.recipe.seeds[:BORDER_SEEDS]:
                    fit_id = (
                        f"probe/pad-cue/pool-{pool}/view-{view}/{policy}/seed-{seed}"
                    )
                    fits.append(ProbeFit(fit_id, "pad_cue", fit_id))
    return tuple(fits)


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


def _content_box(valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(valid_mask)
    if not len(rows):
        raise ValueError("valid mask has no content")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def frame_mask(valid_mask: np.ndarray, width: int) -> np.ndarray:
    """Pixels within ``width`` px of the edge of the resized image content."""
    if width < 1:
        raise ValueError("frame width must be positive")
    top, bottom, left, right = _content_box(np.asarray(valid_mask, dtype=bool))
    frame = np.zeros(valid_mask.shape, dtype=bool)
    frame[top:bottom, left:right] = True
    frame[top + width : bottom - width, left + width : right - width] = False
    return frame


def frame_only_image(
    pixels: np.ndarray, valid_mask: np.ndarray, width: int
) -> np.ndarray:
    """Keep the frame, blank the interior and mask the letterbox padding."""
    out = np.zeros_like(pixels)
    keep = frame_mask(valid_mask, width)
    out[keep] = pixels[keep]
    return out


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


def permutation_null(
    system: PredictionSet, permutations: int = PERMUTATIONS, seed: int = 42
) -> np.ndarray:
    """Macro-F1 under permuted family labels, with no refits.

    Labels are permuted among families, so each family keeps one label, matching
    the family clustering of the #106 intervals. Predictions are not changed.
    """
    families = sorted(set(system.groups))
    position = {family: i for i, family in enumerate(families)}
    family_of = np.array([position[g] for g in system.groups])
    truth = system.truth_index()
    family_label = np.zeros(len(families), dtype=np.int64)
    family_label[family_of] = truth
    k = len(system.classes)
    predicted = [system.prediction_index(s) for s in system.seeds]
    rng = np.random.default_rng(seed)
    out = np.empty(permutations)
    for i in range(permutations):
        permuted = rng.permutation(family_label)[family_of]
        scores = []
        for pred in predicted:
            confusion = np.bincount(permuted * k + pred, minlength=k * k).reshape(k, k)
            scores.append(float(metric_values(confusion, "macro_f1")))
        out[i] = np.mean(scores)
    return out


@dataclass(frozen=True)
class ProbeResult:
    """Macro-F1 and accuracy against the chance baselines."""

    n: int
    macro_f1: Interval
    accuracy: Interval
    majority_accuracy: float
    null_p95: float
    permutations: int
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
    """Apply the prespecified rule: lower CI of macro-F1 above the null's 95th pct."""
    wrapped = {"probe": {0: system}}
    macro = resample_metrics(wrapped, "macro_f1", resamples, seed).interval("probe")
    accuracy = resample_metrics(wrapped, "accuracy", resamples, seed).interval("probe")
    counts = np.bincount(system.truth_index(), minlength=len(system.classes))
    null = permutation_null(system, permutations, seed)
    p95 = float(np.percentile(null, 95))
    detected = macro.lower > p95
    return ProbeResult(
        n=len(system.image_ids),
        macro_f1=macro,
        accuracy=accuracy,
        majority_accuracy=float(counts.max() / counts.sum()),
        null_p95=p95,
        permutations=permutations,
        detected=detected,
        statement=detected_statement if detected else none_statement,
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
