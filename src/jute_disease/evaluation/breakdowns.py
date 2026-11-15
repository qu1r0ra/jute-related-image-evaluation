"""Class-by-scene cells, scene macro-F1 and the leaf-only test sample."""

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from jute_disease.evaluation.resampling import Interval
from jute_disease.evaluation.scores import PredictionSet
from jute_disease.evaluation.statistics import wilson_interval

SCENE_MIN_PER_CLASS = 30
LEAF_SAMPLE_SIZE = 2000
UNKNOWN_SCENE = "unknown"


@dataclass(frozen=True)
class SceneCell:
    label: str
    scene: str
    n: int
    accuracy: Interval


@dataclass(frozen=True)
class SceneSummary:
    scene: str
    n: int
    class_counts: Mapping[str, int]
    macro_f1: float | None
    reason: str | None


def class_scene_cells(
    system: PredictionSet, scenes: Mapping[str, str]
) -> tuple[SceneCell, ...]:
    """Counts and seed-mean accuracy per class and scene, with Wilson intervals.

    The Wilson interval uses the image count and the seed-mean number correct, so
    repeated seeds do not inflate the effective sample size.
    """
    truth = system.truth_index()
    correct = np.mean(
        [system.prediction_index(s) == truth for s in system.seeds], axis=0
    )
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for image_id, label, hit in zip(
        system.image_ids, system.truth, correct, strict=True
    ):
        cells[(label, scenes.get(image_id, UNKNOWN_SCENE))].append(float(hit))
    return tuple(
        SceneCell(
            label,
            scene,
            len(hits),
            wilson_interval(sum(hits), len(hits)),
        )
        for (label, scene), hits in sorted(cells.items())
    )


def scene_summaries(
    system: PredictionSet, scenes: Mapping[str, str]
) -> tuple[SceneSummary, ...]:
    """Scene macro-F1 (seed mean), only where every class in the scene has 30 images."""
    k = len(system.classes)
    truth = system.truth_index()
    predicted = [system.prediction_index(s) for s in system.seeds]
    tags = np.array([scenes.get(i, UNKNOWN_SCENE) for i in system.image_ids])
    out = []
    for scene in sorted(set(tags)):
        rows = tags == scene
        counts = np.bincount(truth[rows], minlength=k)
        present_indices = np.flatnonzero(counts)
        present = {system.classes[i]: int(c) for i, c in enumerate(counts) if c}
        short = sorted(
            system.classes[i]
            for i in present_indices
            if counts[i] < SCENE_MIN_PER_CLASS
        )
        if short:
            out.append(
                SceneSummary(
                    scene,
                    int(rows.sum()),
                    present,
                    None,
                    f"classes below {SCENE_MIN_PER_CLASS} test images: {short}",
                )
            )
            continue
        scores = [
            _macro_f1_for_present_classes(
                np.bincount(truth[rows] * k + p[rows], minlength=k * k).reshape(k, k),
                present_indices,
            )
            for p in predicted
        ]
        out.append(
            SceneSummary(scene, int(rows.sum()), present, float(np.mean(scores)), None)
        )
    return tuple(out)


def _macro_f1_for_present_classes(
    confusion: np.ndarray, class_indices: np.ndarray
) -> float:
    true_positive = np.diag(confusion)[class_indices]
    support = confusion.sum(axis=1)[class_indices]
    predicted = confusion.sum(axis=0)[class_indices]
    denominator = support + predicted
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator > 0,
    )
    return float(np.mean(f1))


def stratified_sample(
    image_ids: Sequence[str],
    labels: Sequence[str],
    size: int = LEAF_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[str, ...]:
    """Class-proportional sample by largest remainder, ranked by a seeded hash."""
    if len(image_ids) != len(labels) or size < 1:
        raise ValueError("invalid sample inputs")
    if size >= len(image_ids):
        return tuple(sorted(image_ids))
    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id, label in zip(image_ids, labels, strict=True):
        by_class[label].append(image_id)
    total = len(image_ids)
    exact = {label: size * len(ids) / total for label, ids in by_class.items()}
    quota = {label: int(value) for label, value in exact.items()}
    spare = size - sum(quota.values())
    for label in sorted(exact, key=lambda x: (-(exact[x] - quota[x]), x))[:spare]:
        quota[label] += 1
    chosen: list[str] = []
    for label, ids in by_class.items():
        ranked = sorted(
            ids,
            key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(),
        )
        chosen.extend(ranked[: quota[label]])
    return tuple(sorted(chosen))


def class_covering_sample(
    image_ids: Sequence[str], labels: Sequence[str], size: int, seed: int
) -> tuple[str, ...]:
    """Sample ``size`` images with at least one per class, ranked by a seeded hash.

    One image per class is reserved first, so no class is dropped while the sample
    has room. The rest is filled class-proportionally. When ``size`` is smaller than
    the number of classes, only the first ``size`` classes by hash rank are covered;
    the caller reports which classes are missing.
    """
    if len(image_ids) != len(labels) or size < 1:
        raise ValueError("invalid sample inputs")
    if size >= len(image_ids):
        return tuple(sorted(image_ids))

    def rank(image_id: str) -> str:
        return hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()

    by_class: dict[str, list[str]] = defaultdict(list)
    for image_id, label in zip(image_ids, labels, strict=True):
        by_class[label].append(image_id)
    reserved = sorted((min(ids, key=rank) for ids in by_class.values()), key=rank)[
        :size
    ]
    reserved_set = set(reserved)
    rest = [i for i in image_ids if i not in reserved_set]
    rest_labels = [
        label
        for i, label in zip(image_ids, labels, strict=True)
        if i not in reserved_set
    ]
    remaining = size - len(reserved)
    filled = stratified_sample(rest, rest_labels, remaining, seed) if remaining else ()
    return tuple(sorted({*reserved, *filled}))
