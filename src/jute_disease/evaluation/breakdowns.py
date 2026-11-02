"""Class-by-scene cells, scene macro-F1 and the leaf-only test sample."""

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from jute_disease.evaluation.resampling import Interval
from jute_disease.evaluation.scores import PredictionSet, metric_values
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
        present = {system.classes[i]: int(c) for i, c in enumerate(counts) if c}
        short = sorted(label for label, c in present.items() if c < SCENE_MIN_PER_CLASS)
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
            float(
                metric_values(
                    np.bincount(truth[rows] * k + p[rows], minlength=k * k).reshape(
                        k, k
                    ),
                    "macro_f1",
                )
            )
            for p in predicted
        ]
        out.append(
            SceneSummary(scene, int(rows.sum()), present, float(np.mean(scores)), None)
        )
    return tuple(out)


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
