"""Hand-checkable classification scores over frozen predictions.

The scores operate on plain label sequences, never on pixels or checkpoints, so
the same definitions serve a fixture, a frozen validation set or the final test
set. A class with no true images is unsupported: it is listed, kept in the
confusion matrix and left out of the macro average rather than scored as zero.
A supported class that is never predicted has zero recall and F1 and an
undefined precision.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

METRICS = ("accuracy", "macro_f1")
CLASS_METRICS = ("precision", "recall", "f1")


@dataclass(frozen=True)
class PredictionSet:
    """Frozen predictions for one system, aligned image by image across seeds.

    ``groups`` are the conservative constraint components of the frozen split;
    ``predictions`` maps each training seed to one predicted label per image.
    """

    classes: tuple[str, ...]
    image_ids: tuple[str, ...]
    truth: tuple[str, ...]
    groups: tuple[str, ...]
    sources: tuple[str, ...]
    predictions: Mapping[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        count = len(self.image_ids)
        if count == 0:
            raise ValueError("prediction set has no images")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("class labels must be unique")
        if len(set(self.image_ids)) != count:
            raise ValueError("image ids must be unique")
        for name in ("truth", "groups", "sources"):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} must have one entry per image")
        if not self.predictions:
            raise ValueError("prediction set has no seeds")
        known = set(self.classes)
        for label in self.truth:
            if label not in known:
                raise ValueError(f"unknown true label {label!r}")
        for seed, predicted in self.predictions.items():
            if len(predicted) != count:
                raise ValueError(f"seed {seed} must predict every image")
            for label in predicted:
                if label not in known:
                    raise ValueError(f"unknown predicted label {label!r}")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted(self.predictions))

    def truth_index(self) -> np.ndarray:
        return self._index(self.truth)

    def prediction_index(self, seed: int) -> np.ndarray:
        if seed not in self.predictions:
            raise ValueError(f"no predictions for seed {seed}")
        return self._index(self.predictions[seed])

    def _index(self, labels: Sequence[str]) -> np.ndarray:
        position = {label: i for i, label in enumerate(self.classes)}
        return np.array([position[label] for label in labels], dtype=np.int64)


@dataclass(frozen=True)
class ClassScore:
    """One class's scores; undefined values are None."""

    label: str
    support: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class RunScores:
    """Scores for one seed over all images or one source."""

    n: int
    accuracy: float
    macro_f1: float
    classes: tuple[ClassScore, ...]
    unsupported: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SeedSummary:
    """Mean and sample SD (n - 1) over the seeds where a value is defined."""

    mean: float | None
    sd: float | None
    n: int


@dataclass(frozen=True)
class Evaluation:
    """Per-seed scores with seed summaries, overall and per source."""

    overall: dict[int, RunScores]
    by_source: dict[str, dict[int, RunScores]]
    summary: dict[str, SeedSummary]
    source_summary: dict[str, dict[str, SeedSummary]]
    class_summary: dict[str, dict[str, SeedSummary]]


def confusion_counts(
    truth: np.ndarray, predicted: np.ndarray, num_classes: int
) -> np.ndarray:
    """Return counts with true classes as rows and predicted classes as columns."""
    flat = np.bincount(
        truth * num_classes + predicted, minlength=num_classes * num_classes
    )
    return flat.reshape(num_classes, num_classes)


def metric_values(confusions: np.ndarray, metric: str) -> np.ndarray:
    """Evaluate a metric on confusion matrices shaped ``(..., classes, classes)``."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    correct = np.trace(confusions, axis1=-2, axis2=-1)
    if metric == "accuracy":
        return correct / confusions.sum(axis=(-2, -1))
    tp = np.diagonal(confusions, axis1=-2, axis2=-1)
    support = confusions.sum(axis=-1)
    predicted = confusions.sum(axis=-2)
    supported = support > 0
    f1 = np.where(supported, 2 * tp / np.maximum(support + predicted, 1), 0.0)
    return f1.sum(axis=-1) / supported.sum(axis=-1)


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _run_scores(confusion: np.ndarray, classes: tuple[str, ...]) -> RunScores:
    tp = np.diagonal(confusion)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    scores = []
    for i, label in enumerate(classes):
        supported = bool(support[i])
        scores.append(
            ClassScore(
                label=label,
                support=int(support[i]),
                precision=_ratio(int(tp[i]), int(predicted[i])) if supported else None,
                recall=_ratio(int(tp[i]), int(support[i])),
                f1=_ratio(2 * int(tp[i]), int(support[i] + predicted[i]))
                if supported
                else None,
            )
        )
    return RunScores(
        n=int(confusion.sum()),
        accuracy=float(metric_values(confusion, "accuracy")),
        macro_f1=float(metric_values(confusion, "macro_f1")),
        classes=tuple(scores),
        unsupported=tuple(s.label for s in scores if s.support == 0),
        confusion=tuple(tuple(int(v) for v in row) for row in confusion),
    )


def score_run(
    predictions: PredictionSet, seed: int, source: str | None = None
) -> RunScores:
    """Score one seed over every image, or over one source's images."""
    truth = predictions.truth_index()
    predicted = predictions.prediction_index(seed)
    if source is not None:
        keep = np.array([s == source for s in predictions.sources])
        if not keep.any():
            raise ValueError(f"no images from source {source!r}")
        truth, predicted = truth[keep], predicted[keep]
    confusion = confusion_counts(truth, predicted, len(predictions.classes))
    return _run_scores(confusion, predictions.classes)


def summarize_seeds(values: Sequence[float | None]) -> SeedSummary:
    defined = [v for v in values if v is not None]
    if not defined:
        return SeedSummary(mean=None, sd=None, n=0)
    sd = float(np.std(defined, ddof=1)) if len(defined) > 1 else None
    return SeedSummary(mean=float(np.mean(defined)), sd=sd, n=len(defined))


def _summaries(runs: Mapping[int, RunScores]) -> dict[str, SeedSummary]:
    return {
        metric: summarize_seeds([getattr(run, metric) for run in runs.values()])
        for metric in METRICS
    }


def evaluate_predictions(predictions: PredictionSet) -> Evaluation:
    """Score every seed overall and per source, then summarize across seeds."""
    seeds = predictions.seeds
    overall = {seed: score_run(predictions, seed) for seed in seeds}
    sources = sorted(set(predictions.sources))
    by_source = {
        source: {seed: score_run(predictions, seed, source) for seed in seeds}
        for source in sources
    }
    class_summary = {
        label: {
            metric: summarize_seeds(
                [getattr(run.classes[i], metric) for run in overall.values()]
            )
            for metric in CLASS_METRICS
        }
        for i, label in enumerate(predictions.classes)
    }
    return Evaluation(
        overall=overall,
        by_source=by_source,
        summary=_summaries(overall),
        source_summary={source: _summaries(runs) for source, runs in by_source.items()},
        class_summary=class_summary,
    )
