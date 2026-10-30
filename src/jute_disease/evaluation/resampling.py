"""Paired group bootstrap for seed-mean metrics.

Conservative constraint components, not images, are the resampling unit
because images in one component are not independent. Components are drawn with
replacement within their class and the same draw is applied to every seed, so
each resample keeps the class composition and the seeds stay jointly
resampled. The metric is computed for each seed on the resample and then
averaged over seeds. Systems compared in one call share every draw, which makes
a difference between them a paired difference.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from jute_disease.evaluation.scores import METRICS, PredictionSet, metric_values

RESAMPLES = 10_000
EVAL_SEED = 42
CONFIDENCE_LEVEL = 0.95
_CHUNK = 1_000

_LIMITATIONS = (
    "Seed SD reflects training-seed variation, not data-generalization uncertainty.",
    "Intervals condition on the frozen split; the image pool was exposed during "
    "the submitted study, so they do not support an external-validation claim.",
)


@dataclass(frozen=True)
class Interval:
    """A point estimate with its percentile interval."""

    estimate: float
    lower: float
    upper: float
    level: float


@dataclass(frozen=True)
class Resampling:
    """Per-resample seed-mean metric values for each system, with a receipt."""

    metric: str
    resamples: int
    seed: int
    level: float
    method: str
    groups_per_class: dict[str, int]
    degenerate_classes: tuple[str, ...]
    unsupported_classes: tuple[str, ...]
    limitations: tuple[str, ...]
    seeds: dict[str, tuple[int, ...]]
    estimates: dict[str, float]
    samples: dict[str, np.ndarray]

    def interval(self, system: str) -> Interval:
        self._require(system)
        return self._interval(self.estimates[system], self.samples[system])

    def difference(self, system: str, baseline: str) -> Interval:
        """Paired ``system - baseline`` difference over the shared resamples."""
        self._require(system)
        self._require(baseline)
        return self._interval(
            self.estimates[system] - self.estimates[baseline],
            self.samples[system] - self.samples[baseline],
        )

    def _require(self, system: str) -> None:
        if system not in self.samples:
            raise ValueError(f"unknown system {system!r}")

    def _interval(self, estimate: float, samples: np.ndarray) -> Interval:
        tail = (1 - self.level) / 2 * 100
        lower, upper = np.percentile(samples, [tail, 100 - tail], method="linear")
        return Interval(
            estimate=float(estimate),
            lower=float(lower),
            upper=float(upper),
            level=self.level,
        )


def _check_shared(systems: Mapping[str, PredictionSet]) -> PredictionSet:
    reference = next(iter(systems.values()))
    for system in systems.values():
        if (
            system.classes != reference.classes
            or system.image_ids != reference.image_ids
            or system.truth != reference.truth
            or system.groups != reference.groups
        ):
            raise ValueError(
                "systems must share the same classes, images, truth and groups"
            )
    return reference


def _group_layout(
    reference: PredictionSet,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    """Index each image's group and list each class's groups in sorted order."""
    names = sorted(set(reference.groups))
    position = {name: i for i, name in enumerate(names)}
    members: dict[str, set[str]] = {name: set() for name in names}
    for group, label in zip(reference.groups, reference.truth, strict=True):
        members[group].add(label)
    by_class: dict[str, list[int]] = {label: [] for label in reference.classes}
    for name in names:
        if len(members[name]) > 1:
            raise ValueError(f"group {name!r} spans more than one class")
        (label,) = members[name]
        by_class[label].append(position[name])
    image_group = np.array([position[g] for g in reference.groups], dtype=np.int64)
    return image_group, names, by_class


def _group_confusions(
    system: PredictionSet, image_group: np.ndarray, num_groups: int
) -> np.ndarray:
    """Return counts shaped ``(seeds, groups, classes, classes)``."""
    k = len(system.classes)
    truth = system.truth_index()
    per_seed = []
    for seed in system.seeds:
        cell = image_group * k * k + truth * k + system.prediction_index(seed)
        flat = np.bincount(cell, minlength=num_groups * k * k)
        per_seed.append(flat.reshape(num_groups, k, k))
    return np.stack(per_seed).astype(np.float64)


def _draw_weights(
    rng: np.random.Generator,
    count: int,
    num_groups: int,
    by_class: Mapping[str, list[int]],
) -> np.ndarray:
    """Draw how many times each group appears in each resample, within class."""
    weights = np.zeros((count, num_groups))
    rows = np.arange(count)[:, None]
    for groups in by_class.values():
        n = len(groups)
        if n == 0:
            continue
        draws = rng.integers(0, n, size=(count, n))
        counts = np.bincount((draws + rows * n).ravel(), minlength=count * n)
        weights[:, groups] = counts.reshape(count, n)
    return weights


def resample_metrics(
    systems: Mapping[str, PredictionSet],
    metric: str,
    resamples: int = RESAMPLES,
    seed: int = EVAL_SEED,
    level: float = CONFIDENCE_LEVEL,
) -> Resampling:
    """Resample groups within class and score every system on the same draws."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    if not systems:
        raise ValueError("no systems to resample")
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    reference = _check_shared(systems)
    image_group, names, by_class = _group_layout(reference)
    counts = {
        name: _group_confusions(system, image_group, len(names))
        for name, system in systems.items()
    }
    estimates = {
        name: float(np.mean(metric_values(per_group.sum(axis=1), metric)))
        for name, per_group in counts.items()
    }

    rng = np.random.default_rng(seed)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in systems}
    for start in range(0, resamples, _CHUNK):
        weights = _draw_weights(
            rng, min(_CHUNK, resamples - start), len(names), by_class
        )
        for name, per_group in counts.items():
            confusions = np.einsum("mg,sgij->msij", weights, per_group)
            chunks[name].append(metric_values(confusions, metric).mean(axis=1))

    groups_per_class = {label: len(by_class[label]) for label in reference.classes}
    degenerate = tuple(c for c, n in groups_per_class.items() if n == 1)
    unsupported = tuple(c for c, n in groups_per_class.items() if n == 0)
    method = (
        f"{level:.0%} percentile bootstrap, {resamples} resamples, seed {seed}; "
        "conservative groups drawn with replacement within class and jointly "
        "across seeds; metric computed per seed on each resample, then averaged"
    )
    limitations = _LIMITATIONS
    if degenerate:
        limitations += (
            "Classes with a single group do not vary under resampling: "
            f"{', '.join(degenerate)}.",
        )
    return Resampling(
        metric=metric,
        resamples=resamples,
        seed=seed,
        level=level,
        method=method,
        groups_per_class=groups_per_class,
        degenerate_classes=degenerate,
        unsupported_classes=unsupported,
        limitations=limitations,
        seeds={name: system.seeds for name, system in systems.items()},
        estimates=estimates,
        samples={name: np.concatenate(parts) for name, parts in chunks.items()},
    )
