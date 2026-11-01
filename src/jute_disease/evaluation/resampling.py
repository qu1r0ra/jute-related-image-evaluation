"""Nested paired bootstrap of split, training seed and test-family uncertainty."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from jute_disease.evaluation.scores import METRICS, PredictionSet, metric_values

RESAMPLES = 10_000
EVAL_SEED = 42
CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class Interval:
    """A point estimate and percentile confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float


@dataclass(frozen=True)
class Resampling:
    """Replicate scores and descending rankings, with ties stable by name."""

    metric: str
    resamples: int
    seed: int
    level: float
    estimates: dict[str, float]
    samples: dict[str, np.ndarray]
    rankings: np.ndarray
    method: str

    def interval(self, system: str) -> Interval:
        return self._interval(self.estimates[system], self.samples[system])

    def difference(self, system: str, baseline: str) -> Interval:
        return self._interval(
            self.estimates[system] - self.estimates[baseline],
            self.samples[system] - self.samples[baseline],
        )

    def _interval(self, estimate: float, samples: np.ndarray) -> Interval:
        tail = (1 - self.level) / 2 * 100
        lower, upper = np.percentile(samples, [tail, 100 - tail])
        return Interval(float(estimate), float(lower), float(upper), self.level)


def resample_metrics(
    systems: Mapping[str, Mapping[int, PredictionSet]],
    metric: str,
    resamples: int = RESAMPLES,
    seed: int = EVAL_SEED,
    level: float = CONFIDENCE_LEVEL,
    checkpoints: Mapping[int, Mapping[int, str]] | None = None,
) -> Resampling:
    """Bootstrap shared splits, seeds and family IDs, scoring each run anew.

    Family draws use the union within each split. Draws with no observations in
    any system are rejected, so intervals are conditional on nonempty native
    test sets. Checkpoint assignments, when supplied, are shared across systems:
    seeds are resampled within each checkpoint, run scores are pooled per
    checkpoint across the sampled splits, and checkpoint means are weighted
    equally. Without assignments, each sampled split is weighted equally.
    """
    if not systems or metric not in METRICS or resamples < 1 or not 0 < level < 1:
        raise ValueError("invalid systems, metric, resample count or confidence level")
    names = sorted(systems)
    splits = sorted(systems[names[0]])
    if not splits or any(sorted(systems[n]) != splits for n in names):
        raise ValueError("systems must share nonempty split IDs")
    layouts = {}
    for split in splits:
        ref = systems[names[0]][split]
        if any(
            systems[n][split].classes != ref.classes
            or systems[n][split].seeds != ref.seeds
            for n in names
        ):
            raise ValueError(
                "systems must share classes and seed IDs within each split"
            )
        labels = {}
        for name in names:
            item = systems[name][split]
            for family, label in zip(item.groups, item.truth, strict=True):
                if family in labels and labels[family] != label:
                    raise ValueError("family spans more than one class")
                labels[family] = label
        families = sorted(labels)
        if checkpoints is None:
            strata = {None: tuple(ref.seeds)}
        else:
            assignment = checkpoints[split]
            if set(assignment) != set(ref.seeds):
                raise ValueError("checkpoint assignment must cover every seed")
            strata = {
                cp: tuple(s for s in ref.seeds if assignment[s] == cp)
                for cp in sorted(set(assignment.values()))
            }
        layouts[split] = (families, strata)

    def run_scores(name, split, groups, weights) -> dict:
        item = systems[name][split]
        index = {label: i for i, label in enumerate(item.classes)}
        scores = {}
        for key, seeds in groups.items():
            group_scores = []
            for run_seed in seeds:
                confusion = np.zeros((len(index), len(index)))
                for family, truth, pred in zip(
                    item.groups, item.truth, item.predictions[run_seed], strict=True
                ):
                    confusion[index[truth], index[pred]] += weights.get(family, 0)
                group_scores.append(float(metric_values(confusion, metric)))
            scores[key] = group_scores
        return scores

    def route_score(name, parts) -> float:
        # Pool runs per checkpoint across sampled splits, then weight checkpoints
        # equally. Without assignments, each sampled split is its own group.
        pooled: dict = {}
        for position, (split, groups, weights) in enumerate(parts):
            for cp, scores in run_scores(name, split, groups, weights).items():
                key = cp if checkpoints is not None else position
                pooled.setdefault(key, []).extend(scores)
        return float(np.mean([np.mean(pooled[key]) for key in sorted(pooled)]))

    estimates = {
        name: route_score(
            name,
            [
                (split, layouts[split][1], dict.fromkeys(layouts[split][0], 1))
                for split in splits
            ],
        )
        for name in names
    }
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(resamples) for name in names}
    for i in range(resamples):
        parts = []
        for split in rng.choice(splits, size=len(splits)):
            families, strata = layouts[split]
            groups = {
                key: rng.choice(group, size=len(group)) for key, group in strata.items()
            }
            for _ in range(10000):
                weights = Counter(rng.choice(families, size=len(families)))
                if all(
                    any(weights.get(f, 0) for f in systems[name][split].groups)
                    for name in names
                ):
                    break
            else:
                raise ValueError("could not draw nonempty native test sets")
            parts.append((split, groups, weights))
        for name in names:
            samples[name][i] = route_score(name, parts)
    rankings = np.array(
        [sorted(names, key=lambda n: (-samples[n][i], n)) for i in range(resamples)]
    )
    return Resampling(
        metric,
        resamples,
        seed,
        level,
        estimates,
        samples,
        rankings,
        "Nested split, seed and family percentile bootstrap; "
        "conditional on nonempty native sets",
    )
