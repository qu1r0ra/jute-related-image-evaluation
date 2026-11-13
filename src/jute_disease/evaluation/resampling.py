"""Nested paired bootstrap of split, training seed and test-family uncertainty.

The bootstrap is descriptive (ADR 0004): it summarises how much a score moves
under resampling of splits, seeds and test families, and it carries no
confirmatory weight. Its empirical coverage is checked on synthetic data with a
known truth by ``jute_disease.evaluation.coverage``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from jute_disease.evaluation.scores import (
    METRICS,
    PredictionSet,
    metric_values,
)

RESAMPLES = 10_000
EVAL_SEED = 42
CONFIDENCE_LEVEL = 0.95
FAMILY_RESAMPLING = ("shared", "per-system")


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


def require_class_support(
    classes: Sequence[str], native: np.ndarray, drawn: np.ndarray
) -> None:
    """Assert that a family draw keeps every class that the native set contains.

    ``native`` and ``drawn`` hold one image count (or weight) per class. A class
    with native images and no drawn images would silently change the macro
    average, so the draw is refused instead of being redrawn.
    """
    native, drawn = np.asarray(native), np.asarray(drawn)
    if native.shape != (len(classes),) or drawn.shape != native.shape:
        raise ValueError("class counts must have one entry per class")
    lost = [
        repr(label)
        for label, have, got in zip(classes, native, drawn, strict=True)
        if have > 0 and got == 0
    ]
    if lost:
        raise ValueError(f"family draw lost class support for {', '.join(lost)}")


@dataclass(frozen=True)
class _Layout:
    """One system's test set in one split, prepared for repeated weighted scoring."""

    classes: tuple[str, ...]
    seeds: tuple[int, ...]
    families: tuple[str, ...]
    family_index: np.ndarray
    truth: np.ndarray
    native: np.ndarray
    codes: np.ndarray
    strata: tuple[np.ndarray, ...]

    @property
    def family_classes(self) -> dict[str, tuple[str, ...]]:
        """The sorted class labels carried by each family."""
        found: dict[int, set[int]] = {}
        for family, label in zip(self.family_index, self.truth, strict=True):
            found.setdefault(int(family), set()).add(int(label))
        return {
            self.families[family]: tuple(sorted(self.classes[c] for c in labels))
            for family, labels in found.items()
        }


def _layout(item: PredictionSet) -> _Layout:
    families = tuple(sorted(set(item.groups)))
    position = {family: i for i, family in enumerate(families)}
    family_index = np.array([position[f] for f in item.groups], dtype=np.int64)
    truth = item.truth_index()
    count = len(item.classes)
    codes = np.stack(
        [
            run * count * count + truth * count + item.prediction_index(seed)
            for run, seed in enumerate(item.seeds)
        ]
    )
    # Strata are keyed by each family's class set, so a family that carries two
    # labels is its own stratum and every class keeps a family in every draw.
    by_key: dict[tuple[int, ...], list[int]] = {}
    for family in range(len(families)):
        key = tuple(sorted({int(c) for c in truth[family_index == family]}))
        by_key.setdefault(key, []).append(family)
    strata = tuple(np.array(by_key[key], dtype=np.int64) for key in sorted(by_key))
    return _Layout(
        item.classes,
        item.seeds,
        families,
        family_index,
        truth,
        np.bincount(truth, minlength=count),
        codes,
        strata,
    )


def _draw_family_counts(layout: _Layout, rng: np.random.Generator) -> np.ndarray:
    """Draw each stratum's families with replacement, keeping the stratum size."""
    counts = np.zeros(len(layout.families))
    for members in layout.strata:
        picks = rng.integers(0, len(members), len(members))
        counts[members] += np.bincount(picks, minlength=len(members))
    return counts


def _run_scores(layout: _Layout, family_counts: np.ndarray, metric: str) -> np.ndarray:
    """Score every seed's run at once under the given family multiplicities."""
    weights = family_counts[layout.family_index]
    require_class_support(
        layout.classes,
        layout.native,
        np.bincount(layout.truth, weights=weights, minlength=len(layout.classes)),
    )
    size = len(layout.classes)
    runs = len(layout.seeds)
    flat = np.bincount(
        layout.codes.ravel(), weights=np.tile(weights, runs), minlength=runs * size**2
    )
    return metric_values(flat.reshape(runs, size, size), metric)


def resample_metrics(
    systems: Mapping[str, Mapping[int, PredictionSet]],
    metric: str,
    resamples: int = RESAMPLES,
    seed: int = EVAL_SEED,
    level: float = CONFIDENCE_LEVEL,
    checkpoints: Mapping[int, Mapping[int, str]] | None = None,
    family_resampling: str = "shared",
) -> Resampling:
    """Bootstrap splits, seeds and class-stratified test families, scoring each run.

    Split and seed draws are shared by every system. Family draws are stratified
    by each family's class set, so every class keeps its native families and the
    class support is asserted instead of redrawn.

    ``family_resampling`` chooses how families are shared. ``"shared"`` (View 2)
    needs every system to hold the same test families in each split and gives
    them identical multiplicities. ``"per-system"`` (View 1) lets each system
    resample its own native families independently. Checkpoint assignments, when
    supplied, are shared across systems: seeds are resampled within each
    checkpoint, run scores are pooled per checkpoint across the sampled splits,
    and checkpoint means are weighted equally. Without assignments, each sampled
    split is weighted equally.
    """
    if not systems or metric not in METRICS or resamples < 1 or not 0 < level < 1:
        raise ValueError("invalid systems, metric, resample count or confidence level")
    if family_resampling not in FAMILY_RESAMPLING:
        raise ValueError(f"family_resampling must be one of {FAMILY_RESAMPLING}")
    shared = family_resampling == "shared"
    names = sorted(systems)
    splits = sorted(systems[names[0]])
    if not splits or any(sorted(systems[n]) != splits for n in names):
        raise ValueError("systems must share nonempty split IDs")
    layouts: dict[str, dict[int, _Layout]] = {name: {} for name in names}
    seed_groups: dict[int, dict] = {}
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
        for name in names:
            layouts[name][split] = _layout(systems[name][split])
        if shared and any(
            layouts[n][split].family_classes != layouts[names[0]][split].family_classes
            for n in names
        ):
            raise ValueError(
                "shared family resampling needs the same test families in every system"
            )
        if checkpoints is None:
            groups = {None: np.arange(len(ref.seeds))}
        else:
            assignment = checkpoints[split]
            if set(assignment) != set(ref.seeds):
                raise ValueError("checkpoint assignment must cover every seed")
            groups = {
                cp: np.array(
                    [i for i, s in enumerate(ref.seeds) if assignment[s] == cp]
                )
                for cp in sorted(set(assignment.values()))
            }
        seed_groups[split] = groups

    def route_score(name: str, parts: list) -> float:
        # Pool runs per checkpoint across sampled splits, then weight checkpoints
        # equally. Without assignments, each sampled split is its own group.
        pooled: dict = {}
        for position, (split, draws, counts) in enumerate(parts):
            scores = _run_scores(layouts[name][split], counts[name], metric)
            for key, picks in draws.items():
                pool = key if checkpoints is not None else position
                pooled.setdefault(pool, []).append(scores[picks])
        return float(
            np.mean([np.mean(np.concatenate(pooled[key])) for key in sorted(pooled)])
        )

    def whole(split: int) -> tuple:
        ones = {n: np.ones(len(layouts[n][split].families)) for n in names}
        return split, seed_groups[split], ones

    estimates = {name: route_score(name, [whole(s) for s in splits]) for name in names}
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(resamples) for name in names}
    for i in range(resamples):
        drawn = [int(s) for s in rng.choice(splits, size=len(splits))]
        draws = [
            {
                key: group[rng.integers(0, len(group), len(group))]
                for key, group in seed_groups[split].items()
            }
            for split in drawn
        ]
        parts = []
        for split, picks in zip(drawn, draws, strict=True):
            if shared:
                counts = _draw_family_counts(layouts[names[0]][split], rng)
                per_system = dict.fromkeys(names, counts)
            else:
                per_system = {
                    n: _draw_family_counts(layouts[n][split], rng) for n in names
                }
            parts.append((split, picks, per_system))
        for name in names:
            samples[name][i] = route_score(name, parts)
    order = np.argsort(
        -np.stack([samples[n] for n in names], axis=1), axis=1, kind="stable"
    )
    rankings = np.array(names)[order]
    return Resampling(
        metric,
        resamples,
        seed,
        level,
        estimates,
        samples,
        rankings,
        "Nested split, seed and family percentile bootstrap, descriptive; "
        f"class-stratified family draws, {family_resampling} family resampling",
    )
