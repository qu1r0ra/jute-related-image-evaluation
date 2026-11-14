"""Empirical coverage of the nested bootstrap on synthetic data with a known truth.

The nested bootstrap is descriptive, so its intervals earn no confirmatory
weight; this simulation reports how often they nonetheless contain the truth,
under named synthetic configurations. A coverage figure belongs to the
configuration that produced it and is not a property of the method: it is
reported, never asserted to equal the nominal level.

The synthetic model is deliberately simple, so the truth is exact:

* every split k has a quality q_k drawn uniformly from ``quality_range``, so the
  population mean accuracy is the midpoint of that range (0.75 by default);
* every training seed shifts the split's quality by a symmetric, mean-zero
  offset;
* each split holds its own test families of two same-class images, with classes
  balanced, and every image is right with the run's quality and otherwise
  predicted as the other class.

Intervals are the percentile intervals of ``resample_metrics`` for the accuracy
metric. By default one system is simulated with shared family resampling (the
View 2 setting) and the target is that system's mean. With ``effect`` set, two
systems are simulated, the first better by ``effect`` on average, and the target
is their difference; ``family_resampling="per-system"`` then draws their splits,
seeds and families independently, the View 1 setting.

Why the default sits near nominal. A percentile interval over ten splits
undercovers, because ten split qualities are a small sample. The simulation
counts family variance twice: the ten observed splits already carry it in their
test sets, and the family draws add it again. The extra width offsets the
undercoverage in the default configuration, which is why it reads near 0.94.
The offset is configuration specific: with more families the repeated family
variance shrinks and coverage drops toward (a larger run with 400 families read
0.903, Wilson 0.865 to 0.932), and with a split spread of 0.74 to 0.76 (almost
no split variance) the double-counted family variance dominates and coverage is
about 1.00. ``coverage_grid`` reports these configurations side by side; at 100
replicates each entry carries a Monte Carlo error of up to about 0.03. Nothing
here covers other systems, other metrics, checkpoints or real model errors.
"""

from dataclasses import dataclass

import numpy as np

from jute_disease.evaluation.resampling import (
    CONFIDENCE_LEVEL,
    EVAL_SEED,
    resample_metrics,
)
from jute_disease.evaluation.scores import PredictionSet
from jute_disease.evaluation.statistics import wilson_interval

QUALITY_RANGE = (0.6, 0.9)
SEED_OFFSET = 0.02
IMAGES_PER_FAMILY = 2
CLASSES = ("a", "b")

GRID_FAMILIES = (20, 40, 400)
GRID_QUALITY_RANGES = ((0.74, 0.76), (0.6, 0.9))
GRID_EFFECT = 0.04
GRID_TOLERANCE = 0.10
"""How far below nominal a grid configuration may fall before a test complains.

The grid is reported, not tuned: coverage differs by configuration and the report
lists every one. The tolerance only guards against collapse, and at 100
replicates the Monte Carlo error of a single entry is about 0.03.
"""


@dataclass(frozen=True)
class CoverageReport:
    """How often the bootstrap interval contained the known truth."""

    metric: str
    truth: float
    nominal: float
    replicates: int
    covered: int
    coverage: float
    coverage_interval: tuple[float, float]
    resamples: int
    seed: int
    n_splits: int
    n_seeds: int
    n_families: int
    family_resampling: str
    status: str
    method: str
    target: str = "system mean"
    quality_range: tuple[float, float] = QUALITY_RANGE
    n_systems: int = 1


def _split(
    rng: np.random.Generator,
    split: int,
    n_seeds: int,
    n_families: int,
    quality_range: tuple[float, float] = QUALITY_RANGE,
) -> PredictionSet:
    quality = rng.uniform(*quality_range)
    families = [f"split{split}-family{i}" for i in range(n_families)]
    truth = [CLASSES[i % 2] for i in range(n_families)]
    image_ids, groups, labels = [], [], []
    for family, label in zip(families, truth, strict=True):
        for image in range(IMAGES_PER_FAMILY):
            image_ids.append(f"{family}-image{image}")
            groups.append(family)
            labels.append(label)
    offsets = np.linspace(-SEED_OFFSET, SEED_OFFSET, n_seeds) if n_seeds > 1 else [0.0]
    predictions = {}
    for seed, offset in enumerate(offsets):
        right = rng.uniform(size=len(labels)) < quality + offset
        predictions[seed] = tuple(
            label if ok else CLASSES[1 - CLASSES.index(label)]
            for label, ok in zip(labels, right, strict=True)
        )
    return PredictionSet(
        CLASSES,
        tuple(image_ids),
        tuple(labels),
        tuple(groups),
        ("synthetic",) * len(labels),
        predictions,
    )


def _check_quality(quality_range: tuple[float, float], effect: float | None) -> None:
    """Keep every run's success probability inside [0, 1], so the truth is exact."""
    low, high = quality_range
    shift = effect or 0.0
    if not (
        0 <= low - SEED_OFFSET and low <= high and high + shift + SEED_OFFSET <= 1
    ) or (effect is not None and effect < 0):
        raise ValueError(
            "quality range (and any effect) must keep every run's quality "
            "between 0 and 1 and the range ordered"
        )


def simulate_coverage(
    replicates: int = 200,
    resamples: int = 200,
    seed: int = EVAL_SEED,
    metric: str = "accuracy",
    level: float = CONFIDENCE_LEVEL,
    n_splits: int = 10,
    n_seeds: int = 5,
    n_families: int = 40,
    quality_range: tuple[float, float] = QUALITY_RANGE,
    effect: float | None = None,
    family_resampling: str = "shared",
) -> CoverageReport:
    """Simulate ``replicates`` studies and count intervals containing the truth.

    Only accuracy has an exactly known truth under the synthetic model, so other
    metrics are refused. The report carries a Wilson interval for the coverage
    itself, which has Monte Carlo error of about sqrt(p(1-p)/replicates), and
    names the configuration it describes (target, system count, quality range,
    family count and resampling mode).

    With ``effect`` left as None one system is simulated and the truth is the
    midpoint of ``quality_range``. With a number, two systems are simulated, the
    first with its split quality range shifted up by ``effect``, and the truth
    is that difference (first minus second).
    """
    if metric != "accuracy":
        raise ValueError("coverage is simulated for accuracy only")
    if replicates < 1 or resamples < 1 or n_splits < 2 or n_seeds < 1 or n_families < 2:
        raise ValueError("invalid simulation size")
    _check_quality(quality_range, effect)
    low, high = quality_range
    shifted = (low + effect, high + effect) if effect is not None else quality_range
    truth = effect if effect is not None else (low + high) / 2
    rng = np.random.default_rng(seed)
    covered = 0
    for _ in range(replicates):
        systems = {
            "first": {
                k: _split(rng, k, n_seeds, n_families, shifted) for k in range(n_splits)
            }
        }
        if effect is not None:
            systems["second"] = {
                k: _split(rng, k, n_seeds, n_families, quality_range)
                for k in range(n_splits)
            }
        result = resample_metrics(
            systems,
            metric,
            resamples,
            int(rng.integers(2**31)),
            level,
            family_resampling=family_resampling,
        )
        interval = (
            result.difference("first", "second")
            if effect is not None
            else result.interval("first")
        )
        covered += interval.lower <= truth <= interval.upper
    wilson = wilson_interval(covered, replicates)
    return CoverageReport(
        metric,
        truth,
        level,
        replicates,
        covered,
        covered / replicates,
        (wilson.lower, wilson.upper),
        resamples,
        seed,
        n_splits,
        n_seeds,
        n_families,
        family_resampling,
        "descriptive",
        "Coverage of the nested split, seed and family percentile bootstrap on "
        "synthetic data with known truth, under the named configuration only; "
        "descriptive",
        "two-system difference" if effect is not None else "system mean",
        quality_range,
        len(systems),
    )


def coverage_grid(
    replicates: int = 200, resamples: int = 200, seed: int = EVAL_SEED
) -> list[CoverageReport]:
    """Coverage over family count and split spread, for a mean and for a difference.

    Every combination of ``GRID_FAMILIES`` and ``GRID_QUALITY_RANGES`` is run
    twice: one system with shared family resampling (the mean of View 2), and a
    ``GRID_EFFECT`` difference between two systems resampled independently
    (View 1). The reports are for reading, not for tuning to the nominal level;
    ``GRID_TOLERANCE`` is the only bound a test places on them.
    """
    grid = []
    for families in GRID_FAMILIES:
        for quality_range in GRID_QUALITY_RANGES:
            common = {
                "replicates": replicates,
                "resamples": resamples,
                "seed": seed,
                "n_families": families,
                "quality_range": quality_range,
            }
            grid.append(simulate_coverage(**common))
            grid.append(
                simulate_coverage(
                    **common, effect=GRID_EFFECT, family_resampling="per-system"
                )
            )
    return grid
