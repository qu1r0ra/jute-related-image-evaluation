"""Empirical coverage of the nested bootstrap on synthetic data with a known truth.

The nested bootstrap is descriptive, so its intervals earn no confirmatory
weight; this simulation reports how often they nonetheless contain the truth.

The synthetic model is deliberately simple, so the truth is exact:

* every split k has a quality q_k drawn uniformly from ``QUALITY_RANGE``, so the
  population mean accuracy is the midpoint of that range (0.75 by default);
* every training seed shifts the split's quality by a symmetric, mean-zero
  offset;
* each split holds its own test families of two same-class images, with classes
  balanced, and every image is right with the run's quality and otherwise
  predicted as the other class.

Intervals are the percentile intervals of ``resample_metrics`` with shared family
resampling, the View 2 setting, for the accuracy metric.
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


def _split(
    rng: np.random.Generator, split: int, n_seeds: int, n_families: int
) -> PredictionSet:
    quality = rng.uniform(*QUALITY_RANGE)
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


def simulate_coverage(
    replicates: int = 200,
    resamples: int = 200,
    seed: int = EVAL_SEED,
    metric: str = "accuracy",
    level: float = CONFIDENCE_LEVEL,
    n_splits: int = 10,
    n_seeds: int = 5,
    n_families: int = 40,
) -> CoverageReport:
    """Simulate ``replicates`` studies and count intervals containing the truth.

    Only accuracy has an exactly known truth under the synthetic model, so other
    metrics are refused. The report carries a Wilson interval for the coverage
    itself, which has Monte Carlo error of about sqrt(p(1-p)/replicates).
    """
    if metric != "accuracy":
        raise ValueError("coverage is simulated for accuracy only")
    if replicates < 1 or resamples < 1 or n_splits < 2 or n_seeds < 1 or n_families < 2:
        raise ValueError("invalid simulation size")
    truth = sum(QUALITY_RANGE) / 2
    rng = np.random.default_rng(seed)
    covered = 0
    for _ in range(replicates):
        sets = {k: _split(rng, k, n_seeds, n_families) for k in range(n_splits)}
        result = resample_metrics(
            {"synthetic": sets},
            metric,
            resamples,
            int(rng.integers(2**31)),
            level,
            family_resampling="shared",
        )
        interval = result.interval("synthetic")
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
        "shared",
        "descriptive",
        "Coverage of the nested split, seed and family percentile bootstrap on "
        "synthetic data with known truth; descriptive",
    )
