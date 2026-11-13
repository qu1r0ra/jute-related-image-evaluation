"""The prespecified analysis of ADR 0004, computed on one macro-F1 value per split.

Every test here runs on split means: for a route, pool, view and policy there is
one macro-F1 per split (the mean over that split's training seeds), ten values
in all. The plan fixes in advance:

* one primary Welch test, Pool C, View 1, MobileNetV2 ImageNet route,
  image-random minus near-duplicate;
* at most five secondary tests, listed in ``SECONDARY_TESTS`` and Holm-adjusted
  together;
* two one-sided equivalence tests (TOST), one each for Pools A and B, at the
  recorded margin ``EQUIVALENCE_MARGIN``;
* a leakage claim on the strict View 2 subset, framed as a lower bound for
  Pool C and computed with a paired test, because View 2 is one protected test
  set shared by the policies;
* every other policy contrast as an exploratory row for the forest plot.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from jute_disease.data.policies import POLICIES
from jute_disease.evaluation.resampling import Interval
from jute_disease.evaluation.scores import PredictionSet, score_run
from jute_disease.evaluation.statistics import (
    EQUIVALENCE_MARGIN,
    EquivalenceResult,
    MeanDifference,
    PowerStatement,
    equivalence_test,
    holm_adjust,
    paired_test,
    split_power,
    welch_test,
)


@dataclass(frozen=True)
class Cell:
    """One pool, view, route and policy; its value is a list of split means."""

    pool: str
    view: int
    route: str
    policy: str


@dataclass(frozen=True)
class Contrast:
    """The macro-F1 of ``minuend`` minus that of ``subtrahend`` in one cell group."""

    pool: str
    view: int
    route: str
    minuend: str
    subtrahend: str

    @property
    def label(self) -> str:
        return (
            f"{self.pool} view {self.view} {self.route}: "
            f"{self.minuend} minus {self.subtrahend}"
        )


IMAGENET_ROUTE = "mobilenet_v2_direct"

PRIMARY_CONTRAST = Contrast("C", 1, IMAGENET_ROUTE, "image-random", "near-duplicate")

LEAKAGE_TEST = "leakage-claim-view-2-strict"

SECONDARY_CONTRASTS: dict[str, Contrast] = {
    "pool-c-view-1-image-random-minus-exact-copy": Contrast(
        "C", 1, IMAGENET_ROUTE, "image-random", "exact-copy"
    ),
    "pool-c-view-1-exact-copy-minus-near-duplicate": Contrast(
        "C", 1, IMAGENET_ROUTE, "exact-copy", "near-duplicate"
    ),
}
"""Cell-based secondary tests: the two halves of the primary contrast."""

SECONDARY_TESTS = (*SECONDARY_CONTRASTS, LEAKAGE_TEST)
"""All secondary tests (at most five); Holm adjustment runs over exactly these."""

EQUIVALENCE_POOLS = ("A", "B")
"""Pools expected to show no leakage effect, tested for equivalence."""

LOWER_BOUND_FRAMING = (
    "The strict View 2 subset keeps only images with no near-duplicate partner "
    "across the split, so the estimated image-random minus near-duplicate gap is "
    "a lower bound on the leakage effect in Pool C, not a point estimate of it."
)


@dataclass(frozen=True)
class SecondaryResult:
    """A secondary test with its raw and Holm-adjusted p-values."""

    name: str
    label: str
    method: str
    estimate: float
    statistic: float
    df: float
    p_value: float
    p_adjusted: float
    interval: Interval


@dataclass(frozen=True)
class ExploratoryRow:
    """One contrast outside the prespecified tests; reported, never confirmed."""

    label: str
    pool: str
    view: int
    route: str
    minuend: str
    subtrahend: str
    method: str
    estimate: float
    statistic: float
    df: float
    p_value: float
    interval: Interval
    status: str = "exploratory"


@dataclass(frozen=True)
class LeakageClaim:
    """Paired View 2 difference, image-random minus near-duplicate, on strict images."""

    estimate: float
    statistic: float
    df: float
    p_value: float
    interval: Interval
    n_splits: int
    n_images: int
    view: int
    framing: str
    method: str


@dataclass(frozen=True)
class AnalysisReport:
    """The primary test, Holm-adjusted secondaries, equivalence tests and the rest."""

    primary: MeanDifference
    secondary: dict[str, SecondaryResult]
    equivalence: dict[str, EquivalenceResult]
    exploratory: tuple[ExploratoryRow, ...]
    power: PowerStatement | None = None
    leakage: LeakageClaim | None = field(default=None)


def split_mean_scores(by_split: Mapping[int, PredictionSet]) -> tuple[float, ...]:
    """One macro-F1 per split: the mean over the split's seeds, in split order."""
    if not by_split:
        raise ValueError("no splits to score")
    return tuple(
        float(
            np.mean(
                [
                    score_run(by_split[split], seed).macro_f1
                    for seed in by_split[split].seeds
                ]
            )
        )
        for split in sorted(by_split)
    )


def _restrict(system: PredictionSet, keep: Collection[str]) -> PredictionSet:
    chosen = [i for i, image in enumerate(system.image_ids) if image in keep]
    return PredictionSet(
        system.classes,
        tuple(system.image_ids[i] for i in chosen),
        tuple(system.truth[i] for i in chosen),
        tuple(system.groups[i] for i in chosen),
        tuple(system.sources[i] for i in chosen),
        {
            seed: tuple(labels[i] for i in chosen)
            for seed, labels in system.predictions.items()
        },
    )


def leakage_claim(
    image_random: Mapping[int, PredictionSet],
    near_duplicate: Mapping[int, PredictionSet],
    strict_images: Collection[str],
) -> LeakageClaim:
    """The leakage claim: View 2 only, on the strict subset, paired by split.

    ``strict_images`` are the image ids of the strict View 2 subset, taken from
    the policy receipt. Both policies are scored on exactly those images, one
    macro-F1 per split, and the per-split differences go to a paired t test.
    Pool C's effect is a lower bound (``LOWER_BOUND_FRAMING``).
    """
    if set(image_random) != set(near_duplicate) or len(image_random) < 2:
        raise ValueError("leakage claim needs the same two or more splits per policy")
    keep = set(strict_images)
    if not keep:
        raise ValueError("the strict View 2 subset is empty")
    for policy_sets in (image_random, near_duplicate):
        for split, system in policy_sets.items():
            if not keep <= set(system.image_ids):
                raise ValueError(
                    f"split {split} lacks images of the strict View 2 subset"
                )
    first = split_mean_scores({k: _restrict(s, keep) for k, s in image_random.items()})
    second = split_mean_scores(
        {k: _restrict(s, keep) for k, s in near_duplicate.items()}
    )
    result = paired_test(first, second)
    return LeakageClaim(
        result.estimate,
        result.statistic,
        result.df,
        result.p_value,
        result.interval,
        len(first),
        len(keep),
        2,
        LOWER_BOUND_FRAMING,
        "paired t test on strict View 2 split means",
    )


def _scores(
    cells: Mapping[Cell, Sequence[float]], contrast: Contrast, policy: str
) -> Sequence[float]:
    cell = Cell(contrast.pool, contrast.view, contrast.route, policy)
    if cell not in cells:
        raise ValueError(
            f"missing split means for pool {cell.pool}, view {cell.view}, "
            f"route {cell.route}, policy {cell.policy}"
        )
    return cells[cell]


def _compare(
    cells: Mapping[Cell, Sequence[float]], contrast: Contrast
) -> MeanDifference:
    """Welch for View 1 (policies own their families), paired for View 2 (shared)."""
    first = _scores(cells, contrast, contrast.minuend)
    second = _scores(cells, contrast, contrast.subtrahend)
    if contrast.view == 1:
        return welch_test(first, second)
    if contrast.view == 2:
        return paired_test(first, second)
    raise ValueError(f"unknown view {contrast.view}")


def analyze_split_means(
    cells: Mapping[Cell, Sequence[float]],
    leakage: LeakageClaim,
    validation: tuple[Sequence[float], Sequence[float]] | None = None,
) -> AnalysisReport:
    """Run the prespecified analysis on per-split macro-F1 means.

    ``cells`` maps each pool, view, route and policy to its split means.
    ``validation`` optionally holds the two policies' split means on validation
    draws, from which the power statement for the 2-point effect is computed.
    """
    primary = _compare(cells, PRIMARY_CONTRAST)
    raw = {name: _compare(cells, c) for name, c in SECONDARY_CONTRASTS.items()}
    raw_p = {name: result.p_value for name, result in raw.items()}
    raw_p[LEAKAGE_TEST] = leakage.p_value
    adjusted = holm_adjust(raw_p)
    secondary = {
        name: SecondaryResult(
            name,
            SECONDARY_CONTRASTS[name].label,
            result.method,
            result.estimate,
            result.statistic,
            result.df,
            result.p_value,
            adjusted[name],
            result.interval,
        )
        for name, result in raw.items()
    }
    secondary[LEAKAGE_TEST] = SecondaryResult(
        LEAKAGE_TEST,
        "strict View 2 subset: image-random minus near-duplicate (lower bound)",
        leakage.method,
        leakage.estimate,
        leakage.statistic,
        leakage.df,
        leakage.p_value,
        adjusted[LEAKAGE_TEST],
        leakage.interval,
    )
    equivalence = {}
    prespecified = {PRIMARY_CONTRAST, *SECONDARY_CONTRASTS.values()}
    for pool in EQUIVALENCE_POOLS:
        contrast = Contrast(pool, 1, IMAGENET_ROUTE, "image-random", "near-duplicate")
        prespecified.add(contrast)
        equivalence[pool] = equivalence_test(
            _scores(cells, contrast, contrast.minuend),
            _scores(cells, contrast, contrast.subtrahend),
            EQUIVALENCE_MARGIN,
        )
    exploratory = []
    for pool, view, route in sorted({(c.pool, c.view, c.route) for c in cells}):
        present = [p for p in POLICIES if Cell(pool, view, route, p) in cells]
        for minuend, subtrahend in combinations(present, 2):
            contrast = Contrast(pool, view, route, minuend, subtrahend)
            if contrast in prespecified:
                continue
            result = _compare(cells, contrast)
            exploratory.append(
                ExploratoryRow(
                    contrast.label,
                    pool,
                    view,
                    route,
                    minuend,
                    subtrahend,
                    "welch" if view == 1 else "paired",
                    result.estimate,
                    result.statistic,
                    result.df,
                    result.p_value,
                    result.interval,
                )
            )
    power = split_power(*validation) if validation is not None else None
    return AnalysisReport(
        primary, secondary, equivalence, tuple(exploratory), power, leakage
    )
