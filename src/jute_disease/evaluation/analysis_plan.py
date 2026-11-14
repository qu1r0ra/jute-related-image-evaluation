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
* a leakage claim on the strict View 2 subset of each redraw, with Pool C's
  effect framed as a lower bound under a stated, untested assumption
  (``LEAKAGE_FRAMING``) and computed with a paired test, because View 2 is one
  protected test set per pool and redraw, shared by the policies;
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


SPLIT_COUNT = 10
"""Split means per cell: ADR 0004 fixes ten splits, so every cell has exactly ten."""

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

EQUIVALENCE_CONTRASTS = tuple(
    Contrast(pool, 1, IMAGENET_ROUTE, "image-random", "near-duplicate")
    for pool in EQUIVALENCE_POOLS
)
"""The contrasts tested for equivalence, one per pool in ``EQUIVALENCE_POOLS``."""

PRESPECIFIED_CONTRASTS = (
    PRIMARY_CONTRAST,
    *SECONDARY_CONTRASTS.values(),
    *EQUIVALENCE_CONTRASTS,
)
"""Every cell-based contrast the plan fixes in advance."""


def is_prespecified(contrast: Contrast) -> bool:
    """Whether the contrast is one of the prespecified tests, in either orientation."""
    ends = {contrast.minuend, contrast.subtrahend}
    return any(
        (contrast.pool, contrast.view, contrast.route)
        == (fixed.pool, fixed.view, fixed.route)
        and ends == {fixed.minuend, fixed.subtrahend}
        for fixed in PRESPECIFIED_CONTRASTS
    )


# Why "lower bound" is kept, and what it rests on (#185 and #179 story 25 require
# Pool C's effect to be framed as a lower bound; a statistics review asked that it
# not be worded as a guarantee). The near-duplicate policy still leaks (Pool C
# under-merges), and residual leakage can only raise that policy's score. The
# image-random minus near-duplicate gap D therefore understates the leakage
# effect L (the gap to a leak-free policy), D = L - r with r >= 0, so D is a lower
# bound on L as a population quantity. That needs an assumption the data cannot
# check: that residual leakage is the only thing separating the two policies on
# the strict subset, and that it never lowers the near-duplicate score. The
# sampling error of D is separate: a confidence interval for D has its lower
# limit as a lower confidence limit for L only under the same assumption. The
# wording below keeps the ticket's framing and states the assumption; whether to
# accept it is the owner's decision, and this module does not claim it is met.
LEAKAGE_FRAMING = (
    "Pool C's leakage effect is framed as a lower bound: the near-duplicate "
    "policy still leaks, and residual leakage can only raise its score, so the "
    "estimated image-random minus near-duplicate gap on the strict View 2 subset "
    "is expected to understate the leakage effect. That rests on an assumption "
    "that is not tested here: that residual leakage is the only difference "
    "between the policies on those images and never lowers the near-duplicate "
    "score. The estimate also carries sampling error, and the interval's lower "
    "limit is a lower limit for the strict-subset gap, so it bounds the leakage "
    "effect only if the assumption holds."
)

EQUIVALENCE_NOTE = (
    "Equivalence is declared only when the whole 90% interval of the split-mean "
    "difference lies inside the margin, so the test is weak unless the spread of "
    "the ten split means is small relative to that margin; with a large spread a "
    "failure to establish equivalence is expected and is not evidence of an "
    "effect. The Pool A and Pool B tests are unadjusted for the two pools."
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

    @property
    def contrast(self) -> Contrast:
        """The cell group this row compares, for the prespecified-set check."""
        return Contrast(self.pool, self.view, self.route, self.minuend, self.subtrahend)


@dataclass(frozen=True)
class LeakageClaim:
    """Paired View 2 difference, image-random minus near-duplicate, on strict images."""

    estimate: float
    statistic: float
    df: float
    p_value: float
    interval: Interval
    n_splits: int
    n_images: tuple[int, ...]
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
    equivalence_note: str = EQUIVALENCE_NOTE


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
    strict_images: Mapping[int, Collection[str]],
) -> LeakageClaim:
    """The leakage claim: View 2 only, on the strict subset, paired by split.

    View 2 is one protected test set per pool and redraw (ADR 0001), so the
    strict subset belongs to a split: ``strict_images`` maps each split to the
    image ids of its own strict subset, taken from the policy receipt. Both
    policies are scored on exactly those images in that split, one macro-F1 per
    split, and the per-split differences go to a paired t test. Pool C's effect
    is framed as a lower bound, under an assumption stated in ``LEAKAGE_FRAMING``.
    """
    if set(image_random) != set(near_duplicate) or len(image_random) < 2:
        raise ValueError("leakage claim needs the same two or more splits per policy")
    if set(strict_images) != set(image_random):
        raise ValueError(
            "the strict View 2 subsets must be given for exactly the same splits "
            "as the policies"
        )
    keep = {split: set(images) for split, images in strict_images.items()}
    for split, images in keep.items():
        if not images:
            raise ValueError(f"the strict View 2 subset of split {split} is empty")
    for policy_sets in (image_random, near_duplicate):
        for split, system in policy_sets.items():
            if not keep[split] <= set(system.image_ids):
                raise ValueError(
                    f"split {split} lacks images of the strict View 2 subset"
                )
    first = split_mean_scores(
        {k: _restrict(s, keep[k]) for k, s in image_random.items()}
    )
    second = split_mean_scores(
        {k: _restrict(s, keep[k]) for k, s in near_duplicate.items()}
    )
    result = paired_test(first, second)
    return LeakageClaim(
        result.estimate,
        result.statistic,
        result.df,
        result.p_value,
        result.interval,
        len(first),
        tuple(len(keep[split]) for split in sorted(keep)),
        2,
        LEAKAGE_FRAMING,
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


def _check_cells(cells: Mapping[Cell, Sequence[float]]) -> None:
    """Refuse cells that are not ten split means each, balanced across policies.

    The generic tests accept any two or more values, so this is the boundary that
    holds the plan to ADR 0004: a cell with a missing or extra split would still
    produce a p-value. Policies of one pool, view and route are compared with
    each other, so a group whose policies differ in length is refused first, and
    every cell, prespecified or exploratory, must then have exactly ten means.
    """
    groups: dict[tuple[str, int, str], dict[str, int]] = {}
    for cell, values in cells.items():
        groups.setdefault((cell.pool, cell.view, cell.route), {})[cell.policy] = len(
            values
        )
    for (pool, view, route), counts in groups.items():
        if len(set(counts.values())) > 1:
            raise ValueError(
                f"split means must be balanced across policies for pool {pool}, "
                f"view {view}, route {route}: {dict(sorted(counts.items()))}"
            )
    for cell, values in cells.items():
        if len(values) != SPLIT_COUNT:
            raise ValueError(
                f"pool {cell.pool}, view {cell.view}, route {cell.route}, policy "
                f"{cell.policy} needs exactly {SPLIT_COUNT} split means, "
                f"got {len(values)}"
            )


def analyze_split_means(
    cells: Mapping[Cell, Sequence[float]],
    leakage: LeakageClaim,
    validation: tuple[Sequence[float], Sequence[float]] | None = None,
) -> AnalysisReport:
    """Run the prespecified analysis on per-split macro-F1 means.

    ``cells`` maps each pool, view, route and policy to its split means; every
    cell must hold exactly ``SPLIT_COUNT`` (ten) means and the policies of a
    pool, view and route must agree in count, or ``ValueError`` is raised.
    ``validation`` optionally holds the two policies' split means on validation
    draws, from which the power statement for the 2-point effect is computed.
    """
    _check_cells(cells)
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
        "strict View 2 subset: image-random minus near-duplicate "
        "(framed as a lower bound, under an untested assumption)",
        leakage.method,
        leakage.estimate,
        leakage.statistic,
        leakage.df,
        leakage.p_value,
        adjusted[LEAKAGE_TEST],
        leakage.interval,
    )
    equivalence = {}
    for pool, contrast in zip(EQUIVALENCE_POOLS, EQUIVALENCE_CONTRASTS, strict=True):
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
            if is_prespecified(contrast):
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
