from math import comb

import numpy as np
import pytest

from jute_disease.evaluation import (
    StratumCounts,
    agreement_gate,
    clopper_pearson_interval,
    design_weighted_interval,
)


def test_clopper_pearson_matches_closed_forms() -> None:
    # With no events the exact two-sided 95% upper bound is 1 - 0.025 ** (1 / n).
    zero = clopper_pearson_interval(0, 20)
    assert zero.estimate == 0
    assert zero.lower == 0
    assert zero.upper == pytest.approx(1 - 0.025 ** (1 / 20))
    # With every trial an event the lower bound is 0.025 ** (1 / n).
    full = clopper_pearson_interval(20, 20)
    assert full.upper == 1
    assert full.lower == pytest.approx(0.025 ** (1 / 20))
    # Worked example: 5 of 10 gives (0.18709, 0.81291).
    half = clopper_pearson_interval(5, 10)
    assert (half.lower, half.upper) == pytest.approx((0.187086, 0.812914), abs=1e-6)
    assert half.level == 0.95


def test_clopper_pearson_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        clopper_pearson_interval(3, 2)
    with pytest.raises(ValueError):
        clopper_pearson_interval(0, 0)


def test_design_weighted_estimate_uses_population_weights() -> None:
    strata = [
        StratumCounts(population=900, sampled=30, events=0),
        StratumCounts(population=100, sampled=30, events=15),
    ]
    result = design_weighted_interval(strata)
    # 0.9 * 0 + 0.1 * 0.5
    assert result.estimate == pytest.approx(0.05)
    assert result.lower <= result.estimate <= result.upper


def test_census_strata_add_no_sampling_error() -> None:
    census = [StratumCounts(population=13, sampled=13, events=10)]
    result = design_weighted_interval(census)
    assert (result.estimate, result.lower, result.upper) == (10 / 13,) * 3


def test_zero_events_still_gives_a_positive_upper_bound() -> None:
    strata = [
        StratumCounts(population=400, sampled=40, events=0),
        StratumCounts(population=200, sampled=40, events=0),
    ]
    result = design_weighted_interval(strata)
    assert result.estimate == 0
    assert 0 < result.upper < 0.2


def test_single_large_stratum_matches_the_binomial_closed_form() -> None:
    # In a huge population sampling without replacement is binomial, so the
    # one-stratum bound is the exact two-sided 95% upper bound 1 - 0.025 ** (1 / n).
    result = design_weighted_interval(
        [StratumCounts(population=10_000_000, sampled=20, events=0)]
    )
    assert result.lower == 0
    assert result.upper == pytest.approx(1 - 0.025 ** (1 / 20), abs=1e-4)


def test_single_small_stratum_matches_a_brute_force_exact_bound() -> None:
    population, sampled, events = 30, 10, 2

    # Brute force: the exact 95% two-sided bounds are the extreme population event
    # counts D that the observed count x does not reject at 2.5% in either tail.
    def pmf(count: int, d: int) -> float:
        return (
            comb(d, count)
            * comb(population - d, sampled - count)
            / comb(population, sampled)
        )

    plausible = [
        d
        for d in range(population + 1)
        if sum(pmf(k, d) for k in range(events + 1)) >= 0.025
        and sum(pmf(k, d) for k in range(events, sampled + 1)) >= 0.025
    ]
    result = design_weighted_interval([StratumCounts(population, sampled, events)])
    assert result.lower == pytest.approx(min(plausible) / population)
    assert result.upper == pytest.approx(max(plausible) / population)


# A synthetic design in the audit's shape: a census stratum, strata of very
# different size, and rare to common event rates.
SYNTHETIC_DESIGN = [
    # population, sampled, true event rate
    (13, 13, 0.60),
    (3000, 40, 0.01),
    (1200, 30, 0.00),
    (800, 30, 0.05),
    (450, 25, 0.10),
    (200, 30, 0.30),
    (60, 20, 0.50),
]


def test_design_weighted_interval_covers_known_truth_by_simulation() -> None:
    rng = np.random.default_rng(7)
    # Fix the finite population's true event counts, then redraw samples without
    # replacement, which is how the audit samples inside a stratum.
    truth_events = [round(n * p) for n, _, p in SYNTHETIC_DESIGN]
    truth = sum(truth_events) / sum(n for n, _, _ in SYNTHETIC_DESIGN)
    trials = 1500
    covered = 0
    for _ in range(trials):
        strata = [
            StratumCounts(
                population=n,
                sampled=m,
                events=int(rng.hypergeometric(d, n - d, m)),
            )
            for (n, m, _), d in zip(SYNTHETIC_DESIGN, truth_events, strict=True)
        ]
        interval = design_weighted_interval(strata)
        assert interval.lower <= interval.estimate <= interval.upper
        covered += interval.lower <= truth <= interval.upper
    # The bound is conservative: coverage is at least the nominal 95%.
    assert covered / trials >= 0.95


def test_design_weighted_interval_covers_when_events_are_rare_in_small_strata() -> None:
    rng = np.random.default_rng(11)
    design = [(40, 10, 1), (25, 10, 0), (500, 30, 5), (90, 30, 9)]
    truth = sum(d for _, _, d in design) / sum(n for n, _, _ in design)
    trials = 1500
    covered = 0
    for _ in range(trials):
        strata = [
            StratumCounts(n, m, int(rng.hypergeometric(d, n - d, m)))
            for n, m, d in design
        ]
        interval = design_weighted_interval(strata)
        covered += interval.lower <= truth <= interval.upper
    assert covered / trials >= 0.95


def _pair_ratings(rng: np.random.Generator, count: int) -> tuple[list[int], list[int]]:
    first = rng.integers(0, 2, count)
    flips = rng.random(count) < 0.2
    second = np.where(flips, 1 - first, first)
    return first.tolist(), second.tolist()


def test_clustered_kappa_interval_widens_for_recurring_images() -> None:
    rng = np.random.default_rng(3)
    first, second = _pair_ratings(rng, 40)
    # Each rated pair recurs four times through shared images, so the 160 rows
    # carry the information of 40 independent ones.
    first4 = [value for value in first for _ in range(4)]
    second4 = [value for value in second for _ in range(4)]
    clusters = [index for index in range(40) for _ in range(4)]
    strata = ["C:merged"] * 160
    naive = agreement_gate(first4, second4, seed=1, resamples=2000)
    clustered = agreement_gate(
        first4, second4, seed=1, resamples=2000, strata=strata, clusters=clusters
    )
    assert clustered.kappa == pytest.approx(naive.kappa)
    naive_width = naive.interval.upper - naive.interval.lower
    clustered_width = clustered.interval.upper - clustered.interval.lower
    assert clustered_width > 1.5 * naive_width


def test_clustered_kappa_resamples_within_strata() -> None:
    # One stratum agrees perfectly and the other disagrees perfectly. Resampling
    # inside strata keeps every draw's mix of the two fixed, so the interval stays
    # narrow; ignoring strata would let the mix vary.
    first = [0, 1] * 20 + [0, 1] * 20
    second = [0, 1] * 20 + [1, 0] * 20
    strata = ["A"] * 40 + ["B"] * 40
    clusters = list(range(80))
    stratified = agreement_gate(
        first, second, seed=5, resamples=2000, strata=strata, clusters=clusters
    )
    pooled = agreement_gate(first, second, seed=5, resamples=2000)
    assert stratified.kappa == pytest.approx(0)
    assert (stratified.interval.upper - stratified.interval.lower) < (
        pooled.interval.upper - pooled.interval.lower
    )


def test_clustered_kappa_requires_both_strata_and_clusters() -> None:
    with pytest.raises(ValueError, match="strata and clusters"):
        agreement_gate([0, 1], [0, 1], strata=["a", "a"])
