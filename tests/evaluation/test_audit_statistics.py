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
    assert 0 < result.upper < 0.15


def test_design_weighted_interval_covers_known_truth() -> None:
    rng = np.random.default_rng(7)
    populations = [3000, 800, 200]
    truth_rates = [0.01, 0.05, 0.30]
    truth = sum(n * p for n, p in zip(populations, truth_rates, strict=True)) / sum(
        populations
    )
    sizes = [40, 30, 30]
    covered = 0
    trials = 400
    for _ in range(trials):
        strata = [
            StratumCounts(
                population=n,
                sampled=m,
                events=int(rng.binomial(m, p)),
            )
            for n, m, p in zip(populations, sizes, truth_rates, strict=True)
        ]
        interval = design_weighted_interval(strata)
        covered += interval.lower <= truth <= interval.upper
    assert covered / trials >= 0.93


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
