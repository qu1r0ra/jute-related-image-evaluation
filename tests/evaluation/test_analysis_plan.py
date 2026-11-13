"""The prespecified analysis of ADR 0004 on simulated split means with known truth."""

import math
from itertools import combinations

import pytest

from jute_disease.data.policies import POLICIES
from jute_disease.evaluation import (
    EQUIVALENCE_MARGIN,
    PRIMARY_CONTRAST,
    SECONDARY_TESTS,
    Cell,
    LeakageClaim,
    PredictionSet,
    analyze_split_means,
    holm_adjust,
    leakage_claim,
    paired_test,
    split_mean_scores,
)

ROUTE = PRIMARY_CONTRAST.route
OTHER_ROUTE = "mobilenet_v2_plantvillage"


def _ten(center: float, spread: float = 0.01) -> list[float]:
    """Ten split means with sample sd sqrt(10/9) * spread, centred on ``center``."""
    return [center - spread, center + spread] * 5


def _cells(effects: dict[tuple[str, int, str], float]) -> dict[Cell, list[float]]:
    """Every pool, view, route and policy; ``effects`` shifts the policies.

    ``effects`` maps (pool, view, route) to the near-duplicate shortfall, so
    image-random is centred 0.90, exact-copy halfway and near-duplicate lower.
    """
    cells = {}
    for pool in "ABC":
        for view in (1, 2):
            for route in (ROUTE, OTHER_ROUTE):
                gap = effects.get((pool, view, route), 0.0)
                for policy, share in zip(POLICIES, (0.0, 0.5, 1.0), strict=True):
                    jitter = 0.002 * (POLICIES.index(policy) + 1)
                    cells[Cell(pool, view, route, policy)] = _ten(
                        0.90 - gap * share, 0.01 + jitter
                    )
    return cells


def _quad(predictions: tuple[str, ...]) -> PredictionSet:
    """Four images, two per class, scored for one seed."""
    return PredictionSet(
        ("a", "b"),
        ("i0", "i1", "i2", "i3"),
        ("a", "a", "b", "b"),
        ("f0", "f1", "f2", "f3"),
        ("s",) * 4,
        {1: predictions},
    )


PERFECT = ("a", "a", "b", "b")
# Macro-F1 against PERFECT truth: 1/3 (all "a"), then 11/15 twice.
WEAK = [("a", "a", "a", "a"), ("a", "a", "b", "a"), ("b", "a", "b", "b")]


def _claim() -> LeakageClaim:
    """A three-split View 2 leakage claim with split differences 2/3, 4/15, 4/15."""
    return leakage_claim(
        {k: _quad(PERFECT) for k in range(3)},
        {k: _quad(WEAK[k]) for k in range(3)},
        {"i0", "i1", "i2", "i3"},
    )


def test_the_primary_test_is_welch_on_pool_c_view_1_image_random_minus_near_dup() -> (
    None
):
    assert (PRIMARY_CONTRAST.pool, PRIMARY_CONTRAST.view) == ("C", 1)
    assert PRIMARY_CONTRAST.route == "mobilenet_v2_direct"
    assert (PRIMARY_CONTRAST.minuend, PRIMARY_CONTRAST.subtrahend) == (
        "image-random",
        "near-duplicate",
    )
    cells = _cells({("C", 1, ROUTE): 0.04})
    # Other pools and routes carry different effects, so a wrong cell is visible.
    cells[Cell("C", 1, ROUTE, "image-random")] = [0.90, 0.92] * 5
    cells[Cell("C", 1, ROUTE, "near-duplicate")] = [0.86, 0.88] * 5
    report = analyze_split_means(cells, _claim())
    primary = report.primary
    # Hand computation: both variances are 10 * 0.01^2 / 9, so t = 6 sqrt(2)
    # on 18 df for a 0.04 difference.
    assert primary.estimate == pytest.approx(0.04)
    assert primary.statistic == pytest.approx(6 * math.sqrt(2))
    assert primary.df == pytest.approx(18.0)
    assert primary.p_value < 1e-6
    assert primary.interval.lower < 0.04 < primary.interval.upper
    assert primary.interval.level == 0.95


def test_at_most_five_secondary_tests_are_listed_and_holm_adjusted() -> None:
    assert 1 <= len(SECONDARY_TESTS) <= 5
    assert len(set(SECONDARY_TESTS)) == len(SECONDARY_TESTS)
    report = analyze_split_means(_cells({}), _claim())
    assert set(report.secondary) == set(SECONDARY_TESTS)
    raw = {name: result.p_value for name, result in report.secondary.items()}
    adjusted = holm_adjust(raw)
    for name, result in report.secondary.items():
        assert result.p_adjusted == pytest.approx(adjusted[name])
        assert result.p_adjusted >= result.p_value


def test_pools_a_and_b_get_two_one_sided_equivalence_tests_at_the_margin() -> None:
    cells = _cells({("B", 1, ROUTE): 0.03})  # pool B carries a real 3-point effect
    report = analyze_split_means(cells, _claim())
    assert set(report.equivalence) == {"A", "B"}
    pool_a, pool_b = report.equivalence["A"], report.equivalence["B"]
    assert pool_a.margin == pool_b.margin == EQUIVALENCE_MARGIN
    assert pool_a.equivalent  # no effect, spread far below the margin
    assert not pool_b.equivalent  # 3 points lies outside the 2-point margin
    assert pool_b.estimate == pytest.approx(0.03, abs=1e-9)


def test_every_other_result_is_exploratory_and_nothing_prespecified_leaks_in() -> None:
    report = analyze_split_means(_cells({}), _claim())
    # 3 pools x 2 views x 2 routes x 3 policy pairs = 36 contrasts; the primary,
    # the two cell-based secondary tests and the two equivalence tests are
    # prespecified, which leaves 31.
    assert len(report.exploratory) == 31
    assert all(row.status == "exploratory" for row in report.exploratory)
    labels = [row.label for row in report.exploratory]
    assert len(set(labels)) == len(labels)
    assert not any(
        row.label == f"C view 1 {ROUTE}: image-random minus near-duplicate"
        for row in report.exploratory
    )
    pairs = {(row.minuend, row.subtrahend) for row in report.exploratory}
    assert pairs == set(combinations(POLICIES, 2))


def test_view_1_contrasts_are_unpaired_and_view_2_contrasts_are_paired() -> None:
    report = analyze_split_means(_cells({}), _claim())
    methods = {(row.view, row.method) for row in report.exploratory}
    assert methods == {(1, "welch"), (2, "paired")}


def test_a_missing_prespecified_cell_is_refused() -> None:
    cells = _cells({})
    del cells[Cell("C", 1, ROUTE, "near-duplicate")]
    with pytest.raises(ValueError, match="near-duplicate"):
        analyze_split_means(cells, _claim())


def test_split_mean_scores_average_macro_f1_over_seeds_per_split() -> None:
    def system(predictions) -> PredictionSet:
        return PredictionSet(
            ("a", "b"),
            ("i0", "i1"),
            ("a", "b"),
            ("f0", "f1"),
            ("s", "s"),
            predictions,
        )

    # Split 0: seed 1 perfect (macro-F1 1), seed 2 all "a" (F1 2/3 and 0, so
    # 1/3); the split mean is 2/3. Split 1 is perfect for both seeds.
    by_split = {
        0: system({1: ("a", "b"), 2: ("a", "a")}),
        1: system({1: ("a", "b"), 2: ("a", "b")}),
    }
    assert split_mean_scores(by_split) == pytest.approx((2 / 3, 1.0))


def test_leakage_claim_uses_only_the_strict_view_2_subset_and_pairs_splits() -> None:
    def system(strict_predictions, other_predictions) -> PredictionSet:
        # Four strict images (two per class) and two other images.
        truth = ("a", "a", "b", "b", "a", "b")
        return PredictionSet(
            ("a", "b"),
            ("s0", "s1", "s2", "s3", "o0", "o1"),
            truth,
            tuple(f"f{i}" for i in range(6)),
            ("s",) * 6,
            {1: (*strict_predictions, *other_predictions)},
        )

    strict = {"s0", "s1", "s2", "s3"}
    perfect = ("a", "a", "b", "b")
    # Near-duplicate policy: split 0 predicts "a" for every strict image
    # (macro-F1 1/3), split 1 gets one "b" wrong (F1 4/5 and 2/3, so 11/15), and
    # split 2 is perfect. Image-random is perfect on the strict subset in every
    # split but wrong on both other images, which the claim must ignore.
    image_random = {k: system(perfect, ("b", "a")) for k in range(3)}
    near_duplicate = {
        0: system(("a", "a", "a", "a"), ("a", "b")),
        1: system(("a", "a", "b", "a"), ("a", "b")),
        2: system(perfect, ("a", "b")),
    }
    claim = leakage_claim(image_random, near_duplicate, strict)
    differences = [2 / 3, 1 - 11 / 15, 0.0]
    assert claim.estimate == pytest.approx(sum(differences) / 3)
    assert claim.n_splits == 3
    assert claim.n_images == 4
    reference = paired_test([1.0, 1.0, 1.0], [1 / 3, 11 / 15, 1.0])
    assert claim.p_value == pytest.approx(reference.p_value)
    assert claim.interval.lower == pytest.approx(reference.interval.lower)
    assert claim.view == 2
    assert "lower bound" in claim.framing


def test_leakage_claim_refuses_a_subset_it_cannot_score() -> None:
    sets = {k: _quad(PERFECT) for k in range(3)}
    weak = {k: _quad(WEAK[k]) for k in range(3)}
    with pytest.raises(ValueError, match="strict"):
        leakage_claim(sets, weak, set())
    with pytest.raises(ValueError, match="strict"):
        leakage_claim(sets, weak, {"not-an-image"})
    with pytest.raises(ValueError, match="splits"):
        leakage_claim(sets, {0: weak[0]}, {"i0"})
    with pytest.raises(ValueError, match="splits"):
        leakage_claim({0: sets[0]}, {0: weak[0]}, {"i0", "i1"})
