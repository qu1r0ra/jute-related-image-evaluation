"""The prespecified analysis of ADR 0004 on simulated split means with known truth."""

import math
from itertools import combinations

import pytest

from jute_disease.data.policies import POLICIES
from jute_disease.evaluation import (
    EQUIVALENCE_MARGIN,
    LEAKAGE_TEST,
    PRESPECIFIED_CONTRASTS,
    PRIMARY_CONTRAST,
    SECONDARY_TESTS,
    SPLIT_COUNT,
    Cell,
    Contrast,
    LeakageClaim,
    PredictionSet,
    analyze_split_means,
    holm_adjust,
    is_prespecified,
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
        {k: {"i0", "i1", "i2", "i3"} for k in range(3)},
    )


def _own_images(split: int, predictions: tuple[str, ...]) -> PredictionSet:
    """Four images named for the split: each redraw has its own protected test set."""
    return PredictionSet(
        ("a", "b"),
        tuple(f"r{split}-i{i}" for i in range(4)),
        ("a", "a", "b", "b"),
        tuple(f"r{split}-f{i}" for i in range(4)),
        ("s",) * 4,
        {1: predictions},
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


def test_every_prespecified_cell_needs_exactly_the_ten_split_means() -> None:
    assert SPLIT_COUNT == 10
    for contrast in PRESPECIFIED_CONTRASTS:
        for policy in (contrast.minuend, contrast.subtrahend):
            cell = Cell(contrast.pool, contrast.view, contrast.route, policy)
            for count in (2, 9, 11):
                # Every policy of the group gets the same count, so only the
                # exact-ten rule can refuse it, not the balance rule.
                cells = _cells({})
                for other in POLICIES:
                    key = Cell(cell.pool, cell.view, cell.route, other)
                    cells[key] = _ten(0.9)[:1] * count
                with pytest.raises(ValueError, match="exactly 10"):
                    analyze_split_means(cells, _claim())


def test_cells_must_be_balanced_across_policies() -> None:
    cells = _cells({})
    cells[Cell("C", 1, ROUTE, "near-duplicate")] = _ten(0.86)[:8]
    with pytest.raises(ValueError, match="balanced across policies"):
        analyze_split_means(cells, _claim())


def test_an_exploratory_cell_with_the_wrong_count_is_refused_too() -> None:
    cells = _cells({})
    # Pool A view 2 is outside the prespecified cells; its group is still checked.
    for policy in POLICIES:
        cells[Cell("A", 2, OTHER_ROUTE, policy)] = _ten(0.9)[:7]
    with pytest.raises(ValueError, match="exactly 10"):
        analyze_split_means(cells, _claim())


def test_non_finite_split_means_are_refused() -> None:
    for bad in (math.nan, math.inf):
        cells = _cells({})
        cells[Cell("C", 1, ROUTE, "image-random")] = [bad, *_ten(0.9)[1:]]
        with pytest.raises(ValueError, match="finite"):
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

    strict = {k: {"s0", "s1", "s2", "s3"} for k in range(3)}
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
    assert claim.n_images == (4, 4, 4)
    reference = paired_test([1.0, 1.0, 1.0], [1 / 3, 11 / 15, 1.0])
    assert claim.p_value == pytest.approx(reference.p_value)
    assert claim.interval.lower == pytest.approx(reference.interval.lower)
    assert claim.view == 2


def test_leakage_claim_restricts_each_split_to_its_own_strict_subset() -> None:
    # ADR 0001: View 2 holds one protected test set per pool and redraw, so the
    # images differ between splits and so does the strict subset. A single
    # shared id set would be absent from most splits and would raise.
    image_random = {k: _own_images(k, PERFECT) for k in range(3)}
    near_duplicate = {k: _own_images(k, WEAK[k]) for k in range(3)}
    strict = {
        0: {f"r0-i{i}" for i in range(4)},
        1: {f"r1-i{i}" for i in range(4)},
        # Split 2 keeps only i1 (class a) and i3 (class b); the weak run gets
        # both right there, so that split's difference is zero.
        2: {"r2-i1", "r2-i3"},
    }
    claim = leakage_claim(image_random, near_duplicate, strict)
    assert claim.n_images == (4, 4, 2)
    assert claim.n_splits == 3
    differences = [2 / 3, 1 - 11 / 15, 0.0]
    assert claim.estimate == pytest.approx(sum(differences) / 3)
    reference = paired_test([1.0, 1.0, 1.0], [1 / 3, 11 / 15, 1.0])
    assert claim.p_value == pytest.approx(reference.p_value)


def test_leakage_claim_is_framed_as_a_lower_bound_under_a_stated_assumption() -> None:
    # #185 and #179 (story 25) frame Pool C's effect as a lower bound. The
    # wording keeps that framing and states what it rests on, so the criterion is
    # not claimed as a statistical guarantee.
    claim = _claim()
    assert "lower bound" in claim.framing
    assert "not a lower bound" not in claim.framing
    assert "assumption" in claim.framing
    assert "residual leakage" in claim.framing
    assert "not tested" in claim.framing
    assert "lower limit" in claim.framing
    report = analyze_split_means(_cells({}), claim)
    label = report.secondary[LEAKAGE_TEST].label
    assert "lower bound" in label
    assert "assumption" in label


def test_leakage_claim_refuses_a_subset_it_cannot_score() -> None:
    sets = {k: _quad(PERFECT) for k in range(3)}
    weak = {k: _quad(WEAK[k]) for k in range(3)}
    every = {k: {"i0", "i1", "i2", "i3"} for k in range(3)}
    with pytest.raises(ValueError, match="strict"):
        leakage_claim(sets, weak, {k: set() for k in range(3)})
    with pytest.raises(ValueError, match="strict"):
        leakage_claim(sets, weak, {**every, 1: {"not-an-image"}})
    with pytest.raises(ValueError, match="splits"):
        leakage_claim(sets, {0: weak[0]}, every)
    with pytest.raises(ValueError, match="splits"):
        leakage_claim({0: sets[0]}, {0: weak[0]}, {0: {"i0", "i1"}})
    with pytest.raises(ValueError, match="splits"):
        leakage_claim(sets, weak, {0: {"i0"}, 1: {"i0"}})


def test_prespecified_contrasts_are_recognised_in_either_orientation() -> None:
    assert is_prespecified(PRIMARY_CONTRAST)
    reversed_primary = Contrast(
        "C", 1, ROUTE, PRIMARY_CONTRAST.subtrahend, PRIMARY_CONTRAST.minuend
    )
    assert is_prespecified(reversed_primary)
    for pool in ("A", "B"):
        assert is_prespecified(
            Contrast(pool, 1, ROUTE, "image-random", "near-duplicate")
        )
    assert is_prespecified(Contrast("C", 1, ROUTE, "image-random", "exact-copy"))
    assert not is_prespecified(Contrast("C", 2, ROUTE, "image-random", "exact-copy"))
    assert not is_prespecified(
        Contrast("C", 1, OTHER_ROUTE, "image-random", "near-duplicate")
    )
    report = analyze_split_means(_cells({}), _claim())
    assert not any(is_prespecified(row.contrast) for row in report.exploratory)


def test_the_report_carries_the_equivalence_caveats() -> None:
    note = analyze_split_means(_cells({}), _claim()).equivalence_note
    assert "not evidence of an effect" in note
    assert "unadjusted" in note
    assert "spread" in note
