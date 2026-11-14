import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from jute_disease.evaluation import (
    EQUIVALENCE_MARGIN,
    SMALLEST_EFFECT_OF_INTEREST,
    agreement_gate,
    equivalence_test,
    holm_adjust,
    kendall_tau,
    paired_test,
    split_power,
    welch_test,
    wilson_interval,
)


def test_wilson_interval_matches_reference() -> None:
    # NIST Engineering Statistics Handbook, Wilson score interval: 5 of 10.
    interval = wilson_interval(5, 10)
    assert interval.estimate == 0.5
    assert (interval.lower, interval.upper) == pytest.approx(
        (0.236593, 0.763407), abs=1e-6
    )


def test_kendall_tau_matches_scipy_worked_example() -> None:
    assert kendall_tau([12, 2, 1, 12, 2], [1, 4, 7, 1, 0]) == pytest.approx(
        -0.4714045208
    )


def test_agreement_gate_uses_lower_bound_and_reports_specific_agreements() -> None:
    result = agreement_gate(
        [1] * 40 + [0] * 40 + [1] * 10 + [0] * 10,
        [1] * 40 + [0] * 40 + [0] * 10 + [1] * 10,
        threshold=0.6,
        seed=42,
    )
    assert result.kappa == pytest.approx(0.6)
    assert result.positive_agreement == pytest.approx(0.8)
    assert result.negative_agreement == pytest.approx(0.8)
    assert result.interval.lower < 0.6
    assert not result.passed


def test_agreement_gate_is_unweighted_kappa_over_three_categories() -> None:
    # Confusion rows: a=0 -> (3, 1, 0), a=1 -> (0, 3, 0), a=2 -> (0, 0, 3).
    # Observed agreement 0.9; expected agreement 0.4*0.3 + 0.3*0.4 + 0.3*0.3 = 0.33.
    first = [0] * 4 + [1] * 3 + [2] * 3
    second = [0] * 3 + [1] + [1] * 3 + [2] * 3
    result = agreement_gate(first, second, threshold=0.6, seed=1)
    assert result.kappa == pytest.approx(0.57 / 0.67)
    assert result.positive_agreement is None
    assert result.negative_agreement is None


def test_agreement_gate_passes_at_bound_and_fails_above_it() -> None:
    first = [0, 1] * 50
    result = agreement_gate(first, first, threshold=1)
    assert result.interval.lower == 1
    assert result.passed
    assert result.positive_agreement == result.negative_agreement == 1


# Welch on split means, Holm adjustment, equivalence and power (ADR 0004).


def test_welch_test_matches_hand_computed_statistic_and_df() -> None:
    # Welch (1947): t = (m1 - m2) / sqrt(s1^2/n1 + s2^2/n2), with the
    # Welch-Satterthwaite df. For [2, 4, 6] and [1, 2, 3] the means are 4 and 2,
    # the variances 4 and 1, so v = (4/3, 1/3), se = sqrt(5/3) and
    # df = (5/3)^2 / ((4/3)^2 / 2 + (1/3)^2 / 2) = 50/17.
    result = welch_test([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert result.estimate == pytest.approx(2.0)
    assert result.statistic == pytest.approx(2.0 / math.sqrt(5 / 3))
    assert result.df == pytest.approx(50 / 17)
    reference = scipy_stats.ttest_ind([2, 4, 6], [1, 2, 3], equal_var=False)
    assert result.p_value == pytest.approx(reference.pvalue)
    interval = reference.confidence_interval(0.95)
    assert result.interval.lower == pytest.approx(interval.low)
    assert result.interval.upper == pytest.approx(interval.high)
    assert result.interval.estimate == result.estimate
    assert result.interval.level == 0.95


def test_welch_test_on_ten_split_means_has_eighteen_df_for_equal_variances() -> None:
    image_random = [0.90, 0.92] * 5
    near_duplicate = [0.86, 0.88] * 5
    # Each variance is 10 * 0.01^2 / 9, so se = sqrt(2 * var / 10) and
    # t = 0.04 / se = 6 * sqrt(2); equal variances and sizes give df = 18.
    result = welch_test(image_random, near_duplicate)
    assert result.estimate == pytest.approx(0.04)
    assert result.statistic == pytest.approx(6 * math.sqrt(2))
    assert result.df == pytest.approx(18.0)
    assert result.p_value < 1e-6


def test_welch_test_rejects_undefined_inputs() -> None:
    with pytest.raises(ValueError):
        welch_test([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_test([1.0, 1.0], [2.0, 2.0])  # no spread in either group
    with pytest.raises(ValueError):
        welch_test([1.0, 2.0], [1.0, 2.0], level=1.0)


def test_welch_test_has_nominal_size_and_coverage_with_known_truth() -> None:
    rng = np.random.default_rng(20261113)
    truth, rejections, covered, runs = 0.03, 0, 0, 3000
    for _ in range(runs):
        first = rng.normal(0.90, 0.01, size=10)
        second = rng.normal(0.90 - truth, 0.03, size=10)  # unequal spread
        interval = welch_test(first, second).interval
        covered += interval.lower <= truth <= interval.upper
        null_second = rng.normal(0.90, 0.03, size=10)
        rejections += welch_test(first, null_second).p_value < 0.05
    assert covered / runs == pytest.approx(0.95, abs=0.015)
    assert rejections / runs == pytest.approx(0.05, abs=0.015)


def test_paired_test_is_a_one_sample_t_on_the_split_differences() -> None:
    # Differences 1, 2, 3: mean 2, sd 1, se 1/sqrt(3), t = 2 sqrt(3), df 2.
    result = paired_test([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert result.estimate == pytest.approx(2.0)
    assert result.statistic == pytest.approx(2 * math.sqrt(3))
    assert result.df == 2
    assert result.p_value == pytest.approx(
        scipy_stats.ttest_rel([2, 4, 6], [1, 2, 3]).pvalue
    )
    with pytest.raises(ValueError):
        paired_test([1.0, 2.0], [1.0])


def test_holm_adjust_matches_hand_computed_step_down_values() -> None:
    # Holm (1979): the sorted p-values (0.005, 0.01, 0.03, 0.04) times
    # (4, 3, 2, 1) are (0.02, 0.03, 0.06, 0.04), made monotone by a running max.
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02})


def test_holm_adjust_caps_at_one_and_controls_family_wise_error() -> None:
    assert holm_adjust({"only": 0.2}) == {"only": 0.2}
    assert holm_adjust({"a": 0.6, "b": 0.7}) == pytest.approx({"a": 1.0, "b": 1.0})
    with pytest.raises(ValueError):
        holm_adjust({"a": 1.5})
    rng = np.random.default_rng(7)
    false_alarms, runs = 0, 4000
    for _ in range(runs):
        p_values = {str(i): float(p) for i, p in enumerate(rng.uniform(size=5))}
        false_alarms += any(p < 0.05 for p in holm_adjust(p_values).values())
    assert false_alarms / runs <= 0.05 + 0.015


def test_equivalence_test_is_two_one_sided_welch_tests() -> None:
    # Schuirmann (1987): reject both H0: d <= -m and H0: d >= +m. For
    # [2, 4, 6] and [1, 2, 3] with m = 3 the statistics are (2 + 3) / se and
    # (2 - 3) / se with se = sqrt(5/3); the p-value is the larger one-sided p.
    result = equivalence_test([2.0, 4.0, 6.0], [1.0, 2.0, 3.0], margin=3.0)
    se = math.sqrt(5 / 3)
    assert result.estimate == pytest.approx(2.0)
    assert result.lower_statistic == pytest.approx(5.0 / se)
    assert result.upper_statistic == pytest.approx(-1.0 / se)
    # Shifting the second group by -m (or +m) turns each bound into a test of 0.
    lower = scipy_stats.ttest_ind(
        [2, 4, 6], [-2.0, -1.0, 0.0], equal_var=False, alternative="greater"
    )
    upper = scipy_stats.ttest_ind(
        [2, 4, 6], [4.0, 5.0, 6.0], equal_var=False, alternative="less"
    )
    assert result.lower_p == pytest.approx(lower.pvalue)
    assert result.upper_p == pytest.approx(upper.pvalue)
    assert result.p_value == pytest.approx(max(lower.pvalue, upper.pvalue))
    assert result.margin == 3.0
    assert result.alpha == 0.05
    assert not result.equivalent
    # The 1 - 2 alpha interval is the 90% interval at alpha = 0.05.
    assert result.interval.level == pytest.approx(0.90)


def test_equivalence_margin_is_recorded_and_defaults_to_the_effect_of_interest() -> (
    None
):
    assert SMALLEST_EFFECT_OF_INTEREST == 0.02
    assert EQUIVALENCE_MARGIN == SMALLEST_EFFECT_OF_INTEREST
    result = equivalence_test([0.90, 0.91] * 5, [0.90, 0.91] * 5)
    assert result.margin == EQUIVALENCE_MARGIN
    assert result.equivalent


def test_equivalence_test_with_known_truth_supports_null_and_rejects_effect() -> None:
    rng = np.random.default_rng(99)
    runs = 400
    null_equivalent = effect_equivalent = 0
    for _ in range(runs):
        base = rng.normal(0.9, 0.005, size=10)
        null_equivalent += equivalence_test(
            base, rng.normal(0.9, 0.005, size=10)
        ).equivalent
        effect_equivalent += equivalence_test(
            base, rng.normal(0.9 - 0.03, 0.005, size=10)
        ).equivalent
    assert null_equivalent / runs > 0.95  # sd 0.005 is far below the margin 0.02
    assert effect_equivalent == 0  # a 3-point effect lies outside the 2-point margin


def test_split_power_reproduces_g_power_value_for_unit_spread() -> None:
    # Cohen (1988); G*Power, two-sample t, d = 1, n = 10 per group, two-sided
    # alpha 0.05: power 0.5620. Ten values +-sqrt(9/10) have sample sd exactly 1.
    z = math.sqrt(9 / 10)
    values = [z, -z] * 5
    statement = split_power(values, values, effect=1.0)
    assert statement.sd_first == pytest.approx(1.0)
    assert statement.sd_second == pytest.approx(1.0)
    assert statement.df == pytest.approx(18.0)
    assert statement.power == pytest.approx(0.5620, abs=1e-4)
    assert statement.effect == 1.0
    assert statement.alpha == 0.05


def test_split_power_reports_power_across_an_interval_for_the_spread() -> None:
    # The ten validation means give an estimated, not a known, spread. For unit
    # sample sd and 18 df the 95% chi-square interval is (8.231, 31.526), so the
    # standard error may be off by sqrt(18/31.526) = 0.756 to sqrt(18/8.231) =
    # 1.479 times. Power is reported at both ends.
    z = math.sqrt(9 / 10)
    values = [z, -z] * 5
    statement = split_power(values, values, effect=1.0)
    low, high = statement.sd_factor_interval
    assert low == pytest.approx(0.756, abs=1e-3)
    assert high == pytest.approx(1.479, abs=1e-3)
    worst, best = statement.power_interval
    assert worst < statement.power < best
    # Independent check: noncentral t with noncentrality 1 / (se * factor),
    # se = sqrt(2/10), straight from scipy.
    from scipy import stats

    se = math.sqrt(2 / 10)
    critical = stats.t.ppf(0.975, 18)
    for factor, power in ((high, worst), (low, best)):
        ncp = 1.0 / (se * factor)
        expected = stats.nct.sf(critical, 18, ncp) + stats.nct.cdf(-critical, 18, ncp)
        assert power == pytest.approx(expected, abs=1e-9)
    assert "known" in statement.note
    assert "interval" in statement.note


def test_split_power_uses_the_two_point_effect_by_default_and_grows_with_effect() -> (
    None
):
    spread = [0.90, 0.92] * 5
    default = split_power(spread, spread)
    assert default.effect == SMALLEST_EFFECT_OF_INTEREST
    assert split_power(spread, spread, effect=0.04).power > default.power
    with pytest.raises(ValueError):
        split_power([0.9] * 10, [0.9] * 10)  # no spread: the statement is undefined
