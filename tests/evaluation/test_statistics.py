import pytest

from jute_disease.evaluation import agreement_gate, kendall_tau, wilson_interval


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
