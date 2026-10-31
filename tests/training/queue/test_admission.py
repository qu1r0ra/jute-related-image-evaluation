"""Deadline admission on a simulated clock; no outcome or test metric is involved."""

from datetime import datetime, timedelta

import pytest

from jute_disease.training.queue import (
    Cutoff,
    CutoffUnverifiedError,
    MandatoryShortfallError,
    OptionalChain,
    admit_optional,
)

HOUR = 3600.0
NOW = datetime(2026, 10, 30, 22, 0)
RESERVE = timedelta(hours=2)


def _cutoff(hours_from_now: float, basis: str = "owner-operational") -> Cutoff:
    return Cutoff(
        at=NOW + timedelta(hours=hours_from_now), basis=basis, record="owner note"
    )


def _chains(
    prerequisite: float | None, dependent: float | None
) -> tuple[OptionalChain, ...]:
    return (
        OptionalChain("shared_source", prerequisite),
        OptionalChain("dependent_arm", dependent, requires="shared_source"),
    )


def test_unverified_cutoff_blocks_launch() -> None:
    with pytest.raises(CutoffUnverifiedError):
        Cutoff(at=NOW, basis="assumed", record="").require_launchable()


@pytest.mark.parametrize("basis", ["verified", "owner-operational"])
def test_verified_or_owner_operational_cutoffs_may_launch(basis: str) -> None:
    Cutoff(at=NOW, basis=basis, record="note").require_launchable()


def test_cutoff_needs_a_recorded_basis() -> None:
    with pytest.raises(CutoffUnverifiedError):
        Cutoff(at=NOW, basis="owner-operational", record="").require_launchable()


def test_everything_fits_with_slack() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(40),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ("shared_source", "dependent_arm")
    assert decision.omitted == {}
    assert decision.slack_seconds == pytest.approx(40 * HOUR - 2 * HOUR - 21 * HOUR)


def test_exact_fit_is_admitted() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(2 + 10 + 5 + 6),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ("shared_source", "dependent_arm")
    assert decision.slack_seconds == 0


def test_dependent_arm_is_dropped_after_prerequisite_when_budget_is_short() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(2 + 10 + 5 + 5),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ("shared_source",)
    assert "dependent_arm" in decision.omitted


def test_both_dropped_when_shared_prerequisite_does_not_fit() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(2 + 10 + 4),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ()
    assert set(decision.omitted) == {"shared_source", "dependent_arm"}


def test_mandatory_shortfall_escalates_instead_of_reducing_scope() -> None:
    with pytest.raises(MandatoryShortfallError) as raised:
        admit_optional(
            NOW,
            _cutoff(2 + 9),
            RESERVE,
            mandatory_upper=10 * HOUR,
            chains=_chains(1 * HOUR, 1 * HOUR),
        )
    assert raised.value.shortfall_seconds == pytest.approx(1 * HOUR)


def test_reservation_below_two_hours_is_refused() -> None:
    with pytest.raises(ValueError, match="2 hours"):
        admit_optional(
            NOW,
            _cutoff(100),
            timedelta(hours=1),
            mandatory_upper=1 * HOUR,
            chains=_chains(1 * HOUR, 1 * HOUR),
        )


def test_larger_reservation_is_honoured() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(30),
        timedelta(hours=12),
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ("shared_source",)


def test_unknown_timing_needs_a_probe_before_admission() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(100),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, None),
    )
    assert decision.admitted == ("shared_source",)
    assert decision.needs_probe == ("dependent_arm",)


def test_dependent_arm_cannot_run_without_shared_prerequisite() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(100),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(None, 6 * HOUR),
    )
    assert decision.admitted == ()
    assert decision.needs_probe == ("shared_source",)
    assert "dependent_arm" in decision.omitted


def test_already_past_the_cutoff_is_a_mandatory_shortfall() -> None:
    with pytest.raises(MandatoryShortfallError):
        admit_optional(
            NOW,
            _cutoff(-1),
            RESERVE,
            mandatory_upper=0,
            chains=_chains(1 * HOUR, 1 * HOUR),
        )
