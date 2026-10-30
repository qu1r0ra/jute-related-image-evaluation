"""Deadline admission on a simulated clock; no outcome or test metric is involved."""

from datetime import datetime, timedelta

import pytest

from jute_disease.training import (
    Cutoff,
    CutoffUnverifiedError,
    MandatoryShortfallError,
    OptionalChain,
    admit_optional,
)

HOUR = 3600.0
NOW = datetime(2026, 10, 30, 22, 0)
RESERVE = timedelta(hours=8)


def _cutoff(hours_from_now: float, basis: str = "owner-operational") -> Cutoff:
    return Cutoff(
        at=NOW + timedelta(hours=hours_from_now), basis=basis, record="owner note"
    )


def _chains(
    plantvillage: float | None, plantdoc: float | None
) -> tuple[OptionalChain, ...]:
    return (
        OptionalChain("plantvillage", plantvillage),
        OptionalChain("plantdoc", plantdoc, requires="plantvillage"),
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
    assert decision.admitted == ("plantvillage", "plantdoc")
    assert decision.omitted == {}
    assert decision.slack_seconds == pytest.approx(40 * HOUR - 8 * HOUR - 21 * HOUR)


def test_exact_fit_is_admitted() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(8 + 10 + 5 + 6),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ("plantvillage", "plantdoc")
    assert decision.slack_seconds == 0


def test_plantdoc_is_dropped_before_plantvillage() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(8 + 10 + 5 + 5),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ("plantvillage",)
    assert "plantdoc" in decision.omitted


def test_both_dropped_when_not_even_plantvillage_fits() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(8 + 10 + 4),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, 6 * HOUR),
    )
    assert decision.admitted == ()
    assert set(decision.omitted) == {"plantvillage", "plantdoc"}


def test_mandatory_shortfall_escalates_instead_of_reducing_scope() -> None:
    with pytest.raises(MandatoryShortfallError) as raised:
        admit_optional(
            NOW,
            _cutoff(8 + 9),
            RESERVE,
            mandatory_upper=10 * HOUR,
            chains=_chains(1 * HOUR, 1 * HOUR),
        )
    assert raised.value.shortfall_seconds == pytest.approx(1 * HOUR)


def test_reservation_below_eight_hours_is_refused() -> None:
    with pytest.raises(ValueError, match="8"):
        admit_optional(
            NOW,
            _cutoff(100),
            timedelta(hours=7),
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
    assert decision.admitted == ("plantvillage",)


def test_unknown_timing_needs_a_probe_before_admission() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(100),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(5 * HOUR, None),
    )
    assert decision.admitted == ("plantvillage",)
    assert decision.needs_probe == ("plantdoc",)


def test_plantdoc_cannot_run_without_plantvillage() -> None:
    decision = admit_optional(
        NOW,
        _cutoff(100),
        RESERVE,
        mandatory_upper=10 * HOUR,
        chains=_chains(None, 6 * HOUR),
    )
    assert decision.admitted == ()
    assert decision.needs_probe == ("plantvillage",)
    assert "plantdoc" in decision.omitted


def test_already_past_the_cutoff_is_a_mandatory_shortfall() -> None:
    with pytest.raises(MandatoryShortfallError):
        admit_optional(
            NOW,
            _cutoff(-1),
            RESERVE,
            mandatory_upper=0,
            chains=_chains(1 * HOUR, 1 * HOUR),
        )
