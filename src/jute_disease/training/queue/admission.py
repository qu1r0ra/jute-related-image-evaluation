"""Deadline admission for optional work.

The rule, applied before launch and again at every completed-fit boundary, is

    now + upper(remaining mandatory) + upper(admitted optional)
        <= cutoff - finalization reservation

Only clock readings, the cutoff, and timing upper bounds enter. No validation
or test outcome is consulted, so the choice of what to run cannot be steered by
results. A mandatory shortfall is raised for the owner; it is never absorbed by
quietly trimming the declared scope.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

MIN_RESERVE = timedelta(hours=2)
CUTOFF_BASES = frozenset({"verified", "owner-operational"})


class CutoffUnverifiedError(ValueError):
    """The deadline is neither verified nor recorded as an owner operational one."""


class MandatoryShortfallError(RuntimeError):
    """Mandatory work cannot finish before the reservation; the owner must decide."""

    def __init__(self, shortfall_seconds: float) -> None:
        super().__init__(
            f"Mandatory work overruns the finalization reservation by "
            f"{shortfall_seconds / 3600:.2f} h; escalate scope to the owner"
        )
        self.shortfall_seconds = shortfall_seconds


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now()


@dataclass(frozen=True)
class Cutoff:
    """The deadline the queue plans against, with its recorded basis."""

    at: datetime
    basis: str
    record: str

    def require_launchable(self) -> None:
        """Raise unless the cutoff is verified or an owner operational cutoff.

        Raises:
            CutoffUnverifiedError: the basis is unknown or no record backs it.
        """
        if self.basis not in CUTOFF_BASES:
            raise CutoffUnverifiedError(
                f"Cutoff basis {self.basis!r} is not one of {sorted(CUTOFF_BASES)}"
            )
        if not self.record.strip():
            raise CutoffUnverifiedError("Cutoff has no recorded source")


@dataclass(frozen=True)
class OptionalChain:
    """An optional source-plus-target chain with an upper bound on its duration.

    ``upper_seconds`` is ``None`` while its timing is unknown; such a chain is not
    admitted until a bounded probe supplies a number.
    """

    name: str
    upper_seconds: float | None
    requires: str | None = None


@dataclass(frozen=True)
class Admission:
    admitted: tuple[str, ...]
    omitted: dict[str, str] = field(default_factory=dict)
    needs_probe: tuple[str, ...] = ()
    slack_seconds: float = 0.0


def admit_optional(
    now: datetime,
    cutoff: Cutoff,
    reserve: timedelta,
    mandatory_upper: float,
    chains: Sequence[OptionalChain],
) -> Admission:
    """Admit the largest ordered prefix of ``chains`` that fits the deadline.

    ``chains`` are in fallback order, most essential first, so the last chain is
    dropped first. A chain whose ``requires`` chain is not admitted is dropped too.

    Raises:
        ValueError: ``reserve`` is under the two-hour minimum.
        MandatoryShortfallError: mandatory work alone does not fit.
    """
    if reserve < MIN_RESERVE:
        raise ValueError("The finalization reservation must be at least 2 hours")
    budget = (cutoff.at - reserve - now).total_seconds()
    shortfall = mandatory_upper - budget
    if shortfall > 0:
        raise MandatoryShortfallError(shortfall)

    needs_probe = tuple(chain.name for chain in chains if chain.upper_seconds is None)
    admitted: list[OptionalChain] = []
    omitted: dict[str, str] = {}
    spent = mandatory_upper
    for chain in chains:
        if chain.upper_seconds is None:
            omitted[chain.name] = "timing unknown; a bounded probe is required first"
            continue
        if chain.requires is not None and chain.requires not in {
            a.name for a in admitted
        }:
            omitted[chain.name] = f"requires {chain.requires}, which was not admitted"
            continue
        if spent + chain.upper_seconds <= budget:
            admitted.append(chain)
            spent += chain.upper_seconds
        else:
            omitted[chain.name] = (
                "does not fit before the finalization reservation "
                f"({(spent + chain.upper_seconds - budget) / 3600:.2f} h over)"
            )
            # Later chains are lower priority: once one is dropped, so are they.
            for later in chains[chains.index(chain) + 1 :]:
                omitted.setdefault(later.name, f"dropped after {chain.name}")
            break
    return Admission(
        admitted=tuple(chain.name for chain in admitted),
        omitted=omitted,
        needs_probe=needs_probe,
        slack_seconds=budget - spent,
    )
