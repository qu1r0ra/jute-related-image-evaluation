"""The serial, resumable, deadline-aware queue over the benchmark plan.

Fits run one at a time in plan order. Each fit is resumed or skipped through its
run identity, never through the mere presence of files. Optional arms are
admitted against the deadline using clock readings and timing upper bounds only,
and re-checked before every fit, so a slow fit stops the arm instead of the
finalization reservation. Nothing here reads a validation or test outcome.
"""

import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec
from jute_disease.training.queue.admission import (
    Clock,
    Cutoff,
    MandatoryShortfallError,
    OptionalChain,
    admit_optional,
)
from jute_disease.training.queue.plan import Stage, build_plan
from jute_disease.training.receipts import (
    Resume,
    RunState,
    RunStore,
    write_json_atomic,
)

TIMED_KINDS = (FitKind.SOURCE, FitKind.TARGET)


@dataclass(frozen=True)
class Outcome:
    """How one fit ended. It carries timing and status, never a metric."""

    state: RunState
    seconds: float
    reason: str = ""


Executor = Callable[[FitSpec, float | None], Outcome]
Probe = Callable[[FitSpec, float], float]
IdentityFor = Callable[[FitSpec], str]


def timing_key(fit: FitSpec) -> str:
    """Fits that cost the same per epoch share a key: same model and dataset."""
    return f"{fit.model}:{fit.stage}"


class TimingBook:
    """Per-epoch seconds by key; observed runs supersede probes, worst case wins."""

    def __init__(self, max_epochs: int, margin: float) -> None:
        self.max_epochs = max_epochs
        self.margin = margin
        self._probed: dict[str, float] = {}
        self._observed: dict[str, float] = {}

    def record_probe(self, key: str, seconds_per_epoch: float) -> None:
        self._probed[key] = seconds_per_epoch

    def observe(self, key: str, seconds_per_epoch: float) -> None:
        self._observed[key] = max(self._observed.get(key, 0.0), seconds_per_epoch)

    def per_epoch(self, key: str) -> float | None:
        return self._observed.get(key, self._probed.get(key))

    def upper(self, fit: FitSpec) -> float:
        """Seconds a full-ceiling run of ``fit`` could take, with the margin."""
        per_epoch = self.per_epoch(timing_key(fit))
        if per_epoch is None:
            raise KeyError(f"No timing for {timing_key(fit)}")
        return self.max_epochs * per_epoch * self.margin

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {"probed": dict(self._probed), "observed": dict(self._observed)}


@dataclass
class QueueReport:
    outcomes: dict[str, RunState] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    arms: dict[str, str] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)


def _arm_status(states: Sequence[RunState]) -> str:
    if all(state is RunState.COMPLETED for state in states):
        return "complete"
    if any(state is RunState.COMPLETED for state in states):
        return "incomplete"
    return "omitted"


class _Queue:
    def __init__(
        self,
        benchmark: Benchmark,
        store: RunStore,
        clock: Clock,
        cutoff: Cutoff,
        execute: Executor,
        probe: Probe,
        identity_for: IdentityFor,
        report_path: Path,
    ) -> None:
        self.benchmark = benchmark
        self.limits = benchmark.limits
        self.store = store
        self.clock = clock
        self.cutoff = cutoff
        self.execute = execute
        self.probe = probe
        self.identity_for = identity_for
        self.report_path = report_path
        self.reserve = timedelta(hours=self.limits.reserve_hours)
        self.book = TimingBook(benchmark.recipe.max_epochs, self.limits.forecast_margin)
        self.plan: tuple[Stage, ...] = build_plan(benchmark)
        self.report = QueueReport()
        self.classical_spent = self._load_classical_spent()

    # -- persistence -------------------------------------------------------

    def _load_classical_spent(self) -> float:
        if not self.report_path.exists():
            if any(
                any(self.store.run_dir(fit.fit_id).glob("attempt-*"))
                for fit in self.benchmark.fits()
                if fit.kind is FitKind.CLASSICAL
            ):
                raise ValueError(
                    "Classical budget receipt missing for existing attempts"
                )
            return 0.0
        try:
            payload = json.loads(self.report_path.read_text(encoding="utf-8"))
            spent = float(payload["classical_spent_seconds"])
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError(
                "Classical budget receipt is missing or invalid"
            ) from error
        if not math.isfinite(spent) or spent < 0:
            raise ValueError("Classical budget receipt is invalid")
        return spent

    def save(self) -> None:
        report = self.report
        write_json_atomic(
            self.report_path,
            {
                "cutoff": {
                    "at": self.cutoff.at.isoformat(),
                    "basis": self.cutoff.basis,
                    "record": self.cutoff.record,
                },
                "reserve_hours": self.limits.reserve_hours,
                "classical_spent_seconds": self.classical_spent,
                "updated_at": self.clock.now().isoformat(),
                "outcomes": {k: v.value for k, v in report.outcomes.items()},
                "reasons": report.reasons,
                "skipped": report.skipped,
                "arms": report.arms,
                "decisions": report.decisions,
                "escalations": report.escalations,
                "timings": self.book.snapshot(),
            },
        )

    def escalate(self, message: str) -> None:
        if message not in self.report.escalations:
            self.report.escalations.append(message)

    # -- forecasting -------------------------------------------------------

    def ensure_timing(self, fits: Sequence[FitSpec]) -> None:
        """Probe each unknown timing key once, serially, before it is relied on."""
        for fit in fits:
            if fit.kind in TIMED_KINDS and self.book.per_epoch(timing_key(fit)) is None:
                remaining = (
                    self.cutoff.at - self.reserve - self.clock.now()
                ).total_seconds()
                if remaining <= 0:
                    raise MandatoryShortfallError(max(0.0, -remaining))
                self.book.record_probe(timing_key(fit), self.probe(fit, remaining))

    def observe_receipt(self, fit: FitSpec) -> None:
        if fit.kind not in TIMED_KINDS:
            return
        receipt = self.store.receipt(fit.fit_id, missing_ok=True)
        seconds = (receipt or {}).get("results", {}).get("seconds_per_epoch")
        if isinstance(seconds, int | float) and seconds > 0:
            self.book.observe(timing_key(fit), float(seconds))

    def unfinished(self, fits: Sequence[FitSpec]) -> Iterator[FitSpec]:
        """Fits that would still run: not processed yet and not a current skip."""
        will_run: set[str] = set()
        for fit in fits:
            if fit.fit_id in self.report.outcomes:
                continue
            upstream = fit.upstream
            if upstream is not None:
                state = self.report.outcomes.get(upstream)
                if state is not None and state is not RunState.COMPLETED:
                    continue
                if upstream in will_run or state is None:
                    will_run.add(fit.fit_id)
                    yield fit
                    continue
            if (
                self.store.resolve(fit.fit_id, self.identity_for(fit))
                is Resume.RUN_FRESH
            ):
                will_run.add(fit.fit_id)
                yield fit

    def remaining_upper(self, fits: Sequence[FitSpec]) -> float:
        pending = list(self.unfinished(fits))
        self.ensure_timing(pending)
        total = 0.0
        classical = 0
        for fit in pending:
            if fit.kind is FitKind.PREPARE:
                total += self.limits.shared_preparation_seconds
            elif fit.kind is FitKind.CLASSICAL:
                classical += 1
            else:
                total += self.book.upper(fit)
        room = max(0.0, self.limits.classical_aggregate_seconds - self.classical_spent)
        return total + min(classical * self.limits.classical_fit_seconds, room)

    def check_mandatory(self) -> None:
        fits = [fit for stage in self.plan if stage.mandatory for fit in stage.fits]
        try:
            upper = self.remaining_upper(fits)
            admit_optional(
                self.clock.now(),
                self.cutoff,
                self.reserve,
                upper,
                (),
            )
        except MandatoryShortfallError as error:
            self.escalate(str(error))
            self.save()
            raise

    # -- running -----------------------------------------------------------

    def omit(self, fit: FitSpec, reason: str) -> None:
        self.store.omit(fit.fit_id, "", reason)
        self.report.outcomes[fit.fit_id] = RunState.OMITTED
        self.report.reasons[fit.fit_id] = reason
        self.save()

    def process(
        self, fit: FitSpec, allowed: bool, reason: str, mandatory: bool
    ) -> None:
        upstream = fit.upstream
        if (
            upstream is not None
            and self.report.outcomes.get(upstream) is not RunState.COMPLETED
        ):
            self.omit(fit, f"upstream {upstream} did not complete")
            return
        try:
            identity = self.identity_for(fit)
        except (OSError, ValueError, KeyError) as error:
            if mandatory:
                raise
            self.omit(fit, reason or f"source input unavailable: {error}")
            return
        if self.store.resolve(fit.fit_id, identity) is Resume.SKIP:
            self.report.skipped.append(fit.fit_id)
            self.report.outcomes[fit.fit_id] = RunState.COMPLETED
            self.observe_receipt(fit)
            self.save()
            return
        if not allowed:
            self.omit(fit, reason)
            return
        remaining = (self.cutoff.at - self.reserve - self.clock.now()).total_seconds()
        if remaining <= 0:
            if mandatory:
                error = MandatoryShortfallError(max(0.0, -remaining))
                self.escalate(str(error))
                self.save()
                raise error
            self.omit(fit, "finalization reservation reached")
            return
        timeout: float = remaining
        if fit.kind is FitKind.PREPARE:
            timeout = min(timeout, self.limits.shared_preparation_seconds)
        elif fit.kind is FitKind.CLASSICAL:
            room = self.limits.classical_aggregate_seconds - self.classical_spent
            if room <= 0:
                message = (
                    "classical aggregate fitting limit "
                    f"({self.limits.classical_aggregate_seconds:.0f} s) reached"
                )
                self.escalate(f"{message}; {fit.fit_id} was not run")
                self.omit(fit, message)
                return
            timeout = min(timeout, self.limits.classical_fit_seconds, room)
        self.launch(fit, identity, timeout, mandatory)

    def launch(
        self, fit: FitSpec, identity: str, timeout: float | None, mandatory: bool
    ) -> None:
        reserved = (
            timeout if fit.kind is FitKind.CLASSICAL and timeout is not None else 0.0
        )
        if reserved:
            # Charge the full limit first. An interrupted queue cannot recover
            # unaccounted time by restarting; normal completion settles actual use.
            self.classical_spent += reserved
            self.save()
        try:
            outcome = self.execute(fit, timeout)
        except Exception as error:
            self.store.settle(fit.fit_id)
            outcome = Outcome(
                RunState.FAILED, reserved, f"{type(error).__name__}: {error}"
            )
        state, reason = outcome.state, outcome.reason
        if (
            state is RunState.COMPLETED
            and self.store.resolve(fit.fit_id, identity) is not Resume.SKIP
        ):
            state, reason = RunState.FAILED, "ended without a valid receipt"
        if fit.kind is FitKind.CLASSICAL:
            self.classical_spent += max(0.0, outcome.seconds) - reserved
        self.report.outcomes[fit.fit_id] = state
        self.report.reasons[fit.fit_id] = reason
        if state is RunState.COMPLETED:
            self.observe_receipt(fit)
        elif mandatory:
            self.escalate(f"Mandatory fit {fit.fit_id} ended {state.value}: {reason}")
        self.save()

    def run_mandatory(self, stage: Stage) -> None:
        for fit in stage.fits:
            self.check_mandatory()
            self.process(fit, True, "", mandatory=True)

    def run_optional(self, stage: Stage) -> None:
        fits = stage.fits
        allowed, reason, first = True, "", True
        for index, fit in enumerate(fits):
            if allowed:
                rest = fits[index:]
                try:
                    upper = self.remaining_upper(rest)
                except Exception as error:  # a probe that cannot time a stage
                    allowed, reason = False, f"timing probe failed: {error}"
                    self.report.decisions.append(
                        {
                            "stage": stage.name,
                            "at": self.clock.now().isoformat(),
                            "admitted": False,
                            "reason": reason,
                        }
                    )
                    self.save()
                    upper = 0.0
                if upper > 0:
                    now = self.clock.now()
                    if now >= self.cutoff.at - self.reserve:
                        allowed = False
                        reason = "finalization reservation reached"
                        slack = 0.0
                    else:
                        decision = admit_optional(
                            now,
                            self.cutoff,
                            self.reserve,
                            0.0,
                            (OptionalChain(stage.name, upper),),
                        )
                        slack = decision.slack_seconds
                        if not decision.admitted:
                            allowed = False
                            reason = decision.omitted.get(stage.name, "not admitted")
                    if first or not allowed:
                        self.report.decisions.append(
                            {
                                "stage": stage.name,
                                "at": self.clock.now().isoformat(),
                                "admitted": allowed,
                                "upper_seconds": upper,
                                "reason": reason,
                                "slack_seconds": slack,
                            }
                        )
                        self.save()
                    first = False
            self.process(fit, allowed, reason, mandatory=False)

    def run(self) -> QueueReport:
        self.cutoff.require_launchable()
        self.check_mandatory()
        for stage in self.plan:
            if stage.mandatory:
                self.run_mandatory(stage)
            else:
                self.run_optional(stage)
        for stage in self.plan:
            if not stage.mandatory:
                states = [self.report.outcomes[fit.fit_id] for fit in stage.fits]
                self.report.arms[stage.name] = _arm_status(states)
        self.save()
        return self.report


def run_queue(
    benchmark: Benchmark,
    store: RunStore,
    *,
    clock: Clock,
    cutoff: Cutoff,
    execute: Executor,
    probe: Probe,
    identity_for: IdentityFor,
    report_path: Path | str,
) -> QueueReport:
    """Run the whole benchmark serially and resumably.

    ``execute`` runs one fit and must have started and ended its run in ``store``.
    ``probe`` returns seconds-per-epoch and accepts the remaining deadline as a
    hard upper bound in seconds for a fit whose timing is unknown.
    ``identity_for`` is the compatibility identity of a fit; it is called after the
    fit's upstream has been settled, so it can cover that upstream's checkpoint.

    Raises:
        CutoffUnverifiedError: the cutoff is neither verified nor owner operational.
        MandatoryShortfallError: mandatory work cannot finish before the
            finalization reservation; the queue stops and records the escalation.
    """
    queue = _Queue(
        benchmark,
        store,
        clock,
        cutoff,
        execute,
        probe,
        identity_for,
        Path(report_path),
    )
    return queue.run()
