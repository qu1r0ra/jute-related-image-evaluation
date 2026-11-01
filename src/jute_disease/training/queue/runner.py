"""The serial, resumable queue over the benchmark plan.

Fits run one at a time in plan order. Each fit is resumed or skipped through its
run identity, never through the mere presence of files. Every attempt runs under
the configured timeout for its kind of step. A failed or timed-out fit is retried
once with the same seed; if the retry also fails, the fit is reported as a
persistent failure and its dependants are omitted. Three persistent failures in a
row abort the queue, because they point to a broken environment rather than to
one bad fit. Nothing here reads a validation or test outcome.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec
from jute_disease.training.queue.plan import Stage, build_plan
from jute_disease.training.receipts import (
    Resume,
    RunState,
    RunStore,
    write_json_atomic,
)

RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3


class QueueAbortedError(RuntimeError):
    """Consecutive fits failed persistently; the environment needs the owner."""

    def __init__(self, failures: Sequence[str]) -> None:
        super().__init__(
            f"Queue aborted after {len(failures)} consecutive persistent "
            f"failures: {', '.join(failures)}"
        )
        self.failures = tuple(failures)


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now()


@dataclass(frozen=True)
class Outcome:
    """How one attempt ended. It carries timing and status, never a metric."""

    state: RunState
    seconds: float
    reason: str = ""


Executor = Callable[[FitSpec, float], Outcome]
IdentityFor = Callable[[FitSpec], str]


@dataclass
class QueueReport:
    outcomes: dict[str, RunState] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    arms: dict[str, str] = field(default_factory=dict)
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
        execute: Executor,
        identity_for: IdentityFor,
        report_path: Path,
    ) -> None:
        self.benchmark = benchmark
        self.store = store
        self.clock = clock
        self.execute = execute
        self.identity_for = identity_for
        self.report_path = report_path
        self.plan: tuple[Stage, ...] = build_plan(benchmark)
        self.report = QueueReport()
        # Counted within one run only: restarting the queue after the owner has
        # fixed the environment starts a fresh count.
        self.consecutive: list[str] = []

    def save(self) -> None:
        report = self.report
        write_json_atomic(
            self.report_path,
            {
                "updated_at": self.clock.now().isoformat(),
                "outcomes": {k: v.value for k, v in report.outcomes.items()},
                "reasons": report.reasons,
                "attempts": report.attempts,
                "skipped": report.skipped,
                "arms": report.arms,
                "escalations": report.escalations,
                "consecutive_failures": self.consecutive,
            },
        )

    def escalate(self, message: str) -> None:
        if message not in self.report.escalations:
            self.report.escalations.append(message)

    def record(self, fit: FitSpec, state: RunState, reason: str) -> None:
        self.report.outcomes[fit.fit_id] = state
        self.report.reasons[fit.fit_id] = reason

    def omit(self, fit: FitSpec, reason: str) -> None:
        self.store.omit(fit.fit_id, "", reason)
        self.record(fit, RunState.OMITTED, reason)
        self.save()

    def fail_persistently(self, fit: FitSpec, state: RunState, reason: str) -> None:
        """Record a fit that cannot be made to run, and abort on a failure streak.

        Raises:
            QueueAbortedError: this is the third persistent failure in a row.
        """
        self.record(fit, state, reason)
        self.escalate(f"Fit {fit.fit_id} ended {state.value}: {reason}")
        self.consecutive.append(fit.fit_id)
        self.save()
        if len(self.consecutive) >= MAX_CONSECUTIVE_FAILURES:
            error = QueueAbortedError(self.consecutive)
            self.escalate(str(error))
            self.save()
            raise error

    def attempt(self, fit: FitSpec, identity: str) -> Outcome:
        timeout = self.benchmark.limits.timeout(fit.kind)
        try:
            outcome = self.execute(fit, timeout)
        except Exception as error:
            self.store.settle(fit.fit_id)
            outcome = Outcome(RunState.FAILED, 0.0, f"{type(error).__name__}: {error}")
        if (
            outcome.state is RunState.COMPLETED
            and self.store.resolve(fit.fit_id, identity) is not Resume.SKIP
        ):
            return Outcome(
                RunState.FAILED, outcome.seconds, "ended without a valid receipt"
            )
        return outcome

    def process(self, fit: FitSpec) -> None:
        upstream = fit.upstream
        if (
            upstream is not None
            and self.report.outcomes.get(upstream) is not RunState.COMPLETED
        ):
            self.omit(fit, f"upstream {upstream} did not complete")
            return
        try:
            identity = self.identity_for(fit)
        except FileNotFoundError as error:
            # Absent inputs do not appear on a retry, so none is spent, but the
            # fit still counts towards the failure streak. Any other identity
            # error, such as changed frozen content, propagates and stops the queue.
            reason = f"inputs unavailable: {error}"
            self.store.omit(fit.fit_id, "", reason)
            self.fail_persistently(fit, RunState.OMITTED, reason)
            return
        if self.store.resolve(fit.fit_id, identity) is Resume.SKIP:
            self.report.skipped.append(fit.fit_id)
            self.record(fit, RunState.COMPLETED, "")
            self.save()
            return
        outcome = Outcome(RunState.FAILED, 0.0, "not attempted")
        for number in range(1, RETRIES + 2):
            self.report.attempts[fit.fit_id] = number
            outcome = self.attempt(fit, identity)
            self.save()
            if outcome.state is RunState.COMPLETED:
                break
        if outcome.state is RunState.COMPLETED:
            self.consecutive.clear()
            self.record(fit, RunState.COMPLETED, "")
            self.save()
            return
        self.fail_persistently(fit, outcome.state, outcome.reason)

    def run(self) -> QueueReport:
        for stage in self.plan:
            for fit in stage.fits:
                self.process(fit)
        for stage in self.plan:
            if any(fit.kind is FitKind.SOURCE for fit in stage.fits):
                # An arm is judged by its targets; a completed source alone is
                # not a partial result.
                targets = [fit for fit in stage.fits if fit.kind is FitKind.TARGET]
                states = [self.report.outcomes[fit.fit_id] for fit in targets]
                self.report.arms[stage.name] = _arm_status(states)
        self.save()
        return self.report


def run_queue(
    benchmark: Benchmark,
    store: RunStore,
    *,
    clock: Clock,
    execute: Executor,
    identity_for: IdentityFor,
    report_path: Path | str,
) -> QueueReport:
    """Run the whole benchmark serially and resumably.

    ``execute`` runs one attempt under the given timeout in seconds and must have
    started and ended its run in ``store``. ``identity_for`` is the compatibility
    identity of a fit; it is called after the fit's upstream has been settled, so
    it can cover that upstream's checkpoint.

    Raises:
        QueueAbortedError: three fits in a row failed after their retry; the
            queue stops with the escalation recorded in the report.
        ValueError: ``identity_for`` found inputs that no longer match their
            frozen record; nothing more is run.
    """
    queue = _Queue(benchmark, store, clock, execute, identity_for, Path(report_path))
    return queue.run()
