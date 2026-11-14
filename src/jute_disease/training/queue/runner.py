"""The serial, resumable queue over the benchmark plan.

Fits run one at a time in plan order. Each fit is resumed or skipped through its
run identity, never through the mere presence of files. Every attempt runs under
the configured timeout for its kind of step. Only infrastructure failures and
timeouts get one retry with the same seed. Three consecutive persistent failures
abort the queue. Nothing here reads a test outcome.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from jute_disease.training.benchmark import Benchmark, FitKind, FitSpec
from jute_disease.training.lock import FitLockedError
from jute_disease.training.queue.plan import Stage, build_plan
from jute_disease.training.receipts import (
    Resume,
    RunState,
    RunStore,
    write_json_atomic,
)

RETRIES = 1
MAX_CONSECUTIVE_FAILURES = 3


class FailureKind(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    DIVERGENCE = "divergence"
    OTHER = "other"


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
    failure_kind: FailureKind | None = None

    @property
    def retryable(self) -> bool:
        """Only outcomes classified as infrastructure consume the retry."""
        return (
            self.state in (RunState.FAILED, RunState.TIMED_OUT)
            and self.failure_kind is FailureKind.INFRASTRUCTURE
        )


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
    planned_fits: list[str] = field(default_factory=list)
    confirmatory_ready: bool = False
    confirmatory_blockers: list[str] = field(default_factory=list)

    @property
    def completion_ready(self) -> bool:
        """True only when every planned fit completed in the current queue."""
        return bool(self.planned_fits) and all(
            self.outcomes.get(fit_id) is RunState.COMPLETED
            for fit_id in self.planned_fits
        )


class ConfirmatoryAnalysisBlockedError(RuntimeError):
    """The training receipts do not support confirmatory analysis."""

    def __init__(self, blockers: Sequence[str]) -> None:
        self.blockers = tuple(blockers)
        super().__init__(
            "Confirmatory analysis is blocked: " + "; ".join(self.blockers)
        )


def _runtime_is_compatible(runtime: object, precision: str) -> bool:
    if not isinstance(runtime, dict):
        return False
    backend = runtime.get("backend")
    return (
        {
            "python",
            "torch",
            "torch_cuda",
            "cuda_driver",
            "libraries",
        }.issubset(runtime)
        and runtime.get("cuda_available") is True
        and runtime.get("precision") == precision
        and isinstance(runtime.get("libraries"), dict)
        and bool(runtime["libraries"])
        and isinstance(backend, dict)
        and backend
        == {
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
        }
        and runtime.get("compile_enabled") is False
        and isinstance(runtime.get("deterministic_algorithms"), bool)
        and isinstance(runtime.get("cudnn_deterministic"), bool)
        and isinstance(runtime.get("float32_matmul_precision"), str)
    )


def confirmatory_blockers(
    benchmark: Benchmark, store: RunStore, identity_for: IdentityFor
) -> list[str]:
    """Describe every planned fit that lacks an official, complete receipt."""
    blockers: list[str] = []
    required_metadata = {
        "provenance",
        "official",
        "configuration",
        "environment_lock",
        "preprocessing",
        "assignment_table",
        "source_checkpoint_ancestry",
        "seed",
        "runtime_contract",
        "run_options",
    }
    terminal_attempts = {
        RunState.COMPLETED.value,
        RunState.FAILED.value,
        RunState.TIMED_OUT.value,
        RunState.INTERRUPTED.value,
        RunState.OMITTED.value,
    }
    for fit in benchmark.fits():
        try:
            expected_identity = identity_for(fit)
            receipt = store.receipt(fit.fit_id, missing_ok=True)
        except Exception as error:
            blockers.append(f"{fit.fit_id}: receipt check failed ({error})")
            continue
        if receipt is None:
            blockers.append(f"{fit.fit_id}: no completed receipt")
            continue
        if store.state(fit.fit_id) is not RunState.COMPLETED:
            blockers.append(f"{fit.fit_id}: planned fit is not completed")
            continue
        if receipt.get("identity") != expected_identity:
            blockers.append(f"{fit.fit_id}: receipt identity does not match")
            continue
        metadata = receipt.get("metadata")
        if not isinstance(metadata, dict) or not required_metadata.issubset(metadata):
            blockers.append(f"{fit.fit_id}: receipt is missing contract metadata")
            continue
        provenance = metadata.get("provenance")
        if (
            metadata.get("official") is not True
            or not isinstance(provenance, dict)
            or provenance.get("working_tree_clean") is not True
            or not provenance.get("commit")
        ):
            blockers.append(f"{fit.fit_id}: receipt is not from a clean official tree")
        if metadata.get("seed") != fit.seed:
            blockers.append(f"{fit.fit_id}: receipt seed does not match the fit")
        run_options = metadata.get("run_options")
        if (
            not isinstance(run_options, dict)
            or run_options.get("require_cuda") is not True
            or run_options.get("accelerator") == "cpu"
        ):
            blockers.append(
                f"{fit.fit_id}: receipt is not from the official CUDA profile"
            )
        expected_runtime_contract = {
            "precision": benchmark.recipe.precision,
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
            "torch_compile": False,
            "num_workers": benchmark.recipe.num_workers,
        }
        if metadata.get("runtime_contract") != expected_runtime_contract:
            blockers.append(f"{fit.fit_id}: runtime contract does not match the recipe")
        assignment = metadata.get("assignment_table")
        rows = assignment.get("assignments") if isinstance(assignment, dict) else None
        if (
            not isinstance(assignment, dict)
            or assignment.get("validated") is not True
            or not assignment.get("sha256")
            or not isinstance(rows, list)
            or not rows
            or any(
                not isinstance(row, dict)
                or any(
                    type(row.get(key)) is not int
                    for key in ("split", "seed", "checkpoint")
                )
                or row["checkpoint"] != (row["seed"] + row["split"]) % 3
                for row in rows
            )
        ):
            blockers.append(f"{fit.fit_id}: receipt has no validated assignment table")
        for field_name in ("configuration", "environment_lock", "preprocessing"):
            section = metadata.get(field_name)
            if not isinstance(section, dict) or not section.get("sha256"):
                blockers.append(f"{fit.fit_id}: receipt has no {field_name} digest")
        runtime = metadata.get("runtime")
        if not _runtime_is_compatible(runtime, benchmark.recipe.precision):
            blockers.append(f"{fit.fit_id}: receipt is missing runtime observations")
        attempts = receipt.get("attempts")
        if (
            not isinstance(attempts, list)
            or not attempts
            or not all(isinstance(attempt, dict) for attempt in attempts)
            or attempts[-1].get("state") != RunState.COMPLETED.value
            or any(
                attempt.get("state") not in terminal_attempts
                or not required_metadata.issubset(
                    attempt.get("metadata", {})
                    if isinstance(attempt.get("metadata"), dict)
                    else {}
                )
                for attempt in attempts
            )
        ):
            blockers.append(f"{fit.fit_id}: attempt history is incomplete")
            continue
        durable_attempts = store.attempts(fit.fit_id, identity=expected_identity)
        attempt_numbers = [attempt.get("attempt") for attempt in attempts]
        durable_attempt_numbers = [
            attempt.get("attempt") for attempt in durable_attempts
        ]
        if attempt_numbers != durable_attempt_numbers:
            blockers.append(f"{fit.fit_id}: attempt history has missing attempts")
        if any(attempt.get("identity") != expected_identity for attempt in attempts):
            blockers.append(f"{fit.fit_id}: an attempt has a different run identity")
        if any(
            any(
                attempt.get("metadata", {}).get(key) != metadata.get(key)
                for key in required_metadata
            )
            for attempt in attempts
        ):
            blockers.append(
                f"{fit.fit_id}: attempt metadata does not match its receipt"
            )
        if attempts[-1].get("metadata", {}).get("runtime") != metadata.get("runtime"):
            blockers.append(
                f"{fit.fit_id}: completed attempt runtime does not match receipt"
            )
        if any(
            not _runtime_is_compatible(
                attempt.get("metadata", {}).get("runtime")
                if isinstance(attempt.get("metadata"), dict)
                else None,
                benchmark.recipe.precision,
            )
            for attempt in attempts
        ):
            blockers.append(f"{fit.fit_id}: an attempt is missing runtime observations")
        if any(
            attempt.get("metadata", {}).get("official") is not True
            or attempt.get("metadata", {})
            .get("provenance", {})
            .get("working_tree_clean")
            is not True
            for attempt in attempts
        ):
            blockers.append(f"{fit.fit_id}: an attempt came from a dirty tree")
    return blockers


def require_confirmatory_completion(
    benchmark: Benchmark, store: RunStore, identity_for: IdentityFor
) -> None:
    """Stop confirmatory analysis unless every planned fit has an official receipt.

    This is the shared analysis gate: it checks every planned fit, not just those
    that happened to finish during the latest queue invocation.
    """
    blockers = confirmatory_blockers(benchmark, store, identity_for)
    if blockers:
        raise ConfirmatoryAnalysisBlockedError(blockers)


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
        plan: tuple[Stage, ...] | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.store = store
        self.clock = clock
        self.execute = execute
        self.identity_for = identity_for
        self.report_path = report_path
        self.plan: tuple[Stage, ...] = build_plan(benchmark) if plan is None else plan
        self.report = QueueReport()
        self.unavailable_stages: set[str] = set()
        self.report.planned_fits = [
            fit.fit_id for stage in self.plan for fit in stage.fits
        ]
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
                "planned_fits": self.report.planned_fits,
                "completion_ready": self.report.completion_ready,
                "confirmatory_blockers": self.report.confirmatory_blockers,
                "confirmatory_ready": self.report.confirmatory_ready,
            },
        )

    def escalate(self, message: str) -> None:
        if message not in self.report.escalations:
            self.report.escalations.append(message)

    def record(self, fit: FitSpec, state: RunState, reason: str) -> None:
        self.report.outcomes[fit.fit_id] = state
        self.report.reasons[fit.fit_id] = reason

    def keep_completed(self, fit: FitSpec) -> bool:
        """Count a fit with a valid receipt and a completed state as complete.

        Its inputs may have been culled since it ran, which says nothing against
        a finished fit. Nothing is written, so its completed state stands.
        """
        if not self.store.is_complete(fit.fit_id):
            return False
        self.report.skipped.append(fit.fit_id)
        self.record(fit, RunState.COMPLETED, "")
        self.save()
        return True

    def omit(self, fit: FitSpec, reason: str) -> None:
        """Leave a fit out of this run because a dependency did not complete.

        A completed fit keeps its receipt and state on disk; only this run's
        report calls it omitted, so nothing downstream builds on a stale result.
        """
        if self.store.is_complete(fit.fit_id):
            reason = f"{reason}; its completed receipt is kept"
        else:
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
        timeout = self.benchmark.timeout(fit)
        try:
            outcome = self.execute(fit, timeout)
        except FitLockedError:
            # Another writer owns this fit: not a failure to retry or record.
            raise
        except Exception as error:
            self.store.settle(fit.fit_id)
            failure_kind = (
                FailureKind.INFRASTRUCTURE
                if isinstance(error, OSError | MemoryError | TimeoutError)
                else FailureKind.OTHER
            )
            outcome = Outcome(
                RunState.FAILED,
                0.0,
                f"{type(error).__name__}: {error}",
                failure_kind,
            )
        if (
            outcome.state is RunState.COMPLETED
            and self.store.resolve(fit.fit_id, identity) is not Resume.SKIP
        ):
            return Outcome(
                RunState.FAILED,
                outcome.seconds,
                "ended without a valid receipt",
                FailureKind.INFRASTRUCTURE,
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
        tuning = self.benchmark.tuning_dependency(fit)
        if (
            tuning is not None
            and self.report.outcomes.get(tuning) is not RunState.COMPLETED
        ):
            self.omit(fit, f"tuning {tuning} did not complete")
            return
        try:
            identity = self.identity_for(fit)
        except FileNotFoundError as error:
            # Absent inputs do not appear on a retry, so none is spent, but the
            # fit still counts towards the failure streak. Any other identity
            # error, such as changed frozen content, propagates and stops the queue.
            if self.keep_completed(fit):
                return
            reason = f"inputs unavailable: {error}"
            self.store.omit(fit.fit_id, "", reason)
            if fit.kind is FitKind.SOURCE and fit.stage in self.unavailable_stages:
                # Sibling seeds of a stage whose inputs are already known to be
                # absent are one problem, not a further streak towards the abort.
                self.record(fit, RunState.OMITTED, reason)
                self.save()
                return
            if fit.kind is FitKind.SOURCE:
                self.unavailable_stages.add(fit.stage)
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
            if outcome.state is RunState.COMPLETED or not outcome.retryable:
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
                targets = [
                    fit
                    for fit in stage.fits
                    if fit.kind is FitKind.TARGET and fit.phase != "tune"
                ]
                states = [self.report.outcomes[fit.fit_id] for fit in targets]
                self.report.arms[stage.name] = _arm_status(states)
        self.report.confirmatory_blockers = confirmatory_blockers(
            self.benchmark, self.store, self.identity_for
        )
        self.report.confirmatory_ready = not self.report.confirmatory_blockers
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
    plan: tuple[Stage, ...] | None = None,
) -> QueueReport:
    """Run the whole benchmark, or the given ``plan``, serially and resumably.

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
    queue = _Queue(
        benchmark, store, clock, execute, identity_for, Path(report_path), plan
    )
    return queue.run()
