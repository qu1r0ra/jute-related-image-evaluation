"""The resumable queue with fake fit executors.

The executor stands in for the child process of one fit: it writes the same
run-store records a real fit does, and fails or times out when a test says so.
"""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from datetime import datetime
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import (
    Benchmark,
    FitKind,
    FitSpec,
    RunState,
    RunStore,
    load_benchmark,
)
from jute_disease.training.queue import (
    MAX_CONSECUTIVE_FAILURES,
    RETRIES,
    Outcome,
    QueueAbortedError,
    QueueReport,
    run_queue,
)

CORE = [f"cnn/mobilenet_v2_direct/seed-{s}" for s in (42, 1337, 7)] + [
    f"cnn/efficientnet_b5_direct/seed-{s}" for s in (42, 1337, 7)
]
PLANTVILLAGE_TARGETS = [
    f"cnn/mobilenet_v2_plantvillage/seed-{s}" for s in (42, 1337, 7)
]
PLANTDOC_TARGETS = [
    f"cnn/mobilenet_v2_plantvillage_plantdoc/seed-{s}" for s in (42, 1337, 7)
]


class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 11, 1, 9, 0)


@dataclass
class Harness:
    store: RunStore
    # Attempts of a fit that fail, timed out or raising, before it succeeds.
    failing: Counter[str] = field(default_factory=Counter)
    timing_out: Counter[str] = field(default_factory=Counter)
    raising: Counter[str] = field(default_factory=Counter)
    no_receipt: set[str] = field(default_factory=set)
    identity_tag: str = "v1"
    calls: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    timeouts: dict[str, float] = field(default_factory=dict)
    active: int = 0
    overlapped: bool = False

    def identity(self, fit: FitSpec) -> str:
        return f"{self.identity_tag}:{fit.fit_id}"

    def execute(self, fit: FitSpec, timeout: float) -> Outcome:
        self.active += 1
        self.overlapped |= self.active > 1
        self.calls.append(fit.fit_id)
        self.seeds.append(fit.seed)
        self.timeouts[fit.fit_id] = timeout
        try:
            if self.raising[fit.fit_id]:
                self.raising[fit.fit_id] -= 1
                self.store.start(fit.fit_id, self.identity(fit))
                raise RuntimeError("launcher crashed")
            run = self.store.start(fit.fit_id, self.identity(fit))
            if self.timing_out[fit.fit_id]:
                self.timing_out[fit.fit_id] -= 1
                run.fail(RunState.TIMED_OUT, f"exceeded {timeout:.0f} s")
                return Outcome(RunState.TIMED_OUT, timeout, "timed out")
            if self.failing[fit.fit_id]:
                self.failing[fit.fit_id] -= 1
                run.fail(RunState.FAILED, "boom")
                return Outcome(RunState.FAILED, 1.0, "boom")
            if fit.fit_id in self.no_receipt:
                return Outcome(RunState.COMPLETED, 1.0)
            artifact = run.attempt_dir / "model.bin"
            artifact.write_bytes(fit.fit_id.encode())
            run.complete({"epochs": 50}, artifacts={"model": artifact})
            return Outcome(RunState.COMPLETED, 1.0)
        finally:
            self.active -= 1

    def run(
        self, adjust: Callable[[Benchmark], Benchmark] = lambda b: b
    ) -> QueueReport:
        return run_queue(
            adjust(load_benchmark(BENCHMARK_CONFIG)),
            self.store,
            clock=FixedClock(),
            execute=self.execute,
            identity_for=self.identity,
            report_path=self.store.root / "queue.json",
        )

    def saved(self) -> dict:
        return json.loads((self.store.root / "queue.json").read_text("utf-8"))


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(store=RunStore(tmp_path / "runs"))


def _one_seed(benchmark: Benchmark) -> Benchmark:
    return replace(benchmark, recipe=replace(benchmark.recipe, seeds=(42,)))


def test_outcome_carries_no_metric_fields() -> None:
    assert {f.name for f in fields(Outcome)} == {"state", "seconds", "reason"}


def test_policy_is_one_retry_and_an_abort_after_three_failures() -> None:
    assert (RETRIES, MAX_CONSECUTIVE_FAILURES) == (1, 3)


def test_runs_core_then_classical_then_plantvillage_then_plantdoc(
    harness: Harness,
) -> None:
    report = harness.run()
    ids = harness.calls
    assert ids[:6] == CORE
    assert ids[6] == "classical/prepare"
    assert ids[7:17] == [
        f"classical/{e}-{x}"
        for e in ("gnb", "knn", "lr", "rf", "svm")
        for x in ("raw", "crafted")
    ]
    assert ids[17] == "source/plantvillage"
    assert ids[18:21] == PLANTVILLAGE_TARGETS
    assert ids[21] == "source/plantdoc"
    assert ids[22:] == PLANTDOC_TARGETS
    assert report.arms == {"plantvillage": "complete", "plantdoc": "complete"}
    assert report.escalations == []
    assert harness.overlapped is False


def test_every_step_runs_under_its_configured_timeout(harness: Harness) -> None:
    limits = load_benchmark(BENCHMARK_CONFIG).limits
    harness.run()
    assert harness.timeouts["classical/prepare"] == limits.prepare_seconds
    assert harness.timeouts["classical/lr-raw"] == limits.classical_seconds
    assert harness.timeouts["source/plantvillage"] == limits.source_seconds
    assert harness.timeouts[CORE[0]] == limits.target_seconds


def test_second_run_skips_everything_completed(harness: Harness) -> None:
    harness.run()
    harness.calls.clear()
    report = harness.run()
    assert harness.calls == []
    assert len(report.skipped) == 25


def test_incompatible_identity_is_rerun_not_reused(harness: Harness) -> None:
    harness.run()
    harness.calls.clear()
    harness.identity_tag = "v2"
    harness.run()
    assert len(harness.calls) == 25


def test_a_failed_fit_is_retried_once_with_the_same_seed(harness: Harness) -> None:
    harness.failing[CORE[1]] = 1
    report = harness.run()
    runs = [i for i, fit_id in enumerate(harness.calls) if fit_id == CORE[1]]
    assert len(runs) == 2
    assert harness.seeds[runs[0]] == harness.seeds[runs[1]] == 1337
    assert report.outcomes[CORE[1]] is RunState.COMPLETED
    assert report.attempts[CORE[1]] == 2
    assert report.escalations == []


def test_a_timed_out_fit_is_retried_once(harness: Harness) -> None:
    harness.timing_out["classical/lr-raw"] = 1
    report = harness.run()
    assert harness.calls.count("classical/lr-raw") == 2
    assert report.outcomes["classical/lr-raw"] is RunState.COMPLETED


def test_an_executor_crash_settles_the_run_and_is_retried(harness: Harness) -> None:
    harness.raising[CORE[0]] = 1
    report = harness.run()
    assert harness.calls.count(CORE[0]) == 2
    assert report.outcomes[CORE[0]] is RunState.COMPLETED
    assert harness.store.state(CORE[0]) is RunState.COMPLETED


def test_completion_without_a_valid_receipt_is_a_failure(harness: Harness) -> None:
    harness.no_receipt = {"classical/gnb-raw"}
    report = harness.run()
    assert harness.calls.count("classical/gnb-raw") == 2
    assert report.outcomes["classical/gnb-raw"] is RunState.FAILED
    assert report.reasons["classical/gnb-raw"] == "ended without a valid receipt"


def test_a_persistent_failure_is_reported_and_its_dependants_omitted(
    harness: Harness,
) -> None:
    harness.failing["source/plantvillage"] = 2
    report = harness.run()
    assert harness.calls.count("source/plantvillage") == 1 + RETRIES
    assert report.outcomes["source/plantvillage"] is RunState.FAILED
    assert any("source/plantvillage" in e for e in report.escalations)
    for fit_id in [*PLANTVILLAGE_TARGETS, "source/plantdoc", *PLANTDOC_TARGETS]:
        assert report.outcomes[fit_id] is RunState.OMITTED
        assert fit_id not in harness.calls
    assert report.arms == {"plantvillage": "omitted", "plantdoc": "omitted"}


def test_three_consecutive_persistent_failures_abort_the_queue(
    harness: Harness,
) -> None:
    for fit_id in CORE[:3]:
        harness.failing[fit_id] = 2
    with pytest.raises(QueueAbortedError) as raised:
        harness.run()
    assert raised.value.failures == tuple(CORE[:3])
    assert harness.calls == [fit_id for fit_id in CORE[:3] for _ in range(2)]
    saved = harness.saved()
    assert saved["consecutive_failures"] == CORE[:3]
    assert any("aborted" in e for e in saved["escalations"])


def test_a_completed_fit_resets_the_failure_streak(harness: Harness) -> None:
    for fit_id in (CORE[0], CORE[1], CORE[3], CORE[4]):
        harness.failing[fit_id] = 2
    report = harness.run()
    assert report.outcomes[CORE[2]] is RunState.COMPLETED
    assert len(harness.calls) == 25 + 4  # every failing fit used its retry
    assert len(report.escalations) == 4


def test_a_rerun_after_an_abort_starts_a_fresh_streak(harness: Harness) -> None:
    for fit_id in CORE[:3]:
        harness.failing[fit_id] = 2
    with pytest.raises(QueueAbortedError):
        harness.run()
    harness.calls.clear()
    report = harness.run()
    assert harness.calls[:3] == CORE[:3]
    assert all(state is RunState.COMPLETED for state in report.outcomes.values())


def test_unavailable_plantdoc_inputs_omit_only_its_chain(harness: Harness) -> None:
    original_identity = harness.identity

    def identity(fit: FitSpec) -> str:
        if fit.fit_id == "source/plantdoc":
            raise FileNotFoundError("PlantDoc manifest is missing")
        return original_identity(fit)

    harness.identity = identity  # type: ignore[method-assign]
    report = harness.run()
    assert report.arms == {"plantvillage": "complete", "plantdoc": "omitted"}
    assert "source/plantdoc" not in harness.calls
    assert report.outcomes["source/plantdoc"] is RunState.OMITTED
    assert any("inputs unavailable" in e for e in report.escalations)


def test_changed_frozen_inputs_stop_the_queue(harness: Harness) -> None:
    original_identity = harness.identity

    def identity(fit: FitSpec) -> str:
        if fit.fit_id == CORE[2]:
            raise ValueError("content hash changed")
        return original_identity(fit)

    harness.identity = identity  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="content hash"):
        harness.run()
    assert harness.calls == CORE[:2]


def test_an_arm_is_judged_by_its_targets_not_its_source(harness: Harness) -> None:
    target = "cnn/mobilenet_v2_plantvillage/seed-42"
    harness.failing[target] = 2
    report = harness.run(_one_seed)
    assert report.outcomes["source/plantvillage"] is RunState.COMPLETED
    assert report.arms["plantvillage"] == "omitted"


def test_report_records_attempts_outcomes_and_arms(harness: Harness) -> None:
    harness.failing[CORE[0]] = 1
    harness.run()
    saved = harness.saved()
    assert saved["attempts"][CORE[0]] == 2
    assert saved["attempts"]["classical/prepare"] == 1
    assert saved["outcomes"][CORE[0]] == "completed"
    assert saved["arms"] == {"plantvillage": "complete", "plantdoc": "complete"}
    assert saved["updated_at"] == "2026-11-01T09:00:00"


def test_declared_timeouts_cover_every_kind_of_step() -> None:
    limits = load_benchmark(BENCHMARK_CONFIG).limits
    for kind in FitKind:
        assert limits.timeout(kind) > 0
