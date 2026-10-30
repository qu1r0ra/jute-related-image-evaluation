"""The resumable queue on a simulated clock with fake fit executors.

The executor stands in for the child process of one fit: it advances the fake
clock by a chosen duration and writes the same run-store records a real fit does.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import (
    Cutoff,
    CutoffUnverifiedError,
    FitKind,
    FitSpec,
    MandatoryShortfallError,
    Outcome,
    RunState,
    RunStore,
    load_benchmark,
    run_queue,
    timing_key,
)

HOUR = 3600.0
START = datetime(2026, 10, 30, 22, 0)
EPOCHS_TIMES_MARGIN = 50 * 1.25


class FakeClock:
    def __init__(self) -> None:
        self.current = START

    def now(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


def per_epoch(upper_hours: float) -> float:
    """Seconds per epoch whose forecast fit upper bound is ``upper_hours``."""
    return upper_hours * HOUR / EPOCHS_TIMES_MARGIN


def default_hours(fit: FitSpec) -> float:
    """Each fit takes exactly its forecast upper bound unless a test says otherwise."""
    if fit.kind in (FitKind.PREPARE, FitKind.CLASSICAL):
        return 0.5
    return 1.0


@dataclass
class Harness:
    store: RunStore
    clock: FakeClock = field(default_factory=FakeClock)
    actual_hours: Callable[[FitSpec], float] = default_hours
    probe_hours: dict[str, float] = field(default_factory=dict)
    fail: set[str] = field(default_factory=set)
    identity_tag: str = "v1"
    calls: list[str] = field(default_factory=list)
    timeouts: dict[str, float | None] = field(default_factory=dict)
    probed: list[str] = field(default_factory=list)
    active: int = 0
    overlapped: bool = False

    def identity(self, fit: FitSpec) -> str:
        return f"{self.identity_tag}:{fit.fit_id}"

    def execute(self, fit: FitSpec, timeout: float | None) -> Outcome:
        self.active += 1
        self.overlapped |= self.active > 1
        self.calls.append(fit.fit_id)
        self.timeouts[fit.fit_id] = timeout
        run = self.store.start(fit.fit_id, self.identity(fit))
        seconds = self.actual_hours(fit) * HOUR
        try:
            if timeout is not None and seconds > timeout:
                self.clock.advance(timeout)
                run.fail(RunState.TIMED_OUT, f"exceeded {timeout:.0f} s")
                return Outcome(RunState.TIMED_OUT, timeout, "timed out")
            self.clock.advance(seconds)
            if fit.fit_id in self.fail:
                run.fail(RunState.FAILED, "boom")
                return Outcome(RunState.FAILED, seconds, "boom")
            artifact = run.attempt_dir / "model.bin"
            artifact.write_bytes(fit.fit_id.encode())
            run.complete(
                # Per-epoch time that makes the observed upper bound equal the
                # fit's real duration, matching what the probe reports.
                {"seconds_per_epoch": seconds / EPOCHS_TIMES_MARGIN, "epochs": 50},
                artifacts={"model": artifact},
            )
            return Outcome(RunState.COMPLETED, seconds)
        finally:
            self.active -= 1

    def probe(self, fit: FitSpec) -> float:
        self.probed.append(timing_key(fit))
        return per_epoch(self.probe_hours.get(timing_key(fit), 1.0))

    def run(
        self,
        hours_to_cutoff: float,
        basis: str = "owner-operational",
        aggregate_seconds: float | None = None,
    ):
        benchmark = load_benchmark(BENCHMARK_CONFIG)
        if aggregate_seconds is not None:
            benchmark = replace(
                benchmark,
                limits=replace(
                    benchmark.limits, classical_aggregate_seconds=aggregate_seconds
                ),
            )
        cutoff = Cutoff(
            at=START + timedelta(hours=hours_to_cutoff), basis=basis, record="note"
        )
        return run_queue(
            benchmark,
            self.store,
            clock=self.clock,
            cutoff=cutoff,
            execute=self.execute,
            probe=self.probe,
            identity_for=self.identity,
            report_path=self.store.root / "queue.json",
        )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(store=RunStore(tmp_path / "runs"))


def _known_timings(harness: Harness, hours: float = 1.0) -> None:
    benchmark = load_benchmark(BENCHMARK_CONFIG)
    for fit in benchmark.fits():
        if fit.kind in (FitKind.SOURCE, FitKind.TARGET):
            harness.probe_hours[timing_key(fit)] = hours


def test_outcome_carries_no_metric_fields() -> None:
    assert {f.name for f in fields(Outcome)} == {"state", "seconds", "reason"}


def test_unverified_cutoff_runs_nothing(harness: Harness) -> None:
    with pytest.raises(CutoffUnverifiedError):
        harness.run(500, basis="assumed")
    assert harness.calls == []


def test_runs_core_then_classical_then_plantvillage_then_plantdoc(
    harness: Harness,
) -> None:
    _known_timings(harness)
    report = harness.run(500)
    ids = harness.calls
    assert ids[:6] == (
        [f"cnn/mobilenet_v2_direct/seed-{s}" for s in (42, 1337, 7)]
        + [f"cnn/efficientnet_b5_direct/seed-{s}" for s in (42, 1337, 7)]
    )
    assert ids[6] == "classical/prepare"
    assert ids[7:17] == [
        f"classical/{e}-{x}"
        for e in ("gnb", "knn", "lr", "rf", "svm")
        for x in ("raw", "crafted")
    ]
    assert ids[17] == "source/plantvillage"
    assert ids[18:21] == [
        f"cnn/mobilenet_v2_plantvillage/seed-{s}" for s in (42, 1337, 7)
    ]
    assert ids[21] == "source/plantdoc"
    assert len(ids) == 25
    assert report.arms == {"plantvillage": "complete", "plantdoc": "complete"}
    assert harness.overlapped is False


def test_second_run_skips_everything_completed(harness: Harness) -> None:
    _known_timings(harness)
    harness.run(500)
    harness.calls.clear()
    report = harness.run(500)
    assert harness.calls == []
    assert len(report.skipped) == 25


def test_incompatible_identity_is_rerun_not_reused(harness: Harness) -> None:
    _known_timings(harness)
    harness.run(500)
    harness.calls.clear()
    harness.identity_tag = "v2"
    harness.run(500)
    assert len(harness.calls) == 25


def test_interrupted_queue_resumes_without_repeating_completed_fits(
    harness: Harness,
) -> None:
    _known_timings(harness)
    harness.fail = {"classical/lr-raw"}
    first = harness.run(500)
    assert first.outcomes["classical/lr-raw"] == RunState.FAILED
    harness.calls.clear()
    harness.fail = set()
    harness.run(500)
    assert harness.calls == ["classical/lr-raw"]


def test_mandatory_shortfall_escalates_before_any_work(harness: Harness) -> None:
    _known_timings(harness, hours=2.0)
    with pytest.raises(MandatoryShortfallError):
        harness.run(8 + 5)
    assert harness.calls == []
    saved = json.loads((harness.store.root / "queue.json").read_text("utf-8"))
    assert any("Mandatory" in note for note in saved["escalations"])


def test_plantdoc_is_dropped_first_when_only_plantvillage_fits(
    harness: Harness,
) -> None:
    _known_timings(harness, hours=1.0)
    # mandatory 6 + prepare 0.5 + 10 * 0.5 classical cap, then pv 4, pd 4.
    mandatory = 6 * 1.0 + 0.5 + 10 * 0.5
    report = harness.run(8 + mandatory + 4 + 1)
    assert report.arms["plantvillage"] == "complete"
    assert report.arms["plantdoc"] == "omitted"
    assert "source/plantdoc" not in harness.calls
    assert harness.store.state("source/plantdoc") is RunState.OMITTED


def test_both_optional_arms_are_omitted_when_neither_fits(harness: Harness) -> None:
    _known_timings(harness, hours=1.0)
    mandatory = 6 * 1.0 + 0.5 + 10 * 0.5
    report = harness.run(8 + mandatory + 1)
    assert report.arms == {"plantvillage": "omitted", "plantdoc": "omitted"}
    assert "source/plantvillage" not in harness.calls


def test_overrun_stops_the_arm_and_it_does_not_count_as_complete(
    harness: Harness,
) -> None:
    _known_timings(harness, hours=1.0)
    harness.actual_hours = lambda fit: (
        9.0 if fit.fit_id == "cnn/mobilenet_v2_plantvillage/seed-42" else 1.0
    )
    mandatory = 6 * 1.0 + 0.5 + 10 * 0.5
    report = harness.run(8 + mandatory + 8 + 4 + 0.5)
    assert report.arms["plantvillage"] == "incomplete"
    assert harness.store.state("cnn/mobilenet_v2_plantvillage/seed-42") is (
        RunState.COMPLETED
    )
    assert harness.store.state("cnn/mobilenet_v2_plantvillage/seed-1337") is (
        RunState.OMITTED
    )
    assert report.arms["plantdoc"] == "omitted"


def test_unknown_timing_is_probed_once_before_admission(harness: Harness) -> None:
    harness.probe_hours = {}
    harness.run(500)
    assert len(harness.probed) == len(set(harness.probed))
    assert any(key.endswith(":plantdoc") for key in harness.probed)
    assert harness.calls.index("source/plantvillage") > 0


def test_a_probe_showing_plantdoc_is_too_slow_omits_it(harness: Harness) -> None:
    _known_timings(harness, hours=1.0)
    for fit in load_benchmark(BENCHMARK_CONFIG).fits():
        if fit.stage == "plantdoc" and fit.kind is FitKind.SOURCE:
            harness.probe_hours[timing_key(fit)] = 200.0
    report = harness.run(100)
    assert report.arms == {"plantvillage": "complete", "plantdoc": "omitted"}


def test_a_failed_probe_omits_the_optional_stage_instead_of_aborting(
    harness: Harness,
) -> None:
    _known_timings(harness, hours=1.0)
    harness.probe_hours = {}
    real_probe = harness.probe

    def failing(fit: FitSpec) -> float:
        if fit.stage == "plantdoc":
            raise RuntimeError("probe child crashed")
        return real_probe(fit)

    harness.probe = failing  # type: ignore[method-assign]
    report = harness.run(500)
    assert report.arms["plantdoc"] == "omitted"
    assert any("probe child crashed" in d["reason"] for d in report.decisions)
    assert report.outcomes["cnn/mobilenet_v2_direct/seed-42"] == RunState.COMPLETED


def test_classical_fits_run_under_the_per_fit_limit(harness: Harness) -> None:
    _known_timings(harness)
    harness.run(500)
    assert harness.timeouts["classical/prepare"] == 1800
    assert harness.timeouts["classical/rf-raw"] == 1800
    assert harness.timeouts["cnn/mobilenet_v2_direct/seed-42"] is None


def test_timed_out_classical_fit_is_preserved_not_counted(harness: Harness) -> None:
    _known_timings(harness)
    harness.actual_hours = lambda fit: (
        3.0 if fit.fit_id == "classical/svm-raw" else default_hours(fit)
    )
    report = harness.run(500)
    assert report.outcomes["classical/svm-raw"] == RunState.TIMED_OUT
    assert harness.store.state("classical/svm-raw") is RunState.TIMED_OUT
    assert harness.store.receipt("classical/svm-raw", missing_ok=True) is None
    assert any("svm-raw" in note for note in report.escalations)
    assert report.outcomes["classical/svm-crafted"] == RunState.COMPLETED


def test_aggregate_classical_budget_stops_remaining_fits_and_escalates(
    harness: Harness,
) -> None:
    _known_timings(harness)
    report = harness.run(500, aggregate_seconds=4 * 1800)
    ran = [
        i for i in harness.calls if i.startswith("classical/") and "prepare" not in i
    ]
    assert len(ran) == 4
    omitted = [
        fit_id
        for fit_id, state in report.outcomes.items()
        if state == RunState.OMITTED and fit_id.startswith("classical/")
    ]
    assert len(omitted) == 6
    assert any("aggregate" in note for note in report.escalations)


def test_forecast_constants_match_the_declared_configuration() -> None:
    limits = load_benchmark(BENCHMARK_CONFIG).limits
    recipe = load_benchmark(BENCHMARK_CONFIG).recipe
    assert recipe.max_epochs * limits.forecast_margin == EPOCHS_TIMES_MARGIN
    assert (
        limits.classical_fit_seconds,
        limits.classical_aggregate_seconds,
        limits.shared_preparation_seconds,
    ) == (1800, 18000, 1800)


def test_report_is_written_and_records_the_cutoff_and_decisions(
    harness: Harness,
) -> None:
    _known_timings(harness)
    harness.run(8 + 6 + 0.5 + 5 + 4 + 1)
    saved = json.loads((harness.store.root / "queue.json").read_text("utf-8"))
    assert saved["cutoff"]["basis"] == "owner-operational"
    assert saved["arms"]["plantdoc"] == "omitted"
    stages = {d["stage"]: d["admitted"] for d in saved["decisions"]}
    assert stages["plantvillage"] is True
    assert stages["plantdoc"] is False
