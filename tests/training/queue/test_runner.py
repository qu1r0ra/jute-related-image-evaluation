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
    LEARNING_RATE_MULTIPLIERS,
    Benchmark,
    ConfirmatoryAnalysisBlockedError,
    FailureKind,
    FitKind,
    FitSpec,
    RunState,
    RunStore,
    load_benchmark,
    require_confirmatory_completion,
)
from jute_disease.training.queue import (
    MAX_CONSECUTIVE_FAILURES,
    RETRIES,
    Outcome,
    QueueAbortedError,
    QueueReport,
    run_queue,
)

SEEDS = (42, 1337, 7)
CORE_TUNING = [
    "cnn/tune/mobilenet_v2_direct/pool-A",
    "cnn/tune/efficientnet_b5_imagenet/pool-A",
]
CORE = [f"cnn/mobilenet_v2_direct/seed-{s}" for s in SEEDS] + [
    f"cnn/efficientnet_b5_imagenet/seed-{s}" for s in SEEDS
]
CORE_PLAN = [CORE_TUNING[0], *CORE[:3], CORE_TUNING[1], *CORE[3:]]
CLASSICAL_COMBOS = [
    f"{e}-{x}" for e in ("gnb", "knn", "lr", "svm") for x in ("raw", "crafted")
]
PREPARE_RESERVE = "classical/prepare/pool-A/reserve"
PREPARE_DRAW = "classical/prepare/pool-A/view-1/image-random/split-0"
PLANTVILLAGE_TARGETS = [f"cnn/mobilenet_v2_plantvillage/seed-{s}" for s in SEEDS]
PLANTDOC_TARGETS = [f"cnn/mobilenet_v2_plantvillage_plantdoc/seed-{s}" for s in SEEDS]


def small_benchmark() -> Benchmark:
    """The declared benchmark cut to one pool, view, policy and split."""
    full = load_benchmark(BENCHMARK_CONFIG)
    return replace(
        full,
        recipe=replace(full.recipe, seeds=SEEDS),
        estimators=("gnb", "knn", "lr", "svm"),
        pools=full.pools[:1],
        policies=full.policies[:1],
        views=full.views[:1],
        splits=(0,),
        primary_split=0,
        assignments=((0, 0, 0), (0, 1, 1), (0, 2, 2)),
    )


TOTAL = len(small_benchmark().fits())


class FixedClock:
    def now(self) -> datetime:
        return datetime(2026, 11, 1, 9, 0)


def _receipt_metadata(
    *,
    seed: int,
    official: bool = True,
    assignment_valid: bool = True,
) -> dict:
    return {
        "official": official,
        "seed": seed,
        "provenance": {
            "commit": "test-commit",
            "working_tree_clean": official,
            "official": official,
        },
        "configuration": {"sha256": "config-sha"},
        "environment_lock": {"sha256": "lock-sha"},
        "preprocessing": {"sha256": "preprocessing-sha"},
        "assignment_table": {
            "sha256": "assignment-sha" if assignment_valid else None,
            "validated": assignment_valid,
            "assignments": [
                {"split": 0, "seed": 42, "checkpoint": 0},
                {"split": 0, "seed": 1337, "checkpoint": 2},
                {"split": 0, "seed": 7, "checkpoint": 1},
            ],
        },
        "source_checkpoint_ancestry": [],
        "runtime_contract": {
            "precision": "bf16-mixed",
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
            "torch_compile": False,
            "num_workers": 8,
        },
        "run_options": {
            "pretrained": True,
            "accelerator": "auto",
            "require_cuda": True,
        },
        "runtime": {
            "python": "3.14.0",
            "torch": "2.6.0",
            "torch_cuda": "12.6",
            "cuda_driver": "570.0",
            "cuda_available": True,
            "precision": "bf16-mixed",
            "libraries": {"torchvision": "0.21.0"},
            "backend": {
                "tf32_matmul": True,
                "cudnn_tf32": True,
                "cudnn_benchmark": False,
            },
            "compile_enabled": False,
            "deterministic_algorithms": False,
            "cudnn_deterministic": False,
            "float32_matmul_precision": "highest",
        },
    }


@dataclass
class Harness:
    store: RunStore
    # Attempts of a fit that fail, timed out or raising, before it succeeds.
    failing: Counter[str] = field(default_factory=Counter)
    timing_out: Counter[str] = field(default_factory=Counter)
    raising: Counter[str] = field(default_factory=Counter)
    no_receipt: set[str] = field(default_factory=set)
    identity_tag: str = "v1"
    failure_kind: FailureKind = FailureKind.INFRASTRUCTURE
    official: bool = True
    assignment_valid: bool = True
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
                self.store.start(
                    fit.fit_id,
                    self.identity(fit),
                    metadata=_receipt_metadata(
                        seed=fit.seed,
                        official=self.official,
                        assignment_valid=self.assignment_valid,
                    ),
                )
                raise OSError("launcher crashed")
            run = self.store.start(
                fit.fit_id,
                self.identity(fit),
                metadata=_receipt_metadata(
                    seed=fit.seed,
                    official=self.official,
                    assignment_valid=self.assignment_valid,
                ),
            )
            if self.timing_out[fit.fit_id]:
                self.timing_out[fit.fit_id] -= 1
                run.fail(RunState.TIMED_OUT, f"exceeded {timeout:.0f} s")
                return Outcome(
                    RunState.TIMED_OUT,
                    timeout,
                    "timed out",
                    FailureKind.INFRASTRUCTURE,
                )
            if self.failing[fit.fit_id]:
                self.failing[fit.fit_id] -= 1
                run.fail(RunState.FAILED, "boom")
                return Outcome(
                    RunState.FAILED,
                    1.0,
                    "boom",
                    failure_kind=self.failure_kind,
                )
            if fit.fit_id in self.no_receipt:
                return Outcome(RunState.COMPLETED, 1.0)
            artifact = run.attempt_dir / "model.bin"
            artifact.write_bytes(fit.fit_id.encode())
            run.complete(
                {"epochs": 50},
                artifacts={"model": artifact},
                metadata=_receipt_metadata(
                    seed=fit.seed,
                    official=self.official,
                    assignment_valid=self.assignment_valid,
                ),
            )
            return Outcome(RunState.COMPLETED, 1.0)
        finally:
            self.active -= 1

    def run(
        self, adjust: Callable[[Benchmark], Benchmark] = lambda b: b
    ) -> QueueReport:
        return run_queue(
            adjust(small_benchmark()),
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
    assert {f.name for f in fields(Outcome)} == {
        "state",
        "seconds",
        "reason",
        "failure_kind",
    }


def test_only_infrastructure_outcomes_are_retryable() -> None:
    assert not Outcome(RunState.TIMED_OUT, 1.0).retryable
    assert Outcome(
        RunState.TIMED_OUT, 1.0, failure_kind=FailureKind.INFRASTRUCTURE
    ).retryable
    assert not Outcome(
        RunState.FAILED, 1.0, failure_kind=FailureKind.DIVERGENCE
    ).retryable


def test_policy_is_one_retry_and_an_abort_after_three_failures() -> None:
    assert (RETRIES, MAX_CONSECUTIVE_FAILURES) == (1, 3)


def test_runs_core_then_classical_then_plantvillage_then_plantdoc(
    harness: Harness,
) -> None:
    report = harness.run()
    ids = harness.calls
    assert len(ids) == TOTAL
    assert ids[:8] == CORE_PLAN
    assert ids[8] == PREPARE_RESERVE
    assert ids[9] == PREPARE_DRAW
    assert ids[10:18] == [f"classical/tune/A/{combo}" for combo in CLASSICAL_COMBOS]
    assert ids[18:26] == [f"classical/{combo}" for combo in CLASSICAL_COMBOS]
    assert ids[26] == "source/tune/plantvillage"
    assert ids[27:30] == [f"source/plantvillage/seed-{s}" for s in SEEDS]
    assert ids[30] == "cnn/tune/mobilenet_v2_plantvillage/pool-A"
    assert ids[31:34] == PLANTVILLAGE_TARGETS
    assert ids[34:37] == [f"source/plantdoc/seed-{s}" for s in SEEDS]
    assert ids[37] == "cnn/tune/mobilenet_v2_plantvillage_plantdoc/pool-A"
    assert ids[38:] == PLANTDOC_TARGETS
    assert report.arms == {"plantvillage": "complete", "plantdoc": "complete"}
    assert report.escalations == []
    assert harness.overlapped is False
    assert report.completion_ready is True
    assert report.confirmatory_ready is True


def test_any_incomplete_planned_fit_blocks_confirmatory_analysis(
    harness: Harness,
) -> None:
    harness.failing[CORE[0]] = 1
    harness.failure_kind = FailureKind.OTHER

    report = harness.run()

    assert report.completion_ready is False
    assert report.confirmatory_ready is False
    with pytest.raises(ConfirmatoryAnalysisBlockedError, match=CORE[0]):
        require_confirmatory_completion(
            small_benchmark(), harness.store, harness.identity
        )


@pytest.mark.parametrize(
    ("official", "assignment_valid"), [(False, True), (True, False)]
)
def test_dirty_or_unbound_runs_block_confirmatory_analysis(
    harness: Harness, official: bool, assignment_valid: bool
) -> None:
    harness.official = official
    harness.assignment_valid = assignment_valid

    report = harness.run()

    assert report.completion_ready is True
    assert report.confirmatory_ready is False
    with pytest.raises(ConfirmatoryAnalysisBlockedError):
        require_confirmatory_completion(
            small_benchmark(), harness.store, harness.identity
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "identity",
        "metadata",
        "attempt_number",
        "state",
        "runtime",
        "attempt_runtime",
        "cpu_profile",
    ],
)
def test_confirmatory_gate_checks_every_attempt_binding(
    harness: Harness, tamper: str
) -> None:
    harness.failing[CORE[0]] = 1
    report = harness.run()
    assert report.confirmatory_ready is True

    receipt_path = harness.store.run_dir(CORE[0]) / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    first_attempt = receipt["attempts"][0]
    if tamper == "identity":
        first_attempt["identity"] = "another-run"
    elif tamper == "metadata":
        first_attempt["metadata"]["configuration"]["sha256"] = "another-config"
    elif tamper == "attempt_number":
        first_attempt["attempt"] = 3
    elif tamper == "state":
        state_path = harness.store.run_dir(CORE[0]) / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["state"] = "failed"
        state_path.write_text(json.dumps(state), encoding="utf-8")
    elif tamper == "runtime":
        receipt["metadata"]["runtime"]["backend"]["tf32_matmul"] = False
        receipt["attempts"][-1]["metadata"]["runtime"]["backend"]["tf32_matmul"] = False
    elif tamper == "attempt_runtime":
        first_attempt["metadata"].pop("runtime")
    else:
        receipt["metadata"]["run_options"]["require_cuda"] = False
        for attempt in receipt["attempts"]:
            attempt["metadata"]["run_options"]["require_cuda"] = False
    if tamper != "state":
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ConfirmatoryAnalysisBlockedError):
        require_confirmatory_completion(
            small_benchmark(), harness.store, harness.identity
        )


def test_every_step_runs_under_its_configured_timeout(harness: Harness) -> None:
    limits = load_benchmark(BENCHMARK_CONFIG).limits
    harness.run()
    assert harness.timeouts[PREPARE_RESERVE] == limits.prepare_seconds
    assert harness.timeouts[PREPARE_DRAW] == limits.prepare_seconds
    assert harness.timeouts["classical/lr-raw"] == limits.classical_seconds
    assert harness.timeouts["source/plantvillage/seed-42"] == limits.source_seconds
    assert harness.timeouts[CORE[0]] == limits.target_seconds
    candidates = len(LEARNING_RATE_MULTIPLIERS)
    assert harness.timeouts[CORE_TUNING[0]] == limits.target_seconds * candidates
    assert harness.timeouts["source/tune/plantvillage"] == (
        limits.source_seconds * candidates
    )
    assert harness.timeouts["classical/tune/A/lr-raw"] == limits.classical_seconds


def test_second_run_skips_everything_completed(harness: Harness) -> None:
    harness.run()
    harness.calls.clear()
    report = harness.run()
    assert harness.calls == []
    assert len(report.skipped) == TOTAL


def test_incompatible_identity_is_rerun_not_reused(harness: Harness) -> None:
    harness.run()
    harness.calls.clear()
    harness.identity_tag = "v2"
    report = harness.run()
    assert len(harness.calls) == TOTAL
    assert report.confirmatory_ready is True
    require_confirmatory_completion(small_benchmark(), harness.store, harness.identity)


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
    harness.failure_kind = FailureKind.INFRASTRUCTURE
    report = harness.run()
    assert harness.calls.count(CORE[0]) == 2
    assert report.outcomes[CORE[0]] is RunState.COMPLETED
    assert harness.store.state(CORE[0]) is RunState.COMPLETED


def test_non_infrastructure_failure_is_not_retried(harness: Harness) -> None:
    harness.failing[CORE[0]] = 2
    harness.failure_kind = FailureKind.OTHER

    report = harness.run()

    assert harness.calls.count(CORE[0]) == 1
    assert report.outcomes[CORE[0]] is RunState.FAILED


def test_divergence_is_not_retried(harness: Harness) -> None:
    harness.failing[CORE[0]] = 2
    harness.failure_kind = FailureKind.DIVERGENCE

    report = harness.run()

    assert harness.calls.count(CORE[0]) == 1
    assert report.outcomes[CORE[0]] is RunState.FAILED


def test_completion_without_a_valid_receipt_is_a_failure(harness: Harness) -> None:
    harness.no_receipt = {"classical/gnb-raw"}
    report = harness.run()
    assert harness.calls.count("classical/gnb-raw") == 2
    assert report.outcomes["classical/gnb-raw"] is RunState.FAILED
    assert report.reasons["classical/gnb-raw"] == "ended without a valid receipt"


def test_a_persistent_failure_is_reported_and_its_dependants_omitted(
    harness: Harness,
) -> None:
    harness.failing["source/tune/plantvillage"] = 2
    report = harness.run()
    assert harness.calls.count("source/tune/plantvillage") == 1 + RETRIES
    assert report.outcomes["source/tune/plantvillage"] is RunState.FAILED
    assert any("source/tune/plantvillage" in e for e in report.escalations)
    for fit_id in [
        *(f"source/plantvillage/seed-{s}" for s in SEEDS),
        *PLANTVILLAGE_TARGETS,
        *(f"source/plantdoc/seed-{s}" for s in SEEDS),
        *PLANTDOC_TARGETS,
    ]:
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
    assert harness.calls == [
        CORE_TUNING[0],
        *(fit_id for fit_id in CORE[:3] for _ in range(2)),
    ]
    saved = harness.saved()
    assert saved["consecutive_failures"] == CORE[:3]
    assert any("aborted" in e for e in saved["escalations"])


def test_a_completed_fit_resets_the_failure_streak(harness: Harness) -> None:
    for fit_id in (CORE[0], CORE[1], CORE[3], CORE[4]):
        harness.failing[fit_id] = 2
    report = harness.run()
    assert report.outcomes[CORE[2]] is RunState.COMPLETED
    assert len(harness.calls) == TOTAL + 4  # every failing fit used its retry
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
        if fit.fit_id.startswith("source/plantdoc/"):
            raise FileNotFoundError("PlantDoc manifest is missing")
        return original_identity(fit)

    harness.identity = identity  # type: ignore[method-assign]
    report = harness.run()
    assert report.arms == {"plantvillage": "complete", "plantdoc": "omitted"}
    assert not any(c.startswith("source/plantdoc/") for c in harness.calls)
    assert report.outcomes["source/plantdoc/seed-42"] is RunState.OMITTED
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
    assert harness.calls == [CORE_TUNING[0], *CORE[:2]]


def test_an_arm_is_judged_by_its_targets_not_its_source(harness: Harness) -> None:
    target = "cnn/mobilenet_v2_plantvillage/seed-42"
    harness.failing[target] = 2
    report = harness.run(_one_seed)
    assert report.outcomes["source/plantvillage/seed-42"] is RunState.COMPLETED
    assert report.arms["plantvillage"] == "omitted"


def test_report_records_attempts_outcomes_and_arms(harness: Harness) -> None:
    harness.failing[CORE[0]] = 1
    harness.run()
    saved = harness.saved()
    assert saved["attempts"][CORE[0]] == 2
    assert saved["attempts"][PREPARE_RESERVE] == 1
    assert saved["attempts"][PREPARE_DRAW] == 1
    assert saved["outcomes"][CORE[0]] == "completed"
    assert saved["arms"] == {"plantvillage": "complete", "plantdoc": "complete"}
    assert saved["updated_at"] == "2026-11-01T09:00:00"


def test_declared_timeouts_cover_every_kind_of_step() -> None:
    limits = load_benchmark(BENCHMARK_CONFIG).limits
    for kind in FitKind:
        assert limits.timeout(kind) > 0


def test_a_failed_tuning_fit_omits_the_finals_that_reuse_it(harness: Harness) -> None:
    tuning = "classical/tune/A/lr-raw"
    harness.failing[tuning] = 2
    harness.failure_kind = FailureKind.OTHER
    report = harness.run()
    assert report.outcomes[tuning] is RunState.FAILED
    assert report.outcomes["classical/lr-raw"] is RunState.OMITTED
    assert "classical/lr-raw" not in harness.calls
    assert report.reasons["classical/lr-raw"] == f"tuning {tuning} did not complete"
    assert report.outcomes["classical/lr-crafted"] is RunState.COMPLETED


def test_a_completed_fit_keeps_its_receipt_when_its_upstream_fails_later(
    harness: Harness,
) -> None:
    harness.run()
    harness.calls.clear()
    harness.identity_tag = "v2"
    harness.failing["source/tune/plantvillage"] = 2
    report = harness.run()

    assert report.outcomes["source/tune/plantvillage"] is RunState.FAILED
    for fit_id in [*PLANTVILLAGE_TARGETS, "source/plantvillage/seed-42"]:
        assert fit_id not in harness.calls
        # This run did not carry the fit, but its finished result is not erased.
        assert report.outcomes[fit_id] is RunState.OMITTED
        assert "completed receipt is kept" in report.reasons[fit_id]
        assert harness.store.state(fit_id) is RunState.COMPLETED
        assert harness.store.receipt(fit_id)["identity"] == f"v1:{fit_id}"


def test_a_completed_fit_counts_as_complete_when_its_inputs_are_gone(
    harness: Harness,
) -> None:
    harness.run()
    harness.calls.clear()
    original_identity = harness.identity

    def identity(fit: FitSpec) -> str:
        if fit.fit_id.startswith("source/plantdoc/"):
            raise FileNotFoundError("PlantDoc manifest is missing after a cull")
        return original_identity(fit)

    harness.identity = identity  # type: ignore[method-assign]
    report = harness.run()

    assert report.arms == {"plantvillage": "complete", "plantdoc": "complete"}
    assert not report.escalations
    assert harness.calls == []
    for fit_id in ["source/plantdoc/seed-42", *PLANTDOC_TARGETS]:
        assert report.outcomes[fit_id] is RunState.COMPLETED
        assert harness.store.state(fit_id) is RunState.COMPLETED


def test_tuning_selection_precedes_every_final_that_reuses_it() -> None:
    benchmark = small_benchmark()
    order = {fit.fit_id: index for index, fit in enumerate(benchmark.fits())}
    for fit in benchmark.fits():
        tuning = benchmark.tuning_dependency(fit)
        if tuning is not None:
            assert order[tuning] < order[fit.fit_id]
