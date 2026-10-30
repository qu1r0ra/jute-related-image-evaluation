"""Resumable queue runner: receipts, resume, timeouts, admission and sources.

The runner is driven through its public seam with a fake executor and a fake
clock, so no model trains and no wall-clock time passes.
"""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from jute_disease.config import DEFAULT_SEED
from jute_disease.data import PLANTDOC, PLANTVILLAGE, SOURCES, readiness_blockers
from jute_disease.training.queue import (
    CLASSICAL_AGGREGATE_SECONDS,
    CLASSICAL_FIT_SECONDS,
    ExecResult,
    QueueEscalationError,
    QueueNotAdmittedError,
    Stage,
    build_plan,
    run_queue,
    subprocess_executor,
)

START = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)
HOUR = 3600


class FakeClock:
    def __init__(self) -> None:
        self.current = START

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


@dataclass
class FakeExecutor:
    """Records each call; ``plan`` maps stage name to a scripted outcome."""

    clock: FakeClock
    seconds: float = 60.0
    seconds_by_group: dict[str, float] | None = None
    outcomes: dict[str, ExecResult | BaseException] | None = None
    on_run: Callable[[Stage], None] | None = None

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    def __call__(self, stage: Stage, timeout: float | None) -> ExecResult:
        self.calls.append((stage.name, timeout))
        if self.on_run is not None:
            self.on_run(stage)
        outcome = (self.outcomes or {}).get(stage.name)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not None:
            self.clock.advance(outcome.seconds)
            return outcome
        spent = (self.seconds_by_group or {}).get(stage.group, self.seconds)
        self.clock.advance(spent)
        return ExecResult(returncode=0, seconds=spent)

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


def deadline(hours_to_cutoff: float = 100.0, **extra: object) -> dict[str, object]:
    cutoff = START + timedelta(hours=hours_to_cutoff)
    return {
        "cutoff": cutoff.isoformat(),
        "timezone": "UTC",
        "verified": True,
        "verification": "owner operational cutoff",
        **extra,
    }


@pytest.fixture
def recipes(tmp_path: Path) -> Path:
    root = tmp_path / "recipes"
    for relative in (
        "core/mobilenet_v2.yaml",
        "core/efficientnet_b0.yaml",
        "plantvillage/target.yaml",
        "plantdoc/target.yaml",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("seed_everything: 42\n", encoding="utf-8")
    return root


ESTIMATES = {
    "core_seconds": 2 * HOUR,
    "plantvillage_seconds": HOUR,
    "plantdoc_seconds": HOUR,
}


@pytest.fixture
def plan(tmp_path: Path, recipes: Path) -> list[Stage]:
    return build_plan(
        ESTIMATES,
        recipes_dir=recipes,
        state_dir=tmp_path / "state",
        source_root=tmp_path / "sources",
    )


def receipt(state: Path, name: str) -> dict:
    return json.loads((state / "receipts" / f"{name}.json").read_text("utf-8"))


def statuses(state: Path, plan: list[Stage]) -> dict[str, str]:
    return {stage.name: receipt(state, stage.name)["status"] for stage in plan}


def run(plan, state, executor, clock, config=None, **kwargs):
    return run_queue(
        plan,
        state,
        config if config is not None else deadline(),
        admit=True,
        executor=executor,
        clock=clock,
        **kwargs,
    )


# --- plan -------------------------------------------------------------------


def test_plan_follows_the_ticket_order(plan: list[Stage]) -> None:
    names = [stage.name for stage in plan]
    classical = [name for name in names if name.startswith("classical-")]
    assert names[:2] == ["core-efficientnet_b0", "core-mobilenet_v2"]
    assert len(classical) == 10
    assert names[2:12] == classical
    assert names[12:] == [
        "plantvillage-source",
        "plantvillage-target",
        "plantdoc-source",
        "plantdoc-target",
    ]


def test_plan_runs_only_the_default_seed(plan: list[Stage]) -> None:
    for stage in plan:
        assert str(DEFAULT_SEED) == "42"
        assert "42" in stage.command
        for stray in ("1337", "7"):
            assert stray not in stage.command


def test_plan_never_reads_the_test_partition(plan: list[Stage]) -> None:
    for stage in plan:
        assert not {"test", "predict"} & set(stage.command)
        assert "evaluate" not in stage.command


def test_classical_fits_carry_the_declared_limit(plan: list[Stage]) -> None:
    fits = [stage for stage in plan if stage.group == "classical"]
    assert all(stage.timeout_seconds == CLASSICAL_FIT_SECONDS == 1800 for stage in fits)
    assert CLASSICAL_AGGREGATE_SECONDS == 5 * HOUR
    assert all(stage.mandatory for stage in plan[:12])
    assert not any(stage.mandatory for stage in plan[12:])


def test_plan_refuses_missing_recipes_and_estimates(
    tmp_path: Path, recipes: Path
) -> None:
    (recipes / "plantdoc" / "target.yaml").unlink()
    with pytest.raises(ValueError, match="plantdoc"):
        build_plan(ESTIMATES, recipes_dir=recipes, state_dir=tmp_path / "s")
    with pytest.raises(ValueError, match="core_seconds"):
        build_plan({}, recipes_dir=recipes, state_dir=tmp_path / "s")


# --- admission --------------------------------------------------------------


def test_run_requires_explicit_admission(tmp_path: Path, plan: list[Stage]) -> None:
    clock = FakeClock()
    executor = FakeExecutor(clock)
    with pytest.raises(QueueNotAdmittedError):
        run_queue(
            plan,
            tmp_path / "state",
            deadline(),
            admit=False,
            executor=executor,
            clock=clock,
        )
    assert executor.calls == []
    assert not (tmp_path / "state").exists()


def test_mandatory_shortfall_escalates_before_any_stage(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    executor = FakeExecutor(clock)
    state = tmp_path / "state"
    # core 2 h + classical 5 h + prep 0.5 h + reserve 2 h > 9 h
    with pytest.raises(QueueEscalationError, match="mandatory"):
        run(plan, state, executor, clock, deadline(hours_to_cutoff=9))
    assert executor.calls == []
    assert not (state / "receipts").exists()


def test_runner_needs_a_verified_cutoff(tmp_path: Path, plan: list[Stage]) -> None:
    clock = FakeClock()
    executor = FakeExecutor(clock)
    with pytest.raises(QueueEscalationError, match="verified cutoff"):
        run(plan, tmp_path / "state", executor, clock, {"cutoff": "x"})
    assert executor.calls == []


# --- receipts and order -----------------------------------------------------


def test_stages_run_in_order_and_write_receipts(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    executor = FakeExecutor(clock)
    state = tmp_path / "state"
    result = run(plan, state, executor, clock)
    assert executor.names == [stage.name for stage in plan]
    assert set(statuses(state, plan).values()) == {"completed"}
    first = receipt(state, plan[0].name)
    assert first["returncode"] == 0
    assert first["seconds"] == 60.0
    assert len(first["identity"]) == 64
    assert result["completed"] == [stage.name for stage in plan]
    assert not list((state / "receipts").glob("*.tmp"))


def test_receipts_are_written_atomically(tmp_path: Path, plan: list[Stage]) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    seen: list[str] = []

    def inspect(stage: Stage) -> None:
        seen.append(receipt(state, stage.name)["status"])

    run(plan, state, FakeExecutor(clock, on_run=inspect), clock)
    assert set(seen) == {"running"}


# --- resume -----------------------------------------------------------------


def test_resume_skips_completed_stages(tmp_path: Path, plan: list[Stage]) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    run(plan, state, FakeExecutor(clock), clock)
    again = FakeExecutor(clock)
    result = run(plan, state, again, clock)
    assert again.calls == []
    assert result["skipped"] == [stage.name for stage in plan]


def test_resume_reruns_a_stage_whose_config_changed(
    tmp_path: Path, plan: list[Stage], recipes: Path
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    run(plan, state, FakeExecutor(clock), clock)
    (recipes / "core" / "mobilenet_v2.yaml").write_text("seed_everything: 7\n", "utf-8")
    again = FakeExecutor(clock)
    run(plan, state, again, clock)
    assert again.names == ["core-mobilenet_v2"]


def test_resume_reruns_downstream_of_a_changed_input(
    tmp_path: Path, plan: list[Stage], recipes: Path
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    run(plan, state, FakeExecutor(clock), clock)
    stale = receipt(state, "plantvillage-target")["identity"]
    source = next(stage for stage in plan if stage.name == "plantvillage-source")
    Path(source.inputs[0]).parent.mkdir(parents=True, exist_ok=True)
    Path(source.inputs[0]).write_text("changed manifest", encoding="utf-8")
    again = FakeExecutor(clock)
    run(plan, state, again, clock)
    assert again.names == ["plantvillage-source", "plantvillage-target"]
    assert receipt(state, "plantvillage-target")["identity"] != stale


def test_interruption_is_recorded_and_resumes_after_the_last_completed_stage(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    interrupted = FakeExecutor(
        clock, outcomes={"classical-gnb-crafted": KeyboardInterrupt()}
    )
    with pytest.raises(KeyboardInterrupt):
        run(plan, state, interrupted, clock)
    assert receipt(state, "classical-gnb-crafted")["status"] == "interrupted"
    resumed = FakeExecutor(clock)
    run(plan, state, resumed, clock)
    assert resumed.names[0] == "classical-gnb-crafted"
    assert "core-efficientnet_b0" not in resumed.names
    assert statuses(state, plan)["plantdoc-target"] == "completed"


def test_crash_leaving_a_running_receipt_reruns_that_stage(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    run(plan, state, FakeExecutor(clock), clock)
    path = state / "receipts" / "core-mobilenet_v2.json"
    crashed = json.loads(path.read_text("utf-8"))
    crashed["status"] = "running"
    path.write_text(json.dumps(crashed), encoding="utf-8")
    again = FakeExecutor(clock)
    run(plan, state, again, clock)
    assert again.names == ["core-mobilenet_v2"]


def test_partial_outputs_without_a_completed_receipt_are_rejected(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    stage = plan[0]
    leftover = Path(stage.outputs[0])
    leftover.parent.mkdir(parents=True, exist_ok=True)
    leftover.write_text("half written", encoding="utf-8")
    present: list[bool] = []

    def inspect(running: Stage) -> None:
        if running.name == stage.name:
            present.append(leftover.exists())

    run(plan, state, FakeExecutor(clock, on_run=inspect), clock)
    assert present == [False]
    assert receipt(state, stage.name)["rejected_partial"] == [str(leftover)]


def test_completed_receipt_with_missing_output_reruns(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    stage = plan[0]
    output = Path(stage.outputs[0])

    def produce(running: Stage) -> None:
        for path in map(Path, running.outputs):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("done", encoding="utf-8")

    run(plan, state, FakeExecutor(clock, on_run=produce), clock)
    output.unlink()
    again = FakeExecutor(clock, on_run=produce)
    run(plan, state, again, clock)
    assert again.names == [stage.name]


# --- timeouts ---------------------------------------------------------------


def test_classical_fit_timeout_is_passed_and_escalates(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(
        clock,
        outcomes={
            "classical-svm-raw": ExecResult(returncode=-9, timed_out=True, seconds=1800)
        },
    )
    with pytest.raises(QueueEscalationError, match="classical-svm-raw"):
        run(plan, state, executor, clock)
    assert dict(executor.calls)["classical-svm-raw"] == 1800
    assert receipt(state, "classical-svm-raw")["status"] == "timed_out"
    assert "plantvillage-source" not in executor.names


def test_aggregate_fitting_allowance_caps_the_remaining_fit(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(clock, seconds=1700)
    with pytest.raises(QueueEscalationError, match="aggregate"):
        run(plan, state, executor, clock, aggregate_seconds=4000)
    calls = dict(executor.calls)
    classical = [name for name in executor.names if name.startswith("classical-")]
    assert calls[classical[0]] == 1800
    assert calls[classical[1]] == 1800
    assert calls[classical[2]] == 4000 - 2 * 1700
    assert len(classical) == 3


def test_mandatory_failure_escalates_and_stops(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(
        clock, outcomes={"core-mobilenet_v2": ExecResult(returncode=1, seconds=5)}
    )
    with pytest.raises(QueueEscalationError, match="core-mobilenet_v2"):
        run(plan, state, executor, clock)
    assert receipt(state, "core-mobilenet_v2")["status"] == "failed"
    assert executor.names == ["core-efficientnet_b0", "core-mobilenet_v2"]


# --- optional arms ----------------------------------------------------------


# Mandatory stages consume 2 h of core fits and 10 classical fits of 1800 s.
MANDATORY_SPENT = {"core": 1.0 * HOUR, "classical": 1800.0}
MANDATORY_HOURS = 2 + 5


def arm_run(tmp_path: Path, plan: list[Stage], cutoff_hours: float, **outcomes):
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(
        clock, seconds=30, seconds_by_group=MANDATORY_SPENT, outcomes=outcomes
    )
    run(plan, state, executor, clock, deadline(hours_to_cutoff=cutoff_hours))
    return state, executor


def test_both_arms_run_when_the_forecast_fits(
    tmp_path: Path, plan: list[Stage]
) -> None:
    # arms 2 h + reserve 2 h after the mandatory stages
    state, _ = arm_run(tmp_path, plan, MANDATORY_HOURS + 4.1)
    assert statuses(state, plan)["plantdoc-target"] == "completed"


def test_plantdoc_is_dropped_first_when_both_do_not_fit(
    tmp_path: Path, plan: list[Stage]
) -> None:
    state, _ = arm_run(tmp_path, plan, MANDATORY_HOURS + 3.1)
    result = statuses(state, plan)
    assert result["plantvillage-target"] == "completed"
    assert result["plantdoc-source"] == result["plantdoc-target"] == "omitted"
    assert "deadline" in receipt(state, "plantdoc-source")["cause"]


def test_both_arms_are_omitted_when_neither_fits(
    tmp_path: Path, plan: list[Stage]
) -> None:
    state, executor = arm_run(tmp_path, plan, MANDATORY_HOURS + 2.6)
    assert not [name for name in executor.names if name.endswith("-source")]
    for name in ("plantvillage-source", "plantdoc-target"):
        assert receipt(state, name)["status"] == "omitted"


def test_arm_admission_is_refreshed_at_the_boundary(
    tmp_path: Path, plan: list[Stage]
) -> None:
    # PlantVillage is estimated at one hour but takes two, so PlantDoc no longer fits.
    state, _ = arm_run(
        tmp_path,
        plan,
        MANDATORY_HOURS + 4.1,
        **{"plantvillage-target": ExecResult(returncode=0, seconds=2 * HOUR)},
    )
    assert receipt(state, "plantvillage-target")["status"] == "completed"
    assert receipt(state, "plantdoc-source")["status"] == "omitted"


def test_optional_failure_omits_its_arm_and_continues(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(
        clock, outcomes={"plantvillage-source": ExecResult(returncode=1, seconds=5)}
    )
    run(plan, state, executor, clock)
    result = statuses(state, plan)
    assert result["plantvillage-source"] == "failed"
    assert result["plantvillage-target"] == "omitted"
    assert result["plantdoc-target"] == "completed"
    assert "plantvillage-source" in receipt(state, "plantvillage-target")["cause"]


def test_forecast_does_not_start_a_run(tmp_path: Path, plan: list[Stage]) -> None:
    from jute_disease.training import check_deadline

    config = deadline(mandatory_seconds=HOUR, optional_seconds=0)
    assert check_deadline(config, START.isoformat())["forecast_fits"] is True
    assert not (tmp_path / "state").exists()


# --- source readiness (no publisher or terms gates) -------------------------


def test_real_sources_are_unblocked_despite_unverified_identity_and_terms() -> None:
    for spec in (PLANTVILLAGE, PLANTDOC):
        assert spec.publisher_identity_verified is False
        assert spec.terms_verified is False
        assert readiness_blockers(spec) == ()


def test_unblocked_sources_run_through_the_queue(
    tmp_path: Path, plan: list[Stage]
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(clock)
    run(plan, state, executor, clock, sources=SOURCES)
    assert "plantvillage-source" in executor.names
    assert "plantdoc-source" in executor.names


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"owner_internal_use_authorized": False}, "Owner authorization"),
        ({"owner_authorization_record": ""}, "Owner authorization"),
        ({"expected_inventory_digest": None}, "inventory SHA-256"),
        ({"expected_inventory_digest": "abc"}, "inventory SHA-256"),
    ],
)
def test_blocked_source_arm_is_omitted_with_its_cause(
    tmp_path: Path, plan: list[Stage], change: dict, fragment: str
) -> None:
    clock = FakeClock()
    state = tmp_path / "state"
    executor = FakeExecutor(clock)
    sources = {**SOURCES, "plantvillage": replace(PLANTVILLAGE, **change)}
    run(plan, state, executor, clock, sources=sources)
    assert not [name for name in executor.names if name.startswith("plantvillage")]
    for name in ("plantvillage-source", "plantvillage-target"):
        assert receipt(state, name)["status"] == "omitted"
        assert fragment in receipt(state, name)["cause"]
    assert receipt(state, "plantdoc-target")["status"] == "completed"
    # Mandatory stages are unaffected by a blocked optional source.
    assert receipt(state, "core-mobilenet_v2")["status"] == "completed"


# --- the default executor ---------------------------------------------------


def stage_running(code: str, tmp_path: Path) -> Stage:
    return Stage(
        name="probe",
        group="core",
        command=(sys.executable, "-c", code),
        timeout_seconds=None,
        estimate_seconds=1,
        mandatory=True,
    )


def test_subprocess_executor_reports_exit_codes(tmp_path: Path) -> None:
    ok = subprocess_executor(stage_running("pass", tmp_path), 30)
    bad = subprocess_executor(stage_running("raise SystemExit(3)", tmp_path), 30)
    assert (ok.returncode, ok.timed_out) == (0, False)
    assert (bad.returncode, bad.timed_out) == (3, False)


def test_subprocess_executor_kills_a_stage_at_its_timeout(tmp_path: Path) -> None:
    outcome = subprocess_executor(
        stage_running("import time; time.sleep(60)", tmp_path), 1
    )
    assert outcome.timed_out is True
    assert outcome.returncode != 0
    assert outcome.seconds < 30
