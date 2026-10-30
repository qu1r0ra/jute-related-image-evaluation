"""Real executors for the queue, with the child process replaced by a launcher.

The launcher stands in for ``python -m jute_disease train fit``: it runs the job
body in this process, so the same code paths run without a process per fit.
"""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import file_sha256
from jute_disease.training import (
    ChildResult,
    Cutoff,
    DataInputs,
    FitKind,
    FitSpec,
    Job,
    QueueReport,
    RunState,
    RunStore,
    SystemClock,
    Workspace,
    load_benchmark,
    make_executor,
    make_identity,
    make_probe,
    open_workspace,
    run_job,
    run_queue,
)

from .conftest import Inputs


def _benchmark_json(path: Path) -> Path:
    raw = json.loads(BENCHMARK_CONFIG.read_text(encoding="utf-8"))
    raw["recipe"].update(
        image_size=64,
        seeds=[42],
        max_epochs=1,
        precision="32-true",
        effective_batch=16,
    )
    raw["cnn_systems"] = {
        "tiny_direct": {"model": "resnet18", "micro_batch": 8, "chain": []},
        "tiny_chained": {
            "model": "resnet18",
            "micro_batch": 8,
            "chain": ["plantvillage"],
        },
    }
    raw["classical"] = {"estimators": ["lr"], "features": ["raw"], "seed": 42}
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.fixture
def workspace(inputs: Inputs, tmp_path: Path) -> Workspace:
    return open_workspace(
        benchmark_path=_benchmark_json(tmp_path / "benchmark.json"),
        data=DataInputs(
            image_root=inputs.image_root,
            target_dir=inputs.target_dir,
            sources=inputs.sources,
        ),
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        pretrained=False,
        require_cuda=False,
        accelerator="cpu",
    )


class InProcess:
    """A launcher that runs the job body here and records what it was given."""

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.jobs: list[Job] = []
        self.limits: list[float | None] = []
        self.override: ChildResult | None = None
        self.raises: BaseException | None = None

    def __call__(self, job: Job, timeout: float | None, log: Path) -> ChildResult:
        self.jobs.append(job)
        self.limits.append(timeout)
        if self.raises is not None:
            raise self.raises
        if self.override is not None:
            return self.override
        run_job(self.workspace, job)
        return ChildResult(0, False, 1.0, "")


def _fit(workspace: Workspace, fit_id: str) -> FitSpec:
    return next(f for f in workspace.benchmark.fits() if f.fit_id == fit_id)


def test_workspace_resolves_a_wrapped_source_package(
    inputs: Inputs, tmp_path: Path
) -> None:
    """A downloaded package nests its partitions in a wrapper folder."""
    package = tmp_path / "package"
    wrapped = package / "PlantVillage"
    for partition in ("train", "val"):
        (wrapped / partition).mkdir(parents=True)
    manifest = inputs.sources["plantvillage"][0]
    workspace = open_workspace(
        benchmark_path=_benchmark_json(tmp_path / "benchmark.json"),
        data=DataInputs(
            image_root=inputs.image_root,
            target_dir=inputs.target_dir,
            sources={"plantvillage": (manifest, package)},
        ),
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        pretrained=False,
        require_cuda=False,
    )
    assert workspace.data.sources["plantvillage"][1] == wrapped.resolve()


def test_workspace_round_trips_through_its_file(workspace: Workspace) -> None:
    path = workspace.runs_dir / "workspace.json"
    workspace.save(path)
    assert Workspace.load(path) == workspace


def test_unavailable_plantdoc_does_not_prevent_workspace_or_plantvillage(
    inputs: Inputs, tmp_path: Path
) -> None:
    missing = tmp_path / "missing-plantdoc"
    data = DataInputs(
        image_root=inputs.image_root,
        target_dir=inputs.target_dir,
        sources={**inputs.sources, "plantdoc": (missing / "manifest", missing)},
    )
    workspace = open_workspace(
        benchmark_path=_benchmark_json(tmp_path / "benchmark.json"),
        data=data,
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        pretrained=False,
        require_cuda=False,
        accelerator="cpu",
    )
    store = RunStore(workspace.runs_dir)
    execute = make_executor(
        workspace, store, make_identity(workspace, store), InProcess(workspace)
    )
    assert (
        execute(_fit(workspace, "source/plantvillage"), 1800).state
        is RunState.COMPLETED
    )


def test_execute_completes_prepare_then_a_classical_fit(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    identity = make_identity(workspace, store)
    execute = make_executor(workspace, store, identity, launcher)

    prepare = _fit(workspace, "classical/prepare")
    outcome = execute(prepare, 1800)
    assert outcome.state is RunState.COMPLETED
    assert launcher.limits[-1] == 1800

    fit = _fit(workspace, "classical/lr-raw")
    assert execute(fit, 1800).state is RunState.COMPLETED
    receipt = store.receipt(fit.fit_id)
    assert receipt["results"]["test_accessed"] is False
    assert receipt["results"]["stack"]["compile_enabled"] is False
    assert receipt["artifacts"]["model"]["sha256"]


def test_missing_prepared_cache_invalidates_receipt_and_blocks_fit(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    identity = make_identity(workspace, store)
    execute = make_executor(workspace, store, identity, launcher)
    prepare = _fit(workspace, "classical/prepare")
    assert execute(prepare, 1800).state is RunState.COMPLETED
    next(workspace.cache_dir.glob("*_X.npy")).unlink()
    assert store.state(prepare.fit_id) is RunState.INTERRUPTED
    count = len(launcher.jobs)
    outcome = execute(_fit(workspace, "classical/lr-raw"), 1800)
    assert outcome.state is RunState.FAILED
    assert len(launcher.jobs) == count
    assert execute(prepare, 1800).state is RunState.COMPLETED
    assert (
        execute(_fit(workspace, "classical/lr-raw"), 1800).state is RunState.COMPLETED
    )


def test_changing_cache_location_invalidates_preparation_identity(
    workspace: Workspace,
) -> None:
    prepare = _fit(workspace, "classical/prepare")
    store = RunStore(workspace.runs_dir)
    other = replace(workspace, cache_dir=workspace.cache_dir / "other")
    assert make_identity(workspace, store)(prepare) != make_identity(other, store)(
        prepare
    )


def test_timeout_is_recorded_and_never_reused(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    identity = make_identity(workspace, store)
    execute = make_executor(workspace, store, identity, launcher)
    fit = _fit(workspace, "classical/prepare")

    launcher.override = ChildResult(1, True, 1800.0, "")
    outcome = execute(fit, 1800)
    assert outcome.state is RunState.TIMED_OUT
    assert "1800" in outcome.reason
    assert store.state(fit.fit_id) is RunState.TIMED_OUT
    assert store.resolve(fit.fit_id, identity(fit)).value == "run-fresh"


def test_a_failed_child_records_its_log_tail(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    execute = make_executor(workspace, store, make_identity(workspace, store), launcher)
    fit = _fit(workspace, "classical/prepare")

    launcher.override = ChildResult(2, False, 3.0, "Traceback: boom")
    outcome = execute(fit, 1800)
    assert outcome.state is RunState.FAILED
    assert "boom" in outcome.reason
    assert store.state(fit.fit_id) is RunState.FAILED


def test_a_child_that_exits_zero_without_a_result_fails(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    execute = make_executor(workspace, store, make_identity(workspace, store), launcher)
    launcher.override = ChildResult(0, False, 1.0, "")
    outcome = execute(_fit(workspace, "classical/prepare"), 1800)
    assert outcome.state is RunState.FAILED
    assert "result" in outcome.reason


def test_an_interrupt_marks_the_run_interrupted_and_propagates(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    execute = make_executor(workspace, store, make_identity(workspace, store), launcher)
    fit = _fit(workspace, "classical/prepare")
    launcher.raises = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        execute(fit, 1800)
    assert store.state(fit.fit_id) is RunState.INTERRUPTED


def test_identity_covers_recipe_manifests_and_upstream_bytes(
    workspace: Workspace,
) -> None:
    store = RunStore(workspace.runs_dir)
    identity = make_identity(workspace, store)
    target = _fit(workspace, "cnn/tiny_chained/seed-42")
    before = identity(target)
    assert identity(target) == before

    other = open_workspace(
        benchmark_path=workspace.benchmark_path,
        data=workspace.data,
        runs_dir=workspace.runs_dir,
        cache_dir=workspace.cache_dir,
        pretrained=True,
        require_cuda=False,
        accelerator="cpu",
    )
    assert make_identity(other, store)(target) != before


def test_chain_passes_the_verified_upstream_checkpoint(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    identity = make_identity(workspace, store)
    execute = make_executor(workspace, store, identity, launcher)

    source = _fit(workspace, "source/plantvillage")
    target = _fit(workspace, "cnn/tiny_chained/seed-42")
    assert execute(source, None).state is RunState.COMPLETED
    assert launcher.jobs[-1].upstream_checkpoint is None

    assert execute(target, None).state is RunState.COMPLETED
    checkpoint = launcher.jobs[-1].upstream_checkpoint
    assert checkpoint is not None
    recorded = store.receipt(source.fit_id)["artifacts"]["checkpoint"]["sha256"]
    assert file_sha256(checkpoint) == recorded
    result = store.receipt(target.fit_id)["results"]
    assert result["lineage"]["upstream_checkpoint_sha256"] == recorded
    assert result["lineage"]["upstream_fit"] == "source/plantvillage"
    assert result["pretrained"] is False


def test_a_bounded_target_job_can_check_transfer_from_a_source_checkpoint(
    workspace: Workspace, tmp_path: Path
) -> None:
    """A smoke run proves the handoff without needing a completed source fit."""
    source_dir = tmp_path / "bounded-source"
    source_dir.mkdir()
    source = Job(
        "source/plantvillage", source_dir, source_dir / "r.json", bounds=(2, 1)
    )
    run_job(workspace, source)
    checkpoint = Path(
        json.loads(source.result_path.read_text())["artifacts"]["checkpoint"]
    )

    target_dir = tmp_path / "bounded-target"
    target_dir.mkdir()
    target = Job(
        "cnn/tiny_chained/seed-42",
        target_dir,
        target_dir / "r.json",
        checkpoint,
        bounds=(2, 1),
    )
    run_job(workspace, target)
    result = json.loads(target.result_path.read_text())["results"]
    assert result["bounded"] is True
    assert result["pretrained"] is False
    assert result["transfer"] is not None
    assert result["head"] == {"fresh": True, "equals_upstream": False}
    assert result["test_accessed"] is False


def test_tampered_upstream_checkpoint_is_refused(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    identity = make_identity(workspace, store)
    execute = make_executor(workspace, store, identity, launcher)
    source = _fit(workspace, "source/plantvillage")
    target = _fit(workspace, "cnn/tiny_chained/seed-42")
    execute(source, None)
    before = identity(target)
    checkpoint = (
        store.run_dir(source.fit_id)
        / store.receipt(source.fit_id)["artifacts"]["checkpoint"]["path"]
    )
    checkpoint.write_bytes(b"tampered")

    assert identity(target) != before
    ran = len(launcher.jobs)
    outcome = execute(target, None)
    assert outcome.state is RunState.FAILED
    assert "upstream" in outcome.reason
    assert len(launcher.jobs) == ran  # nothing was launched on bad bytes


def test_probe_scales_a_bounded_run_to_a_full_epoch(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    probe = make_probe(workspace, launcher)
    chained = _fit(workspace, "cnn/tiny_chained/seed-42")

    seconds = probe(chained, 900.0)
    assert seconds > 0
    job = launcher.jobs[-1]
    assert job.bounds is not None
    assert job.upstream_checkpoint is None  # a probe never needs the chain
    assert not store.run_dir(chained.fit_id).exists()  # probes are not fits


def test_probe_respects_a_shorter_deadline(workspace: Workspace) -> None:
    launcher = InProcess(workspace)
    launcher.override = ChildResult(1, True, 12.0, "deadline")
    probe = make_probe(workspace, launcher)
    with pytest.raises(RuntimeError, match="probe"):
        probe(_fit(workspace, "cnn/tiny_direct/seed-42"), 12.0)
    assert launcher.limits[-1] == 12.0


def test_queue_runs_resumes_and_skips_end_to_end(workspace: Workspace) -> None:
    store = RunStore(workspace.runs_dir)
    launcher = InProcess(workspace)
    identity = make_identity(workspace, store)
    cutoff = Cutoff(
        at=datetime.now() + timedelta(days=30), basis="owner-operational", record="n"
    )

    def go() -> QueueReport:
        return run_queue(
            workspace.benchmark,
            store,
            clock=SystemClock(),
            cutoff=cutoff,
            execute=make_executor(workspace, store, identity, launcher),
            probe=make_probe(workspace, launcher),
            identity_for=identity,
            report_path=workspace.runs_dir / "queue.json",
        )

    first = go()
    expected = {f.fit_id for f in workspace.benchmark.fits()}
    assert set(first.outcomes) == expected
    assert all(state is RunState.COMPLETED for state in first.outcomes.values())

    ran = len(launcher.jobs)
    again = go()
    assert set(again.skipped) == expected
    assert len(launcher.jobs) == ran  # a resumed queue launches nothing


def test_benchmark_declares_the_fits_the_workspace_runs(workspace: Workspace) -> None:
    kinds = {fit.kind for fit in workspace.benchmark.fits()}
    assert kinds == {
        FitKind.PREPARE,
        FitKind.CLASSICAL,
        FitKind.SOURCE,
        FitKind.TARGET,
    }
    assert load_benchmark(workspace.benchmark_path) == workspace.benchmark
