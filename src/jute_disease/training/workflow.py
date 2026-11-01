"""Connect the queue to the fit engines through one child process per fit.

The queue decides what runs and when. This module decides how: it opens a run in
the store, launches ``train fit`` as a child so the time limit is enforced from
outside, and turns the child's result into a receipt. The same ``run_job`` body
runs in the child and, in tests, in process, so there is one code path.

Imports of ``jute_disease.data`` and the fit engines are deferred: both import
``jute_disease.training``, which exports this module.
"""

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from jute_disease.training.benchmark import (
    Benchmark,
    FitKind,
    FitSpec,
    fit_identity,
    load_benchmark,
)
from jute_disease.training.inputs import DataInputs
from jute_disease.training.process import ChildResult, run_child
from jute_disease.training.queue import Executor, IdentityFor, Outcome
from jute_disease.training.receipts import Resume, RunState, RunStore, write_json_atomic
from jute_disease.training.stack import verify_stack

RESULT_FILE = "result.json"
LOG_FILE = "child.log"
WORKSPACE_FILE = "workspace.json"
CNN_KINDS = (FitKind.SOURCE, FitKind.TARGET)


class UpstreamError(RuntimeError):
    """The checkpoint a chained fit starts from is missing or no longer verifies."""


@dataclass(frozen=True)
class Workspace:
    """Everything a child process needs to run any fit, saved as one JSON file."""

    benchmark_path: Path
    data: DataInputs
    runs_dir: Path
    cache_dir: Path
    pretrained: bool
    require_cuda: bool
    accelerator: str
    benchmark: Benchmark = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark", load_benchmark(self.benchmark_path))

    def save(self, path: Path | str) -> None:
        sources = {
            name: [str(manifest), str(root)]
            for name, (manifest, root) in self.data.sources.items()
        }
        write_json_atomic(
            path,
            {
                "benchmark_path": str(self.benchmark_path),
                "image_root": str(self.data.image_root),
                "target_dir": str(self.data.target_dir),
                "sources": sources,
                "runs_dir": str(self.runs_dir),
                "cache_dir": str(self.cache_dir),
                "pretrained": self.pretrained,
                "require_cuda": self.require_cuda,
                "accelerator": self.accelerator,
            },
        )

    @classmethod
    def load(cls, path: Path | str) -> Workspace:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            benchmark_path=Path(raw["benchmark_path"]),
            data=DataInputs(
                image_root=Path(raw["image_root"]),
                target_dir=Path(raw["target_dir"]),
                sources={
                    name: (Path(manifest), Path(root))
                    for name, (manifest, root) in raw["sources"].items()
                },
            ),
            runs_dir=Path(raw["runs_dir"]),
            cache_dir=Path(raw["cache_dir"]),
            pretrained=raw["pretrained"],
            require_cuda=raw["require_cuda"],
            accelerator=raw["accelerator"],
        )


def open_workspace(
    *,
    benchmark_path: Path | str,
    data: DataInputs,
    runs_dir: Path | str,
    cache_dir: Path | str,
    pretrained: bool = True,
    require_cuda: bool = True,
    accelerator: str = "auto",
) -> Workspace:
    """Resolve paths to absolute and create the output roots.

    A source package root may be the download folder; the manifest paths are
    relative to the folder that holds the partitions, which is found when
    available. An unavailable optional source is validated at its own stage.
    """
    from jute_disease.data import SourceMismatchError, find_source_root

    def source_root(root: Path | str) -> Path:
        path = Path(root).resolve()
        try:
            return find_source_root(path).resolve()
        except OSError, SourceMismatchError:
            # Optional inputs are checked when their own stage is attempted.
            return path

    runs = Path(runs_dir).resolve()
    cache = Path(cache_dir).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return Workspace(
        benchmark_path=Path(benchmark_path).resolve(),
        data=DataInputs(
            image_root=Path(data.image_root).resolve(),
            target_dir=Path(data.target_dir).resolve(),
            sources={
                name: (Path(manifest).resolve(), source_root(root))
                for name, (manifest, root) in data.sources.items()
            },
        ),
        runs_dir=runs,
        cache_dir=cache,
        pretrained=pretrained,
        require_cuda=require_cuda,
        accelerator=accelerator,
    )


@dataclass(frozen=True)
class Job:
    """One child invocation: a fit, where to work, and what it may start from.

    ``bounds`` are train and validation batch limits; a bounded job is a timing
    or smoke run and is never a fit.
    """

    fit_id: str
    attempt_dir: Path
    result_path: Path
    upstream_checkpoint: Path | None = None
    bounds: tuple[int, int] | None = None

    def to_args(self) -> list[str]:
        args = [
            "--fit-id",
            self.fit_id,
            "--attempt-dir",
            str(self.attempt_dir),
            "--result",
            str(self.result_path),
        ]
        if self.upstream_checkpoint is not None:
            args += ["--upstream-checkpoint", str(self.upstream_checkpoint)]
        if self.bounds is not None:
            args += ["--bounds", f"{self.bounds[0]},{self.bounds[1]}"]
        return args


Launcher = Callable[[Job, float | None, Path], ChildResult]


def subprocess_launcher(workspace_file: Path | str) -> Launcher:
    """Launch ``train fit`` in a child interpreter of the locked environment."""
    workspace_path = str(Path(workspace_file).resolve())

    def launch(job: Job, timeout: float | None, log: Path) -> ChildResult:
        argv = [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "train",
            "fit",
            "--workspace",
            workspace_path,
            *job.to_args(),
        ]
        return run_child(argv, timeout, log)

    return launch


def find_fit(workspace: Workspace, fit_id: str) -> FitSpec:
    """The declared fit named ``fit_id``.

    Raises:
        KeyError: no declared fit has that id.
    """
    for fit in workspace.benchmark.fits():
        if fit.fit_id == fit_id:
            return fit
    raise KeyError(f"No declared fit {fit_id!r}")


def run_job(workspace: Workspace, job: Job) -> None:
    """The child body: verify the stack, run one engine, write the result file.

    Raises:
        StackError: CUDA is required and unavailable, or a library is missing.
        ValueError: the fit is chained but the job has no upstream checkpoint.
    """
    from jute_disease.training.engines.dl.fit import fit_cnn
    from jute_disease.training.engines.ml.fit import fit_classical, prepare_features

    fit = find_fit(workspace, job.fit_id)
    recipe = workspace.benchmark.recipe
    data = workspace.data
    stack = verify_stack(require_cuda=workspace.require_cuda)
    if fit.kind is FitKind.PREPARE:
        result = prepare_features(fit, recipe, data, workspace.cache_dir)
    elif fit.kind is FitKind.CLASSICAL:
        store = RunStore(workspace.runs_dir)
        prepare = find_fit(workspace, "classical/prepare")
        if (
            store.resolve(prepare.fit_id, make_identity(workspace, store)(prepare))
            is not Resume.SKIP
        ):
            raise ValueError("Prepared feature cache has no valid compatible receipt")
        result = fit_classical(
            fit, recipe, data, job.attempt_dir, workspace.cache_dir, require_cache=True
        )
    else:
        bounded: dict[str, Any] = {}
        if job.bounds is not None:
            # A bounded run needs no chain; one given a checkpoint keeps it.
            if job.upstream_checkpoint is None:
                fit = replace(fit, upstream=None)
            recipe = replace(recipe, max_epochs=1)
            bounded = {
                "limit_train_batches": job.bounds[0],
                "limit_val_batches": job.bounds[1],
            }
        result = fit_cnn(
            fit,
            recipe,
            data,
            job.attempt_dir,
            upstream_checkpoint=job.upstream_checkpoint,
            pretrained=workspace.pretrained,
            accelerator=workspace.accelerator,
            **bounded,
        )
    write_json_atomic(
        job.result_path,
        {
            "results": {**result.results, "stack": stack},
            "artifacts": {name: str(path) for name, path in result.artifacts.items()},
        },
    )


def _manifest_digests(
    workspace: Workspace, fit: FitSpec, verifier: Any
) -> dict[str, str]:
    from jute_disease.data import load_manifest, load_source_manifest

    target = load_manifest(workspace.data.target_dir)
    digests = {
        "target": target.digest(),
        "target_content": verifier.verify(
            target, workspace.data.image_root, ("train", "val")
        ),
    }
    if fit.kind is FitKind.SOURCE:
        manifest_dir, image_root = workspace.data.sources[fit.stage]
        source = load_source_manifest(
            manifest_dir,
            image_root,
            workspace.data.target_dir,
            content_verifier=verifier,
        )
        digests[fit.stage] = source.digest()
    return digests


def _upstream_checkpoint(store: RunStore, fit: FitSpec) -> tuple[Path, str] | None:
    if fit.upstream is None:
        return None
    receipt = store.receipt(fit.upstream, missing_ok=True)
    if receipt is None or "checkpoint" not in receipt["artifacts"]:
        raise UpstreamError(f"upstream {fit.upstream} has no verified checkpoint")
    entry = receipt["artifacts"]["checkpoint"]
    return store.run_dir(fit.upstream) / entry["path"], entry["sha256"]


def make_identity(workspace: Workspace, store: RunStore) -> IdentityFor:
    """The compatibility identity of a fit, covering its current upstream bytes.

    An upstream with no verified checkpoint yields the marker ``unavailable``, so
    a fit never matches a receipt made from a checkpoint that cannot be checked.
    """
    from jute_disease.data import ManifestContentVerifier

    verifier = ManifestContentVerifier()

    def identity(fit: FitSpec) -> str:
        manifests = _manifest_digests(workspace, fit, verifier)
        try:
            upstream = _upstream_checkpoint(store, fit)
            sha = None if upstream is None else upstream[1]
        except UpstreamError:
            sha = "unavailable"
        environment: dict[str, Any] = {}
        if fit.kind is FitKind.PREPARE:
            environment["cache_dir"] = str(workspace.cache_dir.resolve())
        if fit.kind in CNN_KINDS:
            environment = {
                "pretrained": workspace.pretrained,
                "accelerator": workspace.accelerator,
            }
        return fit_identity(
            fit,
            workspace.benchmark.recipe,
            manifests,
            sha,
            environment,
        )

    return identity


def make_executor(
    workspace: Workspace, store: RunStore, identity: IdentityFor, launcher: Launcher
) -> Executor:
    """Run one fit as a child, bounded by ``timeout``, and receipt the result."""

    def execute(fit: FitSpec, timeout: float) -> Outcome:
        if fit.kind is FitKind.CLASSICAL:
            prepare = find_fit(workspace, "classical/prepare")
            if store.resolve(prepare.fit_id, identity(prepare)) is not Resume.SKIP:
                return Outcome(
                    RunState.FAILED,
                    0.0,
                    "Prepared feature cache has no valid compatible receipt",
                )
        try:
            upstream = _upstream_checkpoint(store, fit)
        except UpstreamError as error:
            return Outcome(RunState.FAILED, 0.0, str(error))
        run = store.start(fit.fit_id, identity(fit))
        job = Job(
            fit.fit_id,
            run.attempt_dir,
            run.attempt_dir / RESULT_FILE,
            None if upstream is None else upstream[0],
        )
        try:
            child = launcher(job, timeout, run.attempt_dir / LOG_FILE)
        except BaseException as error:
            state = (
                RunState.FAILED
                if isinstance(error, Exception)
                else (RunState.INTERRUPTED)
            )
            run.fail(state, f"{type(error).__name__}: {error}")
            raise
        if child.timed_out:
            reason = f"exceeded the {timeout:.0f} s limit"
            run.fail(RunState.TIMED_OUT, reason)
            return Outcome(RunState.TIMED_OUT, child.seconds, reason)
        if child.returncode != 0:
            reason = f"child exited {child.returncode}: {child.tail}"
            run.fail(RunState.FAILED, reason)
            return Outcome(RunState.FAILED, child.seconds, reason)
        try:
            payload = json.loads(job.result_path.read_text(encoding="utf-8"))
            artifacts = {
                name: Path(path) for name, path in payload["artifacts"].items()
            }
            run.complete(payload["results"], artifacts)
        except (OSError, ValueError, KeyError) as error:
            reason = f"child exited 0 without a usable result: {error}"
            run.fail(RunState.FAILED, reason)
            return Outcome(RunState.FAILED, child.seconds, reason)
        return Outcome(RunState.COMPLETED, child.seconds)

    return execute
