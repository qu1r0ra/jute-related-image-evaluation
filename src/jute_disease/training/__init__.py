from jute_disease.training.benchmark import (
    Benchmark,
    FitKind,
    FitSpec,
    QueueLimits,
    Recipe,
    fit_identity,
    load_benchmark,
)
from jute_disease.training.inputs import DataInputs
from jute_disease.training.logger import (
    experiment_loggers,
    get_logger,
)
from jute_disease.training.process import ChildResult, run_child
from jute_disease.training.queue import (
    MAX_CONSECUTIVE_FAILURES,
    RETRIES,
    Clock,
    Executor,
    IdentityFor,
    Outcome,
    QueueAbortedError,
    QueueReport,
    Stage,
    SystemClock,
    build_plan,
    run_queue,
)
from jute_disease.training.receipts import (
    ActiveRun,
    Resume,
    RunState,
    RunStore,
    write_json_atomic,
)
from jute_disease.training.results import FitResult
from jute_disease.training.resume import (
    ResumePlan,
    mark_finished,
    prepare_checkpoints,
    record_progress,
    verified_progress,
)
from jute_disease.training.seed import seed_everything
from jute_disease.training.stack import StackError, verify_stack
from jute_disease.training.workflow import (
    WORKSPACE_FILE,
    Job,
    Launcher,
    UpstreamError,
    Workspace,
    find_fit,
    make_executor,
    make_identity,
    open_workspace,
    run_job,
    subprocess_launcher,
)

# Resolved on first use: the fit engines import models, which import this package.
_ENGINE_EXPORTS = {
    "fit_cnn": "jute_disease.training.engines.dl.fit",
    "training_callbacks": "jute_disease.training.engines.dl.fit",
    "ProgressRecord": "jute_disease.training.engines.dl.fit",
    "fit_classical": "jute_disease.training.engines.ml.fit",
    "prepare_features": "jute_disease.training.engines.ml.fit",
}


def __getattr__(name: str) -> object:
    if name in _ENGINE_EXPORTS:
        from importlib import import_module

        return getattr(import_module(_ENGINE_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WORKSPACE_FILE",
    "ChildResult",
    "run_child",
    "Executor",
    "IdentityFor",
    "Job",
    "Launcher",
    "UpstreamError",
    "Workspace",
    "find_fit",
    "make_executor",
    "make_identity",
    "open_workspace",
    "run_job",
    "subprocess_launcher",
    "ActiveRun",
    "DataInputs",
    "FitResult",
    "StackError",
    "Benchmark",
    "FitKind",
    "FitSpec",
    "QueueLimits",
    "Recipe",
    "Stage",
    "build_plan",
    "run_queue",
    "fit_classical",
    "fit_cnn",
    "prepare_features",
    "fit_identity",
    "training_callbacks",
    "ProgressRecord",
    "ResumePlan",
    "mark_finished",
    "prepare_checkpoints",
    "record_progress",
    "verified_progress",
    "verify_stack",
    "load_benchmark",
    "Clock",
    "MAX_CONSECUTIVE_FAILURES",
    "QueueAbortedError",
    "RETRIES",
    "Outcome",
    "QueueReport",
    "Resume",
    "RunState",
    "RunStore",
    "SystemClock",
    "write_json_atomic",
    "experiment_loggers",
    "get_logger",
    "seed_everything",
]
