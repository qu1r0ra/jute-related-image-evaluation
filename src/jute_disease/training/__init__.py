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
    resolve_wandb_mode,
)
from jute_disease.training.process import ChildResult, run_child
from jute_disease.training.queue import (
    MIN_RESERVE,
    Admission,
    Clock,
    Cutoff,
    CutoffUnverifiedError,
    Executor,
    IdentityFor,
    MandatoryShortfallError,
    OptionalChain,
    Outcome,
    Probe,
    QueueReport,
    Stage,
    SystemClock,
    TimingBook,
    admit_optional,
    build_plan,
    run_queue,
    timing_key,
)
from jute_disease.training.receipts import (
    ActiveRun,
    Resume,
    RunState,
    RunStore,
    write_json_atomic,
)
from jute_disease.training.results import FitResult
from jute_disease.training.seed import seed_everything
from jute_disease.training.stack import StackError, verify_stack
from jute_disease.training.workflow import (
    WORKSPACE_FILE,
    Job,
    Launcher,
    ProbeError,
    UpstreamError,
    Workspace,
    find_fit,
    make_executor,
    make_identity,
    make_probe,
    open_workspace,
    run_job,
    subprocess_launcher,
)

# Resolved on first use: the fit engines import models, which import this package.
_ENGINE_EXPORTS = {
    "fit_cnn": "jute_disease.training.engines.dl.fit",
    "training_callbacks": "jute_disease.training.engines.dl.fit",
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
    "Probe",
    "ProbeError",
    "UpstreamError",
    "Workspace",
    "find_fit",
    "make_executor",
    "make_identity",
    "make_probe",
    "open_workspace",
    "run_job",
    "subprocess_launcher",
    "ActiveRun",
    "DataInputs",
    "FitResult",
    "StackError",
    "Admission",
    "Benchmark",
    "FitKind",
    "FitSpec",
    "QueueLimits",
    "Recipe",
    "Stage",
    "build_plan",
    "run_queue",
    "timing_key",
    "fit_classical",
    "fit_cnn",
    "prepare_features",
    "fit_identity",
    "training_callbacks",
    "verify_stack",
    "load_benchmark",
    "Clock",
    "Cutoff",
    "CutoffUnverifiedError",
    "MIN_RESERVE",
    "MandatoryShortfallError",
    "OptionalChain",
    "Outcome",
    "QueueReport",
    "Resume",
    "RunState",
    "RunStore",
    "SystemClock",
    "TimingBook",
    "admit_optional",
    "write_json_atomic",
    "experiment_loggers",
    "get_logger",
    "resolve_wandb_mode",
    "seed_everything",
]
