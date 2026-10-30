from jute_disease.training.benchmark import (
    Benchmark,
    FitKind,
    FitSpec,
    QueueLimits,
    Recipe,
    fit_identity,
    load_benchmark,
)
from jute_disease.training.logger import flatten_log_version, get_logger, setup_wandb
from jute_disease.training.queue import (
    MIN_RESERVE,
    Admission,
    Clock,
    Cutoff,
    CutoffUnverifiedError,
    MandatoryShortfallError,
    OptionalChain,
    Outcome,
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
from jute_disease.training.seed import seed_everything
from jute_disease.training.source import source_data_module


def pretrain_main(argv: list[str] | None = None) -> None:
    """Run the optional source pretraining command."""
    from jute_disease.training.engines.dl.pretrain import main

    main(argv)


__all__ = [
    "ActiveRun",
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
    "fit_identity",
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
    "flatten_log_version",
    "get_logger",
    "pretrain_main",
    "seed_everything",
    "setup_wandb",
    "source_data_module",
]
