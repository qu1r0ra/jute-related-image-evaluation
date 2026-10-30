from jute_disease.training.logger import flatten_log_version, get_logger, setup_wandb
from jute_disease.training.queue import (
    QueueEscalationError,
    QueueNotAdmittedError,
    Stage,
    build_plan,
    check_deadline,
    run_queue,
)
from jute_disease.training.seed import seed_everything
from jute_disease.training.source import source_data_module


def pretrain_main(argv: list[str] | None = None) -> None:
    """Run the optional source pretraining command."""
    from jute_disease.training.engines.dl.pretrain import main

    main(argv)


__all__ = [
    "QueueEscalationError",
    "QueueNotAdmittedError",
    "Stage",
    "build_plan",
    "check_deadline",
    "flatten_log_version",
    "get_logger",
    "pretrain_main",
    "run_queue",
    "seed_everything",
    "setup_wandb",
    "source_data_module",
]
