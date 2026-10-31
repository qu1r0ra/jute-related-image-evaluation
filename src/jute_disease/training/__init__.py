from jute_disease.training.logger import (
    flatten_log_version,
    get_logger,
    write_evaluation,
)
from jute_disease.training.seed import seed_everything
from jute_disease.training.source import source_data_module


def pretrain_main(argv: list[str] | None = None) -> None:
    """Run the optional source pretraining command."""
    from jute_disease.training.engines.dl.pretrain import main

    main(argv)


__all__ = [
    "flatten_log_version",
    "get_logger",
    "pretrain_main",
    "seed_everything",
    "source_data_module",
    "write_evaluation",
]
