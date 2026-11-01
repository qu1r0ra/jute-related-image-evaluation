import logging
import sys
from logging import Logger
from pathlib import Path
from typing import Any

METRICS_DIR = "metrics"
METRICS_FILE = "metrics.csv"


def setup_logging(level: int = logging.INFO) -> None:
    """Set up standardized logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Check if handlers already exist to avoid duplicate logs
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    # Silence noisy loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def get_logger(name: str, level: int = logging.INFO) -> Logger:
    """Get a logger with the specified name and ensure logging is set up."""
    setup_logging(level)
    return logging.getLogger(name)


def experiment_loggers(attempt_dir: Path | str) -> list[Any]:
    """Local loggers for one attempt, with a fixed layout and no version folders.

    Metrics always land in ``<attempt>/metrics/version_0/metrics.csv``.
    """
    from lightning.pytorch.loggers import CSVLogger

    return [CSVLogger(Path(attempt_dir), name=METRICS_DIR, version=0)]
