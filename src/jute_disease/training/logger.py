import logging
import os
import sys
from logging import Logger
from pathlib import Path
from typing import Any

WANDB_MODES = ("off", "offline", "online")
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


def get_logger(name: str, level: int = logging.INFO) -> "Logger":
    """Get a logger with the specified name and ensure logging is set up."""
    setup_logging(level)
    return logging.getLogger(name)


def resolve_wandb_mode(requested: str | None) -> str:
    """Decide how W&B may run, without touching the network.

    W&B is optional. Only an explicit ``online`` request with a credential in the
    environment goes online; everything else records locally or not at all, and
    ``WANDB_MODE=disabled`` always wins.

    Raises:
        ValueError: ``requested`` is not off, offline, online or None.
    """
    if requested is not None and requested not in WANDB_MODES:
        raise ValueError(f"Unknown wandb mode {requested!r}; use one of {WANDB_MODES}")
    if requested == "off" or os.getenv("WANDB_MODE") == "disabled":
        return "off"
    if requested == "online" and os.getenv("WANDB_API_KEY"):
        return "online"
    return "offline"


def experiment_loggers(
    attempt_dir: Path | str,
    *,
    wandb_mode: str | None = None,
    project: str = "jute-disease",
    name: str = "run",
) -> tuple[list[Any], str]:
    """Loggers for one attempt, with a fixed layout and no version folders.

    Metrics always land in ``<attempt>/metrics/version_0/metrics.csv``. A W&B
    logger is added unless the resolved mode is ``off``. W&B is optional, so a
    service that cannot start is reported as ``unavailable`` and never fails the
    run.

    Returns:
        The loggers and the W&B status: ``off``, ``offline``, ``online`` or
        ``unavailable``.
    """
    from lightning.pytorch.loggers import CSVLogger

    attempt_dir = Path(attempt_dir)
    loggers: list[Any] = [CSVLogger(attempt_dir, name=METRICS_DIR, version=0)]
    mode = resolve_wandb_mode(wandb_mode)
    if mode == "off":
        return loggers, mode
    from lightning.pytorch.loggers import WandbLogger

    try:
        remote = WandbLogger(
            name=name,
            project=project,
            save_dir=str(attempt_dir),
            offline=mode == "offline",
            log_model=False,
        )
        _ = remote.experiment  # starts the W&B service, which may be unavailable
    except Exception:
        return loggers, "unavailable"
    return [*loggers, remote], mode
