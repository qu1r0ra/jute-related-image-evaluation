import json
import logging
import shutil
import sys
import tempfile
from logging import Logger
from pathlib import Path
from typing import Any


def flatten_log_version(log_dir: Path, target_name: str) -> None:
    """Takes the latest parameter's `version_*` folder and flattens
    its metrics out.
    """
    if not log_dir.exists():
        return
    versions = sorted([d for d in log_dir.glob("version_*") if d.is_dir()])
    if not versions:
        return
    latest_version = versions[-1]
    metrics_file = latest_version / "metrics.csv"
    if metrics_file.exists():
        shutil.move(str(metrics_file), str(log_dir / target_name))
    shutil.rmtree(str(latest_version))


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


def write_evaluation(
    name: str,
    metrics: dict[str, float],
    targets: object,
    predictions: object,
    classes: list[str] | None,
    config: dict[str, Any],
    directory: Path | None = None,
) -> Path:
    """Write local evaluation evidence atomically, without network access."""
    from sklearn.metrics import confusion_matrix

    from jute_disease.config import ML_MODELS_DIR

    directory = directory or ML_MODELS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    class_names = list(classes) if classes is not None else None
    labels = list(range(len(class_names))) if class_names else None
    payload = {
        "config": config,
        "metrics": metrics,
        "classes": class_names,
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=labels
        ).tolist(),
    }
    destination = directory / f"{name}.json"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=directory, suffix=".tmp", delete=False, encoding="utf-8"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
    temporary.replace(destination)
    return destination
