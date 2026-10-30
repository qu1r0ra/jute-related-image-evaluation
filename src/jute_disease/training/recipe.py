"""Training values of the corrected path, read from ``configs/corrected/training.yaml``.

The optimizer, scheduler, early-stopping and compile values are not final until
the owner approves them, so no code or test holds a copy. Both the Lightning
``train dl`` recipes and source pretraining take them from that one file.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from jute_disease.config import ROOT_DIR

CORRECTED_DIR = ROOT_DIR / "configs" / "corrected"
TRAINING_CONFIG = CORRECTED_DIR / "training.yaml"

_CALLBACKS = "lightning.pytorch.callbacks."


@dataclass(frozen=True)
class TrainingValues:
    """Values every corrected fit shares, as named in ``training.yaml``."""

    lr: float
    weight_decay: float
    scheduler_patience: int
    scheduler_factor: float
    compile_model: bool
    monitor: str
    mode: str
    early_stopping_patience: int


def load_training_values(path: Path | str = TRAINING_CONFIG) -> TrainingValues:
    """Read the shared training values.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: an entry is missing.
    """
    config: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    model = config.get("model", {})
    callbacks = {
        item["class_path"].removeprefix(_CALLBACKS): item["init_args"]
        for item in config.get("trainer", {}).get("callbacks", [])
    }
    stopping = callbacks.get("EarlyStopping", {})
    try:
        return TrainingValues(
            lr=float(model["lr"]),
            weight_decay=float(model["weight_decay"]),
            scheduler_patience=int(model["scheduler_patience"]),
            scheduler_factor=float(model["scheduler_factor"]),
            compile_model=bool(model["compile_model"]),
            monitor=str(stopping["monitor"]),
            mode=str(stopping["mode"]),
            early_stopping_patience=int(stopping["patience"]),
        )
    except KeyError as error:
        raise ValueError(f"{path} is missing the entry {error}") from error


def with_training_config(args: Sequence[str]) -> list[str]:
    """Put ``training.yaml`` before the first corrected recipe in ``args``.

    A corrected recipe holds the data, backbone and batch settings only, so a
    fit that named one without the shared file would silently fall back to
    class defaults. Arguments without a corrected recipe pass through unchanged.
    """
    arguments = list(args)
    for index, argument in enumerate(arguments):
        if argument == "--config" and index + 1 < len(arguments):
            value = arguments[index + 1]
        elif argument.startswith("--config="):
            value = argument.partition("=")[2]
        else:
            continue
        path = Path(value).resolve()
        if path == TRAINING_CONFIG.resolve():
            return arguments
        if path.is_relative_to(CORRECTED_DIR):
            return [
                *arguments[:index],
                "--config",
                str(TRAINING_CONFIG),
                *arguments[index:],
            ]
    return arguments
