"""Run cross-validation training for a single DL model config."""

import subprocess
import sys
from pathlib import Path

import yaml

from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


def run_cross_validation(config: Path, folds: int | None = None) -> None:
    """Run K-Fold cross validation by sweep through fold indices."""
    model_name = config.stem

    if folds is None:
        with open(config) as f:
            cfg = yaml.safe_load(f) or {}
        folds = cfg.get("data", {}).get("k_fold", 1)
        logger.info(f"Read k_fold={folds} from config.")
    else:
        logger.info(f"Overriding k_fold with CLI argument: {folds}")

    logger.info(f"Starting cross-validation for {model_name} with {folds} folds.")

    for fold_idx in range(folds):
        logger.info(f"Running fold {fold_idx}/{folds - 1} for {model_name}...")

        cmd = [
            sys.executable, "-m", "jute_disease.cli", "train", "dl", "fit",
            "--config", str(config),
            f"--data.fold_index={fold_idx}",
            f"--data.k_fold={folds}",
            f"--trainer.logger.init_args.name={model_name}_fold_{fold_idx}",
            f"--trainer.callbacks.1.init_args.filename={model_name}-fold{fold_idx}-{{epoch:02d}}-{{val_loss:.4f}}",
            "--trainer.callbacks.1.init_args.monitor=val_loss",
            "--trainer.callbacks.1.init_args.mode=min",
            "--trainer.callbacks.1.init_args.save_top_k=1",
        ]  # fmt: skip

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Fold {fold_idx} failed with exit code {result.returncode}."
            )

        logger.info(f"Finished fold {fold_idx}.")

    logger.info(f"All {folds} folds completed for {model_name}.")
