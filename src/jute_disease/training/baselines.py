"""Fit every DL baseline config sequentially; test scoring is the evaluator's."""

import os
import subprocess
import sys
from pathlib import Path

import wandb
import yaml

from jute_disease.training.logger import flatten_log_version, get_logger

logger = get_logger(__name__)

CONFIGS_DIR = Path("configs/historical/baselines")


def run_all_dl(
    configs_dir: Path = CONFIGS_DIR, config_file: Path | None = None
) -> None:
    """Execute training for a single configuration or iterate through all."""
    if config_file:
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        configs = [config_file]
    else:
        configs = sorted(configs_dir.glob("*.yaml"))

    if not configs:
        raise FileNotFoundError(f"No configs found in {configs_dir}")

    logger.info(f"Starting DL Training Pipeline — {len(configs)} configs found.")

    for config in configs:
        model_name = config.stem

        logger.info(f"Training {model_name} (config: {config})...")

        with open(config) as f:
            cfg = yaml.safe_load(f) or {}

        loggers = cfg.get("trainer", {}).get("logger", [])
        if isinstance(loggers, dict):
            loggers = [loggers]

        csv_save_dir = "artifacts/logs"
        csv_name = model_name
        for logger_cfg in loggers:
            if "CSVLogger" in logger_cfg.get("class_path", ""):
                init_args = logger_cfg.get("init_args", {})
                csv_save_dir = init_args.get("save_dir", csv_save_dir)
                csv_name = init_args.get("name", csv_name)
                break

        log_dir = Path(csv_save_dir) / csv_name

        run_id = wandb.util.generate_id()
        # Ensure WandB and CSVLogger share the same run context
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = run_id

        fit_cmd = [
            sys.executable,
            "-m",
            "jute_disease.cli",
            "train",
            "dl",
            "fit",
            "--config",
            str(config),
        ]
        result = subprocess.run(fit_cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"{model_name} failed during fit with exit code {result.returncode}."
            )
        flatten_log_version(log_dir, "train-metrics.csv")

        logger.info(f"Finished {model_name}.")

    logger.info("All DL experiments completed!")
