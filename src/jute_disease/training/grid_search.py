"""Run hyperparameter optimization searches for Deep Learning models."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from jute_disease.config import ARTIFACTS_DIR, CHECKPOINTS_DIR, LOGS_DIR
from jute_disease.training.logger import flatten_log_version, get_logger

logger = get_logger(__name__)


def _aggregate_metrics(exp_names: list[str], output_csv: Path) -> None:
    """Consolidates metrics from all CSVLogger outputs into a master CSV."""
    results = []

    for exp in exp_names:
        log_dir = LOGS_DIR / exp
        if not log_dir.exists():
            continue

        # Concatenate all versions to merge fit and test outputs
        metrics_files = list(log_dir.glob("version_*/metrics.csv"))
        if not metrics_files:
            continue

        dfs = [pd.read_csv(f) for f in metrics_files]
        if not dfs:
            continue

        df = pd.concat(dfs, ignore_index=True)

        summary = {"Experiment": exp}

        if "val_loss" in df.columns:
            val_df = df.dropna(subset=["val_loss"])
            if not val_df.empty:
                # Extract best metrics based on validation loss
                best_row = val_df.loc[val_df["val_loss"].idxmin()].to_dict()
                for key, val in best_row.items():
                    if (
                        key.startswith("val_")
                        or key.startswith("train_")
                        or key in ["epoch", "step"]
                    ):
                        summary[key] = val

        for col in ["train_acc", "train_loss"]:
            if col in df.columns and (col not in summary or pd.isna(summary[col])):
                summary[col] = df[col].max() if "acc" in col else df[col].min()

        if "test_loss" in df.columns:
            test_df = df.dropna(subset=["test_loss"])
            if not test_df.empty:
                test_metrics = test_df.iloc[-1].to_dict()
                for key, val in test_metrics.items():
                    if key.startswith("test_"):
                        summary[key] = val

        results.append(summary)

    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv(output_csv, index=False)
        logger.info(f"Summary metrics exported to {output_csv}")
    else:
        logger.warning(f"No metrics found to export to {output_csv}")


def _get_modified_base_config(
    base_config_path: str | Path,
    exp_name: str,
    save_dir_suffix: str = "",
    patience: int | None = None,
) -> Path:
    """Read base config, update loggers/checkpoints, and return temp path."""
    with open(base_config_path) as f:
        config = yaml.safe_load(f) or {}

    trainer_cfg = config.setdefault("trainer", {})

    for cb in trainer_cfg.get("callbacks", []):
        class_path = cb.get("class_path", "")
        if "ModelCheckpoint" in class_path:
            cb.setdefault("init_args", {})["dirpath"] = str(CHECKPOINTS_DIR / exp_name)
        elif "EarlyStopping" in class_path and patience is not None:
            cb.setdefault("init_args", {})["patience"] = patience

    loggers = []

    save_dir = f"{LOGS_DIR}/{save_dir_suffix}"
    loggers.append(
        {
            "class_path": "lightning.pytorch.loggers.CSVLogger",
            "init_args": {"save_dir": save_dir, "name": exp_name},
        }
    )
    trainer_cfg["logger"] = loggers

    temp_dir = CHECKPOINTS_DIR / ".temp_configs"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{exp_name}.yaml"
    with open(temp_path, "w") as f:
        yaml.dump(config, f)
    return Path(temp_path)


def run_grid_search(
    grid_config_path: str | Path,
    base_config_path: str | Path | None = None,
    fast_dev_run: bool = False,
) -> None:
    """
    Reads grid search config and executes training runs using a baseline model config.
    """
    logger.info(f"Reading grid config from {grid_config_path}...")
    with open(grid_config_path) as f:
        grid_config = yaml.safe_load(f)

    if base_config_path is None:
        grid_stem = Path(grid_config_path).stem.replace("_grid", "")
        base_config_path = Path("configs/baselines") / f"{grid_stem}.yaml"
        if not base_config_path.exists():
            raise ValueError(
                f"Base config not found at {base_config_path}. "
                "Ensure your grid file is named <model_name>_grid.yaml "
                "or pass --base-config."
            )

    model_name = Path(base_config_path).stem.capitalize()
    logger.info(f"Using base model config: {base_config_path} ({model_name})")

    # Define search space from grid config
    transfer_levels = grid_config.get("transfer_learning_levels", [])
    dropout_rates = grid_config.get("dropout_rates", [])

    learning_rates = grid_config.get("learning_rate", [])
    weight_decays = grid_config.get("weight_decay", [])
    locked_params = grid_config.get("locked_params", {})

    fixed_params = grid_config.get("fixed_params", {})
    if not weight_decays and "weight_decay" in fixed_params:
        weight_decays = [fixed_params["weight_decay"]]

    # Detect Mode: Phase 1 or Phase 2
    is_phase_two = len(learning_rates) > 0 and len(locked_params) > 0

    if is_phase_two:
        logger.info(f"Detected Phase 2 Optimizer Grid Search for {model_name}.")
        level_name = locked_params.get("name", "locked")
        weights_path = locked_params.get("weights_path", "imagenet")
        dropout = locked_params.get("dropout_rate", 0.0)

        run_exp_names = []

        for lr in learning_rates:
            for wd in weight_decays:
                logger.info(
                    f"\n=== Running Phase 2: {model_name} | LR: {lr} | WD: {wd} ==="
                )

                ckpt_arg = weights_path if weights_path != "imagenet" else "null"
                pretrained_arg = str(weights_path == "imagenet")

                short_level = level_name.replace("level_", "l")
                exp_name = f"{model_name.lower()}-{short_level}-lr_{lr}-wd_{wd}"
                log_group = "phase2_finetune_grid"
                run_exp_names.append(exp_name)
                patience = fixed_params.get("early_stopping_patience")
                exp_config_path = _get_modified_base_config(
                    base_config_path,
                    exp_name,
                    log_group,
                    patience=patience,
                )

                cmd = [
                    sys.executable,
                    "-m",
                    "jute_disease.cli",
                    "train",
                    "dl",
                    "fit",
                    "--config",
                    str(exp_config_path),
                    f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                    f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                    f"--model.feature_extractor.init_args.drop_rate={dropout}",
                    f"--model.lr={lr}",
                    f"--model.weight_decay={wd}",
                ]

                if "num_folds" in fixed_params:
                    cmd.append(f"--data.k_fold={fixed_params['num_folds']}")
                if "batch_size" in fixed_params:
                    cmd.append(f"--data.batch_size={fixed_params['batch_size']}")
                if "max_epochs" in fixed_params:
                    cmd.append(f"--trainer.max_epochs={fixed_params['max_epochs']}")

                if fast_dev_run:
                    cmd.append("--trainer.fast_dev_run=True")

                logger.info(f"Command (Fit): {' '.join(cmd)}")
                try:
                    subprocess.run(cmd, check=True)
                    flatten_log_version(
                        LOGS_DIR / log_group / exp_name, "train-metrics.csv"
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error running Phase 2 experiment lr{lr}_wd{wd}: {e}")
                    continue

                logger.info(f"Testing Phase 2 experiment {exp_name}...")

                ckpt_dir = CHECKPOINTS_DIR / exp_name
                # Locate the newly trained model checkpoint
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                if not ckpts:
                    logger.error(
                        f"No checkpoint found for Phase 2 {exp_name} in {ckpt_dir}."
                    )
                    continue

                best_ckpt = ckpts[0]

                test_cmd = [
                    sys.executable,
                    "-m",
                    "jute_disease.cli",
                    "train",
                    "dl",
                    "test",
                    "--config",
                    str(exp_config_path),
                    "--ckpt_path",
                    str(best_ckpt),
                ]

                logger.info(f"Command (Test): {' '.join(test_cmd)}")
                try:
                    subprocess.run(test_cmd, check=True)
                    flatten_log_version(
                        LOGS_DIR / log_group / exp_name, "test-metrics.csv"
                    )
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error testing Phase 2 experiment {exp_name}: {e}")

        if run_exp_names:
            _aggregate_metrics(
                run_exp_names,
                output_csv=LOGS_DIR
                / "phase2_finetune_grid"
                / f"aggregated_grid_metrics_{model_name.lower()}.csv",
            )
        return

    # Phase 1 Execution
    run_exp_names = []
    for level in transfer_levels:
        level_name = level["name"]
        weights_path = level["weights_path"]

        for dropout in dropout_rates:
            logger.info(
                f"\n=== Running Experiment: {model_name} | {level_name} "
                f"| Dropout: {dropout} ==="
            )

            ckpt_arg = weights_path if weights_path != "imagenet" else "null"
            pretrained_arg = str(weights_path == "imagenet")

            short_level = level_name.replace("level_", "l")
            exp_name = f"{model_name.lower()}-{short_level}-dr_{dropout}"
            log_group = "phase1_transfer_grid"
            run_exp_names.append(exp_name)
            patience = fixed_params.get("early_stopping_patience")
            exp_config_path = _get_modified_base_config(
                base_config_path,
                exp_name,
                log_group,
                patience=patience,
            )

            cmd = [
                sys.executable,
                "-m",
                "jute_disease.cli",
                "train",
                "dl",
                "fit",
                "--config",
                str(exp_config_path),
                f"--model.feature_extractor.init_args.checkpoint_path={ckpt_arg}",
                f"--model.feature_extractor.init_args.pretrained={pretrained_arg}",
                f"--model.feature_extractor.init_args.drop_rate={dropout}",
            ]

            if "learning_rate" in fixed_params:
                cmd.append(f"--model.lr={fixed_params['learning_rate']}")
            if "weight_decay" in fixed_params:
                cmd.append(f"--model.weight_decay={fixed_params['weight_decay']}")
            if "num_folds" in fixed_params:
                cmd.append(f"--data.k_fold={fixed_params['num_folds']}")
            if "batch_size" in fixed_params:
                cmd.append(f"--data.batch_size={fixed_params['batch_size']}")
            if "max_epochs" in fixed_params:
                cmd.append(f"--trainer.max_epochs={fixed_params['max_epochs']}")

            if fast_dev_run:
                cmd.append("--trainer.fast_dev_run=True")

            logger.info(f"Command (Fit): {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                flatten_log_version(
                    LOGS_DIR / log_group / exp_name, "train-metrics.csv"
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"Fit failed for {exp_name}: {e}")
                continue

            logger.info(f"Testing Phase 1 experiment {exp_name}...")

            ckpt_dir = CHECKPOINTS_DIR / exp_name
            # Identify best checkpoint for evaluation
            ckpts = list(ckpt_dir.glob("*.ckpt"))
            if not ckpts:
                logger.error(
                    f"No checkpoint found for Phase 1 {exp_name} in {ckpt_dir}."
                )
                continue

            best_ckpt = ckpts[0]

            test_cmd = [
                sys.executable,
                "-m",
                "jute_disease.cli",
                "train",
                "dl",
                "test",
                "--config",
                str(exp_config_path),
                "--ckpt_path",
                str(best_ckpt),
            ]

            logger.info(f"Command (Test): {' '.join(test_cmd)}")
            try:
                subprocess.run(test_cmd, check=True)
                flatten_log_version(LOGS_DIR / log_group / exp_name, "test-metrics.csv")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error testing Phase 1 experiment {exp_name}: {e}")

    _aggregate_metrics(
        run_exp_names,
        output_csv=ARTIFACTS_DIR
        / f"grid_search_{model_name.lower()}_phase1_metrics.csv",
    )
