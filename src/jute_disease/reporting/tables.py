"""Aggregate the validation metrics of finished fits into a summary table."""

from pathlib import Path

import pandas as pd

from jute_disease.config import RUNS_DIR
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)

METRICS_PATH = Path("metrics") / "version_0" / "metrics.csv"


def _latest_metrics(fit_dir: Path) -> Path | None:
    """The metrics file of the highest numbered attempt that wrote one."""
    attempts = sorted(
        (
            int(child.name.removeprefix("attempt-")),
            child,
        )
        for child in fit_dir.glob("attempt-*")
        if child.name.removeprefix("attempt-").isdigit()
    )
    for _, attempt in reversed(attempts):
        if (attempt / METRICS_PATH).is_file():
            return attempt / METRICS_PATH
    return None


def aggregate_metrics(
    fit_ids: list[str], output_csv: Path, runs_dir: Path = RUNS_DIR
) -> None:
    """Write the best-validation-epoch metrics of each fit to one CSV.

    Reads ``<runs_dir>/<fit id>/attempt-N/metrics/version_0/metrics.csv`` from the
    latest attempt. Training never records test metrics, so none appear here.
    """
    results = []
    for fit_id in fit_ids:
        path = _latest_metrics(Path(runs_dir) / fit_id)
        if path is None:
            continue
        df = pd.read_csv(path)
        summary: dict[str, object] = {"Experiment": fit_id}

        if "val_loss" in df.columns:
            val_df = df.dropna(subset=["val_loss"])
            if not val_df.empty:
                best_row = val_df.loc[val_df["val_loss"].idxmin()].to_dict()
                for key, val in best_row.items():
                    if key.startswith(("val_", "train_")) or key in ["epoch", "step"]:
                        summary[key] = val

        for col in ["train_acc", "train_loss"]:
            if col in df.columns and (col not in summary or pd.isna(summary[col])):
                summary[col] = df[col].max() if "acc" in col else df[col].min()

        results.append(summary)

    if results:
        pd.DataFrame(results).to_csv(output_csv, index=False)
        logger.info(f"Summary metrics exported to {output_csv}")
    else:
        logger.warning(f"No metrics found to export to {output_csv}")
