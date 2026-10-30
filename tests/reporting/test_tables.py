"""The summary reads the metrics files that training actually writes."""

from pathlib import Path

import pandas as pd

from jute_disease.reporting.tables import aggregate_metrics


def _write_metrics(runs: Path, fit_id: str, attempt: int, losses: list[float]) -> None:
    folder = runs / fit_id / f"attempt-{attempt}" / "metrics" / "version_0"
    folder.mkdir(parents=True)
    pd.DataFrame(
        {
            "epoch": range(len(losses)),
            "step": range(len(losses)),
            "val_loss": losses,
            "val_acc": [0.5 + 0.1 * i for i in range(len(losses))],
            "train_loss": losses,
        }
    ).to_csv(folder / "metrics.csv", index=False)


def test_summary_picks_the_best_validation_epoch_of_the_latest_attempt(
    tmp_path: Path,
) -> None:
    runs = tmp_path / "runs"
    _write_metrics(runs, "cnn/a/seed-42", 1, [9.0, 8.0])
    _write_metrics(runs, "cnn/a/seed-42", 2, [0.9, 0.4, 0.6])
    output = tmp_path / "summary.csv"

    aggregate_metrics(["cnn/a/seed-42", "cnn/missing/seed-1"], output, runs)

    summary = pd.read_csv(output)
    assert list(summary["Experiment"]) == ["cnn/a/seed-42"]
    assert summary.loc[0, "val_loss"] == 0.4
    assert summary.loc[0, "epoch"] == 1
    assert not [column for column in summary.columns if column.startswith("test_")]
