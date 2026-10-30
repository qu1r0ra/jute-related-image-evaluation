"""Command-shell checks: every command resolves and planned commands fail loudly."""

import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from jute_disease import cli
from jute_disease.cli import build_parser, main

# Forwarded commands (train dl, train pretrain, evaluate predict) are routed by main().
RESOLVABLE_COMMANDS = [
    ["data", "download"],
    ["data", "synthetic", "--per-class", "2", "--size", "8"],
    ["train", "ml"],
    ["train", "ml", "--all", "--classifiers", "rf", "--feature-types", "crafted"],
    ["train", "baselines"],
    ["train", "cv", "configs/historical/baselines/mobilenet_v2.yaml", "--folds", "2"],
    ["train", "grid", "configs/historical/grid/mobilenet_v2_grid.yaml"],
    ["evaluate", "aggregate", "--exp-names", "a"],
    ["evaluate", "ml", "--classifier", "rf"],
    ["freeze", "target"],
    ["queue", "plan", "--config", "queue.json"],
    ["queue", "run", "--plan", "plan.json", "--config", "queue.json", "--admit"],
    ["queue", "check", "--config", "queue.json", "--now", "2026-10-30T12:00:00+00:00"],
    ["freeze", "cache", "--budget-seconds", "5"],
    [
        "freeze",
        "source",
        "plantdoc",
        "some/dir",
        "--package-version",
        "7",
        "--package-digest",
        "0" * 64,
    ],
]

PLANNED_COMMANDS = [
    (["profile"], "#721"),
    (["evaluate"], "#721"),
]


@pytest.mark.parametrize("argv", RESOLVABLE_COMMANDS, ids=lambda a: " ".join(a[:3]))
def test_command_resolves_to_handler(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert callable(args.run)


def test_train_dl_forwards_lightning_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cli = MagicMock()
    monkeypatch.setattr(cli, "LightningCLI", mock_cli)
    main(["train", "dl", "fit", "--config", "x.yaml"])
    assert mock_cli.call_args.kwargs["args"] == ["fit", "--config", "x.yaml"]


def test_train_pretrain_forwards_leading_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_pretrain = MagicMock()
    monkeypatch.setattr(cli, "pretrain_main", mock_pretrain)
    main(["train", "pretrain", "--data_dir", "d", "--output_path", "o.ckpt"])
    mock_pretrain.assert_called_once_with(
        ["--data_dir", "d", "--output_path", "o.ckpt"]
    )


def test_evaluate_predict_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_predict = MagicMock()
    monkeypatch.setattr(cli, "predict_ml", mock_predict)
    main(["evaluate", "predict", "leaf.jpg", "--classifier", "knn"])
    mock_predict.assert_called_once_with(["leaf.jpg", "--classifier", "knn"])


@pytest.mark.parametrize(
    ("argv", "owner"),
    PLANNED_COMMANDS,
    ids=lambda v: " ".join(v) if isinstance(v, list) else v,
)
def test_planned_command_exits_nonzero_naming_owner(
    argv: list[str], owner: str
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert isinstance(excinfo.value.code, str)
    assert owner in excinfo.value.code


def test_missing_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("missing", ["--package-version", "--package-digest"])
def test_source_intake_requires_both_package_identity_inputs(missing: str) -> None:
    argv = ["freeze", "source", "plantdoc", "source-root"]
    if missing != "--package-version":
        argv.extend(["--package-version", "7"])
    if missing != "--package-digest":
        argv.extend(["--package-digest", "0" * 64])
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(argv)
    assert excinfo.value.code == 2


def test_module_entry_point_prints_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "jute_disease.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "jute-disease" in result.stdout
