"""The corrected training path never opens the held-out test partition.

Test pixels belong to the final evaluator (#721) only. These regressions drive
the public training entry points and the command line, and spy on every
partition a manifest dataset is asked to open.
"""

import ast
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jute_disease import cli
from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import SyntheticPlan, freeze_target
from jute_disease.data import readers as data_readers
from jute_disease.models.ml import features
from jute_disease.training import baselines, grid_search
from jute_disease.training.engines.ml.train import train_ml
from jute_disease.training.partitions import (
    HeldOutAccessError,
    open_training_partition,
)

TRAINING_PACKAGE = Path(cli.__file__).parent / "training"


@pytest.fixture(scope="module")
def frozen_target(
    synthetic: SyntheticPlan, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    base = tmp_path_factory.mktemp("held-out")
    rules_path = base / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    freeze_target(
        base / "freeze",
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
    )
    return base / "freeze"


@pytest.fixture
def opened_partitions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    original = data_readers.ManifestDataset.__init__

    def spy(self: object, manifest: object, partition: str, *args: object, **kw: object):
        opened.append(partition)
        original(self, manifest, partition, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(data_readers.ManifestDataset, "__init__", spy)
    return opened


def test_classical_training_reads_only_train_and_validation(
    synthetic: SyntheticPlan,
    frozen_target: Path,
    opened_partitions: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(features, "ML_FEATURES_DIR", tmp_path / "features")
    model_path = train_ml(
        classifier="gnb",
        feature_type="crafted",
        manifest_dir=frozen_target / "target",
        image_root=synthetic.root,
        output_dir=tmp_path / "models",
    )
    assert model_path.is_file()
    assert sorted(set(opened_partitions)) == ["train", "val"]


def test_training_partition_gate_refuses_test_and_unknown_names(
    synthetic: SyntheticPlan, frozen_target: Path
) -> None:
    for name in ("test", "predict", "anything"):
        with pytest.raises(HeldOutAccessError):
            open_training_partition(
                name,
                manifest_dir=frozen_target / "target",
                image_root=synthetic.root,
            )
    for name in ("train", "val"):
        dataset = open_training_partition(
            name, manifest_dir=frozen_target / "target", image_root=synthetic.root
        )
        assert len(dataset) > 0


@pytest.mark.parametrize("subcommand", ["test", "predict"])
def test_train_dl_refuses_heldout_subcommands(
    subcommand: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lightning = MagicMock()
    monkeypatch.setattr(cli, "LightningCLI", lightning)
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "dl", subcommand, "--config", "x.yaml"])
    assert "#721" in str(excinfo.value)
    lightning.assert_not_called()


@pytest.mark.parametrize("subcommand", ["fit", "validate"])
def test_train_dl_still_forwards_training_subcommands(
    subcommand: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lightning = MagicMock()
    monkeypatch.setattr(cli, "LightningCLI", lightning)
    main(["train", "dl", subcommand, "--config", "x.yaml"])
    assert lightning.call_args.kwargs["args"] == [subcommand, "--config", "x.yaml"]


def test_evaluate_ml_is_owned_by_the_final_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["evaluate", "ml", "--classifier", "rf"])
    assert "#721" in str(excinfo.value)


def _commands(mock_run: MagicMock) -> list[list[str]]:
    return [list(call.args[0]) for call in mock_run.call_args_list]


def test_baseline_runner_issues_no_test_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "net.yaml"
    config.write_text("trainer: {}\n", encoding="utf-8")
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(baselines, "flatten_log_version", MagicMock())
    monkeypatch.setattr(baselines.wandb.util, "generate_id", lambda: "id")

    baselines.run_all_dl(config_file=config)

    commands = _commands(run)
    assert commands
    for command in commands:
        assert "test" not in command
        assert "predict" not in command


def test_grid_search_issues_no_test_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "transfer_learning_levels:\n  - {name: l1, weights_path: imagenet}\n"
        "dropout_rates: [0.1]\n"
        "fixed_params: {input_resolution: [256, 256], batch_size: 32, max_epochs: 1}\n",
        encoding="utf-8",
    )
    run = MagicMock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(grid_search, "flatten_log_version", MagicMock())
    monkeypatch.setattr(grid_search, "_aggregate_metrics", MagicMock())
    monkeypatch.setattr(grid_search, "CHECKPOINTS_DIR", tmp_path / "ck")
    monkeypatch.setattr(grid_search, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(grid_search, "ARTIFACTS_DIR", tmp_path / "art")
    monkeypatch.setattr(
        grid_search.wandb.util, "generate_id", lambda: "id", raising=False
    )
    base = tmp_path / "base.yaml"
    base.write_text("model: {}\n", encoding="utf-8")

    grid_search.run_grid_search(str(grid), str(base), fast_dev_run=True)

    for command in _commands(run):
        assert "test" not in command
        assert "predict" not in command


def test_training_package_names_no_test_partition() -> None:
    """Static guard: no training module spells the held-out partition."""
    offenders: list[str] = []
    for path in sorted(TRAINING_PACKAGE.rglob("*.py")):
        if path.name == "partitions.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in ("test", "predict"):
                offenders.append(f"{path.relative_to(TRAINING_PACKAGE)}:{node.lineno}")
    assert offenders == []
