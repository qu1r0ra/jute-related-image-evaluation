"""``train dl`` runs the corrected recipes through Lightning's real CLI parsing.

Nothing here mocks LightningCLI: arguments are parsed, the recipe and the shared
training file are merged, and a bounded fit runs on CPU against a synthetic freeze.
"""

import json
from pathlib import Path

import pytest
import torch
import yaml

from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import SyntheticPlan, freeze_target
from jute_disease.data import readers as data_readers
from jute_disease.training import TRAINING_CONFIG, load_training_values, recipe
from jute_disease.training.queue import RECIPES_DIR

CORRECTED = Path(__file__).resolve().parents[2] / RECIPES_DIR
RECIPE = CORRECTED / "core" / "mobilenet_v2.yaml"


@pytest.fixture(scope="module")
def frozen_target(
    synthetic: SyntheticPlan, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    base = tmp_path_factory.mktemp("train-dl")
    rules_path = base / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    freeze_target(
        base / "freeze",
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
    )
    return base / "freeze" / "target"


@pytest.fixture
def opened_partitions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    original = data_readers.ManifestDataset.__init__

    def spy(
        self: object, manifest: object, partition: str, *args: object, **kw: object
    ) -> None:
        opened.append(partition)
        original(self, manifest, partition, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(data_readers.ManifestDataset, "__init__", spy)
    return opened


def best_checkpoint(root: Path) -> Path:
    """Return the checkpoint a finished fit recorded under ``root``."""
    record = json.loads((root / "best_checkpoint.json").read_text(encoding="utf-8"))
    assert record["monitor"] == "val_loss"
    assert record["mode"] == "min"
    best = Path(record["path"])
    assert best.is_file()
    return best


def bounded_fit(
    synthetic: SyntheticPlan, frozen_target: Path, root: Path, *extra: str
) -> list[str]:
    return [
        "train",
        "dl",
        "fit",
        "--seed_everything",
        "42",
        "--config",
        str(RECIPE),
        "--trainer.default_root_dir",
        str(root),
        "--trainer.logger.init_args.save_dir",
        str(root / "logs"),
        "--trainer.accelerator",
        "cpu",
        "--trainer.max_epochs",
        "2",
        "--trainer.limit_train_batches",
        "3",
        "--trainer.limit_val_batches",
        "2",
        "--model.feature_extractor.init_args.pretrained",
        "false",
        "--data.manifest_dir",
        str(frozen_target),
        "--data.image_root",
        str(synthetic.root),
        "--data.batch_size",
        "4",
        *extra,
    ]


def test_merged_recipe_parses_with_the_shared_training_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["train", "dl", "fit", "--config", str(RECIPE), "--print_config"])
    assert excinfo.value.code == 0

    printed = yaml.safe_load(capsys.readouterr().out)
    values = load_training_values()
    assert printed["model"]["lr"] == values.lr
    assert printed["model"]["weight_decay"] == values.weight_decay
    assert printed["model"]["scheduler_patience"] == values.scheduler_patience
    assert printed["model"]["compile_model"] is values.compile_model
    assert printed["model"]["freeze_backbone"] is False
    assert printed["trainer"]["precision"] == "16-mixed"
    stopping = next(
        item
        for item in printed["trainer"]["callbacks"]
        if item["class_path"].endswith("EarlyStopping")
    )
    assert stopping["init_args"]["patience"] == values.early_stopping_patience
    assert printed["data"]["num_workers"] == 0
    assert printed["data"]["batch_size"] == 32


def test_command_line_overrides_win_over_the_recipe(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "train",
                "dl",
                "fit",
                "--config",
                str(RECIPE),
                "--data.batch_size",
                "16",
                "--print_config",
            ]
        )
    assert yaml.safe_load(capsys.readouterr().out)["data"]["batch_size"] == 16


def test_corrected_fit_trains_and_validates_without_opening_test(
    synthetic: SyntheticPlan,
    frozen_target: Path,
    opened_partitions: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "disabled")

    main(bounded_fit(synthetic, frozen_target, tmp_path))

    assert sorted(set(opened_partitions)) == ["train", "val"]
    best = best_checkpoint(tmp_path)
    hyper = torch.load(best, map_location="cpu", weights_only=False)["hyper_parameters"]
    values = load_training_values()
    assert hyper["lr"] == values.lr
    assert hyper["compile_model"] is values.compile_model


def test_fit_follows_a_changed_training_file(
    synthetic: SyntheticPlan,
    frozen_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The merge uses the shared file, so editing it changes the run."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    changed = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    changed["model"]["lr"] = 0.00321
    override = tmp_path / "training.yaml"
    override.write_text(yaml.safe_dump(changed), encoding="utf-8")
    monkeypatch.setattr(recipe, "TRAINING_CONFIG", override)
    root = tmp_path / "run"

    main(bounded_fit(synthetic, frozen_target, root))

    best = best_checkpoint(root)
    hyper = torch.load(best, map_location="cpu", weights_only=False)["hyper_parameters"]
    assert hyper["lr"] == 0.00321


@pytest.mark.parametrize("subcommand", ["test", "predict"])
def test_corrected_recipes_cannot_reach_the_test_partition(
    subcommand: str,
    synthetic: SyntheticPlan,
    frozen_target: Path,
    opened_partitions: list[str],
    tmp_path: Path,
) -> None:
    argv = bounded_fit(synthetic, frozen_target, tmp_path)
    argv[argv.index("fit")] = subcommand

    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert "#721" in str(excinfo.value)
    assert opened_partitions == []
