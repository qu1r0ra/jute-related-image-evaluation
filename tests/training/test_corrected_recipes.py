"""The corrected recipes keep tunable values in one named file with sources.

The optimizer, scheduler, early-stopping and compile values are placeholders until
the owner approves them, so these tests check where they live and that nothing
else repeats them; they do not pin the numbers.
"""

import inspect
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from jute_disease.models.dl import Classifier
from jute_disease.training import (
    TRAINING_CONFIG,
    load_training_values,
    with_training_config,
)
from jute_disease.training.queue import RECIPES_DIR

CORRECTED = Path(__file__).resolve().parents[2] / RECIPES_DIR
CORE = {
    "mobilenet_v2": ("mobilenetv2_100.ra_in1k", 32, 1),
    "efficientnet_b5": ("efficientnet_b5.sw_in12k_ft_in1k", 8, 4),
}
ARMS = ("plantvillage", "plantdoc")
NAMED_ENTRIES = (
    "lr",
    "weight_decay",
    "scheduler_patience",
    "scheduler_factor",
    "compile_model",
    "patience",
)


def recipe_paths() -> list[Path]:
    return [
        *(CORRECTED / "core" / f"{name}.yaml" for name in CORE),
        *(CORRECTED / arm / "target.yaml" for arm in ARMS),
    ]


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_training_file_names_a_source_above_every_tunable_entry() -> None:
    lines = TRAINING_CONFIG.read_text(encoding="utf-8").splitlines()
    for entry in NAMED_ENTRIES:
        (index,) = [
            i for i, line in enumerate(lines) if re.match(rf"\s+{entry}:", line)
        ]
        assert re.match(r"\s*# Source: \S", lines[index - 1]), (
            f"{entry} needs a '# Source:' comment on the line above"
        )


def test_training_values_load_from_the_file_and_follow_it(tmp_path: Path) -> None:
    changed = tmp_path / "training.yaml"
    config = load(TRAINING_CONFIG)
    config["model"]["lr"] = 0.5
    config["model"]["compile_model"] = True
    for item in config["trainer"]["callbacks"]:
        if item["class_path"].endswith("EarlyStopping"):
            item["init_args"]["patience"] = 99
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")

    values = load_training_values(changed)

    assert values.lr == 0.5
    assert values.compile_model is True
    assert values.early_stopping_patience == 99
    assert values.monitor == "val_loss"
    assert values.mode == "min"


def test_a_missing_training_entry_is_named(tmp_path: Path) -> None:
    config = load(TRAINING_CONFIG)
    del config["model"]["weight_decay"]
    broken = tmp_path / "training.yaml"
    broken.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="weight_decay"):
        load_training_values(broken)


@pytest.mark.parametrize("path", recipe_paths(), ids=lambda p: "/".join(p.parts[-2:]))
def test_recipes_do_not_repeat_the_training_entries(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for entry in NAMED_ENTRIES:
        assert not re.search(rf"^\s+{entry}:", text, re.MULTILINE), entry
    assert "callbacks" not in load(path)["trainer"]


@pytest.mark.parametrize("path", recipe_paths(), ids=lambda p: "/".join(p.parts[-2:]))
def test_every_recipe_fixes_the_shared_setup_and_logs_locally(path: Path) -> None:
    config = load(path)
    model, trainer, data = config["model"], config["trainer"], config["data"]

    assert config["seed_everything"] == 42
    assert model["freeze_backbone"] is False
    assert model["num_classes"] == 6
    assert trainer["max_epochs"] == 50
    assert trainer["precision"] == "16-mixed"
    assert "gradient_clip_val" not in trainer
    assert trainer["logger"]["class_path"] == "lightning.pytorch.loggers.CSVLogger"
    assert data["num_workers"] == 0
    assert data["image_size"] == 256
    assert data["use_weighted_sampler"] is True
    assert data.get("k_fold", 1) == 1


@pytest.mark.parametrize(("name", "expected"), CORE.items())
def test_core_recipes_set_backbone_batch_and_accumulation(
    name: str, expected: tuple[str, int, int]
) -> None:
    config = load(CORRECTED / "core" / f"{name}.yaml")
    backbone = config["model"]["feature_extractor"]["init_args"]
    model_name, batch, accumulation = expected
    assert backbone["model_name"] == model_name
    assert backbone["pretrained"] is True
    assert "checkpoint_path" not in backbone
    assert config["data"]["batch_size"] == batch
    assert config["trainer"]["accumulate_grad_batches"] == accumulation


@pytest.mark.parametrize("arm", ARMS)
def test_arm_target_recipes_are_mobilenet_and_load_the_source_strictly(
    arm: str,
) -> None:
    config = load(CORRECTED / arm / "target.yaml")
    backbone = config["model"]["feature_extractor"]["init_args"]
    assert backbone["model_name"] == "mobilenetv2_100.ra_in1k"
    assert backbone["strict_checkpoint"] is True
    assert config["data"]["batch_size"] == 32
    assert config["trainer"]["accumulate_grad_batches"] == 1


def test_classifier_defaults_stay_historical() -> None:
    """Shared class defaults keep the submitted behavior; the corrected file overrides."""
    defaults = inspect.signature(Classifier.__init__).parameters
    assert defaults["lr"].default == 0.001
    assert defaults["compile_model"].default is True


def test_a_corrected_recipe_gets_the_training_file_merged_in_first() -> None:
    recipe = str(CORRECTED / "core" / "mobilenet_v2.yaml")
    args = ["--seed_everything", "42", "fit", "--config", recipe, "--x", "1"]

    merged = with_training_config(args)

    assert merged == [
        "--seed_everything",
        "42",
        "fit",
        "--config",
        str(TRAINING_CONFIG),
        "--config",
        recipe,
        "--x",
        "1",
    ]
    assert with_training_config(merged) == merged
    historical = ["fit", "--config", "configs/historical/baselines/mobilenet_v2.yaml"]
    assert with_training_config(historical) == historical
