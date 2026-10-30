"""Source pretraining takes its training values from configs/corrected/training.yaml.

These run the real training entry point on CPU against a synthetic source package.
"""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    PLANTDOC,
    SOURCES,
    TARGET_DIR,
    SyntheticPlan,
    build_inventory,
    freeze_target,
    intake_source,
    inventory_digest,
    make_synthetic_data,
    observed_counts,
)
from jute_disease.training import TRAINING_CONFIG, load_training_values
from jute_disease.training.engines import train_pretext_task


@pytest.fixture(scope="module")
def rules_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    path = tmp_path_factory.mktemp("recipe-rules") / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return path


@pytest.fixture
def pinned_source(
    synthetic: SyntheticPlan,
    rules_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    """Return the freeze directory and the package of an owner-authorized source."""
    output = tmp_path / "freeze"
    freeze_target(
        output,
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
    )
    package = tmp_path / "package"
    make_synthetic_data(package / "train", per_class=20, labels=("a", "b"), seed=500)
    make_synthetic_data(package / "test", per_class=10, labels=("a", "b"), seed=600)
    for duplicate in (package / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    records = build_inventory(package, workers=2)
    spec = replace(
        PLANTDOC,
        expected=tuple(sorted(observed_counts(records).items())),
        expected_inventory_digest=inventory_digest(records),
    )
    monkeypatch.setitem(SOURCES, "plantdoc", spec)
    intake_source(
        "plantdoc",
        package,
        output,
        package_version=7,
        package_digest=inventory_digest(records),
    )
    return output, package


def run_pretrain(
    freeze: Path, package: Path, output_path: Path, **overrides: object
) -> None:
    options: dict[str, object] = {
        "pretrained": False,
        "epochs": 2,
        "batch_size": 4,
        "limit_batches": 3,
    }
    options.update(overrides)
    train_pretext_task(
        freeze / "source_plantdoc",
        package,
        freeze / TARGET_DIR,
        output_path,
        **options,
    )


def val_losses(logs: Path) -> list[float]:
    (metrics,) = logs.rglob("metrics.csv")
    with metrics.open(encoding="utf-8") as handle:
        return [
            float(row["val_loss"])
            for row in csv.DictReader(handle)
            if row.get("val_loss")
        ]


def test_pretraining_takes_its_training_values_from_the_corrected_file(
    pinned_source: tuple[Path, Path], tmp_path: Path
) -> None:
    freeze, package = pinned_source
    output = tmp_path / "out" / "source.ckpt"
    values = load_training_values()

    run_pretrain(freeze, package, output)

    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    hyper = checkpoint["hyper_parameters"]
    assert hyper["lr"] == values.lr
    assert hyper["weight_decay"] == values.weight_decay
    assert hyper["compile_model"] is values.compile_model
    assert hyper["scheduler_patience"] == values.scheduler_patience
    assert hyper["scheduler_factor"] == values.scheduler_factor
    assert hyper["freeze_backbone"] is False

    receipt = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt["monitor"] == values.monitor
    assert receipt["mode"] == values.mode
    assert receipt["early_stopping_patience"] == values.early_stopping_patience
    assert receipt["training_config"] == str(TRAINING_CONFIG)
    assert receipt["seed"] == 42
    assert receipt["num_workers"] == 0
    assert receipt["precision"] == "16-mixed"
    assert receipt["best_val_loss"] == pytest.approx(
        min(val_losses(tmp_path / "out")), rel=1e-4
    )
    assert Path(receipt["best_model_path"]).is_file()


def test_pretraining_follows_a_changed_training_file(
    pinned_source: tuple[Path, Path], tmp_path: Path
) -> None:
    freeze, package = pinned_source
    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    config["model"]["lr"] = 0.00123
    config["model"]["scheduler_patience"] = 1
    for item in config["trainer"]["callbacks"]:
        if item["class_path"].endswith("EarlyStopping"):
            item["init_args"]["patience"] = 2
    changed = tmp_path / "training.yaml"
    changed.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "out" / "source.ckpt"

    run_pretrain(freeze, package, output, epochs=1, training_config=changed)

    hyper = torch.load(output, map_location="cpu", weights_only=False)[
        "hyper_parameters"
    ]
    assert hyper["lr"] == 0.00123
    assert hyper["scheduler_patience"] == 1
    receipt = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt["early_stopping_patience"] == 2
    assert receipt["training_config"] == str(changed)


def test_pretraining_never_logs_to_wandb(
    pinned_source: tuple[Path, Path], tmp_path: Path
) -> None:
    freeze, package = pinned_source
    run_pretrain(freeze, package, tmp_path / "out" / "source.ckpt", epochs=1)
    assert not (tmp_path / "out" / "wandb").exists()
    assert not Path("wandb").exists()


def test_pretraining_continues_from_a_checkpoint_strictly(
    pinned_source: tuple[Path, Path], tmp_path: Path
) -> None:
    freeze, package = pinned_source
    first = tmp_path / "first" / "source.ckpt"
    run_pretrain(freeze, package, first, epochs=1)

    run_pretrain(
        freeze, package, tmp_path / "second" / "source.ckpt", base_checkpoint=first
    )
    assert (tmp_path / "second" / "source.ckpt").is_file()

    broken = tmp_path / "broken.ckpt"
    state = torch.load(first, map_location="cpu", weights_only=False)
    state["state_dict"] = {
        key: value
        for key, value in state["state_dict"].items()
        if key != "feature_extractor.backbone.conv_stem.weight"
    }
    torch.save(state, broken)
    with pytest.raises(RuntimeError, match="Missing key"):
        run_pretrain(
            freeze,
            package,
            tmp_path / "third" / "source.ckpt",
            base_checkpoint=broken,
        )
