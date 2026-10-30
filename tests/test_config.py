"""Unit tests for constants: types, values, and path consistency."""

from pathlib import Path

from jute_disease import config as config_module
from jute_disease.config import (
    BATCH_SIZE,
    BY_CLASS_DIR,
    DATA_DIR,
    DEFAULT_SEED,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    ML_MODELS_DIR,
    ML_SPLIT_DIR,
    NUM_WORKERS,
    ROOT_DIR,
    SPLITS,
    WANDB_ENTITY,
    WANDB_PROJECT,
)


def test_paths_are_path_objects() -> None:
    for p in [
        ROOT_DIR,
        DATA_DIR,
        ML_MODELS_DIR,
        BY_CLASS_DIR,
        ML_SPLIT_DIR,
    ]:
        assert isinstance(p, Path), f"{p!r} is not a Path"


def test_root_dir_is_project_root() -> None:
    """ROOT_DIR must be the repo root (contains pyproject.toml)."""
    assert (ROOT_DIR / "pyproject.toml").exists()


def test_paths_are_nested_under_root() -> None:
    assert str(DATA_DIR).startswith(str(ROOT_DIR))
    assert str(ML_MODELS_DIR).startswith(str(ROOT_DIR))


def test_splits_sum_to_one() -> None:
    assert abs(sum(SPLITS.values()) - 1.0) < 1e-9


def test_splits_has_required_keys() -> None:
    assert set(SPLITS.keys()) == {"train", "val", "test"}


def test_numeric_constants_are_positive() -> None:
    assert IMAGE_SIZE > 0
    assert BATCH_SIZE > 0
    assert NUM_WORKERS >= 0


def test_single_target_seed_is_42() -> None:
    """Owner decision: one target seed; the seed stays a command parameter."""
    assert DEFAULT_SEED == 42


def test_corrected_path_does_not_read_the_historical_seed_list() -> None:
    """SEEDS stays as published history; no source module reads it."""
    package = Path(config_module.__file__).parent
    readers = [
        path.relative_to(package).as_posix()
        for path in sorted(package.rglob("*.py"))
        if path.name != "config.py" and "SEEDS" in path.read_text(encoding="utf-8")
    ]
    assert readers == []


def test_image_extensions_are_strings() -> None:
    assert len(IMAGE_EXTENSIONS) > 0
    for ext in IMAGE_EXTENSIONS:
        assert ext.startswith("."), f"Extension {ext!r} must start with '.'"


def test_wandb_strings_are_nonempty() -> None:
    assert isinstance(WANDB_ENTITY, str) and WANDB_ENTITY
    assert isinstance(WANDB_PROJECT, str) and WANDB_PROJECT
