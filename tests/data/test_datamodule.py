"""Unit tests for the DataModule."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from jute_disease.data import DataModule, Manifest, write_manifest
from jute_disease.data.manifest import RECEIPT_FILE
from jute_disease.data.synthetic import SyntheticPlan


@pytest.fixture
def mock_dataset_root(tmp_path: Path) -> Path:
    """Create a mock data directory structure for testing DataModule."""
    data_dir = tmp_path / "data"
    for split in ["train", "val", "test"]:
        for cls in ["healthy", "diseased"]:
            cls_dir = data_dir / split / cls
            cls_dir.mkdir(parents=True)
            for i in range(5):
                img = Image.fromarray(
                    np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
                )
                img.save(cls_dir / f"img_{i}.jpg")
    return data_dir


def test_datamodule_setup_fit(mock_dataset_root: Path) -> None:
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, pin_memory=False)
    dm.setup(stage="fit")

    assert dm.jute_train is not None
    assert dm.jute_val is not None
    assert len(dm.jute_train) == 10  # 2 classes * 5 images
    assert len(dm.jute_val) == 10
    assert dm.classes == ["diseased", "healthy"]
    assert dm.num_classes == 2


def test_datamodule_setup_test(mock_dataset_root: Path) -> None:
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, pin_memory=False)
    dm.setup(stage="test")

    assert dm.jute_test is not None
    assert len(dm.jute_test) == 10
    assert dm.classes == ["diseased", "healthy"]


def test_datamodule_dataloaders(mock_dataset_root: Path) -> None:
    dm = DataModule(data_dir=mock_dataset_root, batch_size=2, pin_memory=False)
    dm.setup()
    dm.setup(stage="test")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    assert len(next(iter(train_loader))[0]) == 2
    assert len(next(iter(val_loader))[0]) == 4  # batch_size * 2 for val
    assert len(next(iter(test_loader))[0]) == 4


def test_data_dir_rejects_kfold(mock_dataset_root: Path) -> None:
    with pytest.raises(ValueError, match="group-aware"):
        DataModule(data_dir=mock_dataset_root, k_fold=5)


def test_source_argument_is_required_and_exclusive(mock_dataset_root: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        DataModule()
    with pytest.raises(ValueError, match="exactly one"):
        DataModule(data_dir=mock_dataset_root, manifest_dir=mock_dataset_root)
    with pytest.raises(ValueError, match="image_root"):
        DataModule(manifest_dir=mock_dataset_root)


def test_datamodule_weighted_sampler(mock_dataset_root: Path) -> None:
    # Imbalance: diseased=5, healthy=10 in train
    for i in range(5, 10):
        img = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
        img.save(mock_dataset_root / "train" / "healthy" / f"extra_{i}.jpg")

    dm = DataModule(
        data_dir=mock_dataset_root,
        batch_size=2,
        use_weighted_sampler=True,
        pin_memory=False,
    )
    dm.setup(stage="fit")

    assert dm.sampler is not None
    assert len(dm.sampler) == 15  # 5 + 10


def test_datamodule_random_split(tmp_path: Path) -> None:
    """Test standard random splitting when subfolders are missing."""
    data_dir = tmp_path / "external_data"
    for cls in ["apple", "banana", "cherry"]:
        cls_dir = data_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            img = Image.fromarray(
                np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"img_{i}.jpg")

    # 30 images total, 20% val -> 24 train, 6 val
    dm = DataModule(data_dir=data_dir, val_split=0.2, batch_size=2, pin_memory=False)
    dm.setup(stage="fit")

    assert dm.jute_train is not None
    assert dm.jute_val is not None
    assert dm.num_classes == 3
    assert len(dm.jute_train) == 24
    assert len(dm.jute_val) == 6


def test_datamodule_custom_resolution(mock_dataset_root: Path) -> None:
    """Test that DataModule respects custom image resolutions."""
    dm = DataModule(data_dir=mock_dataset_root, image_size=512, pin_memory=False)
    dm.setup(stage="fit")

    # Get a batch and check the resolution
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    imgs, _ = batch

    assert imgs.shape[2:] == (512, 512)


# --- Frozen-manifest mode ---


@pytest.fixture(scope="module")
def frozen_dir(
    tmp_path_factory: pytest.TempPathFactory, synthetic: SyntheticPlan, frozen: Manifest
) -> Path:
    directory = tmp_path_factory.mktemp("freeze")
    write_manifest(frozen, directory)
    return directory


def _manifest_module(
    frozen_dir: Path, synthetic: SyntheticPlan, **kwargs
) -> DataModule:
    return DataModule(
        manifest_dir=frozen_dir,
        image_root=synthetic.root,
        batch_size=4,
        pin_memory=False,
        num_workers=0,
        image_size=32,
        **kwargs,
    )


def test_manifest_mode_serves_exactly_the_frozen_partitions(
    frozen_dir: Path, synthetic: SyntheticPlan, frozen: Manifest
) -> None:
    dm = _manifest_module(frozen_dir, synthetic)
    dm.setup()
    assert len(dm.jute_train) == len(frozen.partition("train"))
    assert len(dm.jute_val) == len(frozen.partition("val"))
    assert len(dm.jute_test) == len(frozen.partition("test"))
    assert dm.classes == list(frozen.labels)
    images, _ = next(iter(dm.train_dataloader()))
    assert images.shape[1:] == (3, 32, 32)


def test_manifest_mode_refuses_a_directory_without_receipt(
    tmp_path: Path, synthetic: SyntheticPlan, frozen: Manifest
) -> None:
    write_manifest(frozen, tmp_path)
    (tmp_path / RECEIPT_FILE).unlink()
    dm = _manifest_module(tmp_path, synthetic)
    with pytest.raises((FileNotFoundError, ValueError)):
        dm.setup()


def test_manifest_folds_are_group_aware_and_cover_train_and_val(
    frozen_dir: Path, synthetic: SyntheticPlan, frozen: Manifest
) -> None:
    dm = _manifest_module(frozen_dir, synthetic, k_fold=3, fold_index=0)
    dm.setup(stage="fit")
    pool = len(frozen.partition("train")) + len(frozen.partition("val"))
    seen: set[int] = set()
    for fold in range(3):
        dm.set_fold(fold)
        train, val = set(dm.jute_train.indices), set(dm.jute_val.indices)
        assert not train & val
        assert len(train | val) == pool
        seen |= val
        groups = np.array(
            dm._pool_train.parts[0].groups + dm._pool_train.parts[1].groups
        )
        assert not set(groups[list(train)]) & set(groups[list(val)])
    assert len(seen) == pool
    with pytest.raises(ValueError, match="out of range"):
        dm.set_fold(3)


def test_manifest_loaders_repeat_for_a_seed_and_differ_for_another(
    frozen_dir: Path, synthetic: SyntheticPlan
) -> None:
    def first_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        dm = _manifest_module(frozen_dir, synthetic, seed=seed)
        dm.setup(stage="fit")
        return next(iter(dm.train_dataloader()))

    one, two, other = first_batch(1), first_batch(1), first_batch(2)
    assert torch.equal(one[0], two[0]) and torch.equal(one[1], two[1])
    assert not torch.equal(one[1], other[1]) or not torch.equal(one[0], other[0])


def test_manifest_weighted_sampler_balances_classes(
    frozen_dir: Path, synthetic: SyntheticPlan, frozen: Manifest
) -> None:
    dm = _manifest_module(frozen_dir, synthetic, use_weighted_sampler=True)
    dm.setup(stage="fit")
    assert dm.sampler is not None
    assert len(dm.sampler) == len(frozen.partition("train"))
