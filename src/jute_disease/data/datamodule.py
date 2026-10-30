from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import ImageFolder

from jute_disease.config import BATCH_SIZE, DEFAULT_SEED, IMAGE_SIZE, NUM_WORKERS
from jute_disease.data.freeze import load_source_manifest
from jute_disease.data.manifest import FreezeBlockedError, Manifest, load_manifest
from jute_disease.data.readers import ManifestDataset
from jute_disease.data.sampling import balanced_sampler, loader_generator, seed_worker
from jute_disease.data.transforms import PreprocessingSpec, create_pipeline


class DataModule(LightningDataModule):
    """DataModule for the frozen jute manifest and for plain image folders.

    Two sources, chosen by the arguments:

    * ``manifest_dir`` (+ ``image_root``): the frozen manifest. Partitions come
      from the manifest only and cross-validation folds respect its groups.
    * ``data_dir``: an ``ImageFolder`` tree, either with ``train/``/``val/``
      (and optional ``test/``) folders, or flat with ``val_split`` for a seeded
      split. Used for external pretraining data and historical configs.

    Every pipeline, sampler and loader is built from ``seed``; nothing random is
    shared between instances.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        manifest_dir: Path | str | None = None,
        image_root: Path | str | None = None,
        target_manifest_dir: Path | str | None = None,
        cache_root: Path | str | None = None,
        val_split: float | None = None,
        batch_size: int = BATCH_SIZE,
        use_weighted_sampler: bool = False,
        seed: int = DEFAULT_SEED,
        num_workers: int = NUM_WORKERS,
        pin_memory: bool = True,
        k_fold: int = 1,
        fold_index: int = 0,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        super().__init__()
        if (data_dir is None) == (manifest_dir is None):
            raise ValueError("Provide exactly one of data_dir or manifest_dir")
        if manifest_dir is not None and image_root is None:
            raise ValueError("manifest_dir requires image_root")
        if data_dir is not None and k_fold > 1:
            raise ValueError(
                "k_fold needs group-aware folds; use manifest_dir, not data_dir"
            )
        self.save_hyperparameters()
        self.data_dir = None if data_dir is None else Path(data_dir).resolve()
        self.manifest_dir = None if manifest_dir is None else Path(manifest_dir)
        self.image_root = None if image_root is None else Path(image_root)
        self.target_manifest_dir = (
            None if target_manifest_dir is None else Path(target_manifest_dir)
        )
        self.cache_root = None if cache_root is None else Path(cache_root)

        self.jute_train: Dataset | None = None
        self.jute_val: Dataset | None = None
        self.jute_test: Dataset | None = None
        self.jute_predict: Dataset | None = None
        self.sampler = None
        self._classes: list[str] | None = None
        self.manifest: Manifest | None = None

        self._folds: list[tuple[np.ndarray, np.ndarray]] | None = None
        self._pool_train: Dataset | None = None
        self._pool_val: Dataset | None = None
        self._pool_targets: np.ndarray | None = None

    def _train_transform(self) -> Callable:
        return create_pipeline(
            self.hparams.image_size, True, True, seed=self.hparams.seed
        )

    def _eval_transform(self) -> Callable:
        return create_pipeline(
            self.hparams.image_size, False, True, seed=self.hparams.seed
        )

    @property
    def classes(self) -> list[str] | None:
        return self._classes

    @property
    def num_classes(self) -> int:
        return len(self._classes) if self._classes else 0

    @property
    def preprocessing(self) -> PreprocessingSpec:
        return PreprocessingSpec(size=self.hparams.image_size)

    def _manifest_dataset(self, partition: str, train: bool) -> ManifestDataset:
        assert self.manifest is not None and self.image_root is not None
        return ManifestDataset(
            self.manifest,
            partition,
            self.image_root,
            transform=self._train_transform() if train else self._eval_transform(),
            cache_root=self.cache_root,
            spec=self.preprocessing,
        )

    def setup(self, stage: str | None = None) -> None:
        if self.manifest_dir is not None:
            self._setup_manifest(stage)
        else:
            self._setup_folders(stage)

    def _setup_manifest(self, stage: str | None) -> None:
        assert self.manifest_dir is not None
        manifest = load_manifest(self.manifest_dir)
        if "source_inventory" in dict(manifest.dependencies):
            if self.target_manifest_dir is None:
                raise FreezeBlockedError(
                    "Source manifests require target_manifest_dir for "
                    "readiness verification"
                )
            assert self.image_root is not None
            self.manifest = load_source_manifest(
                self.manifest_dir, self.image_root, self.target_manifest_dir
            )
        else:
            self.manifest = manifest
        self._classes = list(self.manifest.labels)
        if stage in ("fit", None):
            if self.hparams.k_fold > 1:
                self._setup_folds()
            else:
                self.jute_train = self._manifest_dataset("train", True)
                self.jute_val = self._manifest_dataset("val", False)
                self._make_sampler(self.jute_train.targets)
        if stage in ("test", None):
            self.jute_test = self._manifest_dataset("test", False)
        if stage == "predict":
            self.jute_predict = self._manifest_dataset("test", False)

    def _setup_folds(self) -> None:
        augmented = [self._manifest_dataset(name, True) for name in ("train", "val")]
        plain = [self._manifest_dataset(name, False) for name in ("train", "val")]
        self._pool_targets = np.array(augmented[0].targets + augmented[1].targets)
        groups = np.array(augmented[0].groups + augmented[1].groups)
        self._pool_train = _Concat(*augmented)
        self._pool_val = _Concat(*plain)
        splitter = StratifiedGroupKFold(
            n_splits=self.hparams.k_fold, shuffle=True, random_state=self.hparams.seed
        )
        self._folds = list(
            splitter.split(
                np.zeros(len(self._pool_targets)), self._pool_targets, groups
            )
        )
        self.set_fold(self.hparams.fold_index)

    def _setup_folders(self, stage: str | None) -> None:
        assert self.data_dir is not None
        train_dir, val_dir, test_dir = (
            self.data_dir / name for name in ("train", "val", "test")
        )
        if stage in ("fit", None):
            if train_dir.exists() and val_dir.exists():
                self.jute_train = ImageFolder(train_dir, self._train_transform())
                self.jute_val = ImageFolder(val_dir, self._eval_transform())
                self._classes = self.jute_train.classes
                self._make_sampler(self.jute_train.targets)
            elif self.hparams.val_split is not None:
                full = ImageFolder(self.data_dir)
                self._classes = full.classes
                val_size = int(len(full) * self.hparams.val_split)
                train_part, val_part = random_split(
                    full,
                    [len(full) - val_size, val_size],
                    generator=torch.Generator().manual_seed(self.hparams.seed),
                )
                self.jute_train = Subset(
                    ImageFolder(self.data_dir, self._train_transform()),
                    train_part.indices,
                )
                self.jute_val = Subset(
                    ImageFolder(self.data_dir, self._eval_transform()),
                    val_part.indices,
                )
                self._make_sampler([full.targets[i] for i in train_part.indices])
        if stage in ("test", None) and test_dir.exists():
            self.jute_test = ImageFolder(test_dir, self._eval_transform())
            self._classes = self.jute_test.classes
        if stage == "predict":
            root = test_dir if test_dir.exists() else self.data_dir
            self.jute_predict = ImageFolder(root, self._eval_transform())

    def _make_sampler(self, targets: list[int]) -> None:
        self.sampler = (
            balanced_sampler(targets, self.hparams.seed)
            if self.hparams.use_weighted_sampler
            else None
        )

    def set_fold(self, fold_index: int) -> None:
        """Switch the active cross-validation fold (manifest mode only)."""
        if self._folds is None or self._pool_train is None or self._pool_val is None:
            raise RuntimeError("Folds not initialized. Call setup() first.")
        if not (0 <= fold_index < self.hparams.k_fold):
            raise ValueError(
                f"Fold index {fold_index} out of range (0-{self.hparams.k_fold - 1})"
            )
        self.hparams.fold_index = fold_index
        train_idx, val_idx = self._folds[fold_index]
        self.jute_train = Subset(self._pool_train, train_idx)
        self.jute_val = Subset(self._pool_val, val_idx)
        assert self._pool_targets is not None
        self._make_sampler(self._pool_targets[train_idx].tolist())

    def _loader(self, dataset: Dataset | None, name: str, train: bool) -> DataLoader:
        if dataset is None:
            raise RuntimeError(f"{name}_dataloader called before setup()")
        workers = self.hparams.num_workers
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * (1 if train else 2),
            shuffle=train and self.sampler is None,
            sampler=self.sampler if train else None,
            num_workers=workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=seed_worker if workers else None,
            generator=loader_generator(self.hparams.seed) if train else None,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.jute_train, "train", True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.jute_val, "val", False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.jute_test, "test", False)

    def predict_dataloader(self) -> DataLoader:
        return self._loader(self.jute_predict, "predict", False)


class _Concat(Dataset):
    """Two manifest datasets addressed by one index."""

    def __init__(self, first: ManifestDataset, second: ManifestDataset) -> None:
        self.parts = (first, second)

    def __len__(self) -> int:
        return len(self.parts[0]) + len(self.parts[1])

    def __getitem__(self, index: int) -> Any:
        first = len(self.parts[0])
        if index < first:
            return self.parts[0][index]
        return self.parts[1][index - first]
