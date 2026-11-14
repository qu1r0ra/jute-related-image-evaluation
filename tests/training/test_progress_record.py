"""The resume state a fit leaves behind verifies against its progress record.

Regression: ``last.ckpt`` was rewritten only when a best checkpoint was saved, and
the progress hash was taken before that save, so a resumable fit could record a
hash that its own ``last.ckpt`` no longer matched.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from lightning import LightningModule, Trainer
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jute_disease.data.datamodule import DataModule
from jute_disease.training.engines.dl.fit import ProgressRecord
from jute_disease.training.resume import LAST_CHECKPOINT, verified_progress


class _Linear(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 2)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        x, y = batch
        return nn.functional.cross_entropy(self.net(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        x, y = batch
        self.log("val_loss", nn.functional.cross_entropy(self.net(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.5)


def _loader(seed: int, *, shuffle: bool = False) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 4, generator=generator)
    y = torch.randint(0, 2, (8,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=4, shuffle=shuffle)


def _fit(
    directory: Path,
    epochs: int,
    *,
    attempt: str = "attempt-1",
    resume: Path | None = None,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[ProgressRecord(directory, attempt)],
    )
    trainer.fit(_Linear(), _loader(1, shuffle=True), _loader(2), ckpt_path=resume)
    return {
        name: tensor.detach().clone()
        for name, tensor in trainer.lightning_module.state_dict().items()
    }


def test_resumed_fit_matches_uninterrupted_fit_bit_for_bit(tmp_path: Path) -> None:
    uninterrupted = _fit(tmp_path / "uninterrupted", epochs=3)

    interrupted = tmp_path / "interrupted"
    _fit(interrupted, epochs=1)
    resumed = _fit(
        interrupted,
        epochs=3,
        attempt="attempt-2",
        resume=interrupted / LAST_CHECKPOINT,
    )

    assert uninterrupted.keys() == resumed.keys()
    for name, tensor in uninterrupted.items():
        assert torch.equal(tensor, resumed[name]), name


class _Images(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5), nn.Linear(3 * 32 * 32, 2)
        )

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], _: int
    ) -> torch.Tensor:
        x, y = batch
        return nn.functional.cross_entropy(self.net(x), y)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        x, y = batch
        self.log("val_loss", nn.functional.cross_entropy(self.net(x), y))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _image_folder(root: Path) -> Path:
    data = root / "data"
    rng = np.random.default_rng(3)
    for name, count in (("a", 12), ("b", 4)):
        folder = data / name
        folder.mkdir(parents=True)
        for index in range(count):
            pixels = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{index}.png")
    return data


def _fit_images(
    data: Path,
    directory: Path,
    epochs: int,
    *,
    attempt: str,
    resume: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Fit through the DataModule path: weighted sampler, workers, shuffled loaders."""
    torch.manual_seed(7)
    module = DataModule(
        data_dir=str(data),
        val_split=0.25,
        batch_size=4,
        use_weighted_sampler=True,
        seed=7,
        num_workers=2,
        pin_memory=False,
        image_size=32,
    )
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[ProgressRecord(directory, attempt)],
    )
    trainer.fit(_Images(), datamodule=module, ckpt_path=resume)
    return {
        name: tensor.detach().clone()
        for name, tensor in trainer.lightning_module.state_dict().items()
    }


@pytest.mark.slow
def test_resumed_datamodule_fit_matches_uninterrupted_fit_bit_for_bit(
    tmp_path: Path,
) -> None:
    data = _image_folder(tmp_path)
    uninterrupted = _fit_images(data, tmp_path / "uninterrupted", 3, attempt="a1")

    interrupted = tmp_path / "interrupted"
    _fit_images(data, interrupted, 1, attempt="a1")
    resumed = _fit_images(
        data,
        interrupted,
        3,
        attempt="a2",
        resume=interrupted / LAST_CHECKPOINT,
    )

    assert uninterrupted.keys() == resumed.keys()
    for name, tensor in uninterrupted.items():
        assert torch.equal(tensor, resumed[name]), name


@pytest.mark.parametrize("epochs", [1, 3])
def test_last_checkpoint_matches_its_progress_record(
    tmp_path: Path, epochs: int
) -> None:
    directory = tmp_path / "checkpoints"
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[ProgressRecord(directory, "attempt-1")],
    )
    trainer.fit(_Linear(), _loader(1), _loader(2))

    progress = verified_progress(directory)
    assert progress is not None
    assert progress["epochs_completed"] == epochs
    assert (directory / "last.ckpt").is_file()
    assert not (directory / "last.ckpt.tmp").exists()
