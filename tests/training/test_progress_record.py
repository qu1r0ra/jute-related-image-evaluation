"""The resume state a fit leaves behind verifies against its progress record.

Regression: ``last.ckpt`` was rewritten only when a best checkpoint was saved, and
the progress hash was taken before that save, so a resumable fit could record a
hash that its own ``last.ckpt`` no longer matched.
"""

from pathlib import Path

import pytest
import torch
from lightning import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from jute_disease.training.engines.dl.fit import ProgressRecord
from jute_disease.training.resume import verified_progress


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


def _loader(seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(8, 4, generator=generator)
    y = torch.randint(0, 2, (8,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=4)


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
