from pathlib import Path

import torch
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from jute_disease.config import DEFAULT_LR, DEFAULT_WEIGHT_DECAY
from jute_disease.evaluation.metrics import format_metrics
from jute_disease.training.logger import write_evaluation


class Classifier(LightningModule):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        freeze_backbone: bool = True,
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        if not hasattr(feature_extractor, "out_features"):
            raise ValueError("Feature extractor must have an 'out_features' attribute")

        self.feature_extractor = feature_extractor

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        if compile_model and hasattr(torch, "compile"):
            self.feature_extractor = torch.compile(self.feature_extractor)

        self.classifier = nn.Linear(
            in_features=feature_extractor.out_features, out_features=num_classes
        )
        self.loss = nn.CrossEntropyLoss()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters(ignore=["feature_extractor"])

        self.train_metrics = MetricCollection(
            {"acc": Accuracy(task="multiclass", num_classes=num_classes)},
            prefix="train_",
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", num_classes=num_classes),
                "f1": F1Score(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "precision": Precision(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=num_classes, average="macro"
                ),
            },
            prefix="val_",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_")

        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, lr={self.lr}, "
            f"weight_decay={self.weight_decay}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        return self.classifier(features)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)

        self.test_preds.append(y_hat.argmax(dim=-1).detach().cpu())
        self.test_targets.append(y.detach().cpu())

        return loss

    def on_test_epoch_start(self) -> None:
        self.test_preds.clear()
        self.test_targets.clear()

    def on_test_epoch_end(self) -> None:
        if self.test_preds:
            classes = getattr(self.trainer.datamodule, "classes", None)
            log_dir = getattr(self.logger, "log_dir", None)
            write_evaluation(
                "test",
                format_metrics(self.test_metrics.compute()),
                torch.cat(self.test_targets).numpy(),
                torch.cat(self.test_preds).numpy(),
                classes,
                {
                    "checkpoint": str(getattr(self.trainer, "ckpt_path", "")),
                    "model": dict(self.hparams),
                    "precision": str(self.trainer.precision),
                },
                directory=Path(log_dir) if isinstance(log_dir, str) else None,
            )
        self.test_preds.clear()
        self.test_targets.clear()

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, _ = batch
        return self(x)

    def configure_optimizers(self) -> dict[str, object]:
        optimizer = AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, patience=3),
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
