"""The one CNN fit: every source and target stage runs through ``fit_cnn``.

A fit trains the full backbone with a fresh head and fresh optimizer state, from
either the declared timm initialization or the exact backbone tensors of an
upstream source checkpoint. Only the train and validation partitions are ever
opened. The checkpoint is the one the ``ModelCheckpoint`` callback selected on
validation loss, and the evidence needed for a receipt is returned, not written.
"""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from jute_disease.data import DataModule, file_sha256, load_manifest
from jute_disease.models.dl import (
    Classifier,
    TimmBackbone,
    read_checkpoint_state,
    state_digest,
)
from jute_disease.training.benchmark import FitKind, FitSpec, Recipe
from jute_disease.training.inputs import DataInputs
from jute_disease.training.logger import (
    METRICS_DIR,
    METRICS_FILE,
    experiment_loggers,
)
from jute_disease.training.results import FitResult
from jute_disease.training.seed import seed_everything

CHECKPOINT_DIR = "checkpoints"


class EpochTimer(Callback):
    """Seconds per training epoch, validation included."""

    def __init__(self) -> None:
        self.seconds: list[float] = []
        self._start = 0.0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: Any) -> None:
        self._start = time.perf_counter()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
        self.seconds.append(time.perf_counter() - self._start)


def training_callbacks(recipe: Recipe, checkpoint_dir: Path | str) -> list[Callback]:
    """Best-validation-loss checkpointing and patience-based early stopping."""
    return [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:02d}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=recipe.early_stopping_patience,
        ),
    ]


def _data_module(
    fit: FitSpec, recipe: Recipe, data: DataInputs, pin_memory: bool
) -> tuple[DataModule, dict[str, str]]:
    if fit.kind is FitKind.SOURCE:
        manifest_dir, image_root = data.sources[fit.stage]
        target_dir: Path | None = data.target_dir
    else:
        manifest_dir, image_root, target_dir = (
            data.target_dir,
            data.image_root,
            None,
        )
    module = DataModule(
        manifest_dir=manifest_dir,
        image_root=image_root,
        target_manifest_dir=target_dir,
        batch_size=fit.micro_batch,
        use_weighted_sampler=recipe.weighted_sampler,
        seed=fit.seed,
        num_workers=recipe.num_workers,
        pin_memory=pin_memory,
        image_size=recipe.image_size,
        cache_root=None,
        allow_test=False,
    )
    module.setup("fit")
    assert module.manifest is not None
    manifests = {"train": module.manifest.digest()}
    if target_dir is not None:
        manifests["target"] = load_manifest(target_dir).digest()
    return module, manifests


def _uses_cuda(accelerator: str) -> bool:
    return accelerator in ("gpu", "cuda") or (
        accelerator == "auto" and torch.cuda.is_available()
    )


def _head_equals_upstream(checkpoint: Path, head: torch.nn.Linear) -> bool:
    state = read_checkpoint_state(checkpoint)
    weight = state.get("classifier.weight")
    return (
        weight is not None
        and weight.shape == head.weight.shape
        and bool(torch.equal(weight, head.weight.detach().cpu()))
    )


def _backbone_counts(
    initial: dict[str, torch.Tensor], backbone: torch.nn.Module
) -> dict[str, int]:
    parameters = dict(backbone.named_parameters())
    return {
        "total": len(parameters),
        "trainable": sum(p.requires_grad for p in parameters.values()),
        "updated": sum(
            not torch.equal(initial[name], value.detach().cpu())
            for name, value in parameters.items()
        ),
    }


def fit_cnn(
    fit: FitSpec,
    recipe: Recipe,
    data: DataInputs,
    attempt_dir: Path | str,
    *,
    upstream_checkpoint: Path | str | None = None,
    pretrained: bool = True,
    accelerator: str = "auto",
    wandb_mode: str = "off",
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> FitResult:
    """Train one source or target CNN stage and describe exactly what happened.

    Args:
        fit: the stage; a chained fit names ``upstream`` and needs its checkpoint.
        pretrained: load the timm initialization when there is no upstream.
        limit_train_batches: smoke bound, recorded so a bounded run is never
            mistaken for a final one.

    Raises:
        ValueError: ``fit`` has an upstream but no checkpoint was given.
        RuntimeError: no checkpoint was selected, or the backbone never changed.
    """
    attempt = Path(attempt_dir)
    if fit.upstream is not None and upstream_checkpoint is None:
        raise ValueError(
            f"{fit.fit_id} needs the checkpoint of upstream {fit.upstream}"
        )
    if fit.upstream is None:
        upstream_checkpoint = None
    seed_everything(fit.seed)

    cuda = _uses_cuda(accelerator)
    module, manifests = _data_module(fit, recipe, data, cuda)
    backbone = TimmBackbone(
        fit.model,
        pretrained=pretrained and upstream_checkpoint is None,
        checkpoint_path=(
            None if upstream_checkpoint is None else str(upstream_checkpoint)
        ),
    )
    model = Classifier(
        backbone,
        num_classes=module.num_classes,
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
        freeze_backbone=False,
        compile_model=False,
        scheduler_patience=recipe.scheduler_patience,
        scheduler_factor=recipe.scheduler_factor,
    )
    initial_digest = state_digest(backbone)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in backbone.named_parameters()
    }
    head_equals_upstream = (
        False
        if upstream_checkpoint is None
        else _head_equals_upstream(Path(upstream_checkpoint), model.classifier)
    )

    loggers, wandb_status = experiment_loggers(
        attempt, wandb_mode=wandb_mode, name=fit.fit_id
    )
    timer = EpochTimer()
    callbacks = [*training_callbacks(recipe, attempt / CHECKPOINT_DIR), timer]
    stopping = next(c for c in callbacks if isinstance(c, EarlyStopping))
    selection = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=recipe.max_epochs,
        precision=recipe.precision if cuda else "32-true",
        accumulate_grad_batches=fit.accumulation,
        limit_train_batches=1.0 if limit_train_batches is None else limit_train_batches,
        limit_val_batches=1.0 if limit_val_batches is None else limit_val_batches,
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(attempt),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule=module)

    if not selection.best_model_path:
        raise RuntimeError(f"{fit.fit_id}: the checkpoint callback selected nothing")
    best = Path(selection.best_model_path)
    counts = _backbone_counts(initial, backbone)
    if counts["updated"] == 0:
        raise RuntimeError(f"{fit.fit_id}: no backbone tensor changed during training")
    transfer = backbone.transfer
    metrics_file = attempt / METRICS_DIR / "version_0" / METRICS_FILE
    results = {
        "fit": asdict(fit),
        "recipe": asdict(recipe),
        "checkpoint": {
            "path": best.relative_to(attempt).as_posix(),
            "sha256": file_sha256(best),
            "val_loss": float(selection.best_model_score),
        },
        "epochs_run": trainer.current_epoch,
        "early_stopped": stopping.stopped_epoch > 0,
        "seconds_per_epoch": max(timer.seconds),
        "epoch_seconds": timer.seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "metrics": {
            key: float(value)
            for key, value in trainer.callback_metrics.items()
            if not key.startswith("test")
        },
        "num_classes": module.num_classes,
        "pretrained": pretrained and upstream_checkpoint is None,
        "epoch_batches": {
            "train": len(module.train_dataloader()),
            "val": len(module.val_dataloader()),
        },
        "effective_batch": fit.micro_batch * fit.accumulation,
        "micro_batch": fit.micro_batch,
        "accumulate_grad_batches": fit.accumulation,
        "precision": str(trainer.precision),
        "weighted_sampler": recipe.weighted_sampler,
        "compiled": False,
        "backbone": {**counts, "initial_digest": initial_digest},
        "transfer": None if transfer is None else asdict(transfer),
        "head": {"fresh": True, "equals_upstream": head_equals_upstream},
        "training_state": {"resumed": False, "optimizer_loaded": False},
        "lineage": {
            "stage": fit.stage if fit.kind is FitKind.SOURCE else "jute",
            "upstream_fit": fit.upstream,
            "upstream_checkpoint_sha256": (
                None if transfer is None else transfer.sha256
            ),
            "manifests": manifests,
        },
        "bounds": {
            "train_batches": limit_train_batches,
            "val_batches": limit_val_batches,
        },
        "bounded": limit_train_batches is not None or limit_val_batches is not None,
        "wandb": wandb_status,
        "test_accessed": module.jute_test is not None,
    }
    artifacts = {"checkpoint": best}
    if metrics_file.is_file():
        artifacts["metrics"] = metrics_file
    return FitResult(results, artifacts)
