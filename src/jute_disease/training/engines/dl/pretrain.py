import argparse
import json
from pathlib import Path
from shutil import copyfile

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from jute_disease.config import DEFAULT_SEED
from jute_disease.data import (
    FreezeBlockedError,
    SourceMismatchError,
)
from jute_disease.models.dl import Classifier, TimmBackbone
from jute_disease.training.logger import get_logger
from jute_disease.training.recipe import TRAINING_CONFIG, load_training_values
from jute_disease.training.seed import seed_everything
from jute_disease.training.source import source_data_module

logger = get_logger(__name__)

# Corrected-path setup that is not a tunable training value (see training.yaml).
PRECISION = "16-mixed"
NUM_WORKERS = 0


def train_pretext_task(
    manifest_dir: Path | str,
    image_root: Path | str,
    target_manifest_dir: Path | str,
    output_path: Path | str,
    pretrained: bool = True,
    base_checkpoint: Path | str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    model_name: str = "mobilenetv2_100.ra_in1k",
    seed: int = DEFAULT_SEED,
    limit_batches: int | None = None,
    training_config: Path | str = TRAINING_CONFIG,
) -> None:
    """Train a timm backbone on a verified source and save its best checkpoint.

    The optimizer, scheduler, early-stopping and compile values come from
    ``training_config``. The kept checkpoint has the lowest validation loss, and a
    receipt next to ``output_path`` records which one that was. A
    ``base_checkpoint`` initializes the backbone strictly: any missing or extra
    backbone weight raises. ``limit_batches`` bounds each epoch for smoke runs.

    Raises:
        RuntimeError: training produced no checkpoint, or ``base_checkpoint``
            does not match the backbone.
    """
    values = load_training_values(training_config)
    seed_everything(seed)

    dm = source_data_module(
        manifest_dir,
        image_root,
        target_manifest_dir,
        batch_size,
        seed,
        num_workers=NUM_WORKERS,
        use_weighted_sampler=True,
    )
    num_classes = dm.num_classes
    logger.info(f"Found {num_classes} classes.")

    logger.info(f"Initializing {model_name} backbone...")
    backbone = TimmBackbone(
        model_name=model_name,
        pretrained=pretrained,
        checkpoint_path=None if base_checkpoint is None else str(base_checkpoint),
        strict_checkpoint=base_checkpoint is not None,
    )
    model = Classifier(
        feature_extractor=backbone,
        num_classes=num_classes,
        lr=values.lr,
        weight_decay=values.weight_decay,
        freeze_backbone=False,
        compile_model=values.compile_model,
        scheduler_patience=values.scheduler_patience,
        scheduler_factor=values.scheduler_factor,
    )

    output = Path(output_path)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output.parent / "checkpoints",
        monitor=values.monitor,
        mode=values.mode,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
    )
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=PRECISION,
        logger=CSVLogger(save_dir=output.parent, name="logs"),
        callbacks=[
            checkpoint_callback,
            EarlyStopping(
                monitor=values.monitor,
                mode=values.mode,
                patience=values.early_stopping_patience,
            ),
            TQDMProgressBar(refresh_rate=100),
        ],
        log_every_n_steps=25,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
    )

    logger.info("Starting Training...")
    trainer.fit(model, datamodule=dm)

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError("Training finished without saving a checkpoint")
    logger.info(f"Best model saved at: {best_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    copyfile(best_path, output)
    receipt = {
        "checkpoint": str(output),
        "best_model_path": best_path,
        "best_val_loss": float(checkpoint_callback.best_model_score),
        "monitor": values.monitor,
        "mode": values.mode,
        "early_stopping_patience": values.early_stopping_patience,
        "epochs_run": trainer.current_epoch,
        "training_config": str(training_config),
        "model_name": model_name,
        "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
        "seed": seed,
        "precision": PRECISION,
        "num_workers": NUM_WORKERS,
    }
    output.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(f"Copied best model to target path: {output_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest_dir",
        type=str,
        required=True,
        help="Completed source manifest directory",
    )
    parser.add_argument(
        "--image_root", required=True, help="Original source package root"
    )
    parser.add_argument(
        "--target_manifest_dir", required=True, help="Frozen target manifest directory"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Where to save the final .ckpt"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mobilenetv2_100.ra_in1k",
        help="The timm architecture to initialize (e.g. mobilenetv2_100.ra_in1k)",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        help=(
            "Optional: previous stage checkpoint (e.g. the PlantVillage source) "
            "whose backbone is loaded strictly"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--limit_batches",
        type=int,
        default=None,
        help="Bound each epoch to this many batches (smoke runs only)",
    )
    parser.add_argument(
        "--training_config",
        type=str,
        default=str(TRAINING_CONFIG),
        help="YAML holding the shared training values",
    )

    args = parser.parse_args(argv)

    try:
        train_pretext_task(
            manifest_dir=args.manifest_dir,
            image_root=args.image_root,
            target_manifest_dir=args.target_manifest_dir,
            output_path=args.output_path,
            base_checkpoint=args.base_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model_name,
            seed=args.seed,
            limit_batches=args.limit_batches,
            training_config=args.training_config,
        )
    except (FreezeBlockedError, SourceMismatchError, FileNotFoundError) as error:
        parser.error(f"Source pretraining blocked: {error}")
