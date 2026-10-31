import argparse
from pathlib import Path
from shutil import copyfile

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from jute_disease.data import (
    FreezeBlockedError,
    SourceMismatchError,
)
from jute_disease.models.dl import Classifier, TimmBackbone
from jute_disease.training.logger import get_logger
from jute_disease.training.seed import seed_everything
from jute_disease.training.source import source_data_module

logger = get_logger(__name__)


def train_pretext_task(
    manifest_dir: Path | str,
    image_root: Path | str,
    target_manifest_dir: Path | str,
    output_path: Path | str,
    pretrained: bool = True,
    base_backbone_weights: dict[str, torch.Tensor] | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    model_name: str = "mobilenetv2_100",
    seed: int = 42,
) -> None:
    """
    Trains a given timm backbone on a dataset and saves the checkpoint.
    """
    seed_everything(seed)

    dm = source_data_module(
        manifest_dir, image_root, target_manifest_dir, batch_size, seed
    )

    num_classes = dm.num_classes
    logger.info(f"Found {num_classes} classes.")

    # Backbone
    logger.info(f"Initializing {model_name} backbone...")
    backbone = TimmBackbone(model_name=model_name, pretrained=pretrained)

    if base_backbone_weights is not None:
        logger.info("Loading custom backbone weights...")
        msg = backbone.load_state_dict(base_backbone_weights, strict=True)
        logger.info(f"Backbone load status: {msg}")

    # Classifier
    model = Classifier(
        feature_extractor=backbone,
        num_classes=num_classes,
        lr=lr,
        freeze_backbone=False,
    )

    # Trainer
    task_name = Path(output_path).stem
    csv_logger = CSVLogger(save_dir="artifacts/logs/pretraining", name=task_name)

    callbacks = [
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            filename=f"{task_name}-{{epoch:02d}}-{{val_acc:.4f}}",
            save_top_k=1,
            dirpath="artifacts/checkpoints/pretrained/",
        ),
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        TQDMProgressBar(refresh_rate=100),
    ]

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        callbacks=callbacks,
        log_every_n_steps=100,
    )

    logger.info("Starting Training...")
    trainer.fit(model, datamodule=dm)

    if isinstance(trainer.checkpoint_callback, ModelCheckpoint):
        best_path = trainer.checkpoint_callback.best_model_path
    else:
        best_path = None

    if best_path:
        logger.info(f"Best model saved at: {best_path}")
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(best_path, str(target_path))
        logger.info(f"Copied best model to target path: {output_path}")
    else:
        logger.error("Training failed or no checkpoint saved.")


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
        default="mobilenetv2_100",
        help=(
            "The timm architecture to initialize "
            "(e.g. mobilenetv2_100, tf_efficientnet_b7.ns_jft_in1k)"
        ),
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        help=(
            "Optional: Path to PREVIOUS stage checkpoint to resume from (e.g. "
            "PlantVillage ckpt)"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args(argv)

    backbone_weights: dict[str, torch.Tensor] | None = None

    if args.base_weights:
        logger.info(f"Extracting backbone weights from {args.base_weights}...")
        checkpoint = torch.load(args.base_weights, map_location="cpu")
        state_dict: dict[str, torch.Tensor] = checkpoint["state_dict"]

        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("feature_extractor."):
                name = k.replace("feature_extractor.", "")
                if name.startswith("_orig_mod."):
                    name = name.replace("_orig_mod.", "")
                backbone_dict[name] = v

        if backbone_dict:
            logger.info(f"Extracted {len(backbone_dict)} keys for backbone.")
            backbone_weights = backbone_dict
        else:
            logger.warning("No 'feature_extractor' keys found in base checkpoint.")

    try:
        train_pretext_task(
            manifest_dir=args.manifest_dir,
            image_root=args.image_root,
            target_manifest_dir=args.target_manifest_dir,
            output_path=args.output_path,
            pretrained=True,
            base_backbone_weights=backbone_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_name=args.model_name,
        )
    except (FreezeBlockedError, SourceMismatchError, FileNotFoundError) as error:
        parser.error(f"Source pretraining blocked: {error}")
