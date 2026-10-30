"""Lightning callbacks used by the corrected recipes."""

import json
from pathlib import Path

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

RECORD_NAME = "best_checkpoint.json"


class BestCheckpointRecord(Callback):
    """Write which checkpoint a fit kept, so later stages can find it.

    Lightning places checkpoints under the logger's version folder. The record
    sits at a fixed path in ``default_root_dir`` and is the declared output of a
    fit stage: it exists only when training finished and kept a checkpoint.
    """

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        checkpoint = trainer.checkpoint_callback
        if (
            not isinstance(checkpoint, ModelCheckpoint)
            or not checkpoint.best_model_path
        ):
            raise RuntimeError("Training finished without saving a checkpoint")
        record = {
            "path": str(Path(checkpoint.best_model_path).resolve()),
            "monitor": checkpoint.monitor,
            "mode": checkpoint.mode,
            "best_score": float(checkpoint.best_model_score),
            "epochs_run": trainer.current_epoch,
        }
        target = Path(trainer.default_root_dir) / RECORD_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
