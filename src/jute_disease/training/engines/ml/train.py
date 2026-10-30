import os
from pathlib import Path

import numpy as np
import torch
import wandb

from jute_disease.config import (
    DEFAULT_SEED,
    MANIFEST_DIR,
    ML_MODELS_DIR,
    SOURCE_INVENTORY_DIR,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from jute_disease.data import classical_pipeline
from jute_disease.evaluation.metrics import EVAL_METRICS, format_metrics
from jute_disease.models.ml import (
    FEATURE_EXTRACTORS,
    ML_CLASSIFIERS,
    extract_features,
)
from jute_disease.training.logger import get_logger, setup_wandb
from jute_disease.training.partitions import open_training_partition
from jute_disease.training.seed import seed_everything

logger = get_logger(__name__)


def train_ml(
    classifier: str = "rf",
    feature_type: str = "crafted",
    balanced: bool = True,
    seed: int = DEFAULT_SEED,
    manifest_dir: Path = MANIFEST_DIR,
    image_root: Path = SOURCE_INVENTORY_DIR,
    output_dir: Path = ML_MODELS_DIR,
) -> Path:
    """Fit one classical model on the train partition and score validation.

    The test partition is never opened here; the final evaluator owns it.
    Returns the saved model path.
    """
    seed_everything(seed)

    if os.getenv("WANDB_MODE") != "disabled":
        setup_wandb()

        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"{classifier}-{feature_type}",
            config={
                "classifier": classifier,
                "feature_type": feature_type,
                "balanced": balanced,
                "seed": seed,
            },
        )

    extractor_cls = FEATURE_EXTRACTORS[feature_type]
    extractor = extractor_cls()

    # Classical features use the same deterministic preprocessing in every
    # partition; the frozen manifest alone decides which image is where.
    train_ds, val_ds = (
        open_training_partition(name, classical_pipeline(), manifest_dir, image_root)
        for name in ("train", "val")
    )

    X_train, y_train = extract_features(train_ds, extractor=extractor)
    X_val, y_val = extract_features(val_ds, extractor=extractor)

    logger.info(f"Training {classifier}...")
    classifier_cls = ML_CLASSIFIERS[classifier]
    model = classifier_cls(random_state=seed)

    sample_weight = None
    if balanced:
        unique_classes, counts = np.unique(y_train, return_counts=True)
        class_weights = len(y_train) / (len(unique_classes) * counts)
        weight_map = dict(zip(unique_classes, class_weights, strict=True))
        sample_weight = np.array([weight_map[label] for label in y_train])
        logger.info("Calculated balanced sample weights for training.")

    evaluator = EVAL_METRICS.clone()

    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_val_pred = model.predict(X_val)
    val_out = evaluator(torch.tensor(y_val_pred), torch.tensor(y_val))
    val_metrics = format_metrics(val_out, prefix="val_")

    logger.info(f"Validation Accuracy: {val_metrics['val_acc']:.4f}")
    logger.info(f"Validation F1 Macro: {val_metrics['val_f1']:.4f}")

    if os.getenv("WANDB_MODE") != "disabled":
        wandb.log(val_metrics)
        wandb.finish()
    return model.save(f"{classifier}_{feature_type}", directory=output_dir)
