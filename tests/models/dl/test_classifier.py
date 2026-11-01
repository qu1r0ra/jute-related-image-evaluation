"""Unit tests for the DL Classifier (LightningModule)."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from lightning import LightningModule

from jute_disease.config import DEFAULT_LR, NUM_CLASSES
from jute_disease.models.dl import Classifier, TimmBackbone
from jute_disease.models.dl import classifier as classifier_module


@pytest.fixture
def backbone() -> TimmBackbone:
    return TimmBackbone(model_name="resnet18", pretrained=False)


@pytest.fixture
def mock_trainer() -> MagicMock:
    trainer = MagicMock()
    # Mock some expected trainer behaviors
    trainer.should_stop = False
    trainer.global_step = 0
    return trainer


@pytest.fixture
def classifier(backbone: TimmBackbone, mock_trainer: MagicMock) -> Classifier:
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        lr=DEFAULT_LR,
        freeze_backbone=False,
        compile_model=False,
    )
    # Attach mock trainer to satisfy self.log calls in steps
    model.trainer = mock_trainer
    return model


def test_classifier_requires_out_features() -> None:
    """Classifier must reject a feature extractor without out_features."""
    # nn.Identity has no out_features attribute
    bad_extractor = nn.Identity()
    with pytest.raises(ValueError, match="out_features"):
        Classifier(feature_extractor=bad_extractor, num_classes=NUM_CLASSES)


def test_classifier_forward(classifier: Classifier) -> None:
    x = torch.randn(4, 3, 224, 224)
    logits = classifier(x)
    assert logits.shape == (4, NUM_CLASSES)


def test_classifier_frozen_backbone(backbone: TimmBackbone) -> None:
    """Backbone parameters must be frozen when freeze_backbone=True."""
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        compile_model=False,
    )
    for param in model.feature_extractor.parameters():
        assert not param.requires_grad


def test_classifier_unfrozen_backbone(backbone: TimmBackbone) -> None:
    """Backbone parameters must be trainable when freeze_backbone=False."""
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        freeze_backbone=False,
        compile_model=False,
    )
    assert any(p.requires_grad for p in model.feature_extractor.parameters())


def test_classifier_configure_optimizers(classifier: Classifier) -> None:
    optimizers = classifier.configure_optimizers()
    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers


def test_classifier_training_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    loss = classifier.training_step(batch, 0)
    assert loss is not None
    assert loss.item() > 0


def test_classifier_validation_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    loss = classifier.validation_step(batch, 0)
    assert loss is not None


def test_classifier_predict_step(classifier: Classifier) -> None:
    batch = (torch.randn(4, 3, 224, 224), torch.randint(0, NUM_CLASSES, (4,)))
    preds = classifier.predict_step(batch, 0)
    assert preds.shape == (4, NUM_CLASSES)


def test_classifier_extra_repr(classifier: Classifier) -> None:
    rep = classifier.extra_repr()
    assert f"num_classes={NUM_CLASSES}" in rep
    assert f"lr={DEFAULT_LR}" in rep


def test_default_recipe_trains_the_full_uncompiled_backbone(
    backbone: TimmBackbone,
) -> None:
    model = Classifier(feature_extractor=backbone, num_classes=NUM_CLASSES)
    assert all(p.requires_grad for p in model.feature_extractor.parameters())
    assert not hasattr(model.feature_extractor, "_orig_mod")


def test_classifier_has_no_test_hooks_and_no_wandb() -> None:
    assert Classifier.test_step is LightningModule.test_step
    assert Classifier.on_test_epoch_end is LightningModule.on_test_epoch_end
    assert not hasattr(classifier_module, "wandb")


def test_scheduler_and_optimizer_follow_the_declared_recipe(
    backbone: TimmBackbone,
) -> None:
    model = Classifier(
        feature_extractor=backbone,
        num_classes=NUM_CLASSES,
        lr=3e-4,
        weight_decay=0.01,
        scheduler_patience=3,
        scheduler_factor=0.1,
    )
    configured = model.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]
    assert type(optimizer).__name__ == "AdamW"
    assert optimizer.defaults["lr"] == 3e-4
    assert optimizer.defaults["weight_decay"] == 0.01
    assert scheduler["monitor"] == "val_loss"
    assert scheduler["scheduler"].patience == 3
    assert scheduler["scheduler"].factor == 0.1
