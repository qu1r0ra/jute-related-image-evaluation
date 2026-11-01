"""Unit tests for TimmBackbone wrapper."""

from pathlib import Path

import pytest
import torch

from jute_disease.data import file_sha256
from jute_disease.models.dl.backbone import (
    BackboneTransferError,
    TimmBackbone,
    state_digest,
)


@pytest.fixture
def backbone() -> TimmBackbone:
    return TimmBackbone(model_name="mobilenetv2_100", pretrained=False)


def test_timmbackbone_init(backbone: TimmBackbone) -> None:
    assert backbone.backbone is not None
    assert isinstance(backbone.out_features, int)
    assert backbone.out_features > 0


def test_timmbackbone_forward(backbone: TimmBackbone) -> None:
    x = torch.randn(2, 3, 224, 224)
    out = backbone(x)
    assert out.shape == (2, backbone.out_features)


def test_timmbackbone_no_checkpoint() -> None:
    """Instantiating without a checkpoint_path must not raise."""
    model = TimmBackbone(
        model_name="mobilenetv2_100", pretrained=False, checkpoint_path=None
    )
    assert model is not None


def test_timmbackbone_missing_checkpoint_raises(tmp_path: Path) -> None:
    """A non-existent checkpoint_path should raise an error from torch.load."""
    with pytest.raises(FileNotFoundError):
        TimmBackbone(
            model_name="mobilenetv2_100",
            pretrained=False,
            checkpoint_path=str(tmp_path / "nonexistent.ckpt"),
        )


def _classifier_checkpoint(tmp_path: Path, prefix: str = "") -> tuple[Path, dict]:
    source = TimmBackbone(model_name="mobilenetv2_100", pretrained=False)
    state = {
        f"feature_extractor.{prefix}{k}": v for k, v in source.state_dict().items()
    }
    state["classifier.weight"] = torch.randn(3, source.out_features)
    state["classifier.bias"] = torch.randn(3)
    path = tmp_path / "source.ckpt"
    torch.save({"state_dict": state}, path)
    return path, source.state_dict()


@pytest.mark.parametrize("prefix", ["", "_orig_mod."])
def test_checkpoint_transfers_every_backbone_tensor_exactly(
    tmp_path: Path, prefix: str
) -> None:
    path, expected = _classifier_checkpoint(tmp_path, prefix)
    model = TimmBackbone(
        model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(path)
    )
    actual = model.state_dict()
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert model.transfer is not None
    assert model.transfer.tensors == len(expected)
    assert model.transfer.sha256 == file_sha256(path)


def test_transfer_excludes_the_classifier_head(tmp_path: Path) -> None:
    path, _ = _classifier_checkpoint(tmp_path)
    model = TimmBackbone(
        model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(path)
    )
    assert not any(name.startswith("classifier") for name in model.state_dict())


def test_incomplete_backbone_checkpoint_is_rejected(tmp_path: Path) -> None:
    path, _ = _classifier_checkpoint(tmp_path)
    saved = torch.load(path)
    victim = next(k for k in saved["state_dict"] if k.endswith("conv_stem.weight"))
    del saved["state_dict"][victim]
    torch.save(saved, path)
    with pytest.raises(BackboneTransferError, match="conv_stem"):
        TimmBackbone(
            model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(path)
        )


def test_wrong_architecture_checkpoint_is_rejected(tmp_path: Path) -> None:
    path, _ = _classifier_checkpoint(tmp_path)
    with pytest.raises(BackboneTransferError):
        TimmBackbone(model_name="resnet18", pretrained=False, checkpoint_path=str(path))


def test_checkpoint_without_backbone_tensors_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unrelated.ckpt"
    torch.save({"state_dict": {"random.weight": torch.randn(1)}}, path)
    with pytest.raises(BackboneTransferError, match="no backbone tensors"):
        TimmBackbone(
            model_name="mobilenetv2_100", pretrained=False, checkpoint_path=str(path)
        )


def test_state_digest_tracks_every_tensor() -> None:
    model = TimmBackbone(model_name="resnet18", pretrained=False)
    before = state_digest(model)
    assert before == state_digest(model)
    next(model.parameters()).data.add_(1.0)
    assert state_digest(model) != before


def test_backbone_override_features() -> None:
    """Ensure out_features can be manually overridden."""
    model = TimmBackbone(model_name="resnet18", pretrained=False, out_features=123)
    assert model.out_features == 123
