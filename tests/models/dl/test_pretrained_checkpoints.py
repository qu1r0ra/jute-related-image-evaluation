import pytest

from jute_disease.config import CHECKPOINTS_DIR
from jute_disease.models.dl.backbone import TimmBackbone


@pytest.mark.slow
def test_pretrained_checkpoints_transfer_every_backbone_tensor() -> None:
    """Downloaded source checkpoints must load as exact backbone transfers."""
    ckpt_dir = CHECKPOINTS_DIR / "pretrained"
    if not ckpt_dir.exists():
        pytest.skip("Pretrained checkpoints directory not found.")

    checkpoints = list(ckpt_dir.glob("*.ckpt"))
    if not checkpoints:
        pytest.skip("No pretrained checkpoints downloaded.")

    for ckpt_path in checkpoints:
        model = TimmBackbone(
            model_name="mobilenetv2_100",
            pretrained=False,
            checkpoint_path=str(ckpt_path),
        )
        assert model.transfer is not None
        assert model.transfer.tensors > 0, ckpt_path.name
