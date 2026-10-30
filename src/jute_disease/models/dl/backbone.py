import hashlib
from dataclasses import dataclass
from pathlib import (
    Path,
    PosixPath,
    PurePosixPath,
    PureWindowsPath,
    WindowsPath,
)
from typing import Any

import timm
import torch
from torch import nn

from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


class BackboneTransferError(ValueError):
    """A checkpoint does not carry exactly the backbone tensors being initialized."""


@dataclass(frozen=True)
class BackboneTransfer:
    """Evidence that a source checkpoint's backbone was loaded in full."""

    path: str
    sha256: str
    tensors: int
    parameters: int


def state_digest(module: nn.Module) -> str:
    """Digest of every tensor name, dtype, shape and value in ``module``."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(f"{name}|{value.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def read_checkpoint_state(path: Path | str) -> dict[str, Any]:
    """Read a checkpoint's tensors without executing pickled code.

    Lightning checkpoints also store the data module's path hyperparameters, so
    path classes are the only extra types allowed.
    """
    with torch.serialization.safe_globals(
        [Path, PosixPath, WindowsPath, PurePosixPath, PureWindowsPath]
    ):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    return checkpoint.get("state_dict", checkpoint)


def _backbone_name(key: str) -> str | None:
    parts = [part for part in key.split(".") if part != "_orig_mod"]
    if parts and parts[0] == "feature_extractor":
        parts = parts[1:]
    if len(parts) > 1 and parts[0] == "backbone":
        return ".".join(parts[1:])
    return None


def load_backbone_checkpoint(backbone: nn.Module, path: Path | str) -> BackboneTransfer:
    """Load the backbone tensors of a Lightning or plain checkpoint, exactly.

    Only backbone tensors are read; the classifier head and any training state in
    the checkpoint are ignored, so the head starts fresh. Every backbone tensor
    must be present with the right shape.

    Raises:
        BackboneTransferError: no backbone tensors, or they do not match ``backbone``.
    """
    from jute_disease.data import file_sha256

    path = Path(path)
    state = read_checkpoint_state(path)
    extracted = {
        name: value
        for key, value in state.items()
        if (name := _backbone_name(key)) is not None
    }
    if not extracted:
        raise BackboneTransferError(f"{path.name} has no backbone tensors")
    try:
        backbone.load_state_dict(extracted, strict=True)
    except RuntimeError as error:
        raise BackboneTransferError(
            f"{path.name} does not match the backbone: {error}"
        ) from error
    loaded = backbone.state_dict()
    drifted = [k for k, v in extracted.items() if not torch.equal(loaded[k], v)]
    if drifted:
        raise BackboneTransferError(f"Tensors changed during load: {drifted[:3]}")
    return BackboneTransfer(
        path=path.name,
        sha256=file_sha256(path),
        tensors=len(extracted),
        parameters=sum(value.numel() for value in extracted.values()),
    )


class TimmBackbone(nn.Module):
    """Generic timm backbone wrapper."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_features: int | None = None,
        checkpoint_path: str | None = None,
        drop_rate: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            **kwargs,
        )
        self.model_name = model_name
        self.out_features = out_features or self.backbone.num_features
        self.transfer: BackboneTransfer | None = None

        if checkpoint_path is not None and checkpoint_path != "null":
            logger.info(f"Loading TimmBackbone weights from {checkpoint_path}")
            self.transfer = load_backbone_checkpoint(self.backbone, checkpoint_path)

    def extra_repr(self) -> str:
        return f"model_name={self.model_name!r}, out_features={self.out_features}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
