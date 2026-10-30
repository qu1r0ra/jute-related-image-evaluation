from pathlib import Path

import timm
import torch
from torch import nn

from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


class TimmBackbone(nn.Module):
    """Generic timm backbone wrapper."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        out_features: int | None = None,
        checkpoint_path: str | None = None,
        drop_rate: float = 0.0,
        strict_checkpoint: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__()
        if strict_checkpoint and checkpoint_path in (None, "null"):
            raise ValueError("strict_checkpoint requires a checkpoint_path")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
            **kwargs,
        )
        self.model_name = model_name
        self.out_features = out_features or self.backbone.num_features

        if checkpoint_path is not None and checkpoint_path != "null":
            if strict_checkpoint:
                self._load_strict(checkpoint_path)
                return
            logger.info(f"Loading TimmBackbone custom weights from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            backbone_dict = {}
            prefixes = ("feature_extractor.", "_orig_mod.", "backbone.")
            for k, v in state_dict.items():
                name = k
                changed = True
                while changed:
                    changed = False
                    for prefix in prefixes:
                        if name.startswith(prefix):
                            name = name[len(prefix) :]
                            changed = True

                backbone_dict[name] = v

            if backbone_dict:
                msg = self.backbone.load_state_dict(backbone_dict, strict=False)
                logger.info(f"Grid Search backbone load status: {msg}")
            else:
                logger.warning("No 'feature_extractor' keys found dynamically.")

    def _load_strict(self, checkpoint_path: str) -> None:
        """Load a Lightning checkpoint's backbone weights, all of them or fail.

        Only ``feature_extractor.*`` entries are taken, so the source head stays
        behind. Missing or unexpected backbone keys raise instead of leaving a
        silently half-initialized model.

        Raises:
            FileNotFoundError: the checkpoint does not exist.
            ValueError: the checkpoint holds no backbone weights.
        """
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"source checkpoint not found: {path}")
        logger.info(f"Loading TimmBackbone source weights strictly from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        prefix = "feature_extractor."
        weights = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :].replace("_orig_mod.", "")
            weights[name.removeprefix("backbone.")] = value
        if not weights:
            raise ValueError(f"no feature_extractor weights in checkpoint: {path}")
        self.backbone.load_state_dict(weights, strict=True)

    def extra_repr(self) -> str:
        return f"model_name={self.model_name!r}, out_features={self.out_features}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
