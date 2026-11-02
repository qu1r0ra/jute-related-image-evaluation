"""Image pipelines built per use, never shared.

Every call to :func:`create_pipeline` returns a new, independently seeded pipeline:
no module-level ``Compose`` carries random state between datasets, partitions or
runs. Evaluation pipelines are deterministic; training augmentation is seeded and
reseeds itself per DataLoader worker from the loader's generator.

:class:`PreprocessingSpec` is the identity of the deterministic input
preprocessing. Classical-feature and CNN inputs both start from it, and its digest
is a dependency of every cache.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from PIL.Image import Image
from torchvision.transforms import (
    ColorJitter,
    Compose,
    InterpolationMode,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
)
from torchvision.transforms import functional as F

from jute_disease.config import DEFAULT_SEED, IMAGE_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessingSpec:
    """Deterministic preprocessing applied to every partition.

    ``cache_shortest_side`` is the lossless cache's downsize target (images are
    never upscaled); ``size`` is the model input after the evaluation resize and
    letterbox.
    """

    revision: int = 2
    size: int = IMAGE_SIZE
    cache_shortest_side: int = 512
    cache_format: str = "png"
    resize: str = "letterbox"
    interpolation: str = "bilinear"
    padding: str = "rounded-uint8-imagenet-mean-124-116-104"
    normalization: str = "imagenet-mean-std"
    cache_interpolation: str = "bicubic-downsize-no-upscale"

    def digest(self) -> str:
        """Return the SHA-256 of the spec."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ImageWithMask:
    """A letterboxed image and the pixels supplied by its original image."""

    pixels: np.ndarray
    valid_mask: np.ndarray

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Keep NumPy consumers compatible while preserving the validity mask."""
        array = np.asarray(self.pixels, dtype=dtype)
        return array.copy() if copy else array


def letterbox_dimensions(
    width: int, height: int, size: int = IMAGE_SIZE
) -> tuple[int, int]:
    """Return the rounded resized dimensions used by the letterbox transform."""
    if width < 1 or height < 1 or size < 1:
        raise ValueError("Image dimensions and letterbox size must be positive")
    ratio = size / max(width, height)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def letterbox_pad_fraction(width: int, height: int, size: int = IMAGE_SIZE) -> float:
    """Return the fraction of output pixels supplied by letterbox padding."""
    resized_width, resized_height = letterbox_dimensions(width, height, size)
    return 1.0 - (resized_width * resized_height) / (size * size)


class ImagePipeline:
    """Per-dataset torchvision pipeline with independent worker random state."""

    def __init__(
        self,
        size: int,
        is_train: bool,
        is_dl: bool,
        seed: int,
        include_mask: bool,
        *,
        resize: str = "letterbox",
        training_view: str = "raw-training",
        mask_checkpoint: Path | None = None,
        fixture_masks: bool = False,
        border_frame_width_px: int = 0,
    ) -> None:
        if resize not in {"letterbox", "resize-center-crop"}:
            raise ValueError("resize must be letterbox or resize-center-crop")
        if training_view not in {"raw-training", "background-only", "border-frame"}:
            raise ValueError(f"unsupported training view {training_view!r}")
        if training_view == "background-only" and not (
            mask_checkpoint is not None or fixture_masks
        ):
            raise ValueError(
                "background-only training requires a local mask checkpoint"
            )
        if training_view == "border-frame" and not (
            1 <= border_frame_width_px < size / 2
        ):
            raise ValueError(
                "border-frame width must be positive and smaller than half the image"
            )
        self.size = size
        self.is_train = is_train
        self.is_dl = is_dl
        self.include_mask = include_mask
        self.resize = resize
        self.training_view = training_view
        self.mask_checkpoint = mask_checkpoint
        self.fixture_masks = fixture_masks
        self.border_frame_width_px = border_frame_width_px
        self.seed = seed
        self.deterministic = not is_train
        self.description = (
            f"resize={resize};size={size};view={training_view};"
            f"frame={border_frame_width_px};train={is_train};dl={is_dl}"
        )
        self._states: dict[int, torch.Tensor] = {}
        self._mask_backend: Any = None
        self._mask_cache: dict[str, np.ndarray] = {}

    def __call__(
        self, img: Image | np.ndarray
    ) -> torch.Tensor | np.ndarray | ImageWithMask:
        image = PILImage.fromarray(np.asarray(img)).convert("RGB")
        if self.training_view == "background-only":
            pixels = np.asarray(image)
            key = hashlib.sha256(pixels.tobytes()).hexdigest()
            mask = self._mask_cache.get(key)
            if mask is None:
                if self._mask_backend is None:
                    from jute_disease.data.masks import (
                        ColorThresholdBackend,
                        Sam2Refiner,
                    )

                    self._mask_backend = (
                        ColorThresholdBackend()
                        if self.fixture_masks
                        else Sam2Refiner(self.mask_checkpoint)
                    )
                mask = np.asarray(self._mask_backend.predict(image).mask, dtype=bool)
                if mask.shape != pixels.shape[:2]:
                    raise ValueError("leaf mask shape differs from training image")
                if len(self._mask_cache) >= 256:
                    self._mask_cache.pop(next(iter(self._mask_cache)))
                self._mask_cache[key] = mask
            mean = np.asarray(
                [round(value * 255) for value in IMAGENET_MEAN], dtype=np.uint8
            )
            image = PILImage.fromarray(np.where(mask[..., None], mean, pixels))
        width, height = image.size
        if self.resize == "letterbox":
            width, height = letterbox_dimensions(width, height, self.size)
            image = F.resize(
                image, [height, width], interpolation=InterpolationMode.BILINEAR
            )
            horizontal, vertical = self.size - width, self.size - height
            image = F.pad(
                image,
                [
                    horizontal // 2,
                    vertical // 2,
                    horizontal - horizontal // 2,
                    vertical - vertical // 2,
                ],
                fill=tuple(round(value * 255) for value in IMAGENET_MEAN),
            )
            left, top = horizontal // 2, vertical // 2
            valid_mask = np.zeros((self.size, self.size), dtype=bool)
            valid_mask[top : top + height, left : left + width] = True
        else:
            image = F.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
            image = F.center_crop(image, [self.size, self.size])
            valid_mask = np.ones((self.size, self.size), dtype=bool)
        if self.is_train and self.training_view != "border-frame":
            worker = torch.utils.data.get_worker_info()
            worker_seed = worker.seed if worker else self.seed
            with torch.random.fork_rng(devices=[]):
                if worker_seed in self._states:
                    torch.set_rng_state(self._states[worker_seed])
                else:
                    torch.manual_seed(worker_seed)
                image = self.augmentation(image)
                self._states[worker_seed] = torch.get_rng_state()
        if self.training_view == "border-frame":
            pixels = np.asarray(image)
            width = self.border_frame_width_px
            frame = np.zeros((self.size, self.size), dtype=bool)
            valid_y, valid_x = np.where(valid_mask)
            top, bottom = int(valid_y.min()), int(valid_y.max()) + 1
            left, right = int(valid_x.min()), int(valid_x.max()) + 1
            frame[top : top + width, left:right] = True
            frame[bottom - width : bottom, left:right] = True
            frame[top:bottom, left : left + width] = True
            frame[top:bottom, right - width : right] = True
            frame &= valid_mask
            mean = np.asarray(
                [round(value * 255) for value in IMAGENET_MEAN], dtype=np.uint8
            )
            image = PILImage.fromarray(np.where(frame[..., None], pixels, mean))
        if not self.is_dl:
            pixels = np.asarray(image)
            return ImageWithMask(pixels, valid_mask) if self.include_mask else pixels
        return F.normalize(F.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)

    @property
    def augmentation(self) -> Compose:
        return Compose(
            [
                RandomResizedCrop(self.size, scale=(0.5, 1.0)),
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )


def create_pipeline(
    size: int = IMAGE_SIZE,
    is_train: bool = False,
    is_dl: bool = True,
    seed: int = DEFAULT_SEED,
    include_mask: bool = False,
    *,
    resize: str = "letterbox",
    training_view: str = "raw-training",
    mask_checkpoint: Path | None = None,
    fixture_masks: bool = False,
    border_frame_width_px: int = 0,
) -> ImagePipeline:
    """Build the fixed ADR 0002 preprocessing and optional training augmentation."""
    if size < 1:
        raise ValueError("Image size must be positive")
    return ImagePipeline(
        size,
        is_train,
        is_dl,
        seed,
        include_mask,
        resize=resize,
        training_view=training_view,
        mask_checkpoint=mask_checkpoint,
        fixture_masks=fixture_masks,
        border_frame_width_px=border_frame_width_px,
    )


def classical_pipeline(
    size: int = IMAGE_SIZE,
    *,
    include_mask: bool = False,
    resize: str = "letterbox",
    training_view: str = "raw-training",
    mask_checkpoint: Path | None = None,
    fixture_masks: bool = False,
    border_frame_width_px: int = 0,
) -> ImagePipeline:
    """Return deterministic letterboxed RGB arrays for classical features."""
    return create_pipeline(
        size,
        is_dl=False,
        include_mask=include_mask,
        resize=resize,
        training_view=training_view,
        mask_checkpoint=mask_checkpoint,
        fixture_masks=fixture_masks,
        border_frame_width_px=border_frame_width_px,
    )
