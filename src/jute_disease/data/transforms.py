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

    def __init__(self, size: int, is_train: bool, is_dl: bool, seed: int) -> None:
        self.size = size
        self.is_train = is_train
        self.is_dl = is_dl
        self.seed = seed
        self.deterministic = not is_train
        self.description = f"letterbox={size};train={is_train};dl={is_dl}"
        self._states: dict[int, torch.Tensor] = {}

    def __call__(self, img: Image | np.ndarray) -> torch.Tensor | np.ndarray:
        image = PILImage.fromarray(np.asarray(img)).convert("RGB")
        width, height = image.size
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
        if self.is_train:
            worker = torch.utils.data.get_worker_info()
            worker_seed = worker.seed if worker else self.seed
            with torch.random.fork_rng(devices=[]):
                if worker_seed in self._states:
                    torch.set_rng_state(self._states[worker_seed])
                else:
                    torch.manual_seed(worker_seed)
                image = self.augmentation(image)
                self._states[worker_seed] = torch.get_rng_state()
        if not self.is_dl:
            return np.asarray(image)
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
) -> ImagePipeline:
    """Build the fixed ADR 0002 preprocessing and optional training augmentation."""
    if size < 1:
        raise ValueError("Image size must be positive")
    return ImagePipeline(size, is_train, is_dl, seed)


def classical_pipeline(size: int = IMAGE_SIZE) -> ImagePipeline:
    """Return deterministic letterboxed RGB arrays for classical features."""
    return create_pipeline(size, is_dl=False)
