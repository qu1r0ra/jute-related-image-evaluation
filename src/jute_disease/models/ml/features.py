import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from torch.utils.data import Dataset
from tqdm import tqdm

from jute_disease.config import IMAGE_SIZE, ML_FEATURES_DIR
from jute_disease.data.transforms import ImageWithMask
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, img: Image.Image | np.ndarray | ImageWithMask) -> np.ndarray:
        """Extract features from an image."""
        pass


class CraftedFeatureExtractor(BaseFeatureExtractor):
    """Extract the fixed HSV, multi-scale LBP and GLCM descriptors."""

    block_sizes = (512, 28, 8)

    def __init__(self) -> None:
        self.lbp_scales = ((8, 1), (16, 2))
        self.glcm_levels = 32
        self.glcm_distances = (1, 2)
        self.glcm_offsets = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
        self.glcm_properties = ("contrast", "correlation", "energy", "homogeneity")

    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract descriptors while excluding padded pixels and pixel pairs."""
        if isinstance(img, ImageWithMask):
            pixels, valid = img.pixels, img.valid_mask
        else:
            if isinstance(img, Image.Image):
                pixels = np.asarray(img.convert("RGB"))
            else:
                pixels = np.asarray(img)
            valid = np.ones(pixels.shape[:2], dtype=bool)
        if pixels.ndim != 3 or pixels.shape[2] != 3 or valid.shape != pixels.shape[:2]:
            raise ValueError("Expected RGB pixels and a matching two-dimensional mask")

        hsv = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2HSV)
        color_histogram = np.histogramdd(
            hsv[valid], bins=(8, 8, 8), range=((0, 180), (0, 256), (0, 256))
        )[0].ravel()
        color_histogram = self._l1(color_histogram)

        gray = cv2.cvtColor(pixels.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        lbp_features: list[np.ndarray] = []
        for points, radius in self.lbp_scales:
            lbp = local_binary_pattern(gray, points, radius, method="uniform")
            usable = self._erode(valid, radius)
            histogram = np.histogram(
                lbp[usable], bins=points + 2, range=(0, points + 2)
            )[0]
            lbp_features.append(self._l1(histogram))
        lbp_histogram = np.concatenate(lbp_features)
        gray_levels = gray.astype(np.uint16) * self.glcm_levels // 256
        glcm_features = self._glcm_features(gray_levels, valid)
        return np.concatenate((color_histogram, lbp_histogram, glcm_features)).astype(
            np.float32
        )

    @staticmethod
    def _l1(values: np.ndarray) -> np.ndarray:
        total = float(values.sum())
        return values.astype(np.float64) / total if total else values.astype(np.float64)

    @staticmethod
    def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
        return cv2.erode(
            mask.astype(np.uint8),
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

    def _glcm_features(self, gray: np.ndarray, valid: np.ndarray) -> np.ndarray:
        by_distance: list[list[float]] = []
        for distance in self.glcm_distances:
            by_angle: list[list[float]] = []
            for dy_unit, dx_unit in self.glcm_offsets:
                dy, dx = dy_unit * distance, dx_unit * distance
                cooccurrence = self._cooccurrence(gray, valid, dy, dx)
                by_angle.append(self._properties(cooccurrence))
            by_distance.append(np.mean(by_angle, axis=0).tolist())
        return np.asarray(by_distance, dtype=np.float64).ravel()

    def _cooccurrence(
        self, gray: np.ndarray, valid: np.ndarray, dy: int, dx: int
    ) -> np.ndarray:
        height, width = gray.shape
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        left = gray[y0:y1, x0:x1]
        right = gray[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        pair_mask = valid[y0:y1, x0:x1] & valid[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
        pairs = np.bincount(
            (left[pair_mask] * self.glcm_levels + right[pair_mask]).astype(np.int64),
            minlength=self.glcm_levels**2,
        ).reshape(self.glcm_levels, self.glcm_levels)
        symmetric = pairs + pairs.T
        total = symmetric.sum()
        return symmetric / total if total else symmetric.astype(np.float64)

    def _properties(self, matrix: np.ndarray) -> list[float]:
        levels = np.arange(self.glcm_levels, dtype=np.float64)
        difference = levels[:, None] - levels[None, :]
        contrast = float(np.sum(matrix * difference**2))
        energy = float(np.sqrt(np.sum(matrix**2)))
        homogeneity = float(np.sum(matrix / (1 + difference**2)))
        mean_i = float(np.sum(matrix * levels[:, None]))
        mean_j = float(np.sum(matrix * levels[None, :]))
        std_i = float(np.sqrt(np.sum(matrix * (levels[:, None] - mean_i) ** 2)))
        std_j = float(np.sqrt(np.sum(matrix * (levels[None, :] - mean_j) ** 2)))
        correlation = (
            float(
                np.sum(matrix * (levels[:, None] - mean_i) * (levels[None, :] - mean_j))
                / (std_i * std_j)
            )
            if std_i and std_j
            else 1.0
        )
        return [contrast, correlation, energy, homogeneity]


class RawPixelFeatureExtractor(BaseFeatureExtractor):
    """Extracts raw pixel values as features."""

    def __init__(self, img_size: int = IMAGE_SIZE) -> None:
        self.img_size = img_size

    def extract(self, img: Image.Image | np.ndarray | ImageWithMask) -> np.ndarray:
        """Resize image and flatten pixel values."""
        if isinstance(img, ImageWithMask):
            img = img.pixels
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img_resized = img.resize((self.img_size, self.img_size))
        img_array = np.array(img_resized)
        return img_array.flatten().astype(np.float32)


PAD_CUE_FEATURES = "pad-cue"


class PadCueFeatureExtractor(BaseFeatureExtractor):
    """Letterbox geometry alone: the pad fraction and aspect ratio of the image.

    The pad-cue baseline asks how much of a label the shape of the original photo
    already carries, so it reads the validity mask and never a pixel.
    """

    dimensions = 7

    def extract(self, img: Image.Image | np.ndarray | ImageWithMask) -> np.ndarray:
        """Return the pad share per side, content size, pad fraction and aspect."""
        if not isinstance(img, ImageWithMask):
            raise ValueError("The pad-cue features need the letterbox validity mask")
        valid = np.asarray(img.valid_mask, dtype=bool)
        rows, cols = np.nonzero(valid)
        if not len(rows):
            raise ValueError("valid mask has no content")
        height, width = valid.shape
        top, bottom = int(rows.min()), height - int(rows.max()) - 1
        left, right = int(cols.min()), width - int(cols.max()) - 1
        content_h, content_w = height - top - bottom, width - left - right
        return np.array(
            [
                top / height,
                bottom / height,
                left / width,
                right / width,
                content_h / height,
                content_w / width,
                1.0 - float(valid.mean()),
            ],
            dtype=np.float32,
        )


def feature_cache_key(
    dataset_identity: dict[str, str], extractor: BaseFeatureExtractor, length: int
) -> str:
    """Return the SHA-256 identifying one feature matrix.

    The key covers the dataset (manifest, partition, pixel source and its
    preprocessing), the extractor's class and settings, and the sample count, so
    a cached matrix can never be reused for different inputs.
    """
    payload = {
        "dataset": dataset_identity,
        "extractor": {"class": type(extractor).__name__, "config": vars(extractor)},
        "length": length,
    }
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _load_cached(
    x_path: Path, y_path: Path, meta_path: Path, key: str, length: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a whole cached matrix, or None when anything is missing or wrong."""
    if not (x_path.is_file() and y_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") != key or meta.get("length") != length:
            return None
        x = np.load(x_path, allow_pickle=False)
        y = np.load(y_path, allow_pickle=False)
    except OSError, ValueError, EOFError:
        return None
    if x.ndim != 2 or x.shape[0] != length or y.shape != (length,):
        return None
    return x, y


def _publish(
    cache_dir: Path,
    key: str,
    x: np.ndarray,
    y: np.ndarray,
    identity: dict[str, str],
) -> None:
    """Write the matrices atomically and publish the sidecar last.

    A reader only trusts a matrix once its sidecar exists, so an interruption at
    any point leaves nothing that looks complete.
    """
    parts: list[Path] = []
    try:
        for name, array in (("X", x), ("y", y)):
            part = cache_dir / f"{key}_{name}.part{os.getpid()}.npy"
            parts.append(part)
            np.save(part, array)
        meta_part = cache_dir / f"{key}.part{os.getpid()}.json"
        parts.append(meta_part)
        meta_part.write_text(
            json.dumps(
                {"key": key, "length": len(y), "identity": identity}, sort_keys=True
            ),
            encoding="utf-8",
        )
        os.replace(parts[0], cache_dir / f"{key}_X.npy")
        os.replace(parts[1], cache_dir / f"{key}_y.npy")
        os.replace(meta_part, cache_dir / f"{key}.json")
    finally:
        for part in parts:
            part.unlink(missing_ok=True)


def extract_features(
    dataset: Dataset,
    extractor: BaseFeatureExtractor,
    cache: bool = True,
    cache_dir: Path | str | None = None,
    *,
    require_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features with the provided extractor.

    Features are cached only for datasets that expose an ``identity`` mapping
    (such as ``ManifestDataset``). The cache file is named by its identity key
    and carries a sidecar; a missing sidecar, a different key, a different
    sample count or a matrix of the wrong shape means recomputation, never
    reuse. ``cache_dir`` defaults to the shared features directory.
    """
    identity = getattr(dataset, "identity", None) if cache else None
    if identity is not None and identity.get("deterministic") != "true":
        identity = None
    if identity is not None:
        key = feature_cache_key(identity, extractor, len(dataset))
        directory = Path(ML_FEATURES_DIR if cache_dir is None else cache_dir)
        directory.mkdir(parents=True, exist_ok=True)
        cached = _load_cached(
            directory / f"{key}_X.npy",
            directory / f"{key}_y.npy",
            directory / f"{key}.json",
            key,
            len(dataset),
        )
        if cached is not None:
            logger.info(f"Loading cached features: {key[:12]}")
            return cached

    if require_cache:
        raise FileNotFoundError("Prepared feature cache is missing or invalid")

    features_list = []
    labels_list = []

    logger.info(f"Extracting features using {type(extractor).__name__}...")
    for i in tqdm(range(len(dataset)), desc="Feature Extraction"):
        img, label = dataset[i]
        feats = extractor.extract(img)
        features_list.append(feats)
        labels_list.append(label)

    X = np.array(features_list)
    y = np.array(labels_list)

    if identity is not None:
        _publish(directory, key, X, y, identity)
        logger.info(f"Saved cached features: {key[:12]}")

    return X, y


FEATURE_EXTRACTORS: dict[str, type[BaseFeatureExtractor]] = {
    "crafted": CraftedFeatureExtractor,
    "raw": RawPixelFeatureExtractor,
}
