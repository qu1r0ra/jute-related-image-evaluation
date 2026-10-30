import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from torch.utils.data import Dataset
from tqdm import tqdm

from jute_disease.config import IMAGE_SIZE, ML_FEATURES_DIR
from jute_disease.training.logger import get_logger

logger = get_logger(__name__)


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extractors."""

    @abstractmethod
    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        pass


class CraftedFeatureExtractor(BaseFeatureExtractor):
    """Extracts Color (HSV), Texture (LBP, GLCM), and Shape (HOG) features."""

    def __init__(self) -> None:
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        self.lbp_method = "uniform"

        self.glcm_distances = [1, 3]
        self.glcm_angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        self.glcm_props = [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
            "ASM",
        ]

        self.hog_orientations = 9
        self.hog_pixels_per_cell = (16, 16)
        self.hog_cells_per_block = (2, 2)

        self.gabor_kernels = []
        # Orientations: 0, 45, 90, 135 degrees
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel(
                (21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            self.gabor_kernels.append(kernel)

    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Extract combined color and texture feature vector."""
        if isinstance(img, Image.Image):
            img = np.array(img)

        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        # Color Moments
        h_mean, h_std = np.mean(hsv[:, :, 0]), np.std(hsv[:, :, 0])
        s_mean, s_std = np.mean(hsv[:, :, 1]), np.std(hsv[:, :, 1])
        v_mean, v_std = np.mean(hsv[:, :, 2]), np.std(hsv[:, :, 2])
        color_moments = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std])

        # Color Histogram (8 bins per channel)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        color_hist = hist.flatten()

        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(
            gray, self.lbp_n_points, self.lbp_radius, self.lbp_method
        )
        # Number of bins for 'uniform' LBP is points + 2
        n_bins = self.lbp_n_points + 2
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )

        glcm = graycomatrix(
            gray,
            distances=self.glcm_distances,
            angles=self.glcm_angles,
            levels=256,
            symmetric=True,
            normed=True,
        )

        glcm_features = []
        for prop in self.glcm_props:
            val = graycoprops(glcm, prop).flatten()
            glcm_features.extend(val)
        glcm_features = np.array(glcm_features)

        hog_features = hog(
            gray,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm="L2-Hys",
            transform_sqrt=True,
            feature_vector=True,
        )

        combined = np.hstack(
            [color_moments, color_hist, lbp_hist, glcm_features, hog_features]
        )

        return combined.astype(np.float32)


class RawPixelFeatureExtractor(BaseFeatureExtractor):
    """Extracts raw pixel values as features."""

    def __init__(self, img_size: int = IMAGE_SIZE) -> None:
        self.img_size = img_size

    def extract(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Resize image and flatten pixel values."""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        img_resized = img.resize((self.img_size, self.img_size))
        img_array = np.array(img_resized)
        return img_array.flatten().astype(np.float32)


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
    except (OSError, ValueError, EOFError):
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
