"""Colour-threshold leaf masks and scene calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

SCENE_TAGS = ("clean-background", "field", "unknown")
MASK_RUBRIC_VERSION = "leaf-mask-v1"
SCENE_RUBRIC_VERSION = "scene-tag-v1"
MASK_QUALITY_RATINGS = ("pass", "minor", "fail")
MASK_RUBRIC: dict[str, Any] = {
    "version": MASK_RUBRIC_VERSION,
    "categories": list(MASK_QUALITY_RATINGS),
    "criteria": ["leaf coverage", "clipped lesions", "background leakage"],
    "anchors": {
        "pass": "Leaf and visible lesions are covered; "
        "background leakage is negligible.",
        "minor": "Some edge clipping or leakage is visible, but most leaf tissue "
        "and lesions remain covered.",
        "fail": "Substantial leaf or lesion clipping, or substantial background "
        "leakage, is present.",
    },
}
SCENE_RUBRIC: dict[str, Any] = {
    "version": SCENE_RUBRIC_VERSION,
    "categories": list(SCENE_TAGS),
    "anchors": {
        "clean-background": "Most visible background is a uniform, uncluttered "
        "surface.",
        "field": "The background contains substantial outdoor or field structure.",
        "unknown": "Too little background is visible or the scene cannot be "
        "assigned reliably.",
    },
}


@dataclass(frozen=True)
class MaskPrediction:
    """Binary foreground mask and the automatic signals used to make it."""

    mask: np.ndarray
    model: str
    diagnostics: dict[str, float | int | str | bool]

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("a leaf mask must be a two-dimensional array")
        object.__setattr__(self, "mask", mask)

    def with_mask(
        self,
        mask: np.ndarray,
        *,
        model: str | None = None,
        diagnostics: dict[str, float | int | str | bool] | None = None,
    ) -> MaskPrediction:
        return MaskPrediction(
            np.asarray(mask, dtype=bool),
            model or self.model,
            diagnostics if diagnostics is not None else self.diagnostics,
        )


class MaskBackend(Protocol):
    """Callable boundary for deterministic and learned mask generators."""

    model: str
    version: str

    def predict(self, image: Image.Image) -> MaskPrediction: ...


@dataclass(frozen=True)
class SceneFeatures:
    """Mask-derived background coverage and color uniformity in [0, 1]."""

    background_fraction: float
    color_uniformity: float

    def __post_init__(self) -> None:
        if not (0 <= self.background_fraction <= 1 and 0 <= self.color_uniformity <= 1):
            raise ValueError("scene features must be in [0, 1]")


@dataclass(frozen=True)
class SceneObservation:
    """One labeled development image used to calibrate scene cutoffs."""

    image_id: str
    features: SceneFeatures
    scene_tag: str
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("development image ID is required")
        if self.scene_tag not in SCENE_TAGS:
            raise ValueError(f"unknown scene tag: {self.scene_tag!r}")
        if len(self.source_sha256) != 64:
            raise ValueError("development source must have a SHA-256 digest")


@dataclass(frozen=True)
class SceneThresholds:
    """Thresholds and provenance learned only from a held-apart dev set."""

    background_fraction_min: float
    color_uniformity_min: float
    development_ids: tuple[str, ...]
    development_sources: tuple[tuple[str, str], ...]
    development_digest: str
    score: float
    method: str = "100-image grid search maximizing scene macro-F1"
    mask_model: str = "excess-green-otsu-largest-component"
    mask_version: str = "1"

    def __post_init__(self) -> None:
        if not 0 <= self.background_fraction_min <= 1:
            raise ValueError("background cutoff must be in [0, 1]")
        if not 0 <= self.color_uniformity_min <= 1:
            raise ValueError("uniformity cutoff must be in [0, 1]")
        if len(self.development_ids) != 100 or len(set(self.development_ids)) != 100:
            raise ValueError("scene thresholds must record 100 unique development IDs")
        if len(self.development_sources) != 100:
            raise ValueError("scene thresholds must record 100 development sources")
        if {image_id for image_id, _ in self.development_sources} != set(
            self.development_ids
        ):
            raise ValueError("development source IDs must match development IDs")
        source_hashes = [digest for _, digest in self.development_sources]
        if any(len(digest) != 64 for digest in source_hashes):
            raise ValueError("development sources must use SHA-256 digests")
        if len(set(source_hashes)) != 100:
            raise ValueError("development images must have unique source content")
        if len(self.development_digest) != 64:
            raise ValueError("development digest must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCENE_RUBRIC_VERSION,
            "background_fraction_min": self.background_fraction_min,
            "color_uniformity_min": self.color_uniformity_min,
            "development_ids": list(self.development_ids),
            "development_sources": [
                {"image_id": image_id, "sha256": digest}
                for image_id, digest in self.development_sources
            ],
            "development_digest": self.development_digest,
            "development_count": len(self.development_ids),
            "score": self.score,
            "method": self.method,
            "mask_model": self.mask_model,
            "mask_version": self.mask_version,
            "categories": list(SCENE_TAGS),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneThresholds:
        if data.get("version") != SCENE_RUBRIC_VERSION:
            raise ValueError("unsupported scene-threshold version")
        return cls(
            background_fraction_min=float(data["background_fraction_min"]),
            color_uniformity_min=float(data["color_uniformity_min"]),
            development_ids=tuple(data["development_ids"]),
            development_sources=tuple(
                (str(item["image_id"]), str(item["sha256"]))
                for item in data["development_sources"]
            ),
            development_digest=str(data["development_digest"]),
            score=float(data["score"]),
            method=str(data["method"]),
            mask_model=str(data["mask_model"]),
            mask_version=str(data["mask_version"]),
        )


@dataclass(frozen=True)
class MaskArtifacts:
    """Paths for the stored binary mask, overlay and provenance record."""

    mask_path: Path
    overlay_path: Path
    metadata_path: Path

    def read_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class ColorThresholdBackend:
    """Otsu threshold on excess green followed by largest-component selection."""

    model: str = "excess-green-otsu-largest-component"
    version: str = "1"

    def predict(self, image: Image.Image) -> MaskPrediction:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        red, green, blue = (rgb[..., i].astype(np.int16) for i in range(3))
        excess_green = np.clip(2 * green - red - blue, 0, 255).astype(np.uint8)
        threshold, candidate = cv2.threshold(
            excess_green, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )
        if component_count <= 1:
            mask = np.zeros(excess_green.shape, dtype=bool)
            selected_area = 0
        else:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = labels == largest
            selected_area = int(stats[largest, cv2.CC_STAT_AREA])
        return MaskPrediction(
            mask,
            self.model,
            {
                "otsu_threshold": int(threshold),
                "component_count": int(component_count - 1),
                "selected_component_area": selected_area,
            },
        )


def require_colour_threshold(model: str, version: str) -> None:
    """Refuse any recorded mask backend but the colour-threshold backend."""
    if model != ColorThresholdBackend.model or not version:
        raise ValueError(
            f"only the colour-threshold mask backend ({ColorThresholdBackend.model}) "
            f"is audited; got {model!r} version {version!r}"
        )


def color_uniformity(background_pixels: np.ndarray) -> float:
    """Return one minus mean RGB standard deviation, scaled to [0, 1]."""
    pixels = np.asarray(background_pixels)
    if pixels.size == 0:
        return 0.0
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("background pixels must have shape (n, 3)")
    uniformity = 1.0 - float(np.std(pixels.astype(np.float64), axis=0).mean()) / 255
    return float(np.clip(uniformity, 0, 1))


def extract_scene_features(
    image: Image.Image | np.ndarray, mask: np.ndarray
) -> SceneFeatures:
    """Measure visible background coverage and RGB color uniformity."""
    rgb = np.asarray(
        image.convert("RGB") if isinstance(image, Image.Image) else image,
        dtype=np.uint8,
    )
    foreground = np.asarray(mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or foreground.shape != rgb.shape[:2]:
        raise ValueError("RGB image and mask dimensions must match")
    background = ~foreground
    pixels = rgb[background]
    return SceneFeatures(
        background_fraction=float(background.mean()),
        color_uniformity=color_uniformity(pixels),
    )


def classify_scene(features: SceneFeatures, thresholds: SceneThresholds) -> str:
    """Assign the prespecified scene tag from calibrated background features."""
    if features.background_fraction < thresholds.background_fraction_min:
        return "unknown"
    if features.color_uniformity >= thresholds.color_uniformity_min:
        return "clean-background"
    return "field"


def fit_scene_thresholds(
    observations: list[SceneObservation],
    *,
    mask_model: str = ColorThresholdBackend.model,
    mask_version: str = ColorThresholdBackend.version,
) -> SceneThresholds:
    """Fit two scene cutoffs on exactly 100 labeled development images."""
    if len(observations) != 100:
        raise ValueError("scene calibration requires exactly 100 development images")
    ids = [item.image_id for item in observations]
    labels = [item.scene_tag for item in observations]
    if len(set(ids)) != 100:
        raise ValueError("development image IDs must be unique")
    source_hashes = [item.source_sha256 for item in observations]
    if len(set(source_hashes)) != 100:
        raise ValueError("development images must have unique source content")
    if set(labels) != set(SCENE_TAGS):
        raise ValueError("development images must represent all three scene tags")

    background = np.array([o.features.background_fraction for o in observations])
    uniformity = np.array([o.features.color_uniformity for o in observations])
    category = {name: i for i, name in enumerate(SCENE_TAGS)}
    truth = np.array([category[tag] for tag in labels], dtype=np.int64)
    background_cutoffs = _cutoff_candidates(background)
    uniformity_cutoffs = _cutoff_candidates(uniformity)
    best_score = -1.0
    best_cutoffs = (0.0, 0.0)
    for background_min in background_cutoffs:
        for uniformity_min in uniformity_cutoffs:
            predicted = np.where(
                background < background_min,
                category["unknown"],
                np.where(
                    uniformity >= uniformity_min,
                    category["clean-background"],
                    category["field"],
                ),
            )
            score = _macro_f1(truth, predicted, len(SCENE_TAGS))
            candidate = (score, -background_min, -uniformity_min)
            current = (best_score, -best_cutoffs[0], -best_cutoffs[1])
            if candidate > current:
                best_score = score
                best_cutoffs = (background_min, uniformity_min)
    material = [
        {
            "image_id": item.image_id,
            "scene_tag": item.scene_tag,
            "source_sha256": item.source_sha256,
            "background_fraction": item.features.background_fraction,
            "color_uniformity": item.features.color_uniformity,
        }
        for item in sorted(observations, key=lambda item: item.image_id)
    ]
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SceneThresholds(
        background_fraction_min=best_cutoffs[0],
        color_uniformity_min=best_cutoffs[1],
        development_ids=tuple(sorted(ids)),
        development_sources=tuple(
            sorted((item.image_id, item.source_sha256) for item in observations)
        ),
        development_digest=digest,
        score=best_score,
        mask_model=mask_model,
        mask_version=mask_version,
    )


def save_mask_artifacts(
    image: Image.Image,
    result: MaskPrediction,
    output_dir: Path,
    *,
    image_id: str,
    model_version: str,
    rubric_version: str = MASK_RUBRIC_VERSION,
) -> MaskArtifacts:
    """Write an image mask, red overlay and provenance sidecar without clobbering."""
    if not image_id or not model_version or not rubric_version:
        raise ValueError("image ID, model version and rubric version are required")
    rgb = image.convert("RGB")
    if result.mask.shape != (rgb.height, rgb.width):
        raise ValueError("mask dimensions must match the source image")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:20]
    paths = MaskArtifacts(
        output_dir / f"{stem}.mask.png",
        output_dir / f"{stem}.overlay.png",
        output_dir / f"{stem}.json",
    )
    if any(
        path.exists()
        for path in (paths.mask_path, paths.overlay_path, paths.metadata_path)
    ):
        raise FileExistsError(f"mask artifacts already exist for image {image_id!r}")

    created: list[Path] = []
    try:
        mask_image = Image.fromarray(result.mask.astype(np.uint8) * 255, mode="L")
        with paths.mask_path.open("xb") as handle:
            created.append(paths.mask_path)
            mask_image.save(handle, format="PNG")

        overlay = rgb.convert("RGBA")
        tint = Image.new("RGBA", rgb.size, (255, 0, 0, 0))
        alpha = Image.fromarray(result.mask.astype(np.uint8) * 96, mode="L")
        tint.putalpha(alpha)
        composite = Image.alpha_composite(overlay, tint).convert("RGB")
        with paths.overlay_path.open("xb") as handle:
            created.append(paths.overlay_path)
            composite.save(handle, format="PNG")

        metadata = {
            "image_id": image_id,
            "model": result.model,
            "model_version": model_version,
            "rubric_version": rubric_version,
            "rubric": MASK_RUBRIC,
            "diagnostics": result.diagnostics,
            "mask_sha256": hashlib.sha256(paths.mask_path.read_bytes()).hexdigest(),
            "overlay_sha256": hashlib.sha256(
                paths.overlay_path.read_bytes()
            ).hexdigest(),
        }
        with paths.metadata_path.open("x", encoding="utf-8") as handle:
            created.append(paths.metadata_path)
            json.dump(metadata, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except Exception:
        for created_path in created:
            created_path.unlink(missing_ok=True)
        raise
    return paths


def _cutoff_candidates(values: np.ndarray) -> list[float]:
    ordered = np.unique(np.clip(values.astype(float), 0, 1))
    midpoints = (ordered[:-1] + ordered[1:]) / 2
    return sorted({0.0, 1.0, *(float(value) for value in midpoints)})


def _macro_f1(truth: np.ndarray, predicted: np.ndarray, categories: int) -> float:
    scores = []
    for label in range(categories):
        true_positive = int(np.sum((truth == label) & (predicted == label)))
        false_positive = int(np.sum((truth != label) & (predicted == label)))
        false_negative = int(np.sum((truth == label) & (predicted != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return float(np.mean(scores))
