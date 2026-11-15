"""Pixel arithmetic of the probes, shared by training and evaluation.

The border probe asks whether a model can classify from the strip around the edge
of the resized image alone, and the background probe whether it can from the
pixels the leaf mask leaves. The same pixels are cut at fit time and at
inference, so the arithmetic lives with the data pipeline rather than with the
evaluation statistics.
"""

import numpy as np
from PIL import Image

FRAME_WIDTH_PX = 32
BORDER_VARIANT = "border"
PAD_CUE_VARIANT = "pad_cue"
BACKGROUND_VARIANT = "background_only"
PROBE_VARIANTS = (BORDER_VARIANT, PAD_CUE_VARIANT, BACKGROUND_VARIANT)


def _content_box(valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(valid_mask)
    if not len(rows):
        raise ValueError("valid mask has no content")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def frame_mask(valid_mask: np.ndarray, width: int) -> np.ndarray:
    """Pixels within ``width`` px of the edge of the resized image content."""
    if width < 1:
        raise ValueError("frame width must be positive")
    top, bottom, left, right = _content_box(np.asarray(valid_mask, dtype=bool))
    frame = np.zeros(valid_mask.shape, dtype=bool)
    frame[top:bottom, left:right] = True
    if bottom - top > 2 * width and right - left > 2 * width:
        frame[top + width : bottom - width, left + width : right - width] = False
    return frame


def frame_only_image(
    pixels: np.ndarray, valid_mask: np.ndarray, width: int
) -> np.ndarray:
    """Keep the frame, blank the interior and mask the letterbox padding."""
    out = np.zeros_like(pixels)
    keep = frame_mask(valid_mask, width)
    out[keep] = pixels[keep]
    return out


def letterbox_leaf_mask(mask: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Resize an original-image binary mask onto the ADR 0002 letterbox canvas."""
    source = np.asarray(mask, dtype=bool)
    content = np.asarray(content_mask, dtype=bool)
    if source.ndim != 2 or content.ndim != 2 or not content.any():
        raise ValueError("leaf and content masks must be non-empty 2D arrays")
    ys, xs = np.nonzero(content)
    top, bottom = int(ys.min()), int(ys.max()) + 1
    left, right = int(xs.min()), int(xs.max()) + 1
    resized = (
        np.asarray(
            Image.fromarray(source.astype(np.uint8) * 255).resize(
                (right - left, bottom - top), Image.Resampling.NEAREST
            )
        )
        > 0
    )
    result = np.zeros_like(content)
    result[top:bottom, left:right] = resized
    return result


def background_only_image(
    pixels: np.ndarray, leaf_mask: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Mask the leaf and the letterbox padding, keeping only background pixels."""
    out = np.zeros_like(pixels)
    keep = np.asarray(valid_mask, dtype=bool) & ~np.asarray(leaf_mask, dtype=bool)
    out[keep] = pixels[keep]
    return out
