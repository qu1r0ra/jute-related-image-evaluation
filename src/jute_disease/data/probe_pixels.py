"""Pixel arithmetic of the border probe, shared by training and evaluation.

The border probe asks whether a model can classify from the strip around the edge
of the resized image alone. The same frame is cut at fit time and at inference,
so it lives with the data pipeline rather than with the evaluation statistics.
"""

import numpy as np

FRAME_WIDTH_PX = 32
BORDER_VARIANT = "border"
PAD_CUE_VARIANT = "pad_cue"
PROBE_VARIANTS = (BORDER_VARIANT, PAD_CUE_VARIANT)


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
