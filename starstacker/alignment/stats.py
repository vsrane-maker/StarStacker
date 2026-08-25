"""Sub-pixel translational offset estimation between frames via phase correlation.

Bright, sharp stars dominate a frame's frequency content, so cross-correlating
two frames in the Fourier domain (phase correlation) recovers the translation
between them directly, without needing to detect or match individual stars.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starstacker.io.frame import RawFrame


@dataclass
class FrameShift:
    """Sub-pixel (dx, dy) translation, in pixels, of a frame relative to a reference.

    Positive dx/dy mean the frame's content is shifted right/down relative to
    the reference; `align_frame` uses this directly to shift it back.
    """

    dx: float
    dy: float


def _luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data.astype(np.float32)
    elif data.ndim == 3:
        return data.astype(np.float32).mean(axis=-1)
    else:
        raise ValueError(f"Unsupported frame ndim={data.ndim}")


def compute_shift(reference: RawFrame, target: RawFrame) -> FrameShift:
    """Estimate the translation that would register `target` onto `reference`."""
    import cv2

    ref_luma = _luminance(reference.data)
    tgt_luma = _luminance(target.data)
    if ref_luma.shape != tgt_luma.shape:
        raise ValueError(
            f"Shape mismatch aligning {target.source_path} to {reference.source_path}: "
            f"{tgt_luma.shape} vs {ref_luma.shape}"
        )

    window = cv2.createHanningWindow(ref_luma.shape[::-1], cv2.CV_32F)
    (dx, dy), _response = cv2.phaseCorrelate(ref_luma * window, tgt_luma * window)
    return FrameShift(dx=dx, dy=dy)
