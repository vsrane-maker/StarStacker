"""Robust sky-background estimation used to normalize frames before stacking.

Star pixels are bright outliers against the sky background, so a plain mean
or median is skewed by however many stars happen to be in frame. Iterative
sigma-clipping (median absolute deviation, as in `preprocessing.hot_pixels`)
converges on the background level by repeatedly discarding those outliers.
"""

from __future__ import annotations

import numpy as np

from starstacker.io.frame import RawFrame

_DEFAULT_SIGMA = 3.0
_DEFAULT_MAX_ITERS = 10


def background_level(data: np.ndarray, sigma: float = _DEFAULT_SIGMA, max_iters: int = _DEFAULT_MAX_ITERS) -> float:
    """Iteratively sigma-clipped median of a single-channel array.

    Each round drops pixels more than `sigma` robust-std away from the
    current median, then recomputes; stops early once nothing more is
    clipped or a pixel-less std makes further clipping meaningless.
    """
    working = data.astype(np.float64).ravel()

    for _ in range(max_iters):
        median = np.median(working)
        mad = np.median(np.abs(working - median))
        robust_std = mad * 1.4826 if mad > 0 else np.std(working)
        if robust_std == 0:
            break

        mask = np.abs(working - median) <= sigma * robust_std
        if mask.all():
            break
        working = working[mask]

    return float(np.median(working))


def frame_background(frame: RawFrame, sigma: float = _DEFAULT_SIGMA) -> np.ndarray:
    """Background level(s) for a frame: scalar (0-d array) for mono, one per channel for color."""
    data = frame.data
    if data.ndim == 2:
        return np.array(background_level(data, sigma))
    elif data.ndim == 3:
        return np.array([background_level(data[..., c], sigma) for c in range(data.shape[-1])])
    else:
        raise ValueError(f"Unsupported frame ndim={data.ndim} for {frame.source_path}")
