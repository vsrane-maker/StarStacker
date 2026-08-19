"""Hot/dead pixel suppression via median-filter outlier replacement.

This is a cheap, local-statistics pass done before calibration; it is not a
substitute for dark-frame subtraction, just cleanup of single-pixel sensor
defects that would otherwise get treated as stars later in the pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter

from starstacker.io.frame import RawFrame

_DEFAULT_KERNEL_SIZE = 3


def remove_hot_pixels(
    frame: RawFrame, threshold: float = 5.0, kernel_size: int = _DEFAULT_KERNEL_SIZE
) -> RawFrame:
    """Replace pixels that deviate from their local median by more than
    `threshold` robust standard deviations (via median absolute deviation).

    Works on both mono (H, W) and color (H, W, C) data, filtering each
    channel independently.
    """
    data = frame.data
    if data.ndim == 2:
        cleaned = _clean_channel(data, threshold, kernel_size)
    elif data.ndim == 3:
        cleaned = np.stack(
            [_clean_channel(data[..., c], threshold, kernel_size) for c in range(data.shape[-1])],
            axis=-1,
        )
    else:
        raise ValueError(f"Unsupported frame ndim={data.ndim} for {frame.source_path}")

    return frame.with_data(cleaned)


def _clean_channel(channel: np.ndarray, threshold: float, kernel_size: int) -> np.ndarray:
    original_dtype = channel.dtype
    working = channel.astype(np.float32)

    local_median = median_filter(working, size=kernel_size)
    deviation = np.abs(working - local_median)

    mad = np.median(deviation)
    robust_std = mad * 1.4826 if mad > 0 else np.std(working)
    if robust_std == 0:
        return channel

    outliers = deviation > threshold * robust_std
    cleaned = np.where(outliers, local_median, working)
    return cleaned.astype(original_dtype)
