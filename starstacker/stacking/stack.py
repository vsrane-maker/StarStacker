"""Combines a batch of aligned, normalized frames into a single stacked image.

Averaging N frames improves signal-to-noise by roughly sqrt(N), which is the
whole point of stacking - but a plain average is derailed by anything that
only appears in one or two frames (a cosmic ray hit, a satellite trail, a
plane's strobe). Sigma-clipping rejects those outliers per pixel, across the
stack (not spatially within a frame, as in `preprocessing.hot_pixels`), before
averaging what's left.
"""

from __future__ import annotations

import numpy as np

from starstacker.io.frame import RawFrame

_DEFAULT_SIGMA = 3.0
_DEFAULT_MAX_ITERS = 5

# Floor for the per-pixel robust std, in case a majority of frames agree
# exactly and the MAD collapses to 0 - the minority that disagrees is then
# an outlier by definition (not evidence of zero spread), so clip against a
# near-zero threshold instead of falling back to a raw std that the outlier
# itself would inflate.
_MIN_ROBUST_STD = 1e-6


def _stacked_array(frames: list[RawFrame]) -> np.ndarray:
    shapes = {f.data.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(f"Cannot stack frames with mismatched shapes: {shapes}")
    # float32, not float64: frames are already full-resolution (tens of frames
    # at multiple megapixels each is not unusual), and this function's several
    # same-shaped intermediate arrays make peak memory a real constraint.
    return np.stack([f.data for f in frames], axis=0).astype(np.float32)


def stack_mean(frames: list[RawFrame]) -> np.ndarray:
    """Combine frames by a plain per-pixel mean."""
    return _stacked_array(frames).mean(axis=0).astype(np.float32)


def stack_median(frames: list[RawFrame]) -> np.ndarray:
    """Combine frames by a plain per-pixel median."""
    return np.median(_stacked_array(frames), axis=0).astype(np.float32)


def stack_sigma_clipped_mean(
    frames: list[RawFrame], sigma: float = _DEFAULT_SIGMA, max_iters: int = _DEFAULT_MAX_ITERS
) -> np.ndarray:
    """Combine frames by an iterative per-pixel sigma-clipped mean.

    At each pixel, values more than `sigma` robust standard deviations from
    the current median are discarded, then the median/std are recomputed
    from what's left; repeats until nothing more is clipped or `max_iters`
    is reached.
    """
    stack = _stacked_array(frames)
    mask = np.ones(stack.shape, dtype=bool)

    for _ in range(max_iters):
        masked = np.where(mask, stack, np.nan)
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(masked, axis=0)
            mad = np.nanmedian(np.abs(masked - median), axis=0)
        robust_std = np.maximum(mad * 1.4826, _MIN_ROBUST_STD)

        with np.errstate(invalid="ignore"):
            new_mask = np.abs(stack - median) <= sigma * robust_std
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    result = np.nanmean(np.where(mask, stack, np.nan), axis=0)
    return result.astype(np.float32)
