"""Star centroid detection via sigma-clipped background thresholding.

Mirrors the robust background estimation in `normalization.stats` (and
`preprocessing.hot_pixels`): the sky background is estimated by iterative
sigma-clipping, then connected groups of pixels well above it are treated
as stars, ranked by total flux.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

_DEFAULT_SIGMA = 3.0
_DEFAULT_MAX_ITERS = 10


def _background_stats(data: np.ndarray, sigma: float = _DEFAULT_SIGMA, max_iters: int = _DEFAULT_MAX_ITERS) -> tuple[float, float]:
    """Iteratively sigma-clipped (median, robust std) of an array."""
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

    return float(np.median(working)), float(robust_std)


def detect_star_centroids(
    data: np.ndarray,
    threshold_sigma: float = 5.0,
    min_pixels: int = 3,
    max_stars: int = 25,
) -> np.ndarray:
    """(x, y) centroids of the `max_stars` brightest star-like blobs, brightest first.

    A blob is a connected group of at least `min_pixels` pixels more than
    `threshold_sigma` robust standard deviations above the sigma-clipped
    background - bright and large enough to not be a stray hot pixel or
    noise spike. Returns an (N, 2) array, possibly empty.
    """
    background, robust_std = _background_stats(data)
    if robust_std == 0:
        return np.empty((0, 2))

    above_background = data.astype(np.float64) - background
    mask = above_background > threshold_sigma * robust_std

    labeled, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return np.empty((0, 2))

    labels = np.arange(1, num_labels + 1)
    sizes = ndimage.sum_labels(mask, labeled, labels)
    kept_labels = labels[sizes >= min_pixels]
    if kept_labels.size == 0:
        return np.empty((0, 2))

    fluxes = ndimage.sum_labels(above_background, labeled, kept_labels)
    centroids = ndimage.center_of_mass(above_background, labeled, kept_labels)  # (row, col) per label

    brightest_first = np.argsort(fluxes)[::-1][:max_stars]
    return np.array([(centroids[i][1], centroids[i][0]) for i in brightest_first])
