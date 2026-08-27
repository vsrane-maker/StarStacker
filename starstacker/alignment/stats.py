"""Estimates the rotation + scale + translation that registers one frame onto another.

A star pattern ("asterism") looks the same regardless of a frame's rotation,
translation, or minor scale drift, so matching stars between two frames by
their local geometry - not by brightness or position alone - recovers the
transform between them even when it's more than a simple translation (field
rotation from an alt-az mount, minor framing drift, etc).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starstacker.alignment.detection import detect_star_centroids
from starstacker.alignment.matching import fit_similarity_transform, match_point_sets
from starstacker.io.frame import RawFrame

_DEFAULT_MAX_STARS = 25
_DEFAULT_RATIO_TOLERANCE = 0.01
_OUTLIER_REJECTION_PIXELS = 3.0


@dataclass
class FrameTransform:
    """Affine transform registering a target frame's pixels onto a reference's.

    `matrix` is a (2, 3) array; in homogeneous form,
    `reference_xy ~= matrix @ [target_x, target_y, 1]`.
    """

    matrix: np.ndarray


def _luminance(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data.astype(np.float32)
    elif data.ndim == 3:
        return data.astype(np.float32).mean(axis=-1)
    else:
        raise ValueError(f"Unsupported frame ndim={data.ndim}")


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def compute_transform(
    reference: RawFrame,
    target: RawFrame,
    max_stars: int = _DEFAULT_MAX_STARS,
    ratio_tolerance: float = _DEFAULT_RATIO_TOLERANCE,
) -> FrameTransform:
    """Estimate the transform that registers `target` onto `reference` by star-pattern matching.

    A first fit uses every matched star pair; any pair whose residual under
    that fit exceeds `_OUTLIER_REJECTION_PIXELS` is then dropped and the
    transform is refit on the rest, guarding against the rare spurious
    triangle match that survives `match_point_sets`'s vote filter.
    """
    reference_stars = detect_star_centroids(_luminance(reference.data), max_stars=max_stars)
    target_stars = detect_star_centroids(_luminance(target.data), max_stars=max_stars)

    matched_reference, matched_target = match_point_sets(
        reference_stars, target_stars, ratio_tolerance=ratio_tolerance
    )
    matrix = fit_similarity_transform(matched_target, matched_reference)

    residuals = np.linalg.norm(_apply(matrix, matched_target) - matched_reference, axis=1)
    inliers = residuals < _OUTLIER_REJECTION_PIXELS
    if inliers.sum() >= 3 and inliers.sum() < len(matched_reference):
        matrix = fit_similarity_transform(matched_target[inliers], matched_reference[inliers])

    return FrameTransform(matrix=matrix)
