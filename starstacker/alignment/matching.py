"""Point-set matching and rigid/similarity transform fitting for star alignment.

Two star centroid lists have no known correspondence: the same star can sit
at a different index in each frame's list, and the lists can have different
lengths (dropped detections, extra noise blobs). This module recovers the
correspondence from geometry alone and then fits a transform to it:

  1. `match_point_sets` uses triangle invariants (Groth 1986): a triangle's
     *sorted* side-length ratios are unchanged by translation, rotation,
     reflection, and uniform scaling, so two triangles with matching ratios
     are (very likely) the same three stars seen in both frames. Matching
     every triangle and voting on the point pairs it implies is robust to
     the occasional coincidental/spurious match - a genuine correspondence
     shows up in many mutually consistent triangles, a false one rarely does.
  2. `fit_similarity_transform` takes the matched pairs and solves for the
     single rotation + uniform scale + translation that best explains them,
     via Umeyama's (1991) closed-form least-squares solution.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

_MIN_POINTS_FOR_TRIANGLE = 3
_MIN_CORRESPONDENCES_FOR_FIT = 3
_DEGENERATE_SIDE_LENGTH = 1e-6


def _triangle_features(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every triangle of 3 points, compute:

    - vertex indices in canonical order (opposite side length ascending)
    - (r1, r2): the two shorter sides as a fraction of the longest (r1 <= r2 <= 1)
    - winding: +-1, the orientation (cross product sign) of the canonical vertices

    Degenerate triangles (near-zero longest side) are dropped.
    """
    n = len(points)
    triples = np.array(list(combinations(range(n), 3)))
    triangles = points[triples]  # (T, 3, 2)

    side_lengths = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1),
        ],
        axis=1,
    )  # (T, 3), column i = length of side opposite vertex i

    order = np.argsort(side_lengths, axis=1)
    sides_sorted = np.take_along_axis(side_lengths, order, axis=1)
    longest = sides_sorted[:, 2]
    valid = longest > _DEGENERATE_SIDE_LENGTH
    safe_longest = np.where(valid, longest, 1.0)

    ratios = np.stack([sides_sorted[:, 0] / safe_longest, sides_sorted[:, 1] / safe_longest], axis=1)
    vertex_order = np.take_along_axis(triples, order, axis=1)  # global point indices, canonical order

    canonical = points[vertex_order]
    edge1 = canonical[:, 1] - canonical[:, 0]
    edge2 = canonical[:, 2] - canonical[:, 0]
    winding = np.sign(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])

    return vertex_order[valid], ratios[valid], winding[valid]


def match_point_sets(
    reference_points: np.ndarray,
    target_points: np.ndarray,
    ratio_tolerance: float = 0.01,
    vote_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Find corresponding points between two unordered star centroid sets.

    Returns `(matched_reference_points, matched_target_points)`, same length,
    row i in one corresponding to row i in the other (brightest correspondences
    are not guaranteed first - order reflects vote strength).

    `vote_fraction` keeps only point pairs voted for by at least
    `vote_fraction` times the strongest pair's vote count: a true
    correspondence participates in many mutually consistent triangles, while
    a coincidental ratio match does not, so this cleanly separates the two
    (see module docstring).

    Raises ValueError if either set has fewer than 3 points, or if fewer
    than 3 correspondences survive matching (not enough to fit a transform).
    """
    if len(reference_points) < _MIN_POINTS_FOR_TRIANGLE or len(target_points) < _MIN_POINTS_FOR_TRIANGLE:
        raise ValueError(
            f"Need at least {_MIN_POINTS_FOR_TRIANGLE} stars in each frame to match "
            f"triangles; got {len(reference_points)} reference, {len(target_points)} target"
        )

    ref_vertices, ref_ratios, ref_winding = _triangle_features(reference_points)
    tgt_vertices, tgt_ratios, tgt_winding = _triangle_features(target_points)

    votes: dict[tuple[int, int], int] = {}
    for sign in (1.0, -1.0):
        ref_mask = ref_winding == sign
        tgt_mask = tgt_winding == sign
        if not ref_mask.any() or not tgt_mask.any():
            continue

        tgt_tree = cKDTree(tgt_ratios[tgt_mask])
        tgt_triangle_indices = np.nonzero(tgt_mask)[0]
        neighbor_lists = tgt_tree.query_ball_point(ref_ratios[ref_mask], r=ratio_tolerance)

        for ref_triangle_idx, neighbors in zip(np.nonzero(ref_mask)[0], neighbor_lists):
            if not neighbors:
                continue
            ref_triangle = ref_vertices[ref_triangle_idx]
            for local_tgt_idx in neighbors:
                tgt_triangle = tgt_vertices[tgt_triangle_indices[local_tgt_idx]]
                for ref_point_idx, tgt_point_idx in zip(ref_triangle, tgt_triangle):
                    key = (int(ref_point_idx), int(tgt_point_idx))
                    votes[key] = votes.get(key, 0) + 1

    if not votes:
        raise ValueError("No matching star triangles found between frames")

    max_votes = max(votes.values())
    min_votes = max(1, int(vote_fraction * max_votes))
    ranked = sorted(
        (pair for pair in votes.items() if pair[1] >= min_votes),
        key=lambda pair: pair[1],
        reverse=True,
    )

    used_reference: set[int] = set()
    used_target: set[int] = set()
    matched_reference = []
    matched_target = []
    for (ref_idx, tgt_idx), _votes in ranked:
        if ref_idx in used_reference or tgt_idx in used_target:
            continue
        used_reference.add(ref_idx)
        used_target.add(tgt_idx)
        matched_reference.append(reference_points[ref_idx])
        matched_target.append(target_points[tgt_idx])

    if len(matched_reference) < _MIN_CORRESPONDENCES_FOR_FIT:
        raise ValueError(
            f"Only found {len(matched_reference)} confident star correspondence(s); "
            f"need at least {_MIN_CORRESPONDENCES_FOR_FIT}"
        )

    return np.array(matched_reference), np.array(matched_target)


def fit_similarity_transform(source_points: np.ndarray, dest_points: np.ndarray) -> np.ndarray:
    """Least-squares rotation + uniform scale + translation mapping source onto dest.

    Returns a (2, 3) affine matrix `M` such that, in homogeneous form,
    `dest ~= M @ [source_x, source_y, 1]`. Uses Umeyama's (1991) closed-form
    solution (rotation via SVD of the cross-covariance, scale from the ratio
    of variances, translation from the centroids).
    """
    source = np.asarray(source_points, dtype=np.float64)
    dest = np.asarray(dest_points, dtype=np.float64)

    source_mean = source.mean(axis=0)
    dest_mean = dest.mean(axis=0)
    source_centered = source - source_mean
    dest_centered = dest - dest_mean

    source_variance = np.mean(np.sum(source_centered**2, axis=1))
    if source_variance == 0:
        raise ValueError("Source points are coincident; cannot fit a transform")

    covariance = (dest_centered.T @ source_centered) / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)

    sign_fix = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign_fix[-1, -1] = -1

    rotation = u @ sign_fix @ vt
    scale = np.trace(np.diag(singular_values) @ sign_fix) / source_variance
    translation = dest_mean - scale * rotation @ source_mean

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = translation
    return matrix
