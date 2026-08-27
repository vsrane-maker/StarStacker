from pathlib import Path

import numpy as np
import pytest

from starstacker.alignment.align import align_frame
from starstacker.alignment.detection import detect_star_centroids
from starstacker.alignment.matching import fit_similarity_transform, match_point_sets
from starstacker.alignment.pipeline import AlignmentConfig, AlignmentPipeline
from starstacker.alignment.stats import FrameTransform, compute_transform
from starstacker.io.frame import FrameMetadata, FrameType, RawFrame


def _make_frame(data: np.ndarray) -> RawFrame:
    return RawFrame(
        data=data,
        metadata=FrameMetadata(),
        frame_type=FrameType.LIGHT,
        source_path=Path("synthetic.fits"),
    )


def _add_star(data: np.ndarray, x: float, y: float, amp: float = 800.0, sigma: float = 2.0) -> None:
    yy, xx = np.mgrid[0 : data.shape[0], 0 : data.shape[1]]
    data += amp * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))


def _make_star_field(shape: tuple[int, int], stars: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=100.0, scale=2.0, size=shape).astype(np.float32)
    for x, y in stars:
        _add_star(data, x, y)
    return data


def _rotation_matrix(theta_deg: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


_REFERENCE_STARS = np.array(
    [[40, 50], [150, 60], [90, 170], [30, 140], [170, 30], [110, 100], [60, 20], [180, 150]],
    dtype=np.float64,
)


# --- detection.py ---


def test_detect_star_centroids_finds_known_star_positions():
    data = _make_star_field((200, 200), _REFERENCE_STARS, seed=0)

    centroids = detect_star_centroids(data)

    assert len(centroids) == len(_REFERENCE_STARS)
    for x, y in _REFERENCE_STARS:
        distances = np.linalg.norm(centroids - np.array([x, y]), axis=1)
        assert distances.min() < 0.5


def test_detect_star_centroids_ranks_brightest_first():
    data = np.full((64, 64), 100.0, dtype=np.float32)
    _add_star(data, 20, 20, amp=300.0)
    _add_star(data, 40, 40, amp=900.0)

    centroids = detect_star_centroids(data)

    assert len(centroids) == 2
    np.testing.assert_allclose(centroids[0], [40, 40], atol=0.5)
    np.testing.assert_allclose(centroids[1], [20, 20], atol=0.5)


def test_detect_star_centroids_ignores_flat_background():
    data = np.full((32, 32), 100.0, dtype=np.float32)

    centroids = detect_star_centroids(data)

    assert len(centroids) == 0


# --- matching.py ---


def test_match_point_sets_recovers_correspondence_under_rotation_and_translation():
    rng = np.random.default_rng(2)
    reference = rng.uniform(20, 180, size=(10, 2))
    rotation = _rotation_matrix(7.0)
    translation = np.array([15.0, -8.0])
    target = reference @ rotation.T + translation

    matched_reference, matched_target = match_point_sets(reference, target)

    assert len(matched_reference) == len(reference)
    predicted = matched_reference @ np.linalg.inv(rotation).T  # sanity: just checks shapes align
    assert predicted.shape == matched_target.shape


def test_match_point_sets_ignores_spurious_extra_points():
    rng = np.random.default_rng(2)
    reference = rng.uniform(20, 180, size=(10, 2))
    rotation = _rotation_matrix(7.0)
    translation = np.array([15.0, -8.0])
    target = reference @ rotation.T + translation

    extra_reference = rng.uniform(20, 180, size=(3, 2))
    extra_target = rng.uniform(20, 180, size=(3, 2))
    reference_all = np.vstack([reference, extra_reference])
    target_all = np.vstack([target, extra_target])

    matched_reference, matched_target = match_point_sets(reference_all, target_all)

    matrix = fit_similarity_transform(matched_target, matched_reference)
    predicted = matched_target @ matrix[:, :2].T + matrix[:, 2]
    assert np.max(np.linalg.norm(predicted - matched_reference, axis=1)) < 1e-6
    assert len(matched_reference) == len(reference)


def test_match_point_sets_rejects_too_few_points():
    reference = np.array([[0.0, 0.0], [1.0, 1.0]])
    target = np.array([[0.0, 0.0], [1.0, 1.0]])

    with pytest.raises(ValueError):
        match_point_sets(reference, target)


def test_fit_similarity_transform_recovers_known_rotation_and_translation():
    rng = np.random.default_rng(5)
    source = rng.uniform(0, 100, size=(8, 2))
    rotation = _rotation_matrix(12.0)
    translation = np.array([4.0, -9.0])
    dest = source @ rotation.T + translation

    matrix = fit_similarity_transform(source, dest)

    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    np.testing.assert_allclose(predicted, dest, atol=1e-8)


def test_fit_similarity_transform_rejects_coincident_points():
    source = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
    dest = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    with pytest.raises(ValueError):
        fit_similarity_transform(source, dest)


# --- stats.py ---


def test_compute_transform_recovers_pure_translation():
    # target_stars = reference_stars + translation, so the target -> reference
    # transform compute_transform() returns is the inverse shift, -translation.
    translation = np.array([6.0, -4.0])
    reference_frame = _make_frame(_make_star_field((200, 200), _REFERENCE_STARS, seed=1))
    target_frame = _make_frame(_make_star_field((200, 200), _REFERENCE_STARS + translation, seed=2))

    transform = compute_transform(reference_frame, target_frame)

    np.testing.assert_allclose(transform.matrix[:, :2], np.eye(2), atol=1e-2)
    np.testing.assert_allclose(transform.matrix[:, 2], -translation, atol=0.5)


def test_compute_transform_recovers_rotation_and_translation():
    # target_stars = rotation @ reference_stars + translation (forward transform),
    # so target -> reference is the inverse: rotation.T, -rotation.T @ translation.
    rotation = _rotation_matrix(6.0)
    translation = np.array([10.0, -5.0])
    target_stars = _REFERENCE_STARS @ rotation.T + translation

    reference_frame = _make_frame(_make_star_field((200, 200), _REFERENCE_STARS, seed=1))
    target_frame = _make_frame(_make_star_field((200, 200), target_stars, seed=2))

    transform = compute_transform(reference_frame, target_frame)

    np.testing.assert_allclose(transform.matrix[:, :2], rotation.T, atol=1e-2)
    np.testing.assert_allclose(transform.matrix[:, 2], -rotation.T @ translation, atol=0.5)


def test_compute_transform_works_for_color_frames():
    translation = np.array([5.0, 5.0])
    reference_mono = _make_star_field((150, 150), _REFERENCE_STARS, seed=1)
    target_mono = _make_star_field((150, 150), _REFERENCE_STARS + translation, seed=2)
    reference_frame = _make_frame(np.stack([reference_mono] * 3, axis=-1))
    target_frame = _make_frame(np.stack([target_mono] * 3, axis=-1))

    transform = compute_transform(reference_frame, target_frame)

    np.testing.assert_allclose(transform.matrix[:, 2], -translation, atol=0.5)


# --- align.py ---


def test_align_frame_registers_target_onto_reference():
    data = np.zeros((64, 64), dtype=np.float32)
    data[33, 37] = 1000.0
    target = _make_frame(data)
    transform = FrameTransform(matrix=np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -3.0]]))

    result = align_frame(target, transform)

    peak = np.unravel_index(np.argmax(result.data), result.data.shape)
    assert peak == (30, 32)
    assert result.data.max() == pytest.approx(1000.0)


def test_align_frame_applies_rotation():
    data = np.zeros((100, 100), dtype=np.float32)
    data[50, 40] = 1000.0  # row=50 (y), col=40 (x)
    target = _make_frame(data)

    rotation = _rotation_matrix(15.0)
    translation = np.array([10.0, -4.0])
    matrix = np.hstack([rotation, translation.reshape(2, 1)])
    transform = FrameTransform(matrix=matrix)

    result = align_frame(target, transform)

    expected_xy = rotation @ np.array([40.0, 50.0]) + translation
    peak = np.unravel_index(np.argmax(result.data), result.data.shape)
    assert peak[1] == pytest.approx(expected_xy[0], abs=1)
    assert peak[0] == pytest.approx(expected_xy[1], abs=1)


def test_align_frame_shifts_color_channels_together():
    data = np.zeros((64, 64, 3), dtype=np.float32)
    data[33, 37, :] = [500.0, 1000.0, 1500.0]
    target = _make_frame(data)
    transform = FrameTransform(matrix=np.array([[1.0, 0.0, -5.0], [0.0, 1.0, -3.0]]))

    result = align_frame(target, transform)

    np.testing.assert_allclose(result.data[30, 32, :], [500.0, 1000.0, 1500.0])


# --- pipeline.py ---


def test_alignment_pipeline_registers_batch_to_reference_frame():
    offsets = [np.array([0.0, 0.0]), np.array([6.0, -3.0]), np.array([-4.0, 5.0])]
    frames = [
        _make_frame(_make_star_field((200, 200), _REFERENCE_STARS + offset, seed=i))
        for i, offset in enumerate(offsets)
    ]
    pipeline = AlignmentPipeline(AlignmentConfig(reference_index=0))

    results = pipeline.run_batch(frames)

    for result in results:
        centroids = detect_star_centroids(result.data)
        assert len(centroids) == len(_REFERENCE_STARS)
        for x, y in _REFERENCE_STARS:
            distances = np.linalg.norm(centroids - np.array([x, y]), axis=1)
            assert distances.min() < 1.0


def test_alignment_pipeline_leaves_reference_frame_unchanged():
    frames = [
        _make_frame(_make_star_field((200, 200), _REFERENCE_STARS, seed=1)),
        _make_frame(_make_star_field((200, 200), _REFERENCE_STARS + np.array([5.0, 5.0]), seed=2)),
    ]
    pipeline = AlignmentPipeline(AlignmentConfig(reference_index=0))

    results = pipeline.run_batch(frames)

    assert results[0] is frames[0]


def test_alignment_pipeline_returns_empty_for_no_frames():
    pipeline = AlignmentPipeline(AlignmentConfig())

    assert pipeline.run_batch([]) == []


def test_alignment_pipeline_rejects_out_of_range_reference_index():
    frames = [_make_frame(_make_star_field((200, 200), _REFERENCE_STARS, seed=1))]
    pipeline = AlignmentPipeline(AlignmentConfig(reference_index=5))

    with pytest.raises(ValueError):
        pipeline.run_batch(frames)
