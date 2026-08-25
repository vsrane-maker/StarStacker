from pathlib import Path

import numpy as np
import pytest

from starstacker.io.frame import FrameMetadata, FrameType, RawFrame
from starstacker.normalization.normalize import FrameStats, compute_frame_stats, normalize_frame
from starstacker.normalization.pipeline import NormalizationConfig, NormalizationPipeline
from starstacker.normalization.stats import background_level, frame_background


def _make_frame(data: np.ndarray) -> RawFrame:
    return RawFrame(
        data=data,
        metadata=FrameMetadata(),
        frame_type=FrameType.LIGHT,
        source_path=Path("synthetic.fits"),
    )


def test_background_level_ignores_bright_outliers():
    rng = np.random.default_rng(0)
    sky = rng.normal(loc=100.0, scale=1.0, size=(64, 64))
    sky[0, 0] = 60000.0  # star
    sky[10, 10] = 55000.0  # star

    level = background_level(sky)

    assert level == pytest.approx(100.0, abs=1.0)


def test_frame_background_is_scalar_for_mono_frame():
    data = np.full((16, 16), 200.0, dtype=np.float32)
    frame = _make_frame(data)

    background = frame_background(frame)

    assert background.shape == ()
    assert background == pytest.approx(200.0)


def test_frame_background_is_per_channel_for_color_frame():
    data = np.zeros((16, 16, 3), dtype=np.float32)
    data[..., 0] = 100.0
    data[..., 1] = 200.0
    data[..., 2] = 300.0
    frame = _make_frame(data)

    background = frame_background(frame)

    np.testing.assert_allclose(background, [100.0, 200.0, 300.0])


def test_normalize_frame_shifts_background_to_match_reference():
    frame = _make_frame(np.full((8, 8), 150.0, dtype=np.float32))
    reference = FrameStats(background=np.array(100.0))

    result = normalize_frame(frame, reference)

    np.testing.assert_allclose(result.data, np.full((8, 8), 100.0, dtype=np.float32))


def test_normalize_frame_shifts_each_channel_independently():
    data = np.zeros((8, 8, 2), dtype=np.float32)
    data[..., 0] = 50.0
    data[..., 1] = 150.0
    frame = _make_frame(data)
    reference = FrameStats(background=np.array([100.0, 100.0]))

    result = normalize_frame(frame, reference)

    np.testing.assert_allclose(result.data[..., 0], np.full((8, 8), 100.0, dtype=np.float32))
    np.testing.assert_allclose(result.data[..., 1], np.full((8, 8), 100.0, dtype=np.float32))


def test_normalize_frame_accepts_precomputed_stats():
    frame = _make_frame(np.full((4, 4), 150.0, dtype=np.float32))
    stats = compute_frame_stats(frame)
    reference = FrameStats(background=np.array(120.0))

    result = normalize_frame(frame, reference, stats=stats)

    np.testing.assert_allclose(result.data, np.full((4, 4), 120.0, dtype=np.float32))


def test_normalization_pipeline_matches_all_frames_to_first_by_default():
    frames = [
        _make_frame(np.full((4, 4), 100.0, dtype=np.float32)),
        _make_frame(np.full((4, 4), 130.0, dtype=np.float32)),
        _make_frame(np.full((4, 4), 80.0, dtype=np.float32)),
    ]
    pipeline = NormalizationPipeline(NormalizationConfig(reference="first"))

    results = pipeline.run_batch(frames)

    for result in results:
        np.testing.assert_allclose(result.data, np.full((4, 4), 100.0, dtype=np.float32))


def test_normalization_pipeline_mean_reference_matches_batch_average():
    frames = [
        _make_frame(np.full((4, 4), 100.0, dtype=np.float32)),
        _make_frame(np.full((4, 4), 200.0, dtype=np.float32)),
    ]
    pipeline = NormalizationPipeline(NormalizationConfig(reference="mean"))

    results = pipeline.run_batch(frames)

    for result in results:
        np.testing.assert_allclose(result.data, np.full((4, 4), 150.0, dtype=np.float32))


def test_normalization_pipeline_returns_empty_for_no_frames():
    pipeline = NormalizationPipeline(NormalizationConfig())

    assert pipeline.run_batch([]) == []


def test_normalization_pipeline_rejects_unknown_reference():
    with pytest.raises(ValueError):
        NormalizationPipeline(NormalizationConfig(reference="bogus"))
