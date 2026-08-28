from pathlib import Path

import numpy as np
import pytest

from starstacker.io.frame import FrameMetadata, FrameType, RawFrame
from starstacker.stacking.pipeline import StackingConfig, StackingPipeline
from starstacker.stacking.stack import stack_mean, stack_median, stack_sigma_clipped_mean


def _make_frame(data: np.ndarray, source_path: str = "synthetic.fits") -> RawFrame:
    return RawFrame(
        data=data,
        metadata=FrameMetadata(camera="Test Camera"),
        frame_type=FrameType.LIGHT,
        source_path=Path(source_path),
    )


# --- stack.py ---


def test_stack_mean_averages_frames():
    frames = [_make_frame(np.full((4, 4), v, dtype=np.float32)) for v in (10.0, 20.0, 30.0)]

    result = stack_mean(frames)

    np.testing.assert_allclose(result, np.full((4, 4), 20.0, dtype=np.float32))


def test_stack_median_of_frames():
    frames = [_make_frame(np.full((4, 4), v, dtype=np.float32)) for v in (10.0, 20.0, 90.0)]

    result = stack_median(frames)

    np.testing.assert_allclose(result, np.full((4, 4), 20.0, dtype=np.float32))


def test_stack_rejects_mismatched_shapes():
    frames = [
        _make_frame(np.zeros((4, 4), dtype=np.float32)),
        _make_frame(np.zeros((8, 8), dtype=np.float32)),
    ]

    with pytest.raises(ValueError):
        stack_mean(frames)


def test_stack_sigma_clipped_mean_rejects_cosmic_ray_outlier():
    frames = [_make_frame(np.full((4, 4), 100.0, dtype=np.float32)) for _ in range(7)]
    frames[3].data[0, 0] = 60000.0  # cosmic ray hit in one frame only

    result = stack_sigma_clipped_mean(frames)

    assert result[0, 0] == pytest.approx(100.0, abs=1.0)
    assert result[1, 1] == pytest.approx(100.0)


def test_stack_sigma_clipped_mean_matches_plain_mean_without_outliers():
    # With small N, sigma-clipping can legitimately clip a genuine tail value
    # now and then (small-N robust stats have real sampling variance), so
    # this checks the two combinations agree on average rather than pixel-by-pixel.
    rng = np.random.default_rng(0)
    frames = [_make_frame(rng.normal(100.0, 2.0, (8, 8)).astype(np.float32)) for _ in range(6)]

    clipped = stack_sigma_clipped_mean(frames)
    plain = stack_mean(frames)

    assert np.mean(np.abs(clipped - plain)) < 0.3


def test_stack_sigma_clipped_mean_handles_single_frame():
    frames = [_make_frame(np.full((3, 3), 42.0, dtype=np.float32))]

    result = stack_sigma_clipped_mean(frames)

    np.testing.assert_allclose(result, np.full((3, 3), 42.0, dtype=np.float32))


def test_stack_sigma_clipped_mean_rejects_outlier_in_realistic_noise():
    rng = np.random.default_rng(1)
    frames = [_make_frame(rng.normal(100.0, 2.0, (5, 5)).astype(np.float32)) for _ in range(8)]
    frames[2].data[2, 2] = 5000.0

    result = stack_sigma_clipped_mean(frames)

    assert result[2, 2] == pytest.approx(100.0, abs=3.0)


# --- pipeline.py ---


def test_stacking_pipeline_defaults_to_sigma_clip_mean():
    frames = [_make_frame(np.full((4, 4), 100.0, dtype=np.float32)) for _ in range(5)]
    frames[2].data[0, 0] = 9000.0
    pipeline = StackingPipeline(StackingConfig())

    result = pipeline.run(frames)

    assert result.data[0, 0] == pytest.approx(100.0, abs=1.0)


def test_stacking_pipeline_mean_method():
    frames = [_make_frame(np.full((4, 4), v, dtype=np.float32)) for v in (10.0, 20.0, 30.0)]
    pipeline = StackingPipeline(StackingConfig(method="mean"))

    result = pipeline.run(frames)

    np.testing.assert_allclose(result.data, np.full((4, 4), 20.0, dtype=np.float32))


def test_stacking_pipeline_median_method():
    frames = [_make_frame(np.full((4, 4), v, dtype=np.float32)) for v in (10.0, 20.0, 90.0)]
    pipeline = StackingPipeline(StackingConfig(method="median"))

    result = pipeline.run(frames)

    np.testing.assert_allclose(result.data, np.full((4, 4), 20.0, dtype=np.float32))


def test_stacking_pipeline_preserves_metadata_from_first_frame():
    frames = [_make_frame(np.full((4, 4), v, dtype=np.float32), source_path=f"frame{i}.fits") for i, v in enumerate((10.0, 20.0))]
    pipeline = StackingPipeline(StackingConfig(method="mean"))

    result = pipeline.run(frames)

    assert result.metadata.camera == "Test Camera"
    assert result.frame_type == FrameType.LIGHT
    assert result.source_path.name == "stacked.fits"


def test_stacking_pipeline_rejects_empty_batch():
    pipeline = StackingPipeline(StackingConfig())

    with pytest.raises(ValueError):
        pipeline.run([])


def test_stacking_pipeline_rejects_unknown_method():
    with pytest.raises(ValueError):
        StackingPipeline(StackingConfig(method="bogus"))
