from pathlib import Path

import numpy as np
import pytest

from starstacker.calibration.calibrate import calibrate_frame
from starstacker.calibration.master_frames import build_master_bias, build_master_dark, build_master_flat
from starstacker.calibration.pipeline import CalibrationConfig, CalibrationPipeline
from starstacker.io.frame import FrameMetadata, FrameType, RawFrame


def _make_frame(data: np.ndarray) -> RawFrame:
    return RawFrame(
        data=data,
        metadata=FrameMetadata(),
        frame_type=FrameType.LIGHT,
        source_path=Path("synthetic.fits"),
    )


def test_build_master_bias_is_median_of_frames():
    frames = [_make_frame(np.full((4, 4), v, dtype=np.float32)) for v in (10.0, 20.0, 30.0)]

    master_bias = build_master_bias(frames)

    np.testing.assert_allclose(master_bias, np.full((4, 4), 20.0, dtype=np.float32))


def test_build_master_bias_returns_none_when_no_frames():
    assert build_master_bias([]) is None


def test_build_master_dark_subtracts_bias():
    dark_frames = [_make_frame(np.full((4, 4), 50.0, dtype=np.float32))]
    master_bias = np.full((4, 4), 10.0, dtype=np.float32)

    master_dark = build_master_dark(dark_frames, master_bias)

    np.testing.assert_allclose(master_dark, np.full((4, 4), 40.0, dtype=np.float32))


def test_build_master_flat_normalizes_to_mean_one():
    flat_frames = [_make_frame(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32))]

    master_flat = build_master_flat(flat_frames, master_bias=None)

    assert master_flat.mean() == pytest.approx(1.0)


def test_stack_median_rejects_mismatched_shapes():
    frames = [
        _make_frame(np.zeros((4, 4), dtype=np.float32)),
        _make_frame(np.zeros((8, 8), dtype=np.float32)),
    ]

    with pytest.raises(ValueError):
        build_master_bias(frames)


def test_calibrate_frame_applies_bias_dark_and_flat():
    light = _make_frame(np.full((4, 4), 100.0, dtype=np.float32))
    master_bias = np.full((4, 4), 10.0, dtype=np.float32)
    master_dark = np.full((4, 4), 20.0, dtype=np.float32)
    master_flat = np.full((4, 4), 2.0, dtype=np.float32)

    result = calibrate_frame(light, master_bias, master_dark, master_flat)

    # (100 - 10 - 20) / 2 = 35
    np.testing.assert_allclose(result.data, np.full((4, 4), 35.0, dtype=np.float32))


def test_calibration_pipeline_skips_missing_master_types():
    light = _make_frame(np.full((4, 4), 100.0, dtype=np.float32))
    pipeline = CalibrationPipeline(CalibrationConfig())

    pipeline.build_masters(bias_frames=[], dark_frames=[], flat_frames=[])
    result = pipeline.run(light)

    np.testing.assert_allclose(result.data, np.full((4, 4), 100.0, dtype=np.float32))


def test_calibration_pipeline_end_to_end():
    bias_frames = [_make_frame(np.full((4, 4), 10.0, dtype=np.float32))]
    dark_frames = [_make_frame(np.full((4, 4), 30.0, dtype=np.float32))]
    flat_frames = [_make_frame(np.full((4, 4), 5.0, dtype=np.float32))]
    light = _make_frame(np.full((4, 4), 100.0, dtype=np.float32))

    pipeline = CalibrationPipeline(CalibrationConfig())
    pipeline.build_masters(bias_frames, dark_frames, flat_frames)
    result = pipeline.run(light)

    # master_dark = 30 - 10 = 20; master_flat = (5 - 10) normalized -> mean(-5)=-5 -> flat/-5 = 1.0
    # (100 - 10 - 20) / 1.0 = 70
    np.testing.assert_allclose(result.data, np.full((4, 4), 70.0, dtype=np.float32))
