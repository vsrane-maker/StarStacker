from pathlib import Path

import numpy as np
import pytest

from starstacker.alignment.align import align_frame
from starstacker.alignment.pipeline import AlignmentConfig, AlignmentPipeline
from starstacker.alignment.stats import FrameShift, compute_shift
from starstacker.io.frame import FrameMetadata, FrameType, RawFrame


def _make_frame(data: np.ndarray) -> RawFrame:
    return RawFrame(
        data=data,
        metadata=FrameMetadata(),
        frame_type=FrameType.LIGHT,
        source_path=Path("synthetic.fits"),
    )


def _make_star_frame(shape: tuple[int, int], star: tuple[int, int], value: float = 1000.0) -> RawFrame:
    data = np.zeros(shape, dtype=np.float32)
    data[star] = value
    return _make_frame(data)


def _peak(data: np.ndarray) -> tuple[int, int]:
    return np.unravel_index(np.argmax(data), data.shape)


def test_compute_shift_recovers_known_translation():
    reference = _make_star_frame((64, 64), star=(30, 32))
    target = _make_star_frame((64, 64), star=(33, 37))  # +3 rows, +5 cols

    shift = compute_shift(reference, target)

    assert shift.dx == pytest.approx(5.0, abs=0.05)
    assert shift.dy == pytest.approx(3.0, abs=0.05)


def test_compute_shift_is_zero_for_identical_frames():
    reference = _make_star_frame((64, 64), star=(20, 20))
    target = _make_star_frame((64, 64), star=(20, 20))

    shift = compute_shift(reference, target)

    assert shift.dx == pytest.approx(0.0, abs=0.05)
    assert shift.dy == pytest.approx(0.0, abs=0.05)


def test_compute_shift_rejects_mismatched_shapes():
    reference = _make_star_frame((64, 64), star=(20, 20))
    target = _make_star_frame((32, 32), star=(10, 10))

    with pytest.raises(ValueError):
        compute_shift(reference, target)


def test_align_frame_registers_target_onto_reference():
    target = _make_star_frame((64, 64), star=(33, 37))
    shift = FrameShift(dx=5.0, dy=3.0)

    result = align_frame(target, shift)

    assert _peak(result.data) == (30, 32)
    assert result.data.max() == pytest.approx(1000.0)


def test_align_frame_shifts_color_channels_together():
    data = np.zeros((64, 64, 3), dtype=np.float32)
    data[33, 37, :] = [500.0, 1000.0, 1500.0]
    target = _make_frame(data)
    shift = FrameShift(dx=5.0, dy=3.0)

    result = align_frame(target, shift)

    np.testing.assert_allclose(result.data[30, 32, :], [500.0, 1000.0, 1500.0])


def test_alignment_pipeline_registers_batch_to_reference_frame():
    frames = [
        _make_star_frame((64, 64), star=(30, 32)),
        _make_star_frame((64, 64), star=(33, 37)),
        _make_star_frame((64, 64), star=(25, 28)),
    ]
    pipeline = AlignmentPipeline(AlignmentConfig(reference_index=0))

    results = pipeline.run_batch(frames)

    for result in results:
        assert _peak(result.data) == (30, 32)


def test_alignment_pipeline_leaves_reference_frame_unchanged():
    frames = [
        _make_star_frame((64, 64), star=(30, 32)),
        _make_star_frame((64, 64), star=(33, 37)),
    ]
    pipeline = AlignmentPipeline(AlignmentConfig(reference_index=0))

    results = pipeline.run_batch(frames)

    assert results[0] is frames[0]


def test_alignment_pipeline_returns_empty_for_no_frames():
    pipeline = AlignmentPipeline(AlignmentConfig())

    assert pipeline.run_batch([]) == []


def test_alignment_pipeline_rejects_out_of_range_reference_index():
    frames = [_make_star_frame((64, 64), star=(30, 32))]
    pipeline = AlignmentPipeline(AlignmentConfig(reference_index=5))

    with pytest.raises(ValueError):
        pipeline.run_batch(frames)
