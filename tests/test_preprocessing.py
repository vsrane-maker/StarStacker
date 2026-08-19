from pathlib import Path

import numpy as np

from starstacker.io.frame import FrameMetadata, FrameType, RawFrame
from starstacker.preprocessing.debayer import debayer_frame
from starstacker.preprocessing.hot_pixels import remove_hot_pixels
from starstacker.preprocessing.normalize import normalize_bit_depth
from starstacker.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline


def _make_frame(data: np.ndarray, bayer_pattern: str | None = None, bit_depth: int = 16) -> RawFrame:
    return RawFrame(
        data=data,
        metadata=FrameMetadata(bayer_pattern=bayer_pattern, bit_depth=bit_depth),
        frame_type=FrameType.LIGHT,
        source_path=Path("synthetic.fits"),
    )


def test_debayer_produces_rgb_from_bayer_mosaic():
    mosaic = np.random.randint(0, 65535, size=(32, 32), dtype=np.uint16)
    frame = _make_frame(mosaic, bayer_pattern="RGGB")

    result = debayer_frame(frame)

    assert result.data.shape == (32, 32, 3)


def test_debayer_passes_through_mono_frame():
    mono = np.random.randint(0, 65535, size=(32, 32), dtype=np.uint16)
    frame = _make_frame(mono, bayer_pattern=None)

    result = debayer_frame(frame)

    assert result.data.shape == (32, 32)
    assert np.array_equal(result.data, mono)


def test_remove_hot_pixels_suppresses_spike():
    data = np.full((20, 20), 1000, dtype=np.uint16)
    data[10, 10] = 60000  # single hot pixel
    frame = _make_frame(data)

    result = remove_hot_pixels(frame, threshold=3.0)

    assert result.data[10, 10] < 60000
    # Untouched region stays the same
    assert result.data[0, 0] == 1000


def test_normalize_bit_depth_scales_to_unit_range():
    data = np.array([[0, 65535], [16384, 32768]], dtype=np.uint16)
    frame = _make_frame(data, bit_depth=16)

    result = normalize_bit_depth(frame)

    assert result.data.dtype == np.float32
    assert result.data.max() <= 1.0
    assert result.data.min() >= 0.0
    np.testing.assert_allclose(result.data[0, 0], 0.0)
    np.testing.assert_allclose(result.data[0, 1], 1.0)


def test_pipeline_runs_full_sequence_on_bayer_frame():
    mosaic = np.random.randint(0, 65535, size=(16, 16), dtype=np.uint16)
    mosaic[8, 8] = 65535
    frame = _make_frame(mosaic, bayer_pattern="RGGB")
    pipeline = PreprocessingPipeline(PreprocessingConfig())

    result = pipeline.run(frame)

    assert result.data.shape == (16, 16, 3)
    assert result.data.dtype == np.float32
    assert result.data.max() <= 1.0
