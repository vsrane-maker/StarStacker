import numpy as np
import pytest
from astropy.io import fits

from starstacker.io.frame import FrameType
from starstacker.io.loaders import load_fits, load_frame, load_frame_set


def _write_fits(path, data, header_updates=None):
    hdu = fits.PrimaryHDU(data=data)
    for key, value in (header_updates or {}).items():
        hdu.header[key] = value
    hdu.writeto(path)


def test_load_fits_reads_data_and_metadata(tmp_path):
    path = tmp_path / "frame001.fits"
    data = np.random.randint(0, 65535, size=(64, 64), dtype=np.uint16)
    _write_fits(
        path,
        data,
        {
            "EXPTIME": 30.0,
            "GAIN": 100.0,
            "CCD-TEMP": -10.5,
            "FILTER": "L",
            "IMAGETYP": "Light Frame",
            "DATE-OBS": "2026-01-15T03:22:11.500",
        },
    )

    frame = load_fits(path)

    assert frame.data.shape == (64, 64)
    assert frame.frame_type == FrameType.LIGHT
    assert frame.metadata.exposure_time == 30.0
    assert frame.metadata.gain == 100.0
    assert frame.metadata.temperature == -10.5
    assert frame.metadata.filter_name == "L"
    assert frame.metadata.width == 64
    assert frame.metadata.height == 64
    assert frame.metadata.timestamp is not None


def test_frame_type_inferred_from_directory_when_header_missing(tmp_path):
    darks_dir = tmp_path / "darks"
    darks_dir.mkdir()
    path = darks_dir / "dark001.fits"
    data = np.zeros((16, 16), dtype=np.uint16)
    _write_fits(path, data)

    frame = load_fits(path)

    assert frame.frame_type == FrameType.DARK


def test_load_frame_dispatches_by_extension(tmp_path):
    path = tmp_path / "frame.fits"
    _write_fits(path, np.zeros((8, 8), dtype=np.uint16))

    frame = load_frame(path)

    assert frame.data.shape == (8, 8)


def test_load_frame_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "frame.txt"
    path.write_text("not an image")

    with pytest.raises(ValueError):
        load_frame(path)


def test_load_frame_set_recurses_and_sorts(tmp_path):
    lights_dir = tmp_path / "lights"
    lights_dir.mkdir()
    _write_fits(lights_dir / "b.fits", np.zeros((4, 4), dtype=np.uint16))
    _write_fits(lights_dir / "a.fits", np.zeros((4, 4), dtype=np.uint16))

    frames = load_frame_set(tmp_path)

    assert len(frames) == 2
    assert frames[0].source_path.name == "a.fits"
    assert frames[1].source_path.name == "b.fits"
