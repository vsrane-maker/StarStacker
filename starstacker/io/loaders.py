"""Loaders that turn files on disk into RawFrame objects.

Two source families are supported:
  - FITS (.fits/.fit/.fts): produced by dedicated astro cameras (ZWO, QHY, ...).
    Metadata comes from the FITS header, which is standardized and rich.
  - DSLR/mirrorless RAW (.cr2/.cr3/.nef/.arw/.dng): decoded with rawpy to get
    the untouched Bayer mosaic, with EXIF metadata read via exifread.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from starstacker.io.frame import FrameMetadata, FrameType, RawFrame

FITS_EXTENSIONS = {".fits", ".fit", ".fts"}
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng"}
SUPPORTED_EXTENSIONS = FITS_EXTENSIONS | RAW_EXTENSIONS

# Directory names (case-insensitive) used to infer frame type when header
# metadata doesn't specify it, e.g. data/raw/darks/*.fits
_FRAME_TYPE_DIR_HINTS = {
    "light": FrameType.LIGHT,
    "lights": FrameType.LIGHT,
    "dark": FrameType.DARK,
    "darks": FrameType.DARK,
    "flat": FrameType.FLAT,
    "flats": FrameType.FLAT,
    "bias": FrameType.BIAS,
}

_FITS_IMAGETYP_HINTS = {
    "light": FrameType.LIGHT,
    "light frame": FrameType.LIGHT,
    "dark": FrameType.DARK,
    "dark frame": FrameType.DARK,
    "flat": FrameType.FLAT,
    "flat field": FrameType.FLAT,
    "bias": FrameType.BIAS,
    "bias frame": FrameType.BIAS,
}


def load_frame(path: str | Path) -> RawFrame:
    """Load a single frame, dispatching on file extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in FITS_EXTENSIONS:
        return load_fits(path)
    if suffix in RAW_EXTENSIONS:
        return load_raw(path)
    raise ValueError(
        f"Unsupported frame extension '{suffix}' for {path}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def iter_frame_paths(directory: str | Path) -> list[Path]:
    """Recursively list every supported frame file under `directory`, sorted.

    Does not load any pixel data; use with `load_frame()` to process frames
    one at a time instead of holding a whole batch in memory at once.
    """
    directory = Path(directory)
    return sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_frame_set(directory: str | Path) -> list[RawFrame]:
    """Recursively load every supported frame file under `directory`.

    Loads all frames into memory at once; for large batches of raw sensor
    data, prefer `iter_frame_paths()` + `load_frame()` to stream one at a
    time instead.
    """
    return [load_frame(p) for p in iter_frame_paths(directory)]


def load_fits(path: str | Path) -> RawFrame:
    from astropy.io import fits

    path = Path(path)
    with fits.open(path) as hdul:
        hdu = hdul[0]
        data = np.asarray(hdu.data)
        header = dict(hdu.header)

    metadata = FrameMetadata(
        camera=header.get("INSTRUME"),
        exposure_time=_first_float(header, ("EXPTIME", "EXPOSURE")),
        gain=_to_float(header.get("GAIN")),
        temperature=_to_float(header.get("CCD-TEMP")),
        timestamp=_parse_fits_timestamp(header.get("DATE-OBS")),
        filter_name=header.get("FILTER"),
        width=data.shape[-1] if data.ndim >= 2 else None,
        height=data.shape[-2] if data.ndim >= 2 else None,
        bit_depth=_bit_depth_from_dtype(data.dtype),
        bayer_pattern=header.get("BAYERPAT"),
    )
    frame_type = _infer_frame_type(path, header.get("IMAGETYP"))

    return RawFrame(
        data=data,
        metadata=metadata,
        frame_type=frame_type,
        source_path=path,
        header=header,
    )


def load_raw(path: str | Path) -> RawFrame:
    import rawpy

    path = Path(path)
    with rawpy.imread(str(path)) as raw:
        data = np.array(raw.raw_image_visible, copy=True)
        bayer_pattern = _rawpy_bayer_pattern(raw)
        width, height = raw.sizes.raw_width, raw.sizes.raw_height

    exif = _read_exif(path)

    metadata = FrameMetadata(
        camera=exif.get("camera"),
        exposure_time=exif.get("exposure_time"),
        iso=exif.get("iso"),
        timestamp=exif.get("timestamp"),
        width=width,
        height=height,
        bit_depth=_bit_depth_from_dtype(data.dtype),
        bayer_pattern=bayer_pattern,
    )
    frame_type = _infer_frame_type(path, None)

    return RawFrame(
        data=data,
        metadata=metadata,
        frame_type=frame_type,
        source_path=path,
        header=exif,
    )


def _infer_frame_type(path: Path, imagetyp: str | None) -> FrameType:
    if imagetyp:
        hit = _FITS_IMAGETYP_HINTS.get(imagetyp.strip().lower())
        if hit is not None:
            return hit
    for part in path.parts:
        hit = _FRAME_TYPE_DIR_HINTS.get(part.strip().lower())
        if hit is not None:
            return hit
    return FrameType.UNKNOWN


def _rawpy_bayer_pattern(raw) -> str | None:
    try:
        desc = raw.color_desc.decode("ascii")  # e.g. "RGBG"
        pattern_indices = raw.raw_pattern.flatten()
        return "".join(desc[i] for i in pattern_indices)
    except Exception:
        return None


def _read_exif(path: Path) -> dict:
    import exifread

    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    result: dict = {}
    if "EXIF ExposureTime" in tags:
        result["exposure_time"] = _ratio_to_float(str(tags["EXIF ExposureTime"]))
    if "EXIF ISOSpeedRatings" in tags:
        result["iso"] = _to_int(str(tags["EXIF ISOSpeedRatings"]))
    if "Image Model" in tags:
        result["camera"] = str(tags["Image Model"]).strip()
    if "EXIF DateTimeOriginal" in tags:
        result["timestamp"] = _parse_exif_timestamp(str(tags["EXIF DateTimeOriginal"]))
    return result


def _ratio_to_float(value: str) -> float | None:
    try:
        if "/" in value:
            num, denom = value.split("/")
            return float(num) / float(denom)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_float(header: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in header:
            value = _to_float(header[key])
            if value is not None:
                return value
    return None


def _parse_fits_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_exif_timestamp(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _bit_depth_from_dtype(dtype: np.dtype) -> int | None:
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).bits
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).bits
    return None
