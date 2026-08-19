"""Bayer-mosaic demosaicing for one-shot-color (OSC) sensors.

FITS frames from mono cameras and already-RGB sources have no Bayer pattern
(`metadata.bayer_pattern is None`) and pass through untouched.
"""

from __future__ import annotations

from starstacker.io.frame import RawFrame

# OpenCV's BayerXX2RGB codes are named for the pixel pair one row/column
# *after* the true top-left of the mosaic (a well-known OpenCV quirk), so
# the code name doesn't match the pattern string directly.
_PATTERN_TO_CV2_CODE = {
    "RGGB": "COLOR_BayerBG2RGB",
    "BGGR": "COLOR_BayerRG2RGB",
    "GRBG": "COLOR_BayerGB2RGB",
    "GBRG": "COLOR_BayerGR2RGB",
}


def debayer_frame(frame: RawFrame) -> RawFrame:
    """Demosaic a Bayer-mosaic frame into an (H, W, 3) RGB frame.

    Frames without a bayer_pattern (mono FITS, already-RGB sources) are
    returned unchanged.
    """
    pattern = frame.metadata.bayer_pattern
    if not pattern:
        return frame

    import cv2

    pattern = pattern.upper()
    code_name = _PATTERN_TO_CV2_CODE.get(pattern)
    if code_name is None:
        raise ValueError(
            f"Unsupported Bayer pattern '{pattern}' for {frame.source_path}. "
            f"Supported: {sorted(_PATTERN_TO_CV2_CODE)}"
        )

    rgb = cv2.cvtColor(frame.data, getattr(cv2, code_name))
    return frame.with_data(rgb)
