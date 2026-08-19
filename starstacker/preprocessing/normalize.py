"""Bit-depth normalization: convert raw integer pixel data to float32 in [0, 1].

Downstream stages (calibration, stacking, AI models) all assume float32 data
on a common scale regardless of whether the source sensor was 12, 14, or 16-bit.
"""

from __future__ import annotations

import numpy as np

from starstacker.io.frame import RawFrame


def normalize_bit_depth(frame: RawFrame) -> RawFrame:
    data = frame.data
    if np.issubdtype(data.dtype, np.floating):
        return frame.with_data(data.astype(np.float32))

    max_value = _max_value(frame)
    normalized = data.astype(np.float32) / max_value
    return frame.with_data(normalized)


def _max_value(frame: RawFrame) -> float:
    bit_depth = frame.metadata.bit_depth
    if bit_depth:
        return float((2**bit_depth) - 1)
    return float(np.iinfo(frame.data.dtype).max)
