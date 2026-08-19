"""Applies master bias/dark/flat correction to a single light frame."""

from __future__ import annotations

import numpy as np

from starstacker.io.frame import RawFrame


def calibrate_frame(
    frame: RawFrame,
    master_bias: np.ndarray | None,
    master_dark: np.ndarray | None,
    master_flat: np.ndarray | None,
) -> RawFrame:
    data = frame.data.astype(np.float32)

    if master_bias is not None:
        data = data - master_bias
    if master_dark is not None:
        data = data - master_dark
    if master_flat is not None:
        data = data / master_flat

    return frame.with_data(data)
