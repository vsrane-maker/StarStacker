"""Applies a measured translation to register a frame onto a reference frame's grid.

Stars must land on the same pixel across frames before stacking, or the
result comes out trailed/soft rather than sharp; this is the geometric
counterpart to `normalization`, which matches brightness levels instead of
pixel positions.
"""

from __future__ import annotations

import numpy as np

from starstacker.alignment.stats import FrameShift
from starstacker.io.frame import RawFrame


def align_frame(frame: RawFrame, shift: FrameShift) -> RawFrame:
    """Shift `frame` by `-shift` so its content lands where the reference's does.

    Pixels shifted in from outside the original frame are filled with 0
    (`cv2.warpAffine`'s default border), rather than wrapped or mirrored.
    """
    import cv2

    data = frame.data.astype(np.float32)
    height, width = data.shape[:2]
    matrix = np.array([[1.0, 0.0, -shift.dx], [0.0, 1.0, -shift.dy]], dtype=np.float32)
    aligned = cv2.warpAffine(data, matrix, (width, height))

    return frame.with_data(aligned)
