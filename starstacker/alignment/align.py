"""Applies a measured geometric transform to register a frame onto a reference frame's grid.

Stars must land on the same pixel across frames before stacking, or the
result comes out trailed/soft rather than sharp; this is the geometric
counterpart to `normalization`, which matches brightness levels instead of
pixel positions.
"""

from __future__ import annotations

import numpy as np

from starstacker.alignment.stats import FrameTransform
from starstacker.io.frame import RawFrame


def align_frame(frame: RawFrame, transform: FrameTransform) -> RawFrame:
    """Warp `frame` by `transform` so its stars land where the reference's do.

    `transform.matrix` already maps `frame`'s pixel coordinates onto the
    reference's, matching `cv2.warpAffine`'s forward-mapping convention, so
    it's passed through unmodified. Pixels warped in from outside the
    original frame are filled with 0 (`cv2.warpAffine`'s default border).
    """
    import cv2

    data = frame.data.astype(np.float32)
    height, width = data.shape[:2]
    aligned = cv2.warpAffine(data, transform.matrix.astype(np.float32), (width, height))

    return frame.with_data(aligned)
