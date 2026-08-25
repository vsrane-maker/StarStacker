"""Matches per-frame sky background levels so frames combine cleanly when stacked.

Even after calibration, frames can carry different residual background levels
(moon glow, thin cloud, gradient light pollution varying with sky position).
Stacking frames with mismatched backgrounds either washes out faint detail or
leaves visible seams, so each frame's background is shifted to match a common
reference before stacking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starstacker.io.frame import RawFrame
from starstacker.normalization.stats import frame_background

_DEFAULT_SIGMA = 3.0


@dataclass
class FrameStats:
    """Background level(s) for one frame: scalar (0-d array) for mono, one per channel for color."""

    background: np.ndarray


def compute_frame_stats(frame: RawFrame, sigma: float = _DEFAULT_SIGMA) -> FrameStats:
    return FrameStats(background=frame_background(frame, sigma))


def normalize_frame(frame: RawFrame, reference: FrameStats, stats: FrameStats | None = None) -> RawFrame:
    """Shift `frame` so its background matches `reference`'s background.

    `stats` can be passed in when already computed (e.g. by a pipeline that
    computed stats for every frame up front) to avoid recomputing it here.
    """
    if stats is None:
        stats = compute_frame_stats(frame)

    offset = reference.background - stats.background
    data = frame.data.astype(np.float32)
    if data.ndim == 3:
        offset = offset.reshape((1, 1, -1))
    data = data + offset.astype(np.float32)

    return frame.with_data(data)
