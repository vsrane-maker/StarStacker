"""Orchestrates background normalization across a batch of frames.

All frames are measured before any are adjusted, so the chosen reference
reflects the whole batch rather than being biased by normalizing in place
one frame at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from starstacker.io.frame import RawFrame
from starstacker.normalization.normalize import FrameStats, compute_frame_stats, normalize_frame

_VALID_REFERENCES = ("first", "mean")


@dataclass
class NormalizationConfig:
    sigma: float = 3.0
    reference: str = "first"  # "first" matches every frame to frames[0]; "mean" matches to the batch average


class NormalizationPipeline:
    def __init__(self, config: NormalizationConfig | None = None):
        self.config = config or NormalizationConfig()
        if self.config.reference not in _VALID_REFERENCES:
            raise ValueError(
                f"Unknown reference '{self.config.reference}'. Supported: {_VALID_REFERENCES}"
            )

    def run_batch(self, frames: list[RawFrame]) -> list[RawFrame]:
        if not frames:
            return []

        stats = [compute_frame_stats(frame, self.config.sigma) for frame in frames]
        reference = self._select_reference(stats)
        return [normalize_frame(frame, reference, s) for frame, s in zip(frames, stats)]

    def _select_reference(self, stats: list[FrameStats]) -> FrameStats:
        if self.config.reference == "mean":
            backgrounds = np.stack([s.background for s in stats])
            return FrameStats(background=backgrounds.mean(axis=0))
        return stats[0]
