"""Orchestrates star alignment/registration across a batch of frames.

Every frame is registered against a single reference frame from the batch
(`reference_index`, default the first) so all frames share the same pixel
grid before stacking. Unlike normalization's reference, this can't be a
"mean" of the batch - averaging misaligned pixel data has no geometric
meaning, so a single frame must be chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

from starstacker.alignment.align import align_frame
from starstacker.alignment.stats import compute_shift
from starstacker.io.frame import RawFrame


@dataclass
class AlignmentConfig:
    reference_index: int = 0


class AlignmentPipeline:
    def __init__(self, config: AlignmentConfig | None = None):
        self.config = config or AlignmentConfig()

    def run_batch(self, frames: list[RawFrame]) -> list[RawFrame]:
        if not frames:
            return []
        if not 0 <= self.config.reference_index < len(frames):
            raise ValueError(
                f"reference_index {self.config.reference_index} out of range for "
                f"{len(frames)} frame(s)"
            )

        reference = frames[self.config.reference_index]
        aligned = []
        for frame in frames:
            if frame is reference:
                aligned.append(frame)
                continue
            shift = compute_shift(reference, frame)
            aligned.append(align_frame(frame, shift))
        return aligned
