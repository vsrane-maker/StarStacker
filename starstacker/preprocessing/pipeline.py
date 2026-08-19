"""Orchestrates the preprocessing steps applied to each raw frame.

Order matters: debayering must happen before hot-pixel removal (which needs
correct neighbor structure), and normalization runs last so every downstream
stage sees a consistent float32 [0, 1] scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from starstacker.io.frame import RawFrame
from starstacker.preprocessing.debayer import debayer_frame
from starstacker.preprocessing.hot_pixels import remove_hot_pixels
from starstacker.preprocessing.normalize import normalize_bit_depth


@dataclass
class PreprocessingConfig:
    debayer: bool = True
    remove_hot_pixels: bool = True
    hot_pixel_threshold: float = 5.0
    normalize: bool = True


class PreprocessingPipeline:
    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()

    def run(self, frame: RawFrame) -> RawFrame:
        if self.config.debayer:
            frame = debayer_frame(frame)
        if self.config.remove_hot_pixels:
            frame = remove_hot_pixels(frame, threshold=self.config.hot_pixel_threshold)
        if self.config.normalize:
            frame = normalize_bit_depth(frame)
        return frame

    def run_batch(self, frames: list[RawFrame]) -> list[RawFrame]:
        return [self.run(frame) for frame in frames]
