"""Orchestrates final combination of aligned, normalized frames into one stacked image."""

from __future__ import annotations

from dataclasses import dataclass

from starstacker.io.frame import RawFrame
from starstacker.stacking.stack import stack_mean, stack_median, stack_sigma_clipped_mean

_METHODS = {
    "mean": stack_mean,
    "median": stack_median,
    "sigma_clip_mean": stack_sigma_clipped_mean,
}


@dataclass
class StackingConfig:
    method: str = "sigma_clip_mean"  # "mean" | "median" | "sigma_clip_mean"
    sigma: float = 3.0
    max_iters: int = 5


class StackingPipeline:
    def __init__(self, config: StackingConfig | None = None):
        self.config = config or StackingConfig()
        if self.config.method not in _METHODS:
            raise ValueError(f"Unknown method '{self.config.method}'. Supported: {sorted(_METHODS)}")

    def run(self, frames: list[RawFrame]) -> RawFrame:
        if not frames:
            raise ValueError("Cannot stack an empty batch of frames")

        if self.config.method == "sigma_clip_mean":
            data = stack_sigma_clipped_mean(frames, sigma=self.config.sigma, max_iters=self.config.max_iters)
        else:
            data = _METHODS[self.config.method](frames)

        return self._build_stacked_frame(frames, data)

    def _build_stacked_frame(self, frames: list[RawFrame], data) -> RawFrame:
        reference = frames[0]
        return RawFrame(
            data=data,
            metadata=reference.metadata,
            frame_type=reference.frame_type,
            source_path=reference.source_path.parent / "stacked.fits",
            header=reference.header,
        )
