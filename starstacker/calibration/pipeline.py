"""Orchestrates master-frame building and calibration of light frames.

Master bias/dark are built from frames with hot-pixel removal skipped -
that step would erase the very hot pixels dark/bias frames exist to
characterize. Master flat uses the caller's normal preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from starstacker.calibration.calibrate import calibrate_frame
from starstacker.calibration.master_frames import build_master_bias, build_master_dark, build_master_flat
from starstacker.io.frame import RawFrame


@dataclass
class CalibrationConfig:
    use_bias: bool = True
    use_dark: bool = True
    use_flat: bool = True


class CalibrationPipeline:
    def __init__(self, config: CalibrationConfig | None = None):
        self.config = config or CalibrationConfig()
        self.master_bias = None
        self.master_dark = None
        self.master_flat = None

    def build_masters(
        self,
        bias_frames: list[RawFrame],
        dark_frames: list[RawFrame],
        flat_frames: list[RawFrame],
    ) -> None:
        self.master_bias = build_master_bias(bias_frames) if self.config.use_bias else None
        self.master_dark = (
            build_master_dark(dark_frames, self.master_bias) if self.config.use_dark else None
        )
        self.master_flat = (
            build_master_flat(flat_frames, self.master_bias) if self.config.use_flat else None
        )

    def run(self, frame: RawFrame) -> RawFrame:
        return calibrate_frame(frame, self.master_bias, self.master_dark, self.master_flat)

    def run_batch(self, frames: list[RawFrame]) -> list[RawFrame]:
        return [self.run(frame) for frame in frames]
