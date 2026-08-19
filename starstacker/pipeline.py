"""Top-level pipeline orchestrator.

Full planned architecture:

    Raw Frames -> Preprocessing -> [AI Frame Rejection] -> Calibration
        -> [AI-Assisted Alignment] -> Normalization -> Stacking
        -> [AI Denoising] -> Output

Only "Raw Frames", "Preprocessing", and "Calibration" are implemented so
far. Each later stage is stubbed below in call order so the pipeline shape
stays visible; they raise NotImplementedError until built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from starstacker.calibration.pipeline import CalibrationConfig, CalibrationPipeline
from starstacker.io.frame import RawFrame
from starstacker.io.loaders import load_frame_set
from starstacker.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline


@dataclass
class PipelineConfig:
    raw_frames_dir: Path
    dark_frames_dir: Path | None = None
    flat_frames_dir: Path | None = None
    bias_frames_dir: Path | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)


class StarStackerPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline(config.preprocessing)
        self.calibration_pipeline = CalibrationPipeline(config.calibration)

    def load_raw_frames(self) -> list[RawFrame]:
        return load_frame_set(self.config.raw_frames_dir)

    def preprocess(self, frames: list[RawFrame]) -> list[RawFrame]:
        return self.preprocessing_pipeline.run_batch(frames)

    def reject_frames(self, frames: list[RawFrame]) -> list[RawFrame]:
        """Optional AI-assisted frame rejection (drop blurry/clouded/trailed frames)."""
        raise NotImplementedError("AI frame rejection is not implemented yet")

    def calibrate(self, frames: list[RawFrame]) -> list[RawFrame]:
        """Dark/flat/bias calibration.

        `frames` must already be preprocessed (debayered/normalized) so they
        share a common scale with the master frames built here. Calibration
        frames go through the same debayer/normalize steps, but *not*
        hot-pixel removal for bias/dark - that would erase the hot pixels
        those frames exist to characterize.
        """
        calibration_preprocessing = PreprocessingPipeline(
            PreprocessingConfig(
                debayer=self.config.preprocessing.debayer,
                remove_hot_pixels=False,
                normalize=self.config.preprocessing.normalize,
            )
        )

        bias_frames = self._load_and_preprocess(self.config.bias_frames_dir, calibration_preprocessing)
        dark_frames = self._load_and_preprocess(self.config.dark_frames_dir, calibration_preprocessing)
        flat_frames = self._load_and_preprocess(self.config.flat_frames_dir, self.preprocessing_pipeline)

        self.calibration_pipeline.build_masters(bias_frames, dark_frames, flat_frames)
        return self.calibration_pipeline.run_batch(frames)

    def _load_and_preprocess(
        self, directory: Path | None, preprocessing_pipeline: PreprocessingPipeline
    ) -> list[RawFrame]:
        if directory is None:
            return []
        return preprocessing_pipeline.run_batch(load_frame_set(directory))

    def align(self, frames: list[RawFrame]) -> list[RawFrame]:
        """Optional AI-assisted star alignment/registration."""
        raise NotImplementedError("Alignment is not implemented yet")

    def normalize(self, frames: list[RawFrame]) -> list[RawFrame]:
        """Background/flux normalization across frames prior to stacking."""
        raise NotImplementedError("Normalization is not implemented yet")

    def stack(self, frames: list[RawFrame]) -> RawFrame:
        raise NotImplementedError("Stacking is not implemented yet")

    def denoise(self, stacked: RawFrame) -> RawFrame:
        """Optional AI denoising of the final stacked image."""
        raise NotImplementedError("AI denoising is not implemented yet")

    def run(self) -> list[RawFrame]:
        """Runs the implemented prefix of the pipeline: load + preprocess + calibrate.

        Returns calibrated frames; later stages will extend this once built.
        """
        frames = self.load_raw_frames()
        frames = self.preprocess(frames)
        return self.calibrate(frames)
