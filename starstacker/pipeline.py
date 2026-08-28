"""Top-level pipeline orchestrator.

Full planned architecture:

    Raw Frames -> Preprocessing -> [AI Frame Rejection] -> Calibration
        -> [AI-Assisted Alignment] -> Normalization -> Stacking
        -> [AI Denoising] -> Output

"Raw Frames", "Preprocessing", "Calibration", "Alignment", "Normalization",
and "Stacking" are implemented so far and reachable from `run()`. Each
remaining stage is stubbed below in call order so the pipeline shape stays
visible; they raise NotImplementedError until built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from starstacker.alignment.pipeline import AlignmentConfig, AlignmentPipeline
from starstacker.calibration.pipeline import CalibrationConfig, CalibrationPipeline
from starstacker.io.frame import RawFrame
from starstacker.io.loaders import load_frame_set
from starstacker.normalization.pipeline import NormalizationConfig, NormalizationPipeline
from starstacker.preprocessing.pipeline import PreprocessingConfig, PreprocessingPipeline
from starstacker.stacking.pipeline import StackingConfig, StackingPipeline


@dataclass
class PipelineConfig:
    raw_frames_dir: Path
    dark_frames_dir: Path | None = None
    flat_frames_dir: Path | None = None
    bias_frames_dir: Path | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    stacking: StackingConfig = field(default_factory=StackingConfig)


class StarStackerPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline(config.preprocessing)
        self.calibration_pipeline = CalibrationPipeline(config.calibration)
        self.alignment_pipeline = AlignmentPipeline(config.alignment)
        self.normalization_pipeline = NormalizationPipeline(config.normalization)
        self.stacking_pipeline = StackingPipeline(config.stacking)

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
        """Star alignment/registration: warp each frame onto a common reference frame.

        `frames` must already be calibrated so the detected star pattern
        reflects real pointing drift/field rotation between exposures rather
        than uncorrected sensor artifacts. Registration matches star
        triangles between each frame and the reference to recover rotation,
        scale, and translation - not just a plain pixel shift.
        """
        return self.alignment_pipeline.run_batch(frames)

    def normalize(self, frames: list[RawFrame]) -> list[RawFrame]:
        """Background normalization across frames prior to stacking.

        `frames` should already be calibrated and aligned so background
        differences reflect sky conditions rather than uncorrected sensor
        artifacts or misregistration.
        """
        return self.normalization_pipeline.run_batch(frames)

    def stack(self, frames: list[RawFrame]) -> RawFrame:
        """Combine aligned, normalized frames into a single stacked image.

        `frames` should already be aligned and normalized so the combination
        reflects true signal rather than misregistration or background
        drift. Uses a per-pixel sigma-clipped mean by default, which rejects
        outliers like cosmic ray hits or satellite trails before averaging.
        """
        return self.stacking_pipeline.run(frames)

    def denoise(self, stacked: RawFrame) -> RawFrame:
        """Optional AI denoising of the final stacked image."""
        raise NotImplementedError("AI denoising is not implemented yet")

    def run(self) -> RawFrame:
        """Runs the implemented prefix of the pipeline: load, preprocess, calibrate,
        align, normalize, then stack.

        Returns the final stacked image; denoising is not yet built.
        """
        frames = self.load_raw_frames()
        frames = self.preprocess(frames)
        frames = self.calibrate(frames)
        frames = self.align(frames)
        frames = self.normalize(frames)
        return self.stack(frames)
