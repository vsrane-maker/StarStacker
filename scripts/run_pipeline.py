#!/usr/bin/env python3
"""CLI entry point: run raw frames from a directory through the pipeline
(preprocess, calibrate, align, normalize) and write the results as .npy arrays.

Dark/flat/bias directories are optional; calibration is skipped for whichever
of them isn't provided.

Usage:
    python scripts/run_pipeline.py --raw-dir data/raw/test_lights --output-dir data/processed/test_lights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from starstacker.io.loaders import iter_frame_paths
from starstacker.pipeline import PipelineConfig, StarStackerPipeline
from starstacker.preprocessing.pipeline import PreprocessingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw astro frames through the pipeline.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory of raw light frames")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write .npy outputs")
    parser.add_argument("--dark-dir", type=Path, default=None, help="Directory of dark frames")
    parser.add_argument("--flat-dir", type=Path, default=None, help="Directory of flat frames")
    parser.add_argument("--bias-dir", type=Path, default=None, help="Directory of bias frames")
    parser.add_argument("--no-debayer", action="store_true")
    parser.add_argument("--no-hot-pixel-removal", action="store_true")
    parser.add_argument("--hot-pixel-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        raw_frames_dir=args.raw_dir,
        dark_frames_dir=args.dark_dir,
        flat_frames_dir=args.flat_dir,
        bias_frames_dir=args.bias_dir,
        preprocessing=PreprocessingConfig(
            debayer=not args.no_debayer,
            remove_hot_pixels=not args.no_hot_pixel_removal,
            hot_pixel_threshold=args.hot_pixel_threshold,
        ),
    )
    pipeline = StarStackerPipeline(config)

    paths = iter_frame_paths(args.raw_dir)
    print(f"Found {len(paths)} raw frame(s) in {args.raw_dir}")

    frames = pipeline.run()

    written = 0
    for frame in tqdm(frames, desc="Writing output"):
        out_path = args.output_dir / f"{frame.source_path.stem}.npy"
        np.save(out_path, frame.data)
        written += 1

    print(f"Wrote {written} processed frame(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
