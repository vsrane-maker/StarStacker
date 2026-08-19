#!/usr/bin/env python3
"""CLI entry point: load raw frames from a directory and preprocess them.

Usage:
    python scripts/run_pipeline.py --raw-dir data/raw/lights --output-dir data/processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from starstacker.io.loaders import iter_frame_paths, load_frame
from starstacker.pipeline import PipelineConfig, StarStackerPipeline
from starstacker.preprocessing.pipeline import PreprocessingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and preprocess raw astro frames.")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory of raw frames")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write .npy outputs")
    parser.add_argument("--no-debayer", action="store_true")
    parser.add_argument("--no-hot-pixel-removal", action="store_true")
    parser.add_argument("--hot-pixel-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        raw_frames_dir=args.raw_dir,
        preprocessing=PreprocessingConfig(
            debayer=not args.no_debayer,
            remove_hot_pixels=not args.no_hot_pixel_removal,
            hot_pixel_threshold=args.hot_pixel_threshold,
        ),
    )
    pipeline = StarStackerPipeline(config)

    paths = iter_frame_paths(args.raw_dir)
    print(f"Found {len(paths)} raw frame(s) in {args.raw_dir}")

    written = 0
    for path in tqdm(paths, desc="Preprocessing"):
        frame = load_frame(path)
        frame = pipeline.preprocessing_pipeline.run(frame)
        out_path = args.output_dir / f"{frame.source_path.stem}.npy"
        np.save(out_path, frame.data)
        written += 1

    print(f"Wrote {written} preprocessed frame(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
