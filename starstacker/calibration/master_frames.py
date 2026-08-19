"""Builds master bias/dark/flat frames by median-combining calibration frames.

Median (not mean) is used because it rejects outliers like cosmic ray hits
or transient sensor noise without needing explicit sigma-clipping.
"""

from __future__ import annotations

import numpy as np

from starstacker.io.frame import RawFrame


def stack_median(frames: list[RawFrame]) -> np.ndarray:
    """Median-combine the pixel data of same-shaped frames."""
    shapes = {f.data.shape for f in frames}
    if len(shapes) > 1:
        raise ValueError(f"Cannot combine frames with mismatched shapes: {shapes}")

    stack = np.stack([f.data for f in frames], axis=0)
    return np.median(stack, axis=0).astype(np.float32)


def build_master_bias(bias_frames: list[RawFrame]) -> np.ndarray | None:
    """Master bias: read noise + fixed sensor pattern, captured at zero exposure."""
    if not bias_frames:
        return None
    return stack_median(bias_frames)


def build_master_dark(dark_frames: list[RawFrame], master_bias: np.ndarray | None) -> np.ndarray | None:
    """Master dark current only, with bias removed if a master bias is available."""
    if not dark_frames:
        return None
    master_dark = stack_median(dark_frames)
    if master_bias is not None:
        master_dark = master_dark - master_bias
    return master_dark


def build_master_flat(flat_frames: list[RawFrame], master_bias: np.ndarray | None) -> np.ndarray | None:
    """Master flat as a relative sensitivity map (vignetting/dust), normalized to mean 1."""
    if not flat_frames:
        return None
    master_flat = stack_median(flat_frames)
    if master_bias is not None:
        master_flat = master_flat - master_bias

    mean = float(master_flat.mean())
    if mean == 0:
        raise ValueError("Master flat has zero mean; cannot normalize")
    return master_flat / mean
