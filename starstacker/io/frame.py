"""Core data model for a single raw astrophotography frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class FrameType(Enum):
    """Role a frame plays in the stacking pipeline."""

    LIGHT = "light"
    DARK = "dark"
    FLAT = "flat"
    BIAS = "bias"
    UNKNOWN = "unknown"


@dataclass
class FrameMetadata:
    """Normalized metadata extracted from a frame's FITS header or EXIF data.

    Fields are optional because coverage varies by source (FITS headers from
    dedicated astro cameras are far more complete than DSLR EXIF data).
    """

    camera: str | None = None
    exposure_time: float | None = None  # seconds
    gain: float | None = None
    iso: int | None = None
    temperature: float | None = None  # degrees Celsius
    timestamp: datetime | None = None
    filter_name: str | None = None
    width: int | None = None
    height: int | None = None
    bit_depth: int | None = None
    bayer_pattern: str | None = None  # e.g. "RGGB"; None means already mono/RGB


@dataclass
class RawFrame:
    """A single loaded frame plus its metadata and provenance."""

    data: np.ndarray
    metadata: FrameMetadata
    frame_type: FrameType
    source_path: Path
    header: dict[str, Any] = field(default_factory=dict)

    def with_data(self, data: np.ndarray) -> "RawFrame":
        """Return a copy of this frame with replaced pixel data.

        Preprocessing steps use this to produce new RawFrame instances
        rather than mutating data in place.
        """
        return RawFrame(
            data=data,
            metadata=self.metadata,
            frame_type=self.frame_type,
            source_path=self.source_path,
            header=self.header,
        )
