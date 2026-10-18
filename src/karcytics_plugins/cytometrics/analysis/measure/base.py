"""Measurement framework: one ``Measurer`` per family, each adding named columns.

Columns follow CellProfiler's naming (``AreaShape_Area``,
``Intensity_MeanIntensity_<channel>``, ...) so tables line up with
CellProfiler output and its documentation. Every column declares a unit
kind; lengths and areas are reported in µm/µm² when the image is
calibrated and px/px² when it isn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..model import ImageSet

# Unit kinds a column can have. "length"/"area" follow the calibration.
NONE, LENGTH, AREA, INTENSITY, PIXELS = "", "length", "area", "intensity", "px"


@dataclass(frozen=True)
class Column:
    name: str
    unit: str = NONE
    help: str = ""


@dataclass
class MeasureOptions:
    background: str = "none"  # "none" | "global" | "local"
    ring_gap_px: int = 2  # local background: gap between object and ring
    ring_width_px: int = 5  # local background: ring width
    texture: bool = False
    texture_distance_px: int = 3
    nucleus_channel: str | None = None  # enables the nucleus/cytoplasm measurer


@dataclass
class ObjectView:
    """One object, cropped: everything a measurer needs, nothing image-sized.

    ``crop`` is the object's bounding box grown by a margin; ``mask`` is the
    object inside it, ``others`` every other object in it, and ``offset`` the
    crop's (row, col) origin in the image.
    """

    number: int
    mask: np.ndarray
    others: np.ndarray
    crop: tuple[slice, slice]
    offset: tuple[int, int]
    image_set: ImageSet
    scale: float  # µm per px; 1 when uncalibrated
    options: MeasureOptions
    # Per-measurer image-wide results of ``prepare``, keyed by measurer id.
    context: dict[str, Any] = field(default_factory=dict)

    def pixels(self, channel_name: str) -> np.ndarray:
        """The channel's crop as float64 (native intensity values)."""
        channel = self.image_set.get(channel_name)
        assert channel is not None
        return channel.data[self.crop].astype(np.float64)


class Measurer(Protocol):
    """A family of measurements.

    Optional class hooks: ``enabled(options) -> bool`` (default: always on)
    and ``prepare(image_set, labels, options)``, run once per image, whose
    result reaches ``measure`` as ``obj.context[id]``.
    """

    id: str
    name: str
    citations: tuple[str, ...]

    def columns(self, image_set: ImageSet, options: MeasureOptions) -> list[Column]: ...

    def measure(self, obj: ObjectView) -> dict[str, Any]: ...


REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    if cls.id in REGISTRY:  # type: ignore[attr-defined]
        raise ValueError(f"Measurer id {cls.id!r} is already registered")  # type: ignore[attr-defined]
    REGISTRY[cls.id] = cls  # type: ignore[attr-defined]
    return cls


def channel_key(name: str) -> str:
    """A channel name made safe for a column name: 'GFP (Ch 1).tif' → 'GFP_Ch_1_tif'."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "channel"
