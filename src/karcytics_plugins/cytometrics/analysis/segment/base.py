"""What every segmenter shares: the protocol, the registry and the label-mask helpers.

A segmenter turns an ``ImageSet`` into a **label mask**: an int32 image where
0 is background and each object's pixels carry its own number (1..n). The
mask is the single source of truth; outlines and measurements are derived
from it by ``cells_from_labels``, the same way for every segmenter.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np
from scipy import ndimage

from ..model import Channel, ImageSet

Progress = Callable[[int], None]


@dataclass(frozen=True)
class Param:
    """One user-settable parameter, enough to build its control and validate it.

    ``unit`` is "length" or "area" for sizes the UI shows in µm (px when
    uncalibrated); the segmenter always receives them in pixels.
    """

    key: str
    label: str
    kind: str  # "float" | "int" | "bool" | "choice" | "channel"
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 1
    unit: str = ""
    choices: tuple[tuple[str, Any], ...] = ()
    advanced: bool = False
    help: str = ""
    special_value_text: str = ""
    enabled_by: str = ""  # key of a bool Param that switches this one on


@dataclass
class SegmentationResult:
    """A label mask plus, when the segmenter has one, a per-pixel probability in 0–1."""

    labels: np.ndarray
    probability: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


class Segmenter(Protocol):
    id: str
    name: str
    summary: str
    citations: tuple[str, ...]

    def parameters(self) -> list[Param]: ...

    def describe(self, params: dict, scale: float) -> tuple[str, dict]: ...

    def segment(
        self, image_set: ImageSet, params: dict, progress: Progress | None = None
    ) -> SegmentationResult: ...


REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Class decorator: make a segmenter available to the panel under ``cls.id``."""
    if cls.id in REGISTRY:  # type: ignore[attr-defined]
        raise ValueError(f"Segmenter id {cls.id!r} is already registered")  # type: ignore[attr-defined]
    REGISTRY[cls.id] = cls  # type: ignore[attr-defined]
    return cls


# ── Shared parameters ────────────────────────────────────────────────────────

TARGET_CHANNEL = Param("target_channel", "Target Channel", "channel")
USE_SEED = Param(
    "use_dual_channel",
    "Use a nuclei channel",
    "bool",
    default=False,
    help="A second, internal marker helps separate touching cells.",
)
SEED_CHANNEL = Param(
    "seed_channel",
    "Nuclei Channel",
    "channel",
    help="A nuclear stain (e.g. DAPI) used to tell touching cells apart.",
    enabled_by="use_dual_channel",
)
EXCLUDE_BORDERS = Param(
    "exclude_borders",
    "Exclude cells touching image edges",
    "bool",
    default=True,
    help="Cells cut by the image edge give wrong sizes.",
)
MIN_AREA = Param("min_area_px", "Min Area", "float", 10.0, 0, 100_000, unit="area")
MAX_AREA = Param("max_area_px", "Max Area", "float", 5000.0, 0, 100_000, unit="area")
COMMON = (EXCLUDE_BORDERS, MIN_AREA, MAX_AREA)


# ── Helpers ──────────────────────────────────────────────────────────────────


def require_channel(image_set: ImageSet, name: str | None) -> Channel:
    channel = image_set.get(name)
    if channel is None:
        raise ValueError(f"Channel {name!r} is not loaded.")
    return channel


def seed_channel(image_set: ImageSet, params: dict) -> Channel | None:
    """The nuclei channel, if the user turned it on and it exists."""
    if not params.get("use_dual_channel"):
        return None
    return image_set.get(params.get("seed_channel"))


def otsu_mask(plane_u8: np.ndarray, blur: int = 5) -> tuple[np.ndarray, float]:
    """Foreground by Otsu's threshold after a Gaussian blur; returns (mask, threshold)."""
    blurred = cv2.GaussianBlur(plane_u8, (blur, blur), 0)
    threshold, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0, float(threshold)


def label_components(mask: np.ndarray) -> np.ndarray:
    """8-connected objects of a binary mask, holes filled, as int32 labels."""
    filled = ndimage.binary_fill_holes(mask)
    labels, _ = ndimage.label(filled, structure=np.ones((3, 3), dtype=bool))
    return labels.astype(np.int32)


def clear_border(labels: np.ndarray, margin: int = 2) -> np.ndarray:
    """Remove objects with any pixel within `margin` px of the image edge."""
    edge = np.zeros(labels.shape, dtype=bool)
    edge[:margin, :] = edge[-margin:, :] = True
    edge[:, :margin] = edge[:, -margin:] = True
    touching = np.unique(labels[edge & (labels > 0)])
    out = labels.copy()
    out[np.isin(out, touching)] = 0
    return out


def relabel(labels: np.ndarray) -> np.ndarray:
    """Renumber objects 1..n in order of first appearance (row-major)."""
    present = np.unique(labels)
    present = present[present > 0]
    lut = np.zeros(int(labels.max()) + 1 if labels.size else 1, dtype=np.int32)
    lut[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return lut[labels]


def cells_from_labels(
    labels: np.ndarray,
    scale: float,
    min_area_px: float = 0.0,
    max_area_px: float = math.inf,
    probability: np.ndarray | None = None,
    progress: Progress | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Outline and measure every object; drop those outside the area range.

    Each object is processed inside its own bounding box
    (``ndimage.find_objects``), so the cost grows with the objects' total
    area, not with objects × image size.

    Returns the filtered, renumbered label mask and one dict per object
    (``label``, ``points``, ``area``, ``perim``, ``circ`` and, with a
    probability map, ``confidence``), in label order. Areas and perimeters
    come from the outline polygon, in µm (µm²) for ``scale`` µm/px.
    """
    labels = labels.astype(np.int32, copy=True)
    slices = ndimage.find_objects(labels)
    n = len(slices)
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        outline = _outline(labels, i, sl)
        area_px = cv2.contourArea(outline) if outline is not None else 0.0
        if outline is None or not (min_area_px <= area_px <= max_area_px):
            labels[sl][labels[sl] == i] = 0
        if progress and n:
            progress(int(100 * i / n))

    labels = relabel(labels)
    cells = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        outline = _outline(labels, i, sl)
        assert outline is not None
        perim = cv2.arcLength(outline, True) * scale
        area = cv2.contourArea(outline) * scale**2
        cell = {
            "label": i,
            "points": [[int(x), int(y)] for x, y in outline[:, 0, :]],
            "area": float(area),
            "perim": float(perim),
            "circ": float(4 * math.pi * area / perim**2) if perim > 0 else 0.0,
        }
        if probability is not None:
            cell["confidence"] = float(probability[sl][labels[sl] == i].mean())
        cells.append(cell)
    return labels, cells


def _outline(labels: np.ndarray, label: int, sl: tuple[slice, slice]) -> np.ndarray | None:
    """The largest external contour of object `label`, in image coordinates."""
    crop = np.pad((labels[sl] == label).astype(np.uint8), 1)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour + np.array([sl[1].start - 1, sl[0].start - 1], dtype=contour.dtype)
