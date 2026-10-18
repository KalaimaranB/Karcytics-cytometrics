"""Size and shape of each object, from its mask (scikit-image ``regionprops``)."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from skimage.measure import regionprops

from ..model import ImageSet
from .base import AREA, LENGTH, NONE, PIXELS, Column, MeasureOptions, ObjectView, register

COLUMNS = [
    Column("AreaShape_Area", AREA, "Number of pixels in the object, times the pixel area."),
    Column(
        "AreaShape_Perimeter",
        LENGTH,
        "Length of the object's outline (scikit-image's estimate, as in CellProfiler).",
    ),
    Column(
        "AreaShape_FormFactor",
        NONE,
        "Circularity, 4π·area / perimeter²: 1 for a perfect circle, lower for irregular shapes.",
    ),
    Column(
        "AreaShape_Solidity",
        NONE,
        "Area / convex-hull area: 1 for convex shapes, lower for lobed or concave ones.",
    ),
    Column(
        "AreaShape_Eccentricity",
        NONE,
        "Of the ellipse with the same second moments: 0 for a circle, towards 1 for elongated.",
    ),
    Column("AreaShape_MajorAxisLength", LENGTH, "Major axis of the same-moments ellipse."),
    Column("AreaShape_MinorAxisLength", LENGTH, "Minor axis of the same-moments ellipse."),
    Column(
        "AreaShape_MaxFeretDiameter",
        LENGTH,
        "Longest caliper (Feret) diameter: the largest distance between two outline points.",
    ),
    Column(
        "AreaShape_MinFeretDiameter",
        LENGTH,
        "Shortest caliper (Feret) diameter: the narrowest width over all rotations.",
    ),
    Column("AreaShape_EquivalentDiameter", LENGTH, "Diameter of a circle with the same area."),
    Column("Location_Center_X", PIXELS, "Centroid x, in image pixels."),
    Column("Location_Center_Y", PIXELS, "Centroid y, in image pixels."),
]


@register
class ShapeMeasurer:
    id = "shape"
    name = "Size & shape"
    citations = ("skimage2014", "cellprofiler4")

    def columns(self, image_set: ImageSet, options: MeasureOptions) -> list[Column]:  # noqa: ARG002
        return list(COLUMNS)

    def measure(self, obj: ObjectView) -> dict[str, Any]:
        s = obj.scale
        props = regionprops(obj.mask.astype(np.uint8))[0]
        area, perimeter = float(props.area), float(props.perimeter)
        min_feret, max_feret = feret_diameters(obj.mask)
        cy, cx = props.centroid
        return {
            "AreaShape_Area": area * s**2,
            "AreaShape_Perimeter": perimeter * s,
            "AreaShape_FormFactor": 4 * math.pi * area / perimeter**2 if perimeter else 0.0,
            "AreaShape_Solidity": float(props.solidity),
            "AreaShape_Eccentricity": float(props.eccentricity),
            "AreaShape_MajorAxisLength": float(props.axis_major_length) * s,
            "AreaShape_MinorAxisLength": float(props.axis_minor_length) * s,
            "AreaShape_MaxFeretDiameter": max_feret * s,
            "AreaShape_MinFeretDiameter": min_feret * s,
            "AreaShape_EquivalentDiameter": 2 * math.sqrt(area / math.pi) * s,
            "Location_Center_X": cx + obj.offset[1],
            "Location_Center_Y": cy + obj.offset[0],
        }


def feret_diameters(mask: np.ndarray) -> tuple[float, float]:
    """(min, max) caliper diameters, in px, of a binary mask.

    Uses the convex hull of the object's pixel *corners*, so a 10 × 20 px
    rectangle measures exactly 10 and √(10² + 20²). The minimum caliper is
    attained perpendicular to a hull edge (rotating calipers).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    corners = np.concatenate(
        [np.stack([xs + dx, ys + dy], axis=1) for dx in (0, 1) for dy in (0, 1)]
    ).astype(np.float32)
    hull = cv2.convexHull(corners)[:, 0, :].astype(np.float64)
    if len(hull) < 3:  # noqa: PLR2004  (degenerate; can't happen for pixel corners)
        return 0.0, 0.0
    diffs = hull[:, None, :] - hull[None, :, :]
    max_feret = float(np.sqrt((diffs**2).sum(-1)).max())
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.hypot(edges[:, 0], edges[:, 1])
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1) / lengths[:, None]
    widths = np.ptp(hull @ normals.T, axis=0)
    return float(widths.min()), max_feret
