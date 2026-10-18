"""Rod-shaped cells (bacteria): length, width, curvature and division.

The cell's mask is thinned to a one-pixel skeleton (Zhang & Suen 1984, via
scikit-image) and the skeleton's longest path is its midline. Length is the
midline's length plus the cell's half-width at each end (the skeleton stops
short of the poles). Width is twice the median distance from the midline to
the cell's edge. Curvature is 1 − end-to-end distance / midline length (0
for a straight rod). A constriction in the middle (the narrowest width in
the middle 60% of the midline, relative to the median width) marks a
dividing cell when it is below 0.7.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from ..model import ImageSet
from .base import LENGTH, NONE, Column, MeasureOptions, ObjectView, register

DIVIDING_BELOW = 0.7
_STEPS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]


@register
class RodMeasurer:
    id = "rods"
    name = "Rod shape (bacteria)"
    citations = ("zhang1984", "skimage2014")

    @staticmethod
    def enabled(options: MeasureOptions) -> bool:
        return bool(options.rods)

    def columns(self, image_set: ImageSet, options: MeasureOptions) -> list[Column]:
        return [
            Column("Rod_Length", LENGTH, "Length along the cell's midline, pole to pole."),
            Column("Rod_Width", LENGTH, "Median width across the midline."),
            Column(
                "Rod_Curvature", NONE, "1 − end-to-end distance / midline length (0 = straight)."
            ),
            Column(
                "Rod_Constriction",
                NONE,
                "Narrowest width in the middle 60% of the cell / median width (1 = no waist).",
            ),
            Column(
                "Rod_Dividing",
                NONE,
                f"1 when the constriction is below {DIVIDING_BELOW} (a septum).",
            ),
        ]

    def measure(self, obj: ObjectView) -> dict[str, Any]:
        length, width, curvature, constriction = rod_shape(obj.mask)
        return {
            "Rod_Length": length * obj.scale,
            "Rod_Width": width * obj.scale,
            "Rod_Curvature": curvature,
            "Rod_Constriction": constriction,
            "Rod_Dividing": float(constriction < DIVIDING_BELOW)
            if np.isfinite(constriction)
            else np.nan,
        }


def rod_shape(mask: np.ndarray) -> tuple[float, float, float, float]:
    """(length px, width px, curvature, constriction) of one object's mask."""
    mask = np.pad(mask.astype(bool), 1)
    distance = ndimage.distance_transform_edt(mask)
    path = _longest_path(skeletonize(mask))
    if len(path) < 3:  # noqa: PLR2004  (a blob, not a rod)
        area = float(mask.sum())
        width = 2 * float(distance.max()) - 1.0
        return (area / width if width else 0.0), width, 0.0, float("nan")
    points = np.asarray(path, dtype=float)
    steps = np.hypot(*np.diff(points, axis=0).T)
    # The distance transform reaches the centre of the first background pixel,
    # half a pixel past the edge: take that half pixel off each side.
    ends = distance[path[0]] + distance[path[-1]] - 1.0
    length = float(steps.sum() + ends)
    widths = 2 * distance[tuple(points.astype(int).T)] - 1.0
    width = float(np.median(widths))
    chord = float(np.hypot(*(points[-1] - points[0]))) + ends
    curvature = max(0.0, 1 - chord / length) if length else 0.0
    n = len(widths)
    middle = widths[int(n * 0.2) : max(int(n * 0.8), int(n * 0.2) + 1)]
    constriction = float(middle.min() / width) if width else float("nan")
    return length, width, curvature, constriction


def _longest_path(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """The skeleton's longest path (two breadth-first searches), as (row, col) pixels."""
    pixels = set(zip(*np.nonzero(skeleton), strict=True))
    if not pixels:
        return []
    start = next(iter(pixels))
    far, _ = _farthest(start, pixels)
    end, parents = _farthest(far, pixels)
    path = [end]
    while path[-1] != far:
        path.append(parents[path[-1]])
    return [(int(r), int(c)) for r, c in path]


def _farthest(start, pixels):
    parents = {start: start}
    queue = deque([start])
    last = start
    while queue:
        last = queue.popleft()
        r, c = last
        for dy, dx in _STEPS:
            nxt = (r + dy, c + dx)
            if nxt in pixels and nxt not in parents:
                parents[nxt] = last
                queue.append(nxt)
    return last, parents
