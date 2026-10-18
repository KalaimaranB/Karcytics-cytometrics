"""Editing cells: merge, split, brush and region re-detection, on outlines.

Cells are stored as outline polygons. Each edit rasterises the cells it
touches into a small mask, changes the mask, and traces the new outline,
so edits behave like painting on a label mask but stay as compact as the
rest of the session.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from scipy import ndimage

from .model import Channel, ImageSet

MIN_PIECE_PX = 4  # a split or brush leaves pieces at least this big
_PAD = 2


def cell_mask(cell: dict, shape: tuple[int, int]) -> np.ndarray:
    """The cell's filled outline as a boolean image of `shape`."""
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.round(np.asarray(cell["points"], dtype=np.float64)).astype(np.int32)
    if len(points) >= 3:  # noqa: PLR2004
        cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def outline_of(mask: np.ndarray) -> list[list[int]] | None:
    """Outline points of the largest connected part of `mask` (None if empty)."""
    contours, _ = cv2.findContours(
        np.pad(mask.astype(np.uint8), _PAD), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return [[int(x) - _PAD, int(y) - _PAD] for x, y in contour[:, 0, :]]


def pieces_of(mask: np.ndarray, min_px: int = MIN_PIECE_PX) -> list[list[list[int]]]:
    """One outline per 8-connected part of `mask` with at least `min_px` pixels."""
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    outlines = []
    for i in range(1, n + 1):
        part = labels == i
        if part.sum() >= min_px:
            outline = outline_of(part)
            if outline:
                outlines.append(outline)
    return outlines


def merge(
    cells: Sequence[dict], shape: tuple[int, int], bridge_px: int = 2
) -> list[list[int]] | None:
    """Outline of the union of `cells`, bridging gaps up to `bridge_px`.

    Returns None if the cells are still apart after bridging (merging
    distant cells would invent a cell that isn't there).
    """
    union = np.zeros(shape, dtype=bool)
    for cell in cells:
        union |= cell_mask(cell, shape)
    if bridge_px:
        kernel = np.ones((2 * bridge_px + 1, 2 * bridge_px + 1), np.uint8)
        union = cv2.morphologyEx(union.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    _, n = ndimage.label(union, structure=np.ones((3, 3), bool))
    return outline_of(union) if n == 1 else None


def split(
    cell: dict, start: Sequence[float], end: Sequence[float], shape: tuple[int, int]
) -> list[list[list[int]]]:
    """Cut `cell` along the line `start`–`end`; outlines of the pieces (≥2 if it was cut).

    The cut is a 1-pixel, 4-connected line: the thinnest line no 8-connected
    path can cross, so it separates the pieces while losing the fewest pixels.
    """
    mask = cell_mask(cell, shape).astype(np.uint8)
    p0 = (int(round(start[0])), int(round(start[1])))
    p1 = (int(round(end[0])), int(round(end[1])))
    cv2.line(mask, p0, p1, 0, thickness=1, lineType=cv2.LINE_4)
    return pieces_of(mask.astype(bool))


def paint(
    cell: dict, stroke: Sequence[Sequence[float]], radius: int, add: bool, shape: tuple[int, int]
) -> list[list[list[int]]]:
    """Brush `stroke` (with `radius`) onto `cell`: add pixels or erase them.

    Returns the outlines of what's left: one normally, several if erasing
    cut the cell in two, none if it was erased completely. Added pixels
    keep only the part connected to the cell.
    """
    mask = cell_mask(cell, shape).astype(np.uint8)
    brush = np.zeros(shape, dtype=np.uint8)
    points = [(int(round(x)), int(round(y))) for x, y in stroke]
    for a, b in zip(points, points[1:] or points, strict=False):
        cv2.line(brush, a, b, 1, thickness=max(1, 2 * radius))
    for p in points:
        cv2.circle(brush, p, radius, 1, -1)
    if add:
        original = mask.astype(bool)
        labels, _ = ndimage.label(original | brush.astype(bool), structure=np.ones((3, 3), bool))
        touching = np.unique(labels[original])
        outline = outline_of(np.isin(labels, touching[touching > 0]))
        return [outline] if outline else []
    mask &= ~brush.astype(bool)
    return pieces_of(mask.astype(bool))


def crop(image_set: ImageSet, box: tuple[int, int, int, int]) -> ImageSet:
    """The channels inside `box` = (x0, y0, x1, y1), clipped to the image."""
    x0, y0, x1, y1 = clip_box(box, image_set.shape or (0, 0))
    return ImageSet(
        [
            Channel(c.name, c.data[y0:y1, x0:x1], c.color, c.path, c.visible, c.display_range)
            for c in image_set.channels
        ],
        image_set.pixel_size,
    )


def clip_box(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    h, w = shape
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def centroid(cell: dict) -> tuple[float, float]:
    points = np.asarray(cell["points"], dtype=np.float64)
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def offset_cells(cells: list[dict], dx: int, dy: int) -> list[dict]:
    """`cells` found in a crop, moved into image coordinates."""
    return [{**c, "points": [[x + dx, y + dy] for x, y in c["points"]]} for c in cells]


def inside(cell: dict, box: tuple[int, int, int, int]) -> bool:
    x, y = centroid(cell)
    x0, y0, x1, y1 = box
    return x0 <= x < x1 and y0 <= y < y1
