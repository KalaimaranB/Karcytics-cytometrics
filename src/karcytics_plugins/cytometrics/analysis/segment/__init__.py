"""Segmenters: ImageSet → label mask. Importing this package registers all of them."""

from . import cellpose, otsu, watershed  # noqa: F401  (registration side effect)
from .base import (
    REGISTRY,
    Param,
    SegmentationResult,
    Segmenter,
    cells_from_labels,
    clear_border,
    register,
)

__all__ = [
    "REGISTRY",
    "Param",
    "SegmentationResult",
    "Segmenter",
    "cells_from_labels",
    "clear_border",
    "register",
    "run_segmentation",
]


def run_segmentation(segmenter, image_set, params: dict, scale: float, progress=None):
    """Segment, drop edge cells if asked, filter by area and measure.

    Returns ``(labels, cells)``; see ``cells_from_labels``. Progress runs
    0–90 for the segmenter and 90–100 for measuring.
    """
    report = progress or (lambda _value: None)
    result = segmenter.segment(image_set, params, lambda v: report(int(v * 0.9)))
    labels = result.labels
    if params.get("exclude_borders", True):
        labels = clear_border(labels)
    return cells_from_labels(
        labels,
        scale,
        params.get("min_area_px", 0.0),
        params.get("max_area_px", float("inf")),
        result.probability,
        lambda v: report(90 + v // 10),
    )
