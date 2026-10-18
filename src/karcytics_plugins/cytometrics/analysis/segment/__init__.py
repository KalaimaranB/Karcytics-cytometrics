"""Segmenters: ImageSet → label mask. Importing this package registers all of them."""

from dataclasses import dataclass, field

import numpy as np

from . import (  # noqa: F401  (registration side effect)
    cellpose,
    colonies,
    otsu,
    spheroid,
    texture,
    watershed,
)
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
    "Segmentation",
]


@dataclass
class Segmentation:
    """One run's outcome: the final label mask, one dict per cell, and the probability map."""

    labels: np.ndarray
    cells: list[dict]
    probability: np.ndarray | None = None
    extras: dict = field(default_factory=dict)


def run_segmentation(segmenter, image_set, params: dict, progress=None) -> Segmentation:
    """Segment, drop edge cells if asked, filter by pixel area and outline each cell.

    See ``cells_from_labels`` for the cell dicts. Progress runs 0–90 for the
    segmenter and 90–100 for outlining.
    """
    report = progress or (lambda _value: None)
    result = segmenter.segment(image_set, params, lambda v: report(int(v * 0.9)))
    labels = result.labels
    if params.get("exclude_borders", True):
        labels = clear_border(labels)
    labels, cells = cells_from_labels(
        labels,
        params.get("min_area_px", 0.0),
        params.get("max_area_px", float("inf")),
        result.probability,
        lambda v: report(90 + v // 10),
    )
    return Segmentation(labels, cells, result.probability, dict(result.extras))
