"""Spheroids and organoids: a few large, compact objects on a plain background.

A spheroid in bright field is a large dark body (lighter in some setups);
smoothing removes its texture, Otsu's threshold separates it from the
well, holes are filled and only the largest objects are kept, which leaves
out debris and single cells around it. The measurements made from it (area,
equivalent diameter, volume, circularity) are those Zanoni et al. (2016)
recommend for spheroid screening.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import skew

from ..model import ImageSet
from ..units import area_range, effective_scale, length_unit
from .base import (
    EXCLUDE_BORDERS,
    MAX_AREA,
    MIN_AREA,
    TARGET_CHANNEL,
    Param,
    Progress,
    SegmentationResult,
    clear_border,
    otsu_level,
    register,
    require_channel,
)

POLARITY = Param(
    "polarity",
    "Spheroids are",
    "choice",
    "auto",
    choices=(
        ("Found automatically", "auto"),
        ("Darker than the background", "darker"),
        ("Lighter than the background", "lighter"),
    ),
    help="Auto: the side the brightness distribution leans to (the spheroid is the minority).",
)
SMOOTHING = Param(
    "smoothing_px",
    "Smoothing",
    "float",
    4.0,
    0.0,
    100.0,
    unit="length",
    help="Blur this much first, so the spheroid's own texture doesn't break it up.",
)
KEEP = Param(
    "keep_largest",
    "Keep the largest",
    "int",
    1,
    0,
    1000,
    decimals=0,
    special_value_text="All",
    help="How many spheroids per image: the largest objects are kept, the rest is debris.",
)
SENSITIVITY = Param(
    "sensitivity",
    "Sensitivity",
    "float",
    0.0,
    -0.9,
    0.9,
    step=0.1,
    decimals=2,
    help="Raise to include a paler rim, lower to keep only the dense core. 0: Otsu's threshold.",
)


@register
class SpheroidSegmenter:
    id = "spheroid"
    name = "Spheroids (largest objects)"
    summary = "Finds the few large spheroids or organoids in each image and ignores debris."
    citations = ("otsu1979", "opencv2000")
    for_assays = ("spheroids",)

    def parameters(self) -> list[Param]:
        return [
            TARGET_CHANNEL,
            POLARITY,
            SMOOTHING,
            KEEP,
            SENSITIVITY,
            Param(
                EXCLUDE_BORDERS.key, EXCLUDE_BORDERS.label, "bool", True, help=EXCLUDE_BORDERS.help
            ),
            Param(MIN_AREA.key, MIN_AREA.label, "float", 500.0, 0, 1e9, unit="area"),
            Param(MAX_AREA.key, MAX_AREA.label, "float", 1e9, 0, 1e10, unit="area"),
        ]

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        keep = int(params.get("keep_largest", KEEP.default))
        smoothing = params.get("smoothing_px", SMOOTHING.default) * effective_scale(scale)
        parameters: dict = {
            "smoothing": f"{smoothing:g} {length_unit(scale)}",
            "objects kept per image": keep or "all",
            "size filter": area_range(params, scale),
        }
        return (
            f"Spheroids in the '{params.get('target_channel')}' channel were segmented by "
            "Gaussian smoothing and Otsu's threshold with holes filled, keeping the largest "
            "objects in each image",
            parameters,
        )

    def segment(
        self, image_set: ImageSet, params: dict, progress: Progress | None = None
    ) -> SegmentationResult:
        report = progress or (lambda _v: None)
        plane = require_channel(image_set, params.get("target_channel")).data.astype(np.float32)
        sigma = float(params.get("smoothing_px", SMOOTHING.default))
        smooth = cv2.GaussianBlur(plane, (0, 0), sigma) if sigma > 0 else plane
        polarity = params.get("polarity", "auto")
        if polarity == "auto":
            polarity = "lighter" if skew(smooth.ravel()) >= 0 else "darker"
        signal = smooth if polarity == "lighter" else -smooth
        report(40)
        level = otsu_level(signal.ravel())
        lo = float(signal.min())
        level = lo + (level - lo) * (1 - float(params.get("sensitivity", 0.0)))
        mask = ndimage.binary_fill_holes(signal > level)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
        report(80)
        keep = int(params.get("keep_largest", KEEP.default))
        if params.get("exclude_borders", True):  # before choosing, so a cut-off one isn't kept
            labels = clear_border(labels)
        if keep and n > keep:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest = np.argsort(sizes)[::-1][:keep]
            labels = np.where(np.isin(labels, largest[sizes[largest] > 0]), labels, 0)
        return SegmentationResult(labels.astype(np.int32), extras={"polarity": polarity})
