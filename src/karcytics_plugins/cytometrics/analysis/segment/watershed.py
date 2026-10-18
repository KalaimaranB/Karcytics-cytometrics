"""Marker-based watershed: splits touching, roughly round cells."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from ..model import ImageSet
from ..units import area_range, effective_scale, length_unit
from .base import (
    COMMON,
    SEED_CHANNEL,
    TARGET_CHANNEL,
    USE_SEED,
    Param,
    Progress,
    SegmentationResult,
    label_components,
    otsu_mask,
    register,
    require_channel,
    seed_channel,
)

FOOTPRINT_FRACTION = 0.3  # of the Otsu level: catches dim cell edges and extensions
NUCLEUS_CORE_FRACTION = 0.2  # of each nucleus mask's max distance: splits touching nuclei

SEPARATION = Param(
    "min_separation_px",
    "Min. Cell Separation",
    "float",
    default=10.0,
    minimum=1.0,
    maximum=500.0,
    unit="length",
    help="Two cell centres closer than this are treated as one cell. Roughly a cell radius.",
)


@register
class WatershedSegmenter:
    id = "watershed"
    name = "Watershed (Clump Splitting)"
    summary = (
        "Best for clustered, roughly round cells. Splits touching cells from their shape, "
        "or from a nuclei channel when you have one."
    )
    citations = ("meyer1992", "otsu1979", "skimage2014")

    def parameters(self) -> list[Param]:
        return [TARGET_CHANNEL, USE_SEED, SEED_CHANNEL, SEPARATION, *COMMON]

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        nuclei = seed_channel_name(params)
        seeds = (
            f"seeded from nuclei in the '{nuclei}' channel"
            if nuclei
            else "seeded from distance-transform peaks"
        )
        parameters = {"size filter": area_range(params, scale)}
        if not nuclei:
            sep = params.get("min_separation_px", SEPARATION.default) * effective_scale(scale)
            parameters["minimum seed separation"] = f"{sep:g} {length_unit(scale)}"
        return (
            f"Cells in the '{params.get('target_channel')}' channel were segmented by marker-based "
            f"watershed on an Otsu-derived foreground, {seeds}",
            parameters,
        )

    def segment(
        self, image_set: ImageSet, params: dict, progress: Progress | None = None
    ) -> SegmentationResult:
        target = require_channel(image_set, params.get("target_channel")).as_uint8()

        # Footprint: everything brighter than 30% of the Otsu level, gaps closed.
        blurred = cv2.GaussianBlur(target, (5, 5), 0)
        _, otsu_level = otsu_mask(target)
        footprint = blurred > max(5.0, otsu_level * FOOTPRINT_FRACTION)
        # OpenCV's closing replicates the border; scipy's would erode objects at the edge.
        footprint = cv2.morphologyEx(
            footprint.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2
        ).astype(bool)
        footprint = ndimage.binary_fill_holes(footprint)
        distance = ndimage.distance_transform_edt(footprint)

        nuclei = seed_channel(image_set, params)
        if nuclei is not None:
            markers = _nucleus_markers(nuclei.as_uint8())
        else:
            markers = _peak_markers(distance, footprint, params)
        if progress:
            progress(50)

        labels = watershed(-distance, markers, mask=footprint)
        # Footprint regions with no seed get no label from watershed; keep them as objects.
        unseeded = footprint & (labels == 0)
        if unseeded.any():
            extra = label_components(unseeded)
            labels = np.where(extra > 0, extra + labels.max(), labels)
        return SegmentationResult(labels.astype(np.int32))


def seed_channel_name(params: dict) -> str | None:
    return params.get("seed_channel") if params.get("use_dual_channel") else None


def _nucleus_markers(nuclei_u8: np.ndarray) -> np.ndarray:
    mask, _ = otsu_mask(nuclei_u8)
    distance = ndimage.distance_transform_edt(mask)
    cores = distance > NUCLEUS_CORE_FRACTION * distance.max() if distance.max() else mask
    markers, _ = ndimage.label(cores)
    return markers


def _peak_markers(distance: np.ndarray, footprint: np.ndarray, params: dict) -> np.ndarray:
    separation = max(1, int(round(params.get("min_separation_px", SEPARATION.default))))
    smooth = ndimage.gaussian_filter(distance, 1.0)
    regions, _ = ndimage.label(footprint)
    peaks = peak_local_max(smooth, min_distance=separation, labels=regions, exclude_border=False)
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return markers
