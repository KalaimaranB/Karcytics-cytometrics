"""Pixel intensities of each object, per channel, in the image's native units."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model import ImageSet
from .base import INTENSITY, Column, MeasureOptions, ObjectView, channel_key, register

_STATS = [
    ("MeanIntensity", "Mean pixel value inside the object."),
    ("MedianIntensity", "Median pixel value inside the object."),
    ("MinIntensity", "Lowest pixel value inside the object."),
    ("MaxIntensity", "Highest pixel value inside the object."),
    ("StdIntensity", "Standard deviation of the pixel values inside the object."),
    ("IntegratedIntensity", "Sum of the pixel values inside the object (total signal)."),
]
_BACKGROUND = {
    "global": "the median of all pixels outside every object",
    "local": "the median of a ring of pixels around the object, away from other objects",
}


@register
class IntensityMeasurer:
    id = "intensity"
    name = "Intensity"
    citations = ("cellprofiler4",)

    def columns(self, image_set: ImageSet, options: MeasureOptions) -> list[Column]:
        cols = []
        for ch in image_set.channels:
            key = channel_key(ch.name)
            cols += [
                Column(f"Intensity_{stat}_{key}", INTENSITY, f"{text} Channel '{ch.name}'.")
                for stat, text in _STATS
            ]
            if options.background in _BACKGROUND:
                how = _BACKGROUND[options.background]
                cols += [
                    Column(f"Intensity_Background_{key}", INTENSITY, f"Background: {how}."),
                    Column(
                        f"Intensity_CorrectedMeanIntensity_{key}",
                        INTENSITY,
                        "Mean intensity minus background.",
                    ),
                    Column(
                        f"Intensity_CorrectedIntegratedIntensity_{key}",
                        INTENSITY,
                        "Integrated intensity minus background × area (in pixels).",
                    ),
                ]
        return cols

    def measure(self, obj: ObjectView) -> dict[str, Any]:
        out: dict[str, Any] = {}
        mode = obj.options.background
        ring = local_ring(obj) if mode == "local" else None
        n = int(obj.mask.sum())
        for ch in obj.image_set.channels:
            key = channel_key(ch.name)
            pixels = obj.pixels(ch.name)
            inside = pixels[obj.mask]
            mean = float(inside.mean())
            integrated = float(inside.sum())
            out[f"Intensity_MeanIntensity_{key}"] = mean
            out[f"Intensity_MedianIntensity_{key}"] = float(np.median(inside))
            out[f"Intensity_MinIntensity_{key}"] = float(inside.min())
            out[f"Intensity_MaxIntensity_{key}"] = float(inside.max())
            out[f"Intensity_StdIntensity_{key}"] = float(inside.std())
            out[f"Intensity_IntegratedIntensity_{key}"] = integrated
            if mode in _BACKGROUND:
                if mode == "global":
                    bg = obj.background.get(ch.name, 0.0)
                else:
                    assert ring is not None
                    bg = float(np.median(pixels[ring])) if ring.any() else float("nan")
                out[f"Intensity_Background_{key}"] = bg
                out[f"Intensity_CorrectedMeanIntensity_{key}"] = mean - bg
                out[f"Intensity_CorrectedIntegratedIntensity_{key}"] = integrated - bg * n
        return out


def local_ring(obj: ObjectView) -> np.ndarray:
    """Pixels between `ring_gap_px` and `ring_gap_px + ring_width_px` around the object,
    excluding (a slightly grown copy of) every other object.
    """
    from scipy import ndimage

    gap, width = obj.options.ring_gap_px, obj.options.ring_width_px
    square = np.ones((3, 3), bool)
    near = ndimage.binary_dilation(obj.mask, square, iterations=gap) if gap else obj.mask
    outer = ndimage.binary_dilation(obj.mask, square, iterations=gap + width)
    others = ndimage.binary_dilation(obj.others, square, iterations=gap) if gap else obj.others
    return outer & ~near & ~others


def global_background(image_set: ImageSet, labels: np.ndarray, gap_px: int = 2) -> dict[str, float]:
    """Median of each channel outside every object (objects grown by `gap_px` first)."""
    from scipy import ndimage

    occupied = labels > 0
    if gap_px:
        occupied = ndimage.binary_dilation(occupied, np.ones((3, 3), bool), iterations=gap_px)
    free = ~occupied
    if not free.any():
        return {ch.name: float("nan") for ch in image_set.channels}
    return {ch.name: float(np.median(ch.data[free])) for ch in image_set.channels}
