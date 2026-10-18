"""Nucleus vs cytoplasm inside each cell, from a nuclear stain.

The nucleus is the part of the cell above the nuclear channel's Otsu
threshold (computed over the whole image); the rest of the cell is
cytoplasm. Reports their areas, the N/C area ratio and, per channel, the
nuclear and cytoplasmic mean intensity and their ratio (e.g. for
translocation assays).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.filters import threshold_otsu

from ..model import ImageSet
from .base import AREA, INTENSITY, NONE, Column, MeasureOptions, ObjectView, channel_key, register


@register
class NucleusMeasurer:
    id = "nucleus"
    name = "Nucleus / cytoplasm"
    citations = ("otsu1979", "skimage2014")

    @staticmethod
    def enabled(options: MeasureOptions) -> bool:
        return bool(options.nucleus_channel)

    def columns(self, image_set: ImageSet, options: MeasureOptions) -> list[Column]:
        cols = [
            Column("Nucleus_Area", AREA, "Area of the cell above the nuclear stain's threshold."),
            Column("Cytoplasm_Area", AREA, "Area of the cell outside the nucleus."),
            Column("Nucleus_NCAreaRatio", NONE, "Nucleus area / cytoplasm area."),
        ]
        for ch in image_set.channels:
            key = channel_key(ch.name)
            cols += [
                Column(
                    f"Intensity_NucleusMean_{key}", INTENSITY, f"Mean '{ch.name}' in the nucleus."
                ),
                Column(
                    f"Intensity_CytoplasmMean_{key}",
                    INTENSITY,
                    f"Mean '{ch.name}' in the cytoplasm.",
                ),
                Column(
                    f"Intensity_NCRatio_{key}",
                    NONE,
                    f"Nuclear / cytoplasmic mean '{ch.name}' (> 1: enriched in the nucleus).",
                ),
            ]
        return cols

    def prepare(self, image_set: ImageSet, labels: np.ndarray, options: MeasureOptions):
        channel = image_set.get(options.nucleus_channel)
        if channel is None:
            raise ValueError(f"Nucleus channel {options.nucleus_channel!r} is not loaded.")
        inside = channel.data[labels > 0]
        values = inside if inside.size and np.ptp(inside) else channel.data
        return float(threshold_otsu(values)) if np.ptp(values) else float(values.max())

    def measure(self, obj: ObjectView) -> dict[str, Any]:
        nuclear = obj.pixels(obj.options.nucleus_channel or "") > obj.context[self.id]
        nucleus = obj.mask & nuclear
        cytoplasm = obj.mask & ~nuclear
        px_area = obj.scale**2
        n_nuc, n_cyt = int(nucleus.sum()), int(cytoplasm.sum())
        out: dict[str, Any] = {
            "Nucleus_Area": n_nuc * px_area,
            "Cytoplasm_Area": n_cyt * px_area,
            "Nucleus_NCAreaRatio": n_nuc / n_cyt if n_cyt else float("nan"),
        }
        for ch in obj.image_set.channels:
            key = channel_key(ch.name)
            pixels = obj.pixels(ch.name)
            nuc = float(pixels[nucleus].mean()) if n_nuc else float("nan")
            cyt = float(pixels[cytoplasm].mean()) if n_cyt else float("nan")
            out[f"Intensity_NucleusMean_{key}"] = nuc
            out[f"Intensity_CytoplasmMean_{key}"] = cyt
            out[f"Intensity_NCRatio_{key}"] = nuc / cyt if cyt else float("nan")
        return out
