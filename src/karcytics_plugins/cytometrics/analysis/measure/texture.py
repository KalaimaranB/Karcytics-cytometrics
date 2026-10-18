"""Haralick texture features from each object's grey-level co-occurrence matrix.

Haralick, Shanmugam & Dinstein (1973), computed with scikit-image's
``graycomatrix``/``graycoprops`` and named as in CellProfiler's
MeasureTexture. Each channel is quantised to 8 bits over the whole image
(so cells are comparable), only pixel pairs inside the object count, and
the four directions (0°, 45°, 90°, 135°) are averaged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from ..model import ImageSet
from .base import NONE, Column, MeasureOptions, ObjectView, channel_key, register

# CellProfiler name → (scikit-image property, explanation)
FEATURES = {
    "AngularSecondMoment": ("ASM", "Uniformity: high when few grey-level pairs dominate."),
    "Contrast": ("contrast", "Local intensity variation between neighbouring pixels."),
    "Correlation": ("correlation", "Linear dependency of neighbouring grey levels."),
    "InverseDifferenceMoment": ("homogeneity", "Homogeneity: high for smooth texture."),
    "Entropy": ("entropy", "Randomness of the grey-level pairs."),
    "Variance": ("variance", "Spread of grey levels in the co-occurrence matrix."),
}
_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
_LEVELS = 256


@register
class TextureMeasurer:
    id = "texture"
    name = "Texture (Haralick)"
    citations = ("haralick1973", "skimage2014", "cellprofiler4")

    @staticmethod
    def enabled(options: MeasureOptions) -> bool:
        return options.texture

    def columns(self, image_set: ImageSet, options: MeasureOptions) -> list[Column]:
        d = options.texture_distance_px
        return [
            Column(f"Texture_{feature}_{channel_key(ch.name)}_{d}", NONE, f"{text} ({d} px pairs)")
            for ch in image_set.channels
            for feature, (_, text) in FEATURES.items()
        ]

    def prepare(self, image_set: ImageSet, labels: np.ndarray, options: MeasureOptions):
        """Each channel quantised to 0–255 over the whole image."""
        return {ch.name: ch.as_uint8() for ch in image_set.channels}

    def measure(self, obj: ObjectView) -> dict[str, Any]:
        d = obj.options.texture_distance_px
        out: dict[str, Any] = {}
        for ch in obj.image_set.channels:
            key = channel_key(ch.name)
            values = haralick(obj.context[self.id][ch.name][obj.crop], obj.mask, d)
            for feature, value in values.items():
                out[f"Texture_{feature}_{key}_{d}"] = value
        return out


def haralick(plane_u8: np.ndarray, mask: np.ndarray, distance: int) -> dict[str, float]:
    """Direction-averaged Haralick features over pixel pairs inside `mask`.

    Pixels outside the mask are moved to an extra grey level that is then
    dropped from the matrix, so they never pair with object pixels.
    """
    image = plane_u8.astype(np.uint16) + 1
    image[~mask] = 0
    glcm = graycomatrix(image, [distance], _ANGLES, levels=_LEVELS + 1, symmetric=True)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)
    totals = glcm.sum(axis=(0, 1), keepdims=True)
    if not totals.any():  # object thinner than `distance`: no pairs
        return {feature: float("nan") for feature in FEATURES}
    with np.errstate(invalid="ignore", divide="ignore"):
        glcm = np.where(totals > 0, glcm / np.where(totals > 0, totals, 1), 0.0)
        valid = totals[0, 0, 0, :] > 0
        return {
            feature: float(np.nanmean(graycoprops(glcm, prop)[0, valid]))
            for feature, (prop, _) in FEATURES.items()
        }
