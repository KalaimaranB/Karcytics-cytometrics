"""Global Otsu threshold: fast, for sparse, well-separated objects."""

from __future__ import annotations

from ..model import ImageSet
from ..units import area_range
from .base import (
    COMMON,
    TARGET_CHANNEL,
    Param,
    Progress,
    SegmentationResult,
    label_components,
    otsu_mask,
    register,
    require_channel,
)


@register
class OtsuSegmenter:
    id = "otsu"
    name = "Basic Threshold (Otsu)"
    summary = "Best for sparse, isolated cells. Very fast, but will merge touching cells together."
    citations = ("otsu1979", "opencv2000")

    def parameters(self) -> list[Param]:
        return [TARGET_CHANNEL, *COMMON]

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        return (
            f"Objects in the '{params.get('target_channel')}' channel were segmented by Otsu "
            "thresholding after a 5 × 5 Gaussian blur",
            {"size filter": area_range(params, scale)},
        )

    def segment(
        self,
        image_set: ImageSet,
        params: dict,
        progress: Progress | None = None,
    ) -> SegmentationResult:
        target = require_channel(image_set, params.get("target_channel"))
        mask, _ = otsu_mask(target.as_uint8())
        return SegmentationResult(label_components(mask))
