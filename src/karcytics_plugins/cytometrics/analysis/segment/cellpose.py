"""Cellpose-SAM: the generalist deep-learning segmenter.

Pachitariu, Rariden & Stringer (2025), built on Cellpose (Stringer et al.
2021). Code and weights are BSD-3 (see EXPANSION_PLAN.md §7). torch and
cellpose are imported on first use, so registering this segmenter is cheap.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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
    register,
    require_channel,
    seed_channel,
)

MODEL_NAME = "cpsam"

DIAMETER = Param(
    "diameter",
    "Cell Diameter",
    "float",
    default=0.0,
    minimum=0.0,
    maximum=500.0,
    unit="length",
    special_value_text="Auto",
    help="Typical cell diameter. Auto works for most images; set it if cells are much "
    "smaller or larger than ~30 px.",
)
FLOW_THRESHOLD = Param(
    "flow_threshold",
    "Splitting Sensitivity",
    "float",
    default=0.4,
    minimum=0.0,
    maximum=3.0,
    step=0.1,
    decimals=2,
    help="Flow error threshold. Higher keeps more (and less regular) shapes; lower is stricter.",
)


@register
class CellposeSegmenter:
    id = "cellpose"
    name = "AI Smart Detect (Cellpose)"
    summary = (
        "Best for complex clusters and faint extensions. Requires a one-time ~1.3 GB model "
        "download."
    )
    citations = ("cpsam2025", "cellpose2021", "pytorch2019")

    def __init__(self) -> None:
        self.model: Any = None

    def parameters(self) -> list[Param]:
        return [TARGET_CHANNEL, USE_SEED, SEED_CHANNEL, DIAMETER, FLOW_THRESHOLD, *COMMON]

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        nuclei = params.get("seed_channel") if params.get("use_dual_channel") else None
        second = f", with the '{nuclei}' channel as a second input" if nuclei else ""
        diameter_px = params.get("diameter")
        parameters = {
            "cell diameter": (
                f"{diameter_px * effective_scale(scale):g} {length_unit(scale)}"
                if diameter_px
                else "automatic"
            ),
            "flow threshold": params.get("flow_threshold", FLOW_THRESHOLD.default),
            "size filter": area_range(params, scale),
            "border cells": "excluded" if params.get("exclude_borders", True) else "kept",
        }
        return (
            f"Cells in the '{params.get('target_channel')}' channel were segmented with the "
            f"Cellpose-SAM generalist model ({MODEL_NAME}){second}",
            parameters,
        )

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        import torch
        from cellpose import models

        if torch.cuda.is_available():
            device, gpu = torch.device("cuda"), True
        elif torch.backends.mps.is_available():
            device, gpu = torch.device("mps"), True
        else:
            device, gpu = torch.device("cpu"), False
        self.model = models.CellposeModel(pretrained_model=MODEL_NAME, gpu=gpu, device=device)

    def segment(
        self, image_set: ImageSet, params: dict, progress: Progress | None = None
    ) -> SegmentationResult:
        report = progress or (lambda _value: None)
        target = require_channel(image_set, params.get("target_channel")).data
        nuclei = seed_channel(image_set, params)
        # Cellpose normalises internally, so it gets native bit depth. Cellpose-SAM
        # takes (channels, H, W) with `channel_axis`; the old `channels=` index
        # argument is ignored from v4 on.
        image = np.stack([target, nuclei.data]) if nuclei is not None else target

        report(5)
        self._ensure_model()
        report(15)
        diameter = params.get("diameter") or None
        masks, flows, _styles = self.model.eval(
            image,
            channel_axis=0 if image.ndim == 3 else None,  # noqa: PLR2004
            diameter=diameter,
            flow_threshold=params.get("flow_threshold", FLOW_THRESHOLD.default),
        )
        report(85)
        # flows[2] is the cell-probability logit map.
        probability = 1.0 / (1.0 + np.exp(-np.asarray(flows[2], dtype=np.float32)))
        return SegmentationResult(np.asarray(masks, dtype=np.int32), probability)
