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
    "Shape Tolerance",
    "float",
    default=0.4,
    minimum=0.0,
    maximum=3.0,
    step=0.1,
    decimals=2,
    help="Cellpose's flow threshold. Higher keeps more, less regular shapes; lower keeps only "
    "well-formed cells.",
)
CELLPROB_THRESHOLD = Param(
    "cellprob_threshold",
    "Detection Threshold",
    "float",
    default=0.0,
    minimum=-6.0,
    maximum=6.0,
    step=0.5,
    decimals=1,
    help="Cellpose's cell-probability threshold. Lower finds more and larger cells (including "
    "dim ones); higher finds fewer, brighter ones.",
)
NORM_LOW = Param(
    "norm_percentile_low",
    "Contrast: Low Percentile",
    "float",
    default=1.0,
    minimum=0.0,
    maximum=50.0,
    step=0.5,
    advanced=True,
    help="Intensity percentile mapped to black before segmentation.",
)
NORM_HIGH = Param(
    "norm_percentile_high",
    "Contrast: High Percentile",
    "float",
    default=99.0,
    minimum=50.0,
    maximum=100.0,
    step=0.5,
    advanced=True,
    help="Intensity percentile mapped to white. Lower it if a few very bright cells wash out "
    "the rest.",
)
TILE_NORM = Param(
    "tile_norm_blocksize",
    "Uneven Illumination: Tile Size",
    "float",
    default=0.0,
    minimum=0.0,
    maximum=5000.0,
    decimals=0,
    unit="length",
    advanced=True,
    special_value_text="Off",
    help="Normalise contrast in tiles of this size, so cells in dark corners are found too. "
    "Try about 3–5 cell diameters.",
)
AUGMENT = Param(
    "augment",
    "High Quality (slower)",
    "bool",
    default=False,
    advanced=True,
    help="Also runs the network on flipped copies of each tile and averages them.",
)
NITER = Param(
    "niter",
    "Mask Iterations",
    "int",
    default=0,
    minimum=0,
    maximum=2000,
    advanced=True,
    special_value_text="Auto",
    help="Steps used to follow the flows into cells. Raise it for long, thin cells (e.g. "
    "bacteria) that come out fragmented.",
)
MAX_SIZE_FRACTION = Param(
    "max_size_fraction",
    "Largest Object (fraction of image)",
    "float",
    default=0.4,
    minimum=0.01,
    maximum=1.0,
    step=0.05,
    decimals=2,
    advanced=True,
    help="Cellpose discards masks bigger than this fraction of the image.",
)
BATCH_SIZE = Param(
    "batch_size",
    "GPU Batch Size",
    "int",
    default=8,
    minimum=1,
    maximum=64,
    advanced=True,
    help="Image tiles processed at once. Lower it if you run out of memory.",
)


@register
class CellposeSegmenter:
    id = "cellpose"
    name = "AI Smart Detect (Cellpose)"
    summary = (
        "Best for complex clusters and faint extensions. Requires a one-time ~1.2 GB model "
        "download."
    )
    citations = ("cpsam2025", "cellpose2021", "pytorch2019")

    def __init__(self) -> None:
        self.model: Any = None

    def parameters(self) -> list[Param]:
        return [
            TARGET_CHANNEL,
            USE_SEED,
            SEED_CHANNEL,
            DIAMETER,
            CELLPROB_THRESHOLD,
            FLOW_THRESHOLD,
            *COMMON,
            NORM_LOW,
            NORM_HIGH,
            TILE_NORM,
            AUGMENT,
            NITER,
            MAX_SIZE_FRACTION,
            BATCH_SIZE,
        ]

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
            "cell probability threshold": params.get(
                "cellprob_threshold", CELLPROB_THRESHOLD.default
            ),
            "normalisation percentiles": (
                f"{params.get('norm_percentile_low', NORM_LOW.default):g}–"
                f"{params.get('norm_percentile_high', NORM_HIGH.default):g}"
            ),
            "size filter": area_range(params, scale),
            "border cells": "excluded" if params.get("exclude_borders", True) else "kept",
        }
        tile = params.get("tile_norm_blocksize") or 0
        if tile:
            parameters["tile normalisation"] = (
                f"{tile * effective_scale(scale):g} {length_unit(scale)} blocks"
            )
        if params.get("augment"):
            parameters["test-time augmentation"] = "on"
        if params.get("niter"):
            parameters["dynamics iterations"] = params["niter"]
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
        masks, flows, _styles = self.model.eval(image, **eval_kwargs(params, image.ndim))
        report(85)
        # flows[2] is the cell-probability logit map.
        probability = 1.0 / (1.0 + np.exp(-np.asarray(flows[2], dtype=np.float32)))
        return SegmentationResult(np.asarray(masks, dtype=np.int32), probability)


def eval_kwargs(params: dict, ndim: int) -> dict:
    """``CellposeModel.eval`` keyword arguments for `params` (sizes in pixels)."""
    return {
        "channel_axis": 0 if ndim == 3 else None,  # noqa: PLR2004  (C, H, W)
        "diameter": params.get("diameter") or None,
        "flow_threshold": params.get("flow_threshold", FLOW_THRESHOLD.default),
        "cellprob_threshold": params.get("cellprob_threshold", CELLPROB_THRESHOLD.default),
        # Always a dict: passing a bool makes Cellpose 4.2 overwrite its module defaults.
        "normalize": {
            "normalize": True,
            "percentile": [
                params.get("norm_percentile_low", NORM_LOW.default),
                params.get("norm_percentile_high", NORM_HIGH.default),
            ],
            "tile_norm_blocksize": int(round(params.get("tile_norm_blocksize") or 0)),
        },
        "augment": bool(params.get("augment", AUGMENT.default)),
        "niter": params.get("niter") or None,
        "max_size_fraction": params.get("max_size_fraction", MAX_SIZE_FRACTION.default),
        "batch_size": int(params.get("batch_size", BATCH_SIZE.default)),
    }
