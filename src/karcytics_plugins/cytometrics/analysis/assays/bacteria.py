"""Bacteria: count rod-shaped cells and measure length, width, curvature and division.

Cellpose-SAM segments bacteria well out of the box (its training data
include bacterial images); the rod measurements are in ``measure/rods.py``.
"""

from __future__ import annotations

import pandas as pd

from ..gating import Gate
from ..units import length_unit
from .base import Assay, AssayAnalysis, AssayContext, ResultValue, register
from .common import percent


class BacteriaSummary(AssayAnalysis):
    citations = ("zhang1984",)
    suggests_gates = True

    def suggested_gates(self, df: pd.DataFrame, settings: dict) -> list[Gate]:
        return [
            Gate("Dividing", "threshold", x="Rod_Dividing", params={"value": 0.5}, color="#D55E00")
        ]

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        if "Rod_Length" not in df or not len(df):
            return []
        unit = length_unit(context.scale)
        length = pd.to_numeric(df["Rod_Length"], errors="coerce")
        width = pd.to_numeric(df["Rod_Width"], errors="coerce")
        dividing = pd.to_numeric(df["Rod_Dividing"], errors="coerce").to_numpy() > 0.5  # noqa: PLR2004
        return [
            ResultValue("Cells", "Cells", len(df)),
            ResultValue("MedianLength", "Median length", float(length.median()), unit),
            ResultValue("MedianWidth", "Median width", float(width.median()), unit),
            ResultValue(
                "MedianCurvature",
                "Median curvature",
                float(pd.to_numeric(df["Rod_Curvature"], errors="coerce").median()),
                "",
                "0 = straight rods.",
            ),
            ResultValue(
                "PctDividing", "Dividing", percent(dividing), "%", "Cells pinched in the middle."
            ),
        ]


BACTERIA = register(
    Assay(
        id="bacteria",
        name="Bacteria",
        description="Count rod-shaped bacteria in phase-contrast or fluorescence images: length, "
        "width, curvature and the share that are dividing.",
        icon="🦠",
        segmenter="cellpose",
        group="Microbiology",
        steps=(
            "load",
            "calibrate",
            "detect",
            "review",
            "measure",
            "gate",
            "results",
            "report",
            "batch",
        ),
        settings={"exclude_borders": True, "min_area_px": 4.0},
        measure={"background": "none", "rods": True},
        noun="Bacterium",
        load_hint="Load phase-contrast or fluorescence images of the bacteria, ideally at 60–100×.",
        analysis=BacteriaSummary(),
        citations=("cpsam2025",),
    )
)
