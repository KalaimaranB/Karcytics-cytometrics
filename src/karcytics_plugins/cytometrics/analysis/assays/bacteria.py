"""Bacteria: count rod-shaped cells and measure length, width, curvature and division.

Cellpose-SAM segments bacteria well out of the box (its training data
include bacterial images); the rod measurements are in ``measure/rods.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..experiment import PALETTE, Comparison, Plot, Role, Sample, Series, by_condition
from ..gating import Gate
from ..units import length_unit
from .base import Assay, AssayAnalysis, AssayContext, ResultValue, register
from .common import percent

CONDITION = "condition"


class BacteriaSummary(AssayAnalysis):
    citations = ("zhang1984",)
    suggests_gates = True
    roles = (
        Role(
            CONDITION,
            "image of a condition",
            named=True,
            groups=True,
            help="Name the condition (Control, Drug A, ...). Images with the same name are "
            "replicates and are averaged.",
        ),
    )

    def compare(
        self, samples: list[Sample], settings: dict, context: AssayContext
    ) -> Comparison | None:
        unit = length_unit(context.scale)
        lengths, dividing = by_condition(samples, "MedianLength", "median length", unit)
        pct, _ = by_condition(samples, "PctDividing", "dividing", "%", ".1f")
        names = list(dict.fromkeys(s.name for s in samples))
        colors = dict(
            zip(names, (PALETTE[i % len(PALETTE)] for i in range(len(names))), strict=True)
        )
        series = []
        for name in names:
            pooled = [
                pd.to_numeric(s.table.get("Rod_Length", pd.Series(dtype=float)), errors="coerce")
                for s in samples
                if s.name == name
            ]
            values = pd.concat(pooled).to_numpy(float) if pooled else np.zeros(0)
            series.append(Series(name, values, colors[name]))
        rows = [
            {
                "Image": s.name,
                "Cells": len(s.table),
                f"Median length ({unit})": s.results.get("MedianLength"),
                f"Median width ({unit})": s.results.get("MedianWidth"),
                "% dividing": s.results.get("PctDividing"),
            }
            for s in samples
        ]
        return Comparison(
            values=lengths + pct,
            table=pd.DataFrame(rows),
            plot=Plot("histogram", label=f"Cell length ({unit})", series=series),
            note="Lengths of every cell, pooled per condition; values are means of the images.",
        )

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
