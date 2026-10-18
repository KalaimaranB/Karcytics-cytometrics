"""Transfection / % positive: the share of cells expressing a reporter (GFP, mCherry, …).

The cut-off comes from a negative control: load an untransfected (or
unstained) image, measure it, and "Use this image as the negative control"
sets the cut-off at a high percentile of those cells' reporter intensity
(99th by default, so 1% of control cells would be called positive). Every
image measured afterwards, here or in a batch, is scored against it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from .. import gating
from ..gating import Gate
from ..segment.base import Param
from .base import ActionResult, Assay, AssayAnalysis, AssayContext, ResultValue, register
from .common import intensity_column, percent

POSITIVE = "Positive"
STATS = (("Mean intensity", "MeanIntensity"), ("Integrated intensity", "IntegratedIntensity"))


class PercentPositive(AssayAnalysis):
    channel_stains = {"reporter_channel": "green"}
    actions = (
        (
            "set_control",
            "🎯 Use this image as the negative control",
            "Set the cut-off from the cells in this image (untransfected or unstained), then "
            "load your samples.",
        ),
    )

    def parameters(self) -> list[Param]:
        return [
            Param("reporter_channel", "Reporter", "channel", help="The transfection reporter."),
            Param(
                "statistic",
                "Score by",
                "choice",
                "MeanIntensity",
                choices=STATS,
                help="Mean: brightness. Integrated: total signal (also grows with cell size).",
            ),
            Param(
                "percentile",
                "Control percentile",
                "float",
                99.0,
                50.0,
                99.99,
                decimals=2,
                help="The cut-off is this percentile of the control cells: 99 calls 1% of them "
                "positive.",
            ),
            Param(
                "cutoff",
                "Cut-off",
                "float",
                0.0,
                0.0,
                1e15,
                decimals=2,
                special_value_text="Not set",
                help="Cells above this are positive. Set from a negative control, or type it.",
            ),
        ]

    def _column(self, df: pd.DataFrame, settings: dict) -> str | None:
        return intensity_column(
            df, settings.get("reporter_channel"), settings.get("statistic") or "MeanIntensity"
        )

    def run_action(self, action, df, settings, context, box=None) -> ActionResult:
        column = self._column(df, settings)
        if column is None or not len(df):
            raise ValueError("Detect and measure the control image's cells first.")
        pct = float(settings.get("percentile") or 99)
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        cutoff = float(np.percentile(values, pct))
        return ActionResult(
            settings={"cutoff": cutoff, "control_cells": len(values)},
            gates=[self._gate(column, cutoff)],
            message=f"Cut-off set to {cutoff:.4g} from {len(values)} control cells "
            f"({pct:g}th percentile). Now load a sample.",
        )

    def _gate(self, column: str, cutoff: float) -> Gate:
        return Gate(POSITIVE, "threshold", x=column, params={"value": cutoff}, color="#009E73")

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        column = self._column(df, settings)
        cutoff = float(settings.get("cutoff") or 0)
        if column is None or not len(df):
            return []
        if not cutoff:
            return [
                ResultValue(
                    "PctPositive",
                    "Positive",
                    float("nan"),
                    "%",
                    warning="set a cut-off: measure a negative control and click 🎯",
                )
            ]
        values = pd.to_numeric(df[column], errors="coerce").to_numpy()
        if any(g.name == POSITIVE for g in gates):
            positive = gating.evaluate(gates, df)[POSITIVE]
        else:
            positive = values > cutoff
        out = [
            ResultValue("Cells", "Cells", len(df)),
            ResultValue("PctPositive", "Positive", percent(positive), "%"),
            ResultValue("Positive", "Positive cells", int(positive.sum())),
            ResultValue("Cutoff", "Cut-off", cutoff, "a.u."),
        ]
        if positive.any():
            out.append(
                ResultValue(
                    "MeanPositive",
                    "Mean reporter in positive cells",
                    float(np.nanmean(values[positive])),
                    "a.u.",
                )
            )
        return out

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        cutoff = float(settings.get("cutoff") or 0)
        if not cutoff:
            return []
        stat = {v: k for k, v in STATS}.get(str(settings.get("statistic")), "mean intensity")
        source = (
            f", the {float(settings.get('percentile') or 99):g}th percentile of "
            f"{settings['control_cells']} negative-control cells"
            if settings.get("control_cells")
            else ""
        )
        return [
            MethodStep(
                f"Cells were scored positive when their {stat.lower()} of "
                f"'{settings.get('reporter_channel')}' exceeded {cutoff:.4g}{source}"
            )
        ]


TRANSFECTION = register(
    Assay(
        id="transfection",
        name="Transfection / % positive",
        description="The share of cells expressing a reporter (GFP, mCherry, …), with the cut-off "
        "set from a negative-control image.",
        icon="💡",
        segmenter="cellpose",
        group="Cells",
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
        measure={"background": "local"},
        analysis=PercentPositive(),
        citations=("cpsam2025",),
    )
)
