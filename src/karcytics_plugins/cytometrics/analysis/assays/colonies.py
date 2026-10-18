"""Colony counting: colonies per plate and CFU/mL.

CFU/mL = colonies × dilution factor ÷ volume plated (mL). Plates with
30–300 colonies are the countable range (Breed & Dotterrer 1916): fewer is
statistically unreliable, more and colonies merge and compete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from ..gating import Gate
from ..segment.base import Param
from ..units import is_calibrated, length_unit
from .base import Assay, AssayAnalysis, AssayContext, Overlay, ResultValue, register

COUNTABLE = (30, 300)


class ColonyCount(AssayAnalysis):
    citations = ("opencfu2013", "breed1916")

    def parameters(self) -> list[Param]:
        return [
            Param(
                "dilution_factor",
                "Dilution factor (1 : …)",
                "float",
                1.0,
                1.0,
                1e15,
                decimals=0,
                help="How many times the sample was diluted before plating: 1e6 for a 10⁻⁶ dilution.",
            ),
            Param(
                "volume_ul",
                "Volume plated (µL)",
                "float",
                100.0,
                0.001,
                1e6,
                decimals=1,
                help="How much of the dilution was spread on the plate.",
            ),
        ]

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        n = len(df)
        lo, hi = COUNTABLE
        warning = ""
        if n < lo:
            warning = f"fewer than {lo} colonies: too few for a reliable count"
        elif n > hi:
            warning = f"more than {hi} colonies: plate a higher dilution"
        dilution = float(settings.get("dilution_factor") or 1)
        volume_ml = float(settings.get("volume_ul") or 100) / 1000
        out = [
            ResultValue("Colonies", "Colonies", n, "", "Colonies counted on the plate."),
            ResultValue(
                "CFU_per_mL",
                "CFU / mL",
                n * dilution / volume_ml if volume_ml else float("nan"),
                "CFU/mL",
                "Colonies × dilution factor ÷ volume plated.",
                warning,
            ),
        ]
        if "AreaShape_EquivalentDiameter" in df and n:
            diameter = float(pd.to_numeric(df["AreaShape_EquivalentDiameter"]).median())
            out.append(
                ResultValue(
                    "MedianDiameter",
                    "Median colony diameter",
                    diameter,
                    length_unit(context.scale),
                    "Diameter of a circle with the colony's area.",
                )
            )
        dish = context.extras.get("dish")
        if dish and n and "AreaShape_Area" in df:
            area_px = float(pd.to_numeric(df["AreaShape_Area"]).sum())
            if is_calibrated(context.scale):
                area_px /= context.scale**2  # the table's areas are in µm²
            out.append(
                ResultValue(
                    "Coverage",
                    "Plate covered",
                    100 * area_px / (np.pi * dish[2] ** 2),
                    "%",
                    "Total colony area over the plate area inside the rim.",
                )
            )
        return out

    def overlays(self, settings: dict, context: AssayContext) -> list[Overlay]:
        dish = context.extras.get("dish")
        return [Overlay("circle", tuple(dish), "#ffd166", "Plate")] if dish else []

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        dilution = float(settings.get("dilution_factor") or 1)
        volume = float(settings.get("volume_ul") or 100)
        return [
            MethodStep(
                f"Colony-forming units per mL were calculated as colonies × {dilution:g} "
                f"(dilution factor) ÷ {volume:g} µL plated; plates outside 30–300 colonies "
                "were flagged as outside the countable range",
                ("breed1916",),
            )
        ]


COLONIES = register(
    Assay(
        id="colonies",
        name="Colony counter",
        description="Count bacterial or yeast colonies on an agar plate photo and get CFU/mL "
        "from the dilution and the volume plated.",
        icon="🧫",
        segmenter="colonies",
        group="Microbiology",
        steps=("load", "calibrate", "detect", "review", "results", "report", "batch"),
        settings={"exclude_borders": False},
        measure={"background": "none"},
        noun="Colony",
        analysis=ColonyCount(),
        citations=("opencfu2013",),
    )
)
