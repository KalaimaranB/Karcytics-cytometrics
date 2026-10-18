"""Spheroids and organoids: size, volume and shape (``segment.spheroid``).

A spheroid seen from above is taken to be a sphere with the diameter of a
circle of its projected area (the equivalent diameter, d), so its volume is
π d³ / 6 (Zanoni 2016). Circularity (4π area / perimeter², 1 for a circle)
and solidity flag spheroids that are irregular or falling apart, which
makes the volume estimate unreliable.
"""

from __future__ import annotations

import math

import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from ..experiment import Comparison, Plot, Role, Sample, by_condition
from ..gating import Gate
from ..units import area_unit, is_calibrated, length_unit
from .base import PHOTO, Assay, AssayAnalysis, AssayContext, ResultValue, register, with_columns

VOLUME = "Spheroid_Volume"
DIAMETER = "AreaShape_EquivalentDiameter"
IRREGULAR = 0.8  # circularity below this: not round enough for a sphere's volume


def volume_unit(scale: float) -> str:
    return "µm³" if is_calibrated(scale) else "px³"


CONDITION = "condition"


class SpheroidSize(AssayAnalysis):
    citations = ("zanoni2016",)
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
        diameter, _ = by_condition(samples, "Diameter", "diameter", length_unit(context.scale))
        volume, points = by_condition(samples, "Volume", "volume", volume_unit(context.scale))
        rows = [
            {
                "Image": s.name,
                "Spheroids": s.results.get("Spheroids"),
                f"Diameter ({length_unit(context.scale)})": s.results.get("Diameter"),
                f"Volume ({volume_unit(context.scale)})": s.results.get("Volume"),
                "Circularity": s.results.get("Circularity"),
            }
            for s in samples
        ]
        return Comparison(
            values=diameter + volume,
            table=pd.DataFrame(rows),
            plot=Plot("bars", label=f"Volume ({volume_unit(context.scale)})", points=points),
            note="Per condition: the mean of its images (one spheroid per image: its own value; "
            "several: their median).",
        )

    def derive(self, df: pd.DataFrame, settings: dict, context: AssayContext) -> pd.DataFrame:
        if DIAMETER not in df:
            return df
        d = pd.to_numeric(df[DIAMETER], errors="coerce")
        return with_columns(
            df,
            {VOLUME: math.pi * d**3 / 6},
            {VOLUME: volume_unit(context.scale)},
            {VOLUME: "π d³ / 6, a sphere with the equivalent diameter d."},
        )

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        n = len(df)
        out = [ResultValue("Spheroids", "Spheroids", n)]
        if not n or VOLUME not in df:
            return out

        def label(name: str) -> str:  # one spheroid: its value; several: their median
            return name.capitalize() if n == 1 else f"Median {name}"

        def median(column: str) -> float:
            return float(pd.to_numeric(df[column], errors="coerce").median())

        circularity = median("AreaShape_FormFactor") if "AreaShape_FormFactor" in df else math.nan
        out += [
            ResultValue(
                "Diameter",
                label("diameter"),
                median(DIAMETER),
                length_unit(context.scale),
                "Diameter of a circle with the spheroid's projected area.",
            ),
            ResultValue(
                "Area",
                label("area"),
                median("AreaShape_Area"),
                area_unit(context.scale),
                "Projected area.",
            ),
            ResultValue(
                "Volume",
                label("volume"),
                median(VOLUME),
                volume_unit(context.scale),
                "π d³ / 6: the volume of a sphere with that diameter.",
                "irregular shape: the volume is a rough estimate"
                if circularity < IRREGULAR
                else "",
            ),
            ResultValue(
                "Circularity",
                label("circularity"),
                circularity,
                "",
                "4π area / perimeter²: 1 for a perfect circle.",
            ),
        ]
        if "AreaShape_Solidity" in df:
            out.append(
                ResultValue(
                    "Solidity",
                    label("solidity"),
                    median("AreaShape_Solidity"),
                    "",
                    "Area / convex hull area: below ~0.9 the spheroid has gaps or protrusions.",
                )
            )
        return out

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        return [
            MethodStep(
                "Spheroid volume was estimated as π d³ / 6 from the equivalent diameter d of the "
                "projected area, with circularity and solidity reported to flag irregular "
                "spheroids",
                ("zanoni2016",),
            )
        ]


SPHEROIDS = register(
    Assay(
        id="spheroids",
        name="Spheroids & organoids",
        description="Size, volume and shape of 3D spheroids or organoids imaged from above.",
        icon="⚪",
        segmenter="spheroid",
        group="Growth & migration",
        steps=("load", "calibrate", "detect", "review", "results", "report", "batch"),
        measure={"background": "none"},
        noun="Spheroid",
        image_kind=PHOTO,
        load_hint="Load a bright-field image of the well, from above. One spheroid per image works best.",
        analysis=SpheroidSize(),
        citations=("zanoni2016",),
    )
)
