"""Confluence and scratch-wound assays, from the covered area (``segment.texture``).

Confluence is the % of the image covered by cells. A scratch (wound
healing) assay follows a gap scratched into a confluent layer as cells
migrate into it (Jonkman 2014): its open area, its mean width (open area ÷
the wound's length) and its closure relative to the start,

    closure = (area at the start − area now) ÷ area at the start × 100%.

The first image is added to the experiment as the start; each later image
added (named by its time or condition) is measured against it.
"""

from __future__ import annotations

import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from ..experiment import Comparison, Plot, Role, Sample, by_condition
from ..gating import Gate
from ..segment.base import Param
from ..units import area_unit, effective_scale, length_unit
from .base import PHOTO, Assay, AssayAnalysis, AssayContext, ResultValue, register

AREA = "AreaShape_Area"


def _area_px(df: pd.DataFrame, scale: float) -> float:
    """The objects' total area in pixels (the table's areas are in µm² when calibrated)."""
    if AREA not in df or not len(df):
        return 0.0
    return float(pd.to_numeric(df[AREA], errors="coerce").sum()) / effective_scale(scale) ** 2


def _image_px(context: AssayContext) -> float:
    return float(context.shape[0] * context.shape[1]) if context.shape else float("nan")


CONDITION, START, LATER = "condition", "start", "later"


class Confluence(AssayAnalysis):
    citations = ()
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
        values, points = by_condition(samples, "Confluence", "confluence", "%", ".1f")
        rows = [{"Image": s.name, "Confluence %": s.results.get("Confluence")} for s in samples]
        return Comparison(
            values=values,
            table=pd.DataFrame(rows),
            plot=Plot("bars", label="Confluence (%)", points=points),
            note="Per condition: the mean confluence of its images (fields).",
        )

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        coverage = context.extras.get("coverage")
        if coverage is None:
            coverage = _area_px(df, context.scale) / _image_px(context)
        return [
            ResultValue(
                "Confluence",
                "Confluence",
                100 * float(coverage),
                "%",
                "The % of the image covered by cells.",
            ),
            ResultValue("Patches", "Covered patches", len(df), "", "Separate cell-covered areas."),
        ]

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        return [MethodStep("Confluence was reported as the % of the image area covered by cells")]


class Wound(AssayAnalysis):
    citations = ("jonkman2014",)
    roles = (
        Role(
            START,
            "the start (0 h)",
            many=False,
            help="The scratch as it was made: later images' closure is measured from it.",
        ),
        Role(
            LATER,
            "a later image",
            named=True,
            groups=True,
            help="Name it by its time or condition (12 h, Drug A 24 h, ...). Images with the "
            "same name are averaged.",
        ),
    )

    def compare(
        self, samples: list[Sample], settings: dict, context: AssayContext
    ) -> Comparison | None:
        start = next((s for s in samples if s.role == START), None)
        start_area = float(start.results.get("WoundArea") or 0) if start else 0.0
        later = [s for s in samples if s.role == LATER]
        for s in later:  # closure from each image's own area against the start's
            area = float(s.results.get("WoundArea") or 0)
            s.results["Closure"] = 100 * (start_area - area) / start_area if start_area else None
        values, points = by_condition(later, "Closure", "closure", "%", ".1f")
        rows = [
            {
                "Image": s.name,
                "Role": "start" if s.role == START else "later",
                f"Wound area ({area_unit(context.scale)})": s.results.get("WoundArea"),
                f"Mean width ({length_unit(context.scale)})": s.results.get("MeanWidth"),
                "Closure %": 0.0 if s.role == START else s.results.get("Closure"),
            }
            for s in samples
        ]
        derived = {}
        if start_area:
            derived["start_area_px"] = round(start_area / effective_scale(context.scale) ** 2)
        if not start:
            note = "Add the first image (0 h) as the start."
        elif not later:
            note = "Now load a later image of the same scratch and add it."
        else:
            note = "Closure: (area at the start − area now) ÷ area at the start."
        return Comparison(
            values=values,
            table=pd.DataFrame(rows),
            plot=Plot("bars", label="Closure (%)", points=points),
            settings=derived,
            note=note,
        )

    def parameters(self) -> list[Param]:
        return [
            Param(
                "start_area_px",
                "Wound area at the start (px²)",
                "float",
                0.0,
                0.0,
                1e12,
                decimals=0,
                special_value_text="Not set",
                help="The open area at 0 h: set by the start image in the experiment, or typed in.",
            )
        ]

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        scale = context.scale
        size = effective_scale(scale)
        area_px = _area_px(df, scale)
        wound = context.extras.get("wound", {})
        length = wound.get("length_px") or (
            (context.shape[0] if wound.get("direction") != "horizontal" else context.shape[1])
            if context.shape
            else 0
        )
        out = [
            ResultValue(
                "WoundArea", "Wound area", area_px * size**2, area_unit(scale), "Open area."
            ),
            ResultValue(
                "WoundPct",
                "Image open",
                100 * area_px / _image_px(context),
                "%",
                "Open area ÷ image.",
            ),
            ResultValue(
                "MeanWidth",
                "Mean width",
                area_px / length * size if length else float("nan"),
                length_unit(scale),
                "Open area ÷ the wound's length.",
            ),
        ]
        if "min_px" in wound:
            out.append(
                ResultValue(
                    "MinWidth",
                    "Narrowest",
                    wound["min_px"] * size,
                    length_unit(scale),
                    "The narrowest line across the wound, as detected (0: cells have bridged it).",
                )
            )
        start = float(settings.get("start_area_px") or 0)
        out.append(
            ResultValue(
                "Closure",
                "Closure",
                100 * (start - area_px) / start if start > 0 else float("nan"),
                "%",
                "(Area at the start − area now) ÷ area at the start.",
                "" if start > 0 else "add the first image to the experiment as the start",
            )
        )
        return out

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        return [
            MethodStep(
                "Wound closure was measured as the decrease in open (cell-free) area relative to "
                "the start, and the mean wound width as the open area divided by the wound's "
                "length",
                ("jonkman2014",),
            )
        ]


CONFLUENCE = register(
    Assay(
        id="confluence",
        name="Confluence",
        description="How much of the dish cells cover, from phase-contrast or bright-field "
        "images: no stain needed.",
        icon="🟩",
        outputs="% covered",
        segmenter="texture",
        group="Growth & migration",
        steps=("load", "calibrate", "detect", "review", "results", "report", "batch"),
        settings={"detect": "covered", "exclude_borders": False},
        measure={"background": "none"},
        noun="Patch",
        image_kind=PHOTO,
        load_hint="Load a phase-contrast or bright-field image of the dish. No stain needed.",
        analysis=Confluence(),
    )
)

WOUND = register(
    Assay(
        id="wound",
        name="Scratch wound",
        description="Follow a scratch closing as cells migrate: open area, width and % closure "
        "since the start.",
        icon="🩹",
        outputs="Wound width · % closure",
        segmenter="texture",
        group="Growth & migration",
        steps=("load", "calibrate", "detect", "review", "results", "report", "batch"),
        settings={"detect": "gap", "exclude_borders": False},
        measure={"background": "none"},
        noun="Gap",
        image_kind=PHOTO,
        load_hint="Load the scratch as it was made (0 h), detect it and add it as the start in "
        "Results; then load each later image and add it too.",
        analysis=Wound(),
        citations=("jonkman2014",),
    )
)
