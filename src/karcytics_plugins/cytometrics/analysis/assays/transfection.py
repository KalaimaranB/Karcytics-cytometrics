"""Transfection / % positive: the share of cells expressing a reporter (GFP, mCherry, …).

The cut-off comes from a negative control: analyse an untransfected (or
unstained) image and add it to the experiment as the negative control; the
cut-off is a high percentile of those cells' reporter intensity (99th by
default, so 1% of control cells would be called positive). Each sample
added afterwards is scored against it, and compared with the control: %
positive, the brightness of the positive cells, and the shift of the
median (fold change). Several control images are pooled. The cut-off also
applies to the image open now and to batch runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from .. import gating
from ..experiment import Comparison, Plot, Role, Sample, Series, colors_for
from ..gating import Gate
from ..segment.base import Param
from .base import Assay, AssayAnalysis, AssayContext, ResultValue, register
from .common import intensity_column, percent

POSITIVE = "Positive"
STATS = (("Mean intensity", "MeanIntensity"), ("Integrated intensity", "IntegratedIntensity"))


CONTROL, SAMPLE = "control", "sample"


class PercentPositive(AssayAnalysis):
    channel_stains = {"reporter_channel": "green"}
    suggests_gates = True
    roles = (
        Role(
            CONTROL,
            "negative control",
            fields=("reporter_channel",),
            help="Untransfected or unstained cells: they set the cut-off.",
        ),
        Role(
            SAMPLE,
            "sample",
            fields=("reporter_channel",),
            named=True,
            help="Transfected cells, scored against the control's cut-off.",
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

    def suggested_gates(self, df: pd.DataFrame, settings: dict) -> list[Gate]:
        column = self._column(df, settings)
        cutoff = float(settings.get("cutoff") or 0)
        if column is None or not cutoff:
            return []
        return [Gate(POSITIVE, "threshold", x=column, params={"value": cutoff}, color="#009E73")]

    def _values(self, sample: Sample, settings: dict) -> np.ndarray:
        """A sample's reporter values (its own channel: file names differ between images)."""
        column = intensity_column(
            sample.table,
            sample.fields.get("reporter_channel"),
            settings.get("statistic") or "MeanIntensity",
        )
        if column is None:
            return np.zeros(0)
        values = pd.to_numeric(sample.table[column], errors="coerce").to_numpy(float)
        return values[np.isfinite(values)]

    def compare(
        self, samples: list[Sample], settings: dict, context: AssayContext
    ) -> Comparison | None:
        controls = [s for s in samples if s.role == CONTROL]
        pct = float(settings.get("percentile") or 99)
        derived: dict = {}
        if controls:
            pooled = np.concatenate([self._values(s, settings) for s in controls])
            if pooled.size:
                cutoff = float(np.percentile(pooled, pct))
                derived = {"cutoff": cutoff, "control_cells": int(pooled.size)}
        cutoff = float(derived.get("cutoff") or settings.get("cutoff") or 0)
        control_median = (
            float(np.median(np.concatenate([self._values(s, settings) for s in controls])))
            if controls and derived
            else float("nan")
        )
        colors = colors_for(samples, (CONTROL,))
        rows, values, series = [], [], []
        for s in samples:
            v = self._values(s, settings)
            positive = v > cutoff if cutoff else np.zeros(v.size, bool)
            median = float(np.median(v)) if v.size else float("nan")
            row = {
                "Image": s.name,
                "Role": "control" if s.role == CONTROL else "sample",
                "Cells": int(v.size),
                "% positive": percent(positive) if cutoff else float("nan"),
                "Mean of positive": float(v[positive].mean()) if positive.any() else float("nan"),
                "Median": median,
                "Fold change (median)": median / control_median if control_median else float("nan"),
            }
            rows.append(row)
            series.append(Series(s.name, v, colors[s.id]))
            if s.role == SAMPLE:
                values.append(
                    ResultValue(
                        f"PctPositive_{s.name}",
                        f"{s.name}: positive",
                        row["% positive"],
                        "%",
                        f"{row['Cells']} cells; median {row['Fold change (median)']:.2g}× the "
                        "control's"
                        if control_median
                        else f"{row['Cells']} cells",
                        "" if cutoff else "add a negative control",
                    )
                )
        head = []
        if derived:
            head.append(
                ResultValue(
                    "Cutoff",
                    "Cut-off",
                    cutoff,
                    "a.u.",
                    f"The {pct:g}th percentile of {derived['control_cells']} control cells.",
                )
            )
            control_pct = [float(r["% positive"]) for r in rows if r["Role"] == "control"]
            head.append(
                ResultValue(
                    "ControlPositive",
                    "Control: positive",
                    float(np.mean(control_pct)),
                    "%",
                    f"By construction about {100 - pct:g}%; more means the controls differ.",
                )
            )
        if not any(s.role == SAMPLE for s in samples):
            note = "Now load a transfected image, detect its cells and add it as a sample."
        elif not controls:
            note = "Add a negative control to set the cut-off from untransfected cells."
        else:
            note = (
                f"Cut-off: the {pct:g}th percentile of the pooled control cells' "
                f"{_stat_name(settings).lower()}."
            )
        return Comparison(
            values=head + values,
            table=pd.DataFrame(rows),
            plot=Plot(
                "histogram",
                label=f"Reporter {_stat_name(settings).lower()}",
                log=True,
                series=series,
                lines=[(cutoff, "cut-off")] if cutoff else [],
            ),
            settings=derived,
            note=note,
            methods=self._experiment_methods(rows, settings, derived),
        )

    def _experiment_methods(self, rows: list[dict], settings: dict, derived: dict) -> list:
        if not derived:
            return []
        samples = [r for r in rows if r["Role"] == "sample"]
        if not samples:
            return []
        listed = "; ".join(f"{r['Image']}, {r['% positive']:.1f}% of {r['Cells']}" for r in samples)
        return [
            MethodStep(
                f"Positive cells were counted against a cut-off at the "
                f"{float(settings.get('percentile') or 99):g}th percentile of "
                f"{derived['control_cells']} pooled negative-control cells ({listed})"
            )
        ]

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
                    warning="add a negative control to the experiment below (or type a cut-off)",
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
        outputs="% positive vs control",
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
        load_hint="Load a nuclear stain to find the cells and the reporter (GFP, mCherry, ...), as separate "
        "files or one multi-channel file. Start with the untransfected control.",
        analysis=PercentPositive(),
        citations=("cpsam2025",),
    )
)


def _stat_name(settings: dict) -> str:
    return {v: k for k, v in STATS}.get(str(settings.get("statistic")), "Mean intensity")
