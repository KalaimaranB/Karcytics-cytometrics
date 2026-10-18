"""Nuclei: nuclear size, shape, DNA content, chromatin texture and the cell cycle.

Cell cycle staging follows Roukos et al. (2015): a nucleus's DNA content is
its integrated nuclear-stain intensity (DAPI, Hoechst). The most frequent
content, the G1 peak, is taken as 2N, so G2/M nuclei sit near 4N. Nuclei are
then staged with gates on the normalised content, which the user can move
in the Gate step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from .. import gating
from ..gating import Gate
from ..segment.base import Param
from .base import Assay, AssayAnalysis, AssayContext, ResultValue, register, with_columns
from .common import intensity_column, main_peak, percent

DNA = "CellCycle_DNAContent"
# (name, lower, upper) in N, the G1 peak being 2N; colours: Okabe–Ito.
PHASES = (
    ("Sub-G1", 0.0, 1.5, "#999999"),
    ("G1", 1.5, 2.5, "#56B4E9"),
    ("S", 2.5, 3.5, "#E69F00"),
    ("G2/M", 3.5, 5.0, "#009E73"),
    (">4N", 5.0, float("inf"), "#CC79A7"),
)


class CellCycle(AssayAnalysis):
    channel_stains = {"dna_channel": "blue"}
    citations = ("roukos2015",)
    suggests_gates = True

    def parameters(self) -> list[Param]:
        return [
            Param(
                "dna_channel",
                "DNA stain",
                "channel",
                help="The nuclear stain whose integrated intensity measures DNA content.",
            ),
            Param(
                "g1_peak",
                "G1 peak (integrated intensity)",
                "float",
                0.0,
                0,
                1e15,
                decimals=1,
                special_value_text="Auto",
                help="The integrated intensity of G1 nuclei (2N). Auto: the most frequent value.",
            ),
        ]

    def _source(self, df: pd.DataFrame, settings: dict) -> str | None:
        return intensity_column(df, settings.get("dna_channel"), "IntegratedIntensity")

    def g1_peak(self, df: pd.DataFrame, settings: dict) -> float:
        manual = float(settings.get("g1_peak") or 0)
        source = self._source(df, settings)
        if manual > 0 or source is None:
            return manual if manual > 0 else float("nan")
        return main_peak(pd.to_numeric(df[source], errors="coerce").to_numpy())

    def derive(self, df: pd.DataFrame, settings: dict, context: AssayContext) -> pd.DataFrame:
        source = self._source(df, settings)
        if source is None:
            return df
        peak = self.g1_peak(df, settings)
        values = pd.to_numeric(df[source], errors="coerce") / peak * 2 if peak > 0 else np.nan
        return with_columns(
            df,
            {DNA: values},
            {DNA: ""},
            {DNA: f"DNA content in N: {source} relative to the G1 peak (2N)."},
        )

    def suggested_gates(self, df: pd.DataFrame, settings: dict) -> list[Gate]:
        gates = []
        for name, lo, hi, color in PHASES:
            if np.isinf(hi):
                gates.append(Gate(name, "threshold", x=DNA, params={"value": lo}, color=color))
            else:
                gates.append(
                    Gate(name, "range", x=DNA, params={"lo": lo, "hi": hi - 1e-9}, color=color)
                )
        return gates

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        if DNA not in df or not len(df):
            return []
        dna = pd.to_numeric(df[DNA], errors="coerce").to_numpy()
        named = {g.name for g in gates}
        masks = gating.evaluate(gates, df) if gates else {}
        peak = self.g1_peak(df, settings)
        out = [
            ResultValue(
                "G1Peak",
                "G1 peak (2N)",
                peak,
                "a.u.",
                "Integrated intensity of the most frequent DNA content.",
                "" if len(df) >= 50 else "few nuclei: the G1 peak may be unreliable",  # noqa: PLR2004
            )
        ]
        for name, lo, hi, _color in PHASES:
            mask = masks[name] if name in named else (dna >= lo) & (dna < hi)
            out.append(
                ResultValue(
                    f"Pct_{name.replace('/', '').replace('>', 'Over').replace('-', '')}",
                    f"% {name}",
                    percent(mask),
                    "%",
                    f"Nuclei staged {name}"
                    + (" by the gate of that name." if name in named else f" ({lo:g}–{hi:g} N)."),
                )
            )
        return out

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        manual = float(settings.get("g1_peak") or 0)
        peak = (
            f"set to {manual:g}"
            if manual
            else "the most frequent value, found by Gaussian kernel density estimation"
        )
        return [
            MethodStep(
                "DNA content was measured as each nucleus's integrated intensity of the "
                f"'{settings.get('dna_channel') or 'nuclear stain'}' channel and normalised so "
                f"that the G1 peak ({peak}) equals 2N; nuclei were staged as sub-G1 (< 1.5N), "
                "G1 (1.5–2.5N), S (2.5–3.5N), G2/M (3.5–5N) and > 4N (≥ 5N)",
                ("roukos2015",),
            )
        ]


NUCLEI = register(
    Assay(
        id="nuclei",
        name="Nuclei & cell cycle",
        description="Count and measure nuclei from a nuclear stain (DAPI, Hoechst): size, shape, "
        "chromatin texture and DNA content, staged into G1, S and G2/M.",
        icon="🧬",
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
        measure={"background": "local", "texture": True},
        target_stain="blue",
        noun="Nucleus",
        load_hint="Load the nuclear stain (DAPI, Hoechst), plus any other channels to measure in the "
        "nuclei. For the cell cycle, use images that aren't saturated.",
        analysis=CellCycle(),
        citations=("cpsam2025",),
    )
)
