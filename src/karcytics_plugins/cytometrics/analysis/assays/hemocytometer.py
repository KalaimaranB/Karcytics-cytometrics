"""Hemocytometer counts with trypan blue: cells/mL and % viability.

Trypan blue enters only dead cells (Strober 2015), so dead cells look dark
and live ones bright. A cell's *darkness* is 1 − its mean intensity / the
local background around it (0: as bright as the background, 1: black);
cells darker than the cut-off are dead. Concentration:

    cells/mL = cells counted ÷ volume counted (mL) × dilution factor,

where the volume is the counting area (mm²) × the chamber depth (0.1 mm for
a Neubauer chamber), 1 mm³ = 1 µL. The counting area is a box the user
drags; a cell counts when its centre is inside it, which splits cells on the
box's edges fairly, like the usual "count two edges, not the other two" rule.

In an experiment, several squares (or fields) are counted and added: the
concentration is all the cells counted over all the volume counted (the
same as averaging the squares, weighted by their size). Counting is
Poisson: N cells give a precision of about ±1/√N (100 cells: ±10%), and
squares that differ by more than chance allows (index of dispersion,
χ² test) suggest the cells weren't evenly mixed.

The scale can be set from the grid itself: the spacing of the grid lines is
found from the image's line profiles (the distance at which they first
repeat), and the user says what that spacing is (e.g. 0.25 mm).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep
from scipy.stats import chi2

from .. import gating
from ..experiment import Comparison, Plot, Role, Sample
from ..gating import Gate
from ..segment.base import Param
from ..units import is_calibrated
from .base import (
    PHOTO,
    ActionResult,
    Assay,
    AssayAnalysis,
    AssayContext,
    Overlay,
    ResultValue,
    register,
    with_columns,
)
from .common import intensity_column, percent

_UNEVEN_P = 0.05  # squares this unlikely under Poisson counting are flagged

DARKNESS = "Viability_Darkness"
_BOX_VALUES = 4
_MIN_REPEAT = 0.3  # a grid's line profile matches itself this well one spacing along
_NEAR_BEST = 0.8  # a peak this close to the best match counts as the spacing
DEAD = "Dead"
SQUARES = (
    ("1 mm (large square)", 1.0),
    ("0.25 mm", 0.25),
    ("0.2 mm (centre square)", 0.2),
    ("0.05 mm (smallest)", 0.05),
)


SQUARE = "square"


class Viability(AssayAnalysis):
    citations = ("strober2015",)
    suggests_gates = True
    roles = (
        Role(
            SQUARE,
            "counted square",
            help="Draw the counting area over one square (or field), then add it. Add each "
            "square you count; the concentration uses them all.",
        ),
    )
    channel_stains = {"stain_channel": "red"}  # trypan blue absorbs red light most
    actions = (
        (
            "box:counting_area",
            "⬚ Draw the counting area",
            "Drag a box over the squares you count. Cells whose centre is inside are counted.",
        ),
        (
            "calibrate_grid",
            "📐 Calibrate from the grid",
            "Measure the grid spacing in the image and set the scale from the square size below.",
        ),
    )

    def parameters(self) -> list[Param]:
        return [
            Param(
                "stain_channel",
                "Trypan blue seen in",
                "channel",
                help="The channel where dead cells look darkest (red, for a colour photo).",
            ),
            Param(
                "darkness_cutoff",
                "Dead if darker than",
                "float",
                0.35,
                0.0,
                1.0,
                step=0.05,
                decimals=2,
                help="Darkness: 1 − mean intensity / local background. Above this a cell is dead.",
            ),
            Param(
                "dilution_factor",
                "Dilution factor",
                "float",
                2.0,
                1.0,
                1e6,
                decimals=2,
                help="Sample dilution including trypan blue: 2 for equal volumes of cells and dye.",
            ),
            Param(
                "depth_um",
                "Chamber depth (µm)",
                "float",
                100.0,
                1.0,
                1000.0,
                decimals=1,
                help="100 µm for Neubauer and most hemocytometers.",
            ),
            Param(
                "square_mm",
                "Grid line spacing",
                "choice",
                0.25,
                choices=SQUARES,
                help="The distance between neighbouring grid lines in your image, for 📐.",
            ),
        ]

    def derive(self, df: pd.DataFrame, settings: dict, context: AssayContext) -> pd.DataFrame:
        mean = intensity_column(df, settings.get("stain_channel"), "MeanIntensity")
        if mean is None:
            return df
        mean = mean.replace("Corrected", "")  # darkness needs the raw mean and the background
        background = mean.replace("MeanIntensity", "Background")
        if background not in df:
            return df
        raw = pd.to_numeric(df[mean], errors="coerce")
        bg = pd.to_numeric(df[background], errors="coerce").replace(0, np.nan)
        return with_columns(
            df,
            {DARKNESS: (1 - raw / bg).clip(lower=0)},
            {DARKNESS: ""},
            {DARKNESS: "1 − mean intensity / local background: 0 bright (live), 1 black (dead)."},
        )

    def suggested_gates(self, df: pd.DataFrame, settings: dict) -> list[Gate]:
        cutoff = float(settings.get("darkness_cutoff", 0.35))
        return [Gate(DEAD, "threshold", x=DARKNESS, params={"value": cutoff}, color="#0072B2")]

    def _area(self, settings: dict) -> tuple[float, float, float, float] | None:
        box = settings.get("counting_area")
        return tuple(box) if box and len(box) == _BOX_VALUES else None  # type: ignore[return-value]

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        if DARKNESS not in df or not len(df):
            return []
        box = self._area(settings)
        x = pd.to_numeric(df.get("Location_Center_X"), errors="coerce").to_numpy()
        y = pd.to_numeric(df.get("Location_Center_Y"), errors="coerce").to_numpy()
        if box is not None:
            x0, y0, x1, y1 = box
            counted = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        else:
            counted = np.ones(len(df), bool)
        if any(g.name == DEAD for g in gates):
            dead = gating.evaluate(gates, df)[DEAD]
        else:
            darkness = pd.to_numeric(df[DARKNESS], errors="coerce").to_numpy()
            dead = darkness > float(settings.get("darkness_cutoff", 0.35))
        n, n_dead = int(counted.sum()), int((dead & counted).sum())
        out = [
            ResultValue("Counted", "Cells counted", n),
            ResultValue("Live", "Live", n - n_dead),
            ResultValue("Dead", "Dead", n_dead),
            ResultValue("Viability", "Viability", percent(~dead, counted), "%"),
        ]
        warning = ""
        volume_ml = float("nan")
        if box is None:
            warning = "draw the counting area"
        elif not is_calibrated(context.scale):
            warning = "calibrate (📐) to get a concentration"
        else:
            x0, y0, x1, y1 = box
            area_mm2 = (x1 - x0) * (y1 - y0) * (context.scale / 1000) ** 2
            volume_ml = area_mm2 * float(settings.get("depth_um", 100)) / 1000 / 1000
        dilution = float(settings.get("dilution_factor", 2))
        if volume_ml > 0:
            out.append(
                ResultValue("Volume", "Volume counted", volume_ml * 1000, "µL", "Area × depth.")
            )
        for key, label, count in (
            ("LivePerML", "Live cells / mL", n - n_dead),
            ("TotalPerML", "Total cells / mL", n),
        ):
            out.append(
                ResultValue(
                    key,
                    label,
                    count / volume_ml * dilution if volume_ml > 0 else float("nan"),
                    "cells/mL",
                    "Cells counted ÷ volume counted × dilution factor.",
                    warning,
                )
            )
        return out

    def compare(
        self, samples: list[Sample], settings: dict, context: AssayContext
    ) -> Comparison | None:
        counts = np.array([float(s.results.get("Counted") or 0) for s in samples])
        dead = np.array([float(s.results.get("Dead") or 0) for s in samples])
        volumes = np.array([float(s.results.get("Volume") or np.nan) for s in samples])  # µL
        dilution = float(settings.get("dilution_factor", 2))
        total, total_dead = float(counts.sum()), float(dead.sum())
        rows = [
            {
                "Image": s.name,
                "Counted": int(c),
                "Dead": int(d),
                "Viability %": 100 * (c - d) / c if c else float("nan"),
                "Volume (µL)": v,
                "Cells / µL": c / v if v > 0 else float("nan"),
            }
            for s, c, d, v in zip(samples, counts, dead, volumes, strict=True)
        ]
        values = [
            ResultValue("Squares", "Squares counted", len(samples)),
            ResultValue("Counted", "Cells counted", int(total), "", "All squares together."),
            ResultValue(
                "Viability",
                "Viability",
                100 * (total - total_dead) / total if total else float("nan"),
                "%",
            ),
        ]
        measured = np.isfinite(volumes) & (volumes > 0)
        warning = ""
        if not measured.all():
            warning = "draw the counting area and calibrate for every square"
        elif total and len(counts) > 1:
            dispersion = float(((counts - counts.mean()) ** 2).sum() / counts.mean())
            if chi2.sf(dispersion, len(counts) - 1) < _UNEVEN_P:
                warning = "squares differ more than chance: mix the cells and recount"
        per_ul = total / volumes[measured].sum() if measured.any() else float("nan")
        live_per_ul = (
            (total - total_dead) / volumes[measured].sum() if measured.any() else float("nan")
        )
        precision = 100 / np.sqrt(total) if total else float("nan")
        for key, label, value in (
            ("LivePerML", "Live cells / mL", live_per_ul),
            ("TotalPerML", "Total cells / mL", per_ul),
        ):
            values.append(
                ResultValue(
                    key,
                    label,
                    value * 1000 * dilution if measured.all() else float("nan"),
                    "cells/mL",
                    f"All cells counted ÷ all volume counted × {dilution:g} (dilution). Counting "
                    f"precision about ±{precision:.0f}% (1/√N).",
                    warning,
                )
            )
        note = (
            "Count more squares for a more precise answer: ±10% needs about 100 cells."
            if total < 100  # noqa: PLR2004
            else f"{int(total)} cells counted: precision about ±{precision:.0f}%."
        )
        points = pd.DataFrame({"group": [s.name for s in samples], "value": counts})
        return Comparison(
            values=values,
            table=pd.DataFrame(rows),
            plot=Plot("bars", label="Cells counted", points=points),
            note=note,
        )

    def overlays(self, settings: dict, context: AssayContext) -> list[Overlay]:
        box = self._area(settings)
        return [Overlay("rect", box, "#ffd166", "Counting area")] if box else []

    def run_action(self, action, df, settings, context, box=None) -> ActionResult:
        if action == "box:counting_area":
            if box is None:
                return ActionResult()
            x0, y0, x1, y1 = box
            area = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
            return ActionResult(settings={"counting_area": area}, message="Counting area set.")
        if action == "calibrate_grid":
            image_set = context.image_set
            channel = image_set.get(settings.get("stain_channel")) if image_set else None
            if channel is None:
                raise ValueError("Load the hemocytometer image first.")
            period = grid_period(channel.data)
            if period is None:
                raise ValueError("No regular grid found. Calibrate with the scale bar instead.")
            square_um = float(settings.get("square_mm", 0.25)) * 1000
            return ActionResult(
                scale=square_um / period,
                scale_source="hemocytometer grid",
                message=f"Grid lines every {period:.1f} px = {square_um:g} µm: "
                f"{square_um / period:.4g} µm/px.",
            )
        return ActionResult()

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        return [
            MethodStep(
                "Cells were stained with trypan blue and counted in a hemocytometer; a cell was "
                "scored dead when its darkness (1 − mean intensity / local background in the "
                f"'{settings.get('stain_channel')}' channel) exceeded "
                f"{float(settings.get('darkness_cutoff', 0.35)):g}, cells were counted when "
                "their centre lay inside the counting area, and concentrations were calculated "
                f"for a {float(settings.get('depth_um', 100)):g} µm chamber depth and a "
                f"{float(settings.get('dilution_factor', 2)):g}× dilution",
                ("strober2015",),
            )
        ]


def grid_period(plane: np.ndarray, min_px: float = 6.0) -> float | None:
    """The spacing of a regular grid of lines, in pixels (None if there's no clear one).

    Grid lines are edges that line up: summing the image's gradients along
    each column (and each row) gives a profile with a peak at every line.
    The profile repeats every line spacing, so its autocorrelation peaks
    there (and at every multiple of it); the first strong peak beyond `min_px`
    (refined between pixels) is the
    spacing. Averaged over the two directions.
    """
    image = plane.astype(np.float64)
    periods = []
    for axis in (0, 1):
        profile = np.abs(np.diff(image, axis=1 - axis)).sum(axis=axis)
        n = len(profile)
        window = max(3, n // 4) | 1
        kernel = np.ones(window)
        # A running mean, averaged over the part of the window inside the image at the ends.
        trend = np.convolve(profile, kernel, mode="same") / np.convolve(
            np.ones(n), kernel, mode="same"
        )
        profile = profile - trend
        spectrum = np.fft.rfft(profile, 2 * n)
        auto = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
        if auto[0] <= 0:
            continue
        auto = auto / auto[0]
        lo, hi = int(np.ceil(min_px)), n // 3
        if hi <= lo + 2:
            continue
        window_auto = auto[lo:hi]
        best = float(window_auto.max())
        if best < _MIN_REPEAT:  # the profile doesn't repeat: not a grid
            continue
        # Every multiple of the spacing matches about as well: take the first peak that
        # comes close to the best one, not the best itself.
        peaks = [
            i
            for i in range(1, len(window_auto) - 1)
            if window_auto[i] >= window_auto[i - 1]
            and window_auto[i] >= window_auto[i + 1]
            and window_auto[i] >= _NEAR_BEST * best
        ]
        lag = lo + (peaks[0] if peaks else int(np.argmax(window_auto)))
        a, b, c = auto[lag - 1], auto[lag], auto[lag + 1]
        shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
        periods.append(lag + shift)
    return float(np.mean(periods)) if periods else None


HEMOCYTOMETER = register(
    Assay(
        id="hemocytometer",
        name="Hemocytometer & viability",
        description="Count cells in a hemocytometer with trypan blue: live and dead cells, "
        "% viability and cells/mL.",
        icon="🧮",
        outputs="Cells/mL · % viable",
        segmenter="cellpose",
        group="Counting",
        steps=("load", "calibrate", "detect", "review", "gate", "results", "report", "batch"),
        settings={"exclude_borders": False},
        measure={"background": "local"},
        image_kind=PHOTO,
        load_hint="Load a bright-field image of the chamber with the grid lines in view, after mixing the "
        "cells with trypan blue.",
        analysis=Viability(),
        citations=("cpsam2025", "strober2015"),
    )
)
