"""Assays: what the user is analysing, and the workflow, settings and results that go with it.

An assay is a preset over the shared pipeline (load › detect › review ›
measure › gate › report › batch): which steps the user walks through, which
segmenter and settings to start from, which measurements to make. An assay
with an ``analysis`` also has its own settings (a dilution, a chamber depth),
adds derived columns to the measurements (DNA content, colony count, …),
can suggest gates (G1/S/G2-M, live/dead) and reports headline results
(CFU/mL, % viable, % confluence) in the Results step, in batch runs and in
the methods text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from karcytics_sdk.plugin.citations import MethodStep

from ..experiment import Comparison, Role, Sample
from ..gating import Gate
from ..model import color_for_name
from ..segment.base import Param

ALL_STEPS = (
    "load",
    "calibrate",
    "detect",
    "review",
    "measure",
    "gate",
    "results",
    "report",
    "batch",
)
STEP_LABELS = {
    "load": "Load",
    "calibrate": "Calibrate",
    "detect": "Detect",
    "review": "Review",
    "measure": "Measure",
    "gate": "Gate",
    "results": "Results",
    "report": "Report",
    "batch": "Batch",
}
STANDARD_STEPS = tuple(s for s in ALL_STEPS if s != "results")
# What the Load step asks for: one file per stain (fluorescence), or one photo whose colour
# channels are only a means to an end (a plate, a hemocytometer, a well).
FLUORESCENCE, PHOTO = "fluorescence", "photo"


@dataclass(frozen=True)
class ResultValue:
    """One headline result: "Colonies: 142", "CFU/mL: 1.4 × 10⁸"."""

    key: str  # stable id, used as a column name in batch tables ("Result_<key>")
    label: str
    value: float | str
    unit: str = ""
    help: str = ""
    warning: str = ""  # shown next to the value (e.g. "outside the countable range")


@dataclass(frozen=True)
class Overlay:
    """A shape drawn over the image, such as the dish found or the counting area.

    ``coords``: circle (x, y, r); rect (x0, y0, x1, y1); line (x0, y0, x1, y1).
    """

    kind: str
    coords: tuple[float, ...]
    color: str = "#ffd166"
    label: str = ""
    dashed: bool = True


@dataclass
class ActionResult:
    """What an action changed: settings to store, gates to add, a message for the user."""

    settings: dict[str, Any] = field(default_factory=dict)
    gates: list[Gate] = field(default_factory=list)
    message: str = ""
    scale: float | None = None  # a new calibration (µm/px), e.g. measured from a grid
    scale_source: str = ""


@dataclass
class AssayContext:
    """What an analysis may need beyond the measurements."""

    scale: float = 0.0  # µm/px, 0 when uncalibrated
    shape: tuple[int, int] | None = None  # image (height, width) in px
    extras: dict[str, Any] = field(default_factory=dict)  # from the segmenter (e.g. the dish)
    image_set: Any = None  # the images, for actions that look at them (not set in batch runs)


class AssayAnalysis:
    """An assay's own calculations. Every hook is optional (the defaults do nothing).

    ``settings`` are the values of ``parameters()`` as the user set them.
    """

    citations: tuple[str, ...] = ()
    suggests_gates = False  # True when ``suggested_gates`` returns something
    # Default stain colour for each channel setting ({"dna_channel": "blue"}).
    channel_stains: dict[str, str] = {}  # noqa: RUF012  (read-only class default)
    # What an image can be in an experiment (control, sample, plate, square, …); empty: the
    # assay works on one image at a time. See ``analysis.experiment`` and ``compare``.
    roles: tuple[Role, ...] = ()
    # (action id, button text, tooltip): buttons in the Results step; see ``run_action``.
    # An id starting with "box:" first asks the user to drag a box on the image.
    actions: tuple[tuple[str, str, str], ...] = ()

    def parameters(self) -> list[Param]:
        return []

    def derive(self, df: pd.DataFrame, settings: dict, context: AssayContext) -> pd.DataFrame:
        """`df` with the assay's own columns added (or recomputed). Must not change `df`."""
        return df

    def suggested_gates(self, df: pd.DataFrame, settings: dict) -> list[Gate]:
        return []

    def results(
        self, df: pd.DataFrame, gates: list[Gate], settings: dict, context: AssayContext
    ) -> list[ResultValue]:
        return []

    def methods(self, settings: dict, context: AssayContext) -> list[MethodStep]:
        return []

    def overlays(self, settings: dict, context: AssayContext) -> list[Overlay]:
        return []

    def compare(
        self, samples: list[Sample], settings: dict, context: AssayContext
    ) -> Comparison | None:
        """The experiment's result from its images (None: nothing to compare yet)."""
        return None

    def run_action(
        self,
        action: str,
        df: pd.DataFrame,
        settings: dict,
        context: AssayContext,
        box: tuple[float, float, float, float] | None = None,
    ) -> ActionResult:
        """Handle one of ``actions``; `box` is the dragged box for "box:" actions."""
        return ActionResult()


@dataclass(frozen=True)
class Assay:
    id: str
    name: str
    description: str
    icon: str
    segmenter: str
    group: str = "Cells"  # heading in the picker
    steps: tuple[str, ...] = STANDARD_STEPS
    settings: dict[str, Any] = field(default_factory=dict)  # Detect display values
    measure: dict[str, Any] = field(default_factory=dict)  # MeasureOptions fields
    target_stain: str | None = None  # colour of the stain to segment by default ("blue")
    analysis: AssayAnalysis | None = None
    citations: tuple[str, ...] = ()  # shown on the picker card and in the Results step
    noun: str = "Cell"  # what one object is called in the table and the inspector ("Colony 3")
    image_kind: str = FLUORESCENCE  # FLUORESCENCE or PHOTO: how the Load step looks
    load_hint: str = ""  # what to load, shown in the Load step (empty: the general text)
    outputs: str = ""  # the headline results, for the picker card ("CFU/mL · colony size")

    def default_target(self, channel_names: list[str]) -> str | None:
        """The channel to segment by default, from the stain names (None: keep the user's)."""
        if self.target_stain is None:
            return None
        return next((n for n in channel_names if color_for_name(n) == self.target_stain), None)

    def default_settings(self) -> dict[str, Any]:
        return {p.key: p.default for p in self.analysis.parameters()} if self.analysis else {}


REGISTRY: dict[str, Assay] = {}
GROUPS = ("Cells", "Counting", "Microbiology", "Growth & migration")  # picker order


def by_group(assays: list[Assay]) -> list[tuple[str, list[Assay]]]:
    """`assays` grouped for the picker: groups in ``GROUPS`` order (others after), then by name."""
    names = list(dict.fromkeys(a.group for a in assays))
    names.sort(key=lambda g: GROUPS.index(g) if g in GROUPS else len(GROUPS))
    return [(g, [a for a in assays if a.group == g]) for g in names]


def register(assay: Assay) -> Assay:
    if assay.id in REGISTRY:
        raise ValueError(f"Assay id {assay.id!r} is already registered")
    REGISTRY[assay.id] = assay
    return assay


def with_columns(
    df: pd.DataFrame, columns: dict[str, Any], units: dict[str, str], helps: dict[str, str]
) -> pd.DataFrame:
    """A copy of `df` with `columns` set, their units and help recorded in ``attrs``."""
    out = df.copy()
    for name, values in columns.items():
        out[name] = values
    out.attrs = {
        **df.attrs,
        "units": {**df.attrs.get("units", {}), **units},
        "help": {**df.attrs.get("help", {}), **helps},
    }
    return out
