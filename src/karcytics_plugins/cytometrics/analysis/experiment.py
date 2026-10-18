"""An experiment: several analysed images, each with a role, compared the assay's way. No Qt.

At the bench an assay is rarely one image: a transfection is read against
an untransfected control, a hemocytometer count averages several squares, a
plate count picks the plates in the countable range from a dilution series,
a cell-cycle profile is compared with the untreated cells. Each assay
declares the **roles** an image can have (``AssayAnalysis.roles``) and how
to **compare** them (``AssayAnalysis.compare``).

The user analyses one image at a time and adds it with a role; its
measurement table is kept (not the image), so the comparison can be redone
whenever a setting changes. For many images run the same way, the Batch
step is the tool; the experiment is for the handful of images one result
is made from.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .segment.base import Param


@dataclass(frozen=True)
class Role:
    """What an image is in the experiment ("Negative control", "Plate", …)."""

    id: str
    label: str  # shown on the add button: "Add as {label}"
    many: bool = True  # several images may have this role
    fields: tuple[str, ...] = ()  # assay settings recorded with each image (a plate's dilution)
    help: str = ""
    named: bool = False  # ask for a name (a condition: "siRNA 1"); else numbered
    groups: bool = False  # images with the same name are replicates of one condition


@dataclass
class Sample:
    """One analysed image in the experiment: its role, name and measurements."""

    name: str
    role: str
    table: pd.DataFrame
    image: str = ""  # the file name(s), to show where it came from
    fields: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)  # the image's own headline values
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def to_dict(self) -> dict:
        table = self.table.select_dtypes("number")
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "image": self.image,
            "fields": dict(self.fields),
            "results": {k: _json_number(v) for k, v in self.results.items()},
            "columns": {c: [_json_number(v) for v in table[c]] for c in table.columns},
            "units": dict(self.table.attrs.get("units", {})),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Sample:
        table = pd.DataFrame({c: pd.to_numeric(v) for c, v in data.get("columns", {}).items()})
        table = table.astype(float)
        table.attrs = {"units": dict(data.get("units", {}))}
        return cls(
            name=data.get("name", ""),
            role=data.get("role", ""),
            table=table,
            image=data.get("image", ""),
            fields=dict(data.get("fields", {})),
            results=dict(data.get("results", {})),
            id=data.get("id") or uuid.uuid4().hex[:8],
        )


@dataclass
class Series:
    """One distribution in an overlaid histogram."""

    name: str
    values: np.ndarray
    color: str


@dataclass
class Plot:
    """What the Compare tab draws.

    ``histogram``: ``series`` overlaid (each normalised to its own count),
    with labelled vertical ``lines`` (a cut-off, phase boundaries).
    ``bars``: one bar per group of ``points`` (column ``group``, ``value``),
    the mean with SD, a dot per image.
    """

    kind: str
    label: str = ""  # the value axis
    log: bool = False
    series: list[Series] = field(default_factory=list)
    lines: list[tuple[float, str]] = field(default_factory=list)
    points: pd.DataFrame = field(default_factory=pd.DataFrame)
    band: tuple[float, float] | None = None  # a shaded range on the value axis (countable range)


@dataclass
class Comparison:
    """The experiment's outcome: headline values, a row per image, a plot, derived settings."""

    values: list = field(default_factory=list)  # ``assays.ResultValue``
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    plot: Plot | None = None
    settings: dict[str, Any] = field(default_factory=dict)  # e.g. the cut-off from the controls
    note: str = ""  # how the result was made, shown under it
    methods: list = field(default_factory=list)  # ``MethodStep``s for the report


def default_name(role: Role, samples: list[Sample]) -> str:
    """A name for the next image of `role`: "Negative control", then "Negative control 2".

    For a role whose images are grouped by name, the last name used, so replicates
    of one condition are quick to add.
    """
    if role.groups:
        previous = [s.name for s in samples if s.role == role.id]
        if previous:
            return previous[-1]
    base = role.label[:1].upper() + role.label[1:]
    taken = {s.name for s in samples}
    if base not in taken:
        return base
    n = 2
    while f"{base} {n}" in taken:
        n += 1
    return f"{base} {n}"


def by_condition(
    samples: list[Sample], value: str, label: str, unit: str = "", digits: str = ".3g"
) -> tuple[list, pd.DataFrame]:
    """Per condition (sample name): the mean of each image's result `value`, with SD and n.

    Returns the headline ``ResultValue``s and the points for a bar plot.
    """
    from .assays.base import ResultValue  # the assays package imports this module

    points = pd.DataFrame(
        {
            "group": [s.name for s in samples],
            "value": [_float(s.results.get(value)) for s in samples],
        }
    )
    values = []
    for name, group in points.groupby("group", sort=False)["value"]:
        finite = group.dropna()
        sd = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
        spread = f" ± {sd:{digits}} (SD, n = {len(finite)})" if len(finite) > 1 else " (one image)"
        values.append(
            ResultValue(
                f"{value}_{name}",
                f"{name}: {label}",
                float(finite.mean()) if len(finite) else float("nan"),
                unit,
                f"Mean of the images named {name}{spread}.",
            )
        )
    return values, points


def _float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def field_params(params: list[Param], role: Role) -> list[Param]:
    """The assay settings `role` records with each image."""
    return [p for p in params if p.key in role.fields]


PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442")
CONTROL_COLOR = "#9e9e9e"


def colors_for(samples: list[Sample], control_roles: tuple[str, ...] = ()) -> dict[str, str]:
    """A colour per sample: grey for controls, the Okabe–Ito palette for the rest."""
    out, i = {}, 0
    for s in samples:
        if s.role in control_roles:
            out[s.id] = CONTROL_COLOR
        else:
            out[s.id] = PALETTE[i % len(PALETTE)]
            i += 1
    return out


def _json_number(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    return v
