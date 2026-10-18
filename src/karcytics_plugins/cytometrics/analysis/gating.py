"""Gates and populations on the object table, without Qt.

A gate selects cells by their measurements:

- ``threshold``: one measurement above (or below) a value;
- ``range``: one measurement between two values;
- ``rect`` / ``polygon``: a region of a two-measurement scatter plot;
- ``and`` / ``or`` / ``not``: combinations of other gates.

Gates form a hierarchy: a gate with a ``parent`` only looks at the parent's
cells, so its population is always a subset of the parent's (as in flow
cytometry). A cell with a missing value is outside any gate on that value.

Shape gates remember whether each axis was drawn on a log scale; membership
is tested in the same space the gate was drawn in, so what the user drew is
what they get.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.path import Path as _MplPath

KINDS = ("threshold", "range", "rect", "polygon", "and", "or", "not")
BOOLEAN_KINDS = ("and", "or", "not")
ALL = "All cells"  # the root population's name (not a gate)

# Distinct, colour-blind-friendly population colours (Okabe & Ito 2008).
UNGATED_COLOR = "#9e9e9e"  # cells in no gate, on the image and the plots
PALETTE = ("#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")


@dataclass
class Gate:
    name: str
    kind: str
    x: str = ""  # measurement column (threshold, range, rect, polygon)
    y: str = ""  # second column (rect, polygon)
    params: dict[str, Any] = field(default_factory=dict)
    parent: str | None = None
    color: str = PALETTE[0]
    x_log: bool = False
    y_log: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Gate:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def operands(self) -> list[str]:
        return list(self.params.get("of", [])) if self.kind in BOOLEAN_KINDS else []

    def describe(self) -> str:
        """A one-line definition for tables and the methods text."""
        p = self.params
        if self.kind == "threshold":
            return f"{self.x} {'>' if p.get('above', True) else '<'} {p.get('value', 0):.4g}"
        if self.kind == "range":
            return f"{p.get('lo', 0):.4g} ≤ {self.x} ≤ {p.get('hi', 0):.4g}"
        if self.kind == "rect":
            return (
                f"{p.get('x0', 0):.4g} ≤ {self.x} ≤ {p.get('x1', 0):.4g} and "
                f"{p.get('y0', 0):.4g} ≤ {self.y} ≤ {p.get('y1', 0):.4g}"
            )
        if self.kind == "polygon":
            return f"polygon on {self.x} vs {self.y} ({len(p.get('points', []))} vertices)"
        if self.kind == "not":
            return f"not {self.operands[0]}" if self.operands else "not ?"
        return f" {self.kind} ".join(self.operands)


def next_color(gates: list[Gate]) -> str:
    """The first palette colour no gate uses yet (cycling when all are taken)."""
    used = {g.color for g in gates}
    return next((c for c in PALETTE if c not in used), PALETTE[len(gates) % len(PALETTE)])


def unique_name(gates: list[Gate], base: str) -> str:
    names = {g.name for g in gates} | {ALL}
    if base not in names:
        return base
    n = 2
    while f"{base} {n}" in names:
        n += 1
    return f"{base} {n}"


def _values(df: pd.DataFrame, column: str, log: bool) -> np.ndarray:
    if column not in df:
        return np.full(len(df), np.nan)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if log:
        with np.errstate(divide="ignore", invalid="ignore"):
            values = np.where(values > 0, np.log10(values), np.nan)
    return values


def _axis(value: float, log: bool) -> float:
    if not log:
        return value
    return float(np.log10(value)) if value > 0 else -np.inf


def _own_mask(gate: Gate, df: pd.DataFrame, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Cells inside `gate` itself, before its parent is applied."""
    if gate.kind in BOOLEAN_KINDS:
        return _boolean_mask(gate, len(df), masks)
    if gate.kind in ("threshold", "range"):
        return _one_axis_mask(gate, _values(df, gate.x, gate.x_log))
    if gate.kind in ("rect", "polygon"):
        return _two_axis_mask(
            gate, _values(df, gate.x, gate.x_log), _values(df, gate.y, gate.y_log)
        )
    raise ValueError(f"Unknown gate kind {gate.kind!r}")


def _boolean_mask(gate: Gate, n: int, masks: dict[str, np.ndarray]) -> np.ndarray:
    parts = [masks.get(name, np.zeros(n, bool)) for name in gate.operands]
    if not parts:
        return np.zeros(n, bool)
    if gate.kind == "not":
        return ~parts[0]
    combine = np.logical_and if gate.kind == "and" else np.logical_or
    return combine.reduce(parts)


def _one_axis_mask(gate: Gate, x: np.ndarray) -> np.ndarray:
    p = gate.params
    with np.errstate(invalid="ignore"):
        if gate.kind == "threshold":
            value = _axis(float(p.get("value", 0.0)), gate.x_log)
            return x > value if p.get("above", True) else x < value
        lo, hi = sorted((float(p.get("lo", 0.0)), float(p.get("hi", 0.0))))
        return (x >= _axis(lo, gate.x_log)) & (x <= _axis(hi, gate.x_log))


def _two_axis_mask(gate: Gate, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = gate.params
    if gate.kind == "rect":
        x0, x1 = sorted((float(p.get("x0", 0.0)), float(p.get("x1", 0.0))))
        y0, y1 = sorted((float(p.get("y0", 0.0)), float(p.get("y1", 0.0))))
        with np.errstate(invalid="ignore"):
            return (
                (x >= _axis(x0, gate.x_log))
                & (x <= _axis(x1, gate.x_log))
                & (y >= _axis(y0, gate.y_log))
                & (y <= _axis(y1, gate.y_log))
            )
    points = p.get("points", [])
    inside = np.zeros(len(x), bool)
    if len(points) < 3:  # noqa: PLR2004
        return inside
    vertices = [(_axis(px, gate.x_log), _axis(py, gate.y_log)) for px, py in points]
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.any():
        inside[finite] = _MplPath(vertices).contains_points(np.column_stack([x, y])[finite])
    return inside


def evaluate(gates: list[Gate], df: pd.DataFrame) -> dict[str, np.ndarray]:
    """``{gate name: boolean mask over df's rows}``, each within its parent.

    Raises ValueError for a gate that refers to itself through parents or
    operands, or to a gate that doesn't exist.
    """
    by_name = {g.name: g for g in gates}
    masks: dict[str, np.ndarray] = {}
    visiting: set[str] = set()

    def mask_of(name: str) -> np.ndarray:
        if name in masks:
            return masks[name]
        if name not in by_name:
            raise ValueError(f"Gate {name!r} doesn't exist")
        if name in visiting:
            raise ValueError(f"Gate {name!r} depends on itself")
        visiting.add(name)
        gate = by_name[name]
        for operand in gate.operands:
            mask_of(operand)
        own = _own_mask(gate, df, masks)
        if gate.parent and gate.parent != ALL:
            own = own & mask_of(gate.parent)
        visiting.discard(name)
        masks[name] = np.asarray(own, dtype=bool)
        return masks[name]

    for gate in gates:
        mask_of(gate.name)
    return masks


def population(gates: list[Gate], df: pd.DataFrame, name: str | None) -> pd.DataFrame:
    """The rows of `df` in population `name` (all rows for None or ``ALL``)."""
    if not name or name == ALL:
        return df
    return df[evaluate(gates, df)[name]]


def stats(gates: list[Gate], df: pd.DataFrame) -> pd.DataFrame:
    """Count, % of parent and % of all cells for every population, root first."""
    masks = evaluate(gates, df)
    total = len(df)
    rows = [{"Population": ALL, "Parent": "", "Definition": "", "Count": total, "PctParent": 100.0}]
    for gate in gates:
        count = int(masks[gate.name].sum())
        parent = gate.parent or ALL
        parent_count = total if parent == ALL else int(masks[parent].sum())
        rows.append(
            {
                "Population": gate.name,
                "Parent": parent,
                "Definition": gate.describe(),
                "Count": count,
                "PctParent": 100.0 * count / parent_count if parent_count else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["PctTotal"] = 100.0 * out["Count"] / total if total else np.nan
    return out


def depth(gates: list[Gate], name: str) -> int:
    """How many parents a gate has (0 for a gate on all cells)."""
    by_name = {g.name: g for g in gates}
    d, seen = 0, set()
    gate = by_name.get(name)
    while gate and gate.parent and gate.parent != ALL and gate.parent not in seen:
        seen.add(gate.parent)
        gate = by_name.get(gate.parent)
        d += 1
    return d


def colors_by_population(
    gates: list[Gate], df: pd.DataFrame, ids_column: str = "ObjectNumber"
) -> dict[int, str]:
    """Each gated cell's colour: its deepest population's (later gates win ties)."""
    masks = evaluate(gates, df)
    ids = df[ids_column].to_numpy()
    best: dict[int, tuple[int, int, str]] = {}  # id -> (depth, order, colour)
    for order, gate in enumerate(gates):
        rank = (depth(gates, gate.name), order)
        for cell_id in ids[masks[gate.name]]:
            key = int(cell_id)
            if key not in best or rank >= best[key][:2]:
                best[key] = (*rank, gate.color)
    return {k: v[2] for k, v in best.items()}


def with_gate_columns(gates: list[Gate], df: pd.DataFrame) -> pd.DataFrame:
    """`df` plus one boolean ``Gate_<name>`` column per gate (for export)."""
    if not gates:
        return df
    out = df.copy()
    for name, mask in evaluate(gates, df).items():
        out[f"Gate_{name}"] = mask
    out.attrs = df.attrs
    return out


def rename(gates: list[Gate], old: str, new: str) -> list[Gate]:
    """Rename a gate and every reference to it (parents and operands)."""
    out = []
    for g in gates:
        params = dict(g.params)
        if g.operands:
            params["of"] = [new if n == old else n for n in g.operands]
        out.append(
            Gate(
                **{
                    **asdict(g),
                    "name": new if g.name == old else g.name,
                    "parent": new if g.parent == old else g.parent,
                    "params": params,
                }
            )
        )
    return out


def remove(gates: list[Gate], name: str) -> list[Gate]:
    """Delete a gate and every gate that depends on it (children and combinations)."""
    doomed = {name}
    changed = True
    while changed:
        changed = False
        for g in gates:
            if g.name not in doomed and (g.parent in doomed or doomed & set(g.operands)):
                doomed.add(g.name)
                changed = True
    return [g for g in gates if g.name not in doomed]
