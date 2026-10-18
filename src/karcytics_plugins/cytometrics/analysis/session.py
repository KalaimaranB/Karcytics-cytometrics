"""One analysis, without Qt: images, cells, calibration, measurements and the last run.

The panel is wiring around an ``AnalysisSession``: widgets call these
methods and redraw from the result. Everything here is testable without a
display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..report import Measured, build_steps, gating_step
from . import edit, gating
from .assays import Assay, AssayContext, ResultValue
from .experiment import Comparison, Role, Sample
from .gating import Gate
from .measure import MeasureOptions, measure_cells, rescale
from .model import ImageSet
from .state import CytoMetricsState
from .units import is_calibrated


@dataclass
class LastRun:
    """The segmenter and parameters behind the current cells, for the methods text."""

    segmenter: Any
    params: dict


class AnalysisSession:
    def __init__(self) -> None:
        self.image_set = ImageSet()
        self.state = CytoMetricsState()
        self.measure_options = MeasureOptions()
        self.assay: Assay | None = None  # set by the panel; its analysis derives columns
        self._raw: pd.DataFrame = measure_cells([], self.image_set, 0.0)
        self._measurements: pd.DataFrame = self._raw
        self.labels: np.ndarray | None = None  # last run's label mask (not saved)
        self.probability: np.ndarray | None = None  # last Cellpose probability map (not saved)
        self.last_run: LastRun | None = None

    # ── Measurements and the assay ───────────────────────────────────────────

    @property
    def measurements(self) -> pd.DataFrame:
        """The measured table plus the assay's derived columns."""
        return self._measurements

    @measurements.setter
    def measurements(self, df: pd.DataFrame) -> None:
        self._raw = df
        self._measurements = self._derive(df)

    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        analysis = self.assay.analysis if self.assay else None
        if analysis is None:
            return df
        return analysis.derive(df, self.assay_settings, self.context())

    def refresh_derived(self) -> None:
        """Recompute the assay's columns (after its settings or the assay changed)."""
        self._measurements = self._derive(self._raw)

    def context(self) -> AssayContext:
        return AssayContext(
            self.scale, self.image_set.shape, dict(self.state.extras), self.image_set
        )

    @property
    def assay_settings(self) -> dict:
        """The assay's settings: its defaults, overridden by what the user set."""
        defaults = self.assay.default_settings() if self.assay else {}
        return {**defaults, **self.state.assay_settings}

    def set_assay_settings(self, values: dict) -> None:
        self.state.assay_settings = {**self.state.assay_settings, **values}
        self.refresh_derived()

    def assay_overlays(self) -> list:
        analysis = self.assay.analysis if self.assay else None
        return analysis.overlays(self.assay_settings, self.context()) if analysis else []

    def assay_results(self) -> list[ResultValue]:
        analysis = self.assay.analysis if self.assay else None
        if analysis is None:
            return []
        return analysis.results(self.measurements, self.gates, self.assay_settings, self.context())

    # ── Experiment ───────────────────────────────────────────────────────────

    @property
    def roles(self) -> tuple[Role, ...]:
        analysis = self.assay.analysis if self.assay else None
        return analysis.roles if analysis else ()

    @property
    def samples(self) -> list[Sample]:
        """The images added to this assay's experiment, in the order added."""
        if self.assay is None:
            return []
        return [Sample.from_dict(d) for d in self.state.experiment.get(self.assay.id, [])]

    def _store(self, samples: list[Sample]) -> None:
        assert self.assay is not None
        self.state.experiment = {
            **self.state.experiment,
            self.assay.id: [s.to_dict() for s in samples],
        }

    def add_sample(self, role_id: str, name: str, image: str = "") -> Sample:
        """Add the current image's measurements to the experiment as `role_id`."""
        role = next(r for r in self.roles if r.id == role_id)
        settings = self.assay_settings
        sample = Sample(
            name=name,
            role=role_id,
            table=self.export_table(),
            image=image,
            fields={k: settings.get(k) for k in role.fields},
            results={v.key: v.value for v in self.assay_results()},
        )
        samples = self.samples
        if not role.many:
            samples = [s for s in samples if s.role != role_id]
        self._store([*samples, sample])
        return sample

    def remove_sample(self, sample_id: str) -> None:
        self._store([s for s in self.samples if s.id != sample_id])

    def comparison(self) -> Comparison | None:
        analysis = self.assay.analysis if self.assay else None
        samples = self.samples
        if analysis is None or not samples:
            return None
        return analysis.compare(samples, self.assay_settings, self.context())

    # ── Calibration ──────────────────────────────────────────────────────────

    @property
    def scale(self) -> float:
        return self.state.scale

    def set_scale(self, scale: float, source: str) -> None:
        """Change the calibration; lengths and areas already measured are converted."""
        self.measurements = rescale(self._raw, scale)
        self.state.scale = scale
        self.state.scale_source = source if is_calibrated(scale) else ""

    def apply_metadata_calibration(self) -> bool:
        """Use the pixel size stored in the image files, unless a scale is already set."""
        pixel_size = self.image_set.pixel_size
        if pixel_size is None or is_calibrated(self.scale):
            return False
        self.set_scale(pixel_size.x_um, f"metadata ({pixel_size.source})")
        return True

    # ── Cells ────────────────────────────────────────────────────────────────

    @property
    def cells(self) -> list[dict]:
        return self.state.cells

    def cell(self, cell_id: int) -> dict | None:
        return next((c for c in self.state.cells if c["id"] == cell_id), None)

    def replace_cells(
        self,
        cells: list[dict],
        measurements: pd.DataFrame | None = None,
        labels: np.ndarray | None = None,
        probability: np.ndarray | None = None,
    ) -> None:
        """The cells of a new segmentation run (ids already assigned)."""
        self.state.cells = list(cells)
        self.state.cell_counter = max((c["id"] for c in cells), default=0)
        if measurements is not None:
            self.measurements = measurements
        self.labels = labels
        self.probability = probability

    def add_manual_cell(self, points: list) -> dict:
        self.state.cell_counter += 1
        cell = {
            "id": self.state.cell_counter,
            "points": [list(p) for p in points],
            "source": "manual",
        }
        self.state.cells.append(cell)
        return cell

    def delete_cells(self, cell_ids: set[int]) -> None:
        self.state.cells = [c for c in self.state.cells if c["id"] not in cell_ids]
        self._drop_rows(cell_ids)

    # ── Edits (each returns the ids it created; the caller re-measures) ──────

    def _new_cell(self, points: list, source: str = "edited") -> dict:
        self.state.cell_counter += 1
        cell = {"id": self.state.cell_counter, "points": points, "source": source}
        self.state.cells.append(cell)
        return cell

    def _shape(self) -> tuple[int, int]:
        shape = self.image_set.shape
        if shape is None:
            raise ValueError("No image is loaded.")
        return shape

    def merge_cells(self, cell_ids: set[int]) -> int | None:
        """Merge touching cells into one; None if they don't touch (nothing changes)."""
        cells = [c for c in self.state.cells if c["id"] in cell_ids]
        if len(cells) < 2:  # noqa: PLR2004
            return None
        outline = edit.merge(cells, self._shape())
        if outline is None:
            return None
        self.delete_cells(cell_ids)
        return self._new_cell(outline)["id"]

    def split_cell(self, cell_id: int, start, end) -> list[int]:
        """Cut a cell along a line; [] if the line doesn't cut it in two."""
        cell = self.cell(cell_id)
        if cell is None:
            return []
        pieces = edit.split(cell, start, end, self._shape())
        if len(pieces) < 2:  # noqa: PLR2004
            return []
        self.delete_cells({cell_id})
        return [self._new_cell(p)["id"] for p in pieces]

    def paint_cell(self, cell_id: int, stroke, radius: int, add: bool) -> list[int]:
        """Brush pixels onto (or off) a cell; returns the ids of what's left."""
        cell = self.cell(cell_id)
        if cell is None:
            return []
        pieces = edit.paint(cell, stroke, radius, add, self._shape())
        if len(pieces) == 1:
            cell["points"] = pieces[0]
            if cell.get("source") != "manual":
                cell["source"] = "edited"
            cell.pop("confidence", None)  # no longer the model's outline
            self._drop_rows({cell_id})
            return [cell_id]
        self.delete_cells({cell_id})
        return [self._new_cell(p)["id"] for p in pieces]

    def replace_region(self, box: tuple[int, int, int, int], found: list[dict]) -> list[int]:
        """Swap the cells centred in `box` for `found` (already in image coordinates)."""
        old = {c["id"] for c in self.state.cells if edit.inside(c, box)}
        self.delete_cells(old)
        ids = []
        for cell in found:
            new = self._new_cell(cell["points"], "auto")
            if cell.get("confidence") is not None:
                new["confidence"] = cell["confidence"]
            ids.append(new["id"])
        return ids

    def set_flag(self, cell_ids: set[int], flagged: bool) -> None:
        for cell in self.state.cells:
            if cell["id"] in cell_ids:
                if flagged:
                    cell["flagged"] = True
                else:
                    cell.pop("flagged", None)
        if "Flagged" in self._raw.columns:
            raw = self._raw.copy()
            raw.loc[raw["ObjectNumber"].isin(cell_ids), "Flagged"] = flagged
            raw.attrs = self._raw.attrs
            self.measurements = raw

    def _drop_rows(self, cell_ids: set[int]) -> None:
        kept = self._raw[~self._raw["ObjectNumber"].isin(cell_ids)].copy()
        kept.attrs = self._raw.attrs
        self.measurements = kept

    def needs_measuring(self) -> bool:
        measured = set(self.measurements["ObjectNumber"]) if len(self.measurements) else set()
        return measured != {c["id"] for c in self.state.cells}

    def uncertain_cells(self, threshold: float) -> list[dict]:
        """Cells whose confidence is below `threshold`, least confident first."""
        scored = [c for c in self.state.cells if c.get("confidence") is not None]
        return sorted(
            (c for c in scored if c["confidence"] < threshold), key=lambda c: c["confidence"]
        )

    def has_confidence(self) -> bool:
        return any(c.get("confidence") is not None for c in self.state.cells)

    # ── Gates and populations ────────────────────────────────────────────────

    @property
    def gates(self) -> list[Gate]:
        return [Gate.from_dict(g) for g in self.state.gates]

    def set_gates(self, gates: list[Gate]) -> None:
        gating.evaluate(gates, self.measurements)  # raises on cycles or missing gates
        self.state.gates = [g.to_dict() for g in gates]

    def add_gate(self, gate: Gate) -> Gate:
        """Add `gate` under a unique name and the next free colour; returns it."""
        gates = self.gates
        gate.name = gating.unique_name(gates, gate.name)
        if not gate.color or gate.color in {g.color for g in gates}:
            gate.color = gating.next_color(gates)
        self.set_gates([*gates, gate])
        return gate

    def update_gate(self, name: str, **changes) -> None:
        self.set_gates(
            [
                Gate.from_dict({**g.to_dict(), **changes}) if g.name == name else g
                for g in self.gates
            ]
        )

    def rename_gate(self, old: str, new: str) -> str:
        new = gating.unique_name([g for g in self.gates if g.name != old], new.strip() or old)
        self.set_gates(gating.rename(self.gates, old, new))
        return new

    def remove_gate(self, name: str) -> None:
        self.set_gates(gating.remove(self.gates, name))

    def population(self, name: str | None) -> pd.DataFrame:
        """The measurement rows of population `name` (all cells for None)."""
        return gating.population(self.gates, self.measurements, name)

    def gate_stats(self) -> pd.DataFrame:
        return gating.stats(self.gates, self.measurements)

    def population_colors(self) -> dict[int, str]:
        return gating.colors_by_population(self.gates, self.measurements)

    def export_table(self) -> pd.DataFrame:
        """The measurements plus one ``Gate_<name>`` column per gate."""
        return gating.with_gate_columns(self.gates, self.measurements)

    # ── Images ───────────────────────────────────────────────────────────────

    def channels_metadata(self) -> list[dict]:
        return [
            {
                "name": c.name,
                "path": c.path,
                "color": c.color,
                "visible": c.visible,
                "display_range": list(c.display_range) if c.display_range else None,
            }
            for c in self.image_set.channels
        ]

    def reload_images(self, channels_meta: list[dict]) -> list[str]:
        """Reload the saved image files and reapply channel names and colours.

        Returns the paths that couldn't be found or read. Does nothing if the
        same files are already loaded in the same order.
        """
        current = [c.path for c in self.image_set.channels]
        saved = [ch.get("path", "") for ch in channels_meta]
        if current == saved:
            return []
        self.image_set.clear()
        self.probability = None
        missing = []
        for path in dict.fromkeys(saved):  # each file once, in order
            if not path or not Path(path).exists():
                missing.append(path)
                continue
            try:
                self.image_set.add_file(path, Path(path).name)
            except (OSError, ValueError):
                missing.append(path)
        for channel, meta in zip(self.image_set.channels, channels_meta, strict=False):
            channel.name = meta.get("name", channel.name)
            channel.color = meta.get("color", channel.color)
            channel.visible = meta.get("visible", True)
            saved_range = meta.get("display_range")
            channel.display_range = tuple(saved_range) if saved_range else None  # type: ignore[assignment]
        return missing

    # ── Reporting ────────────────────────────────────────────────────────────

    def methods_steps(self) -> list | None:
        """The methods text steps for the current cells, or None before any run."""
        if self.last_run is None:
            return None
        n_manual = sum(1 for c in self.state.cells if c.get("source") == "manual")
        steps = build_steps(
            self.last_run.segmenter,
            self.last_run.params,
            self.scale,
            len(self.state.cells),
            n_manual=n_manual,
            scale_source=self.state.scale_source,
            measured=Measured(
                tuple(c.name for c in self.image_set.channels),
                self.measure_options.background,
                self.measure_options.texture,
                self.measure_options.nucleus_channel,
                self.measure_options.rods,
            ),
        )
        analysis = self.assay.analysis if self.assay else None
        if analysis is not None:
            steps.extend(analysis.methods(self.assay_settings, self.context()))
        if self.state.gates and len(self.measurements):
            steps.append(gating_step(self.gate_stats()))
        comparison = self.comparison()
        if comparison is not None:
            steps.extend(comparison.methods)
        return steps

    def export_stem(self) -> str:
        """The first image's file name without extension, for default export names."""
        path = self.image_set.channels[0].path if self.image_set.channels else ""
        name = Path(path).name if path else "cytometrics"
        for suffix in (".ome.tiff", ".ome.tif"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem
