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

from ..report import Measured, build_steps
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
        self.measurements: pd.DataFrame = measure_cells([], self.image_set, 0.0)
        self.labels: np.ndarray | None = None  # last run's label mask (not saved)
        self.probability: np.ndarray | None = None  # last Cellpose probability map (not saved)
        self.last_run: LastRun | None = None

    # ── Calibration ──────────────────────────────────────────────────────────

    @property
    def scale(self) -> float:
        return self.state.scale

    def set_scale(self, scale: float, source: str) -> None:
        """Change the calibration; lengths and areas already measured are converted."""
        self.measurements = rescale(self.measurements, scale)
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
        kept = self.measurements[~self.measurements["ObjectNumber"].isin(cell_ids)].copy()
        kept.attrs = self.measurements.attrs
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

    # ── Images ───────────────────────────────────────────────────────────────

    def channels_metadata(self) -> list[dict]:
        return [{"name": c.name, "path": c.path, "color": c.color} for c in self.image_set.channels]

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
        return missing

    # ── Reporting ────────────────────────────────────────────────────────────

    def methods_steps(self) -> list | None:
        """The methods text steps for the current cells, or None before any run."""
        if self.last_run is None:
            return None
        n_manual = sum(1 for c in self.state.cells if c.get("source") == "manual")
        return build_steps(
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
            ),
        )

    def export_stem(self) -> str:
        """The first image's file name without extension, for default export names."""
        path = self.image_set.channels[0].path if self.image_set.channels else ""
        name = Path(path).name if path else "cytometrics"
        for suffix in (".ome.tiff", ".ome.tif"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem
