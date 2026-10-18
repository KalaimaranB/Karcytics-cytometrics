"""CytoMetrics analysis state container."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from karcytics_sdk.plugin import PluginState, get_logger

logger = get_logger(__name__, "cytometrics")


@dataclass
class CytoMetricsState(PluginState):
    """Mutable state for one CytoMetrics analysis session.

    Attributes:
        scale:              Imaging scale (microns per pixel); 0 means uncalibrated (pixels).
        scale_source:       Where the scale came from: "metadata (<reader>)", "manual" or "".
        cells:              List of detected cell objects (dict with points, metrics).
        cell_counter:       Running total of cells detected (used for IDs).
        channels_metadata:  List of channel information (name, path, color).
        ui_params:          Configuration from the Smart Detect tab.
    """

    scale: float = 0.0
    scale_source: str = ""
    cells: list[dict] = field(default_factory=list)
    cell_counter: int = 0
    channels_metadata: list[dict] = field(default_factory=list)
    ui_params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CytoMetricsState:
        """Rebuild from a saved dict, ignoring keys this version doesn't know."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning("Ignoring unknown CytoMetrics state keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in data.items() if k in known})
