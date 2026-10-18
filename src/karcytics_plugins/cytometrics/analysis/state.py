"""CytoMetrics analysis state container."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from karcytics_sdk.plugin import PluginState, get_logger

logger = get_logger(__name__, "cytometrics")

STATE_VERSION = 2

# v1 `ui_params` → v2: display names became segmenter ids, flat keys became Param keys.
_V1_PIPELINES = {
    "Basic Threshold (Otsu)": "otsu",
    "Watershed (Clump Splitting)": "watershed",
    "AI Smart Detect (Cellpose)": "cellpose",
}
_V1_KEYS = {
    "target_channel": "target_channel",
    "seed_channel": "seed_channel",
    "use_dual": "use_dual_channel",
    "min_area": "min_area_px",
    "max_area": "max_area_px",
    "diameter": "diameter",
    "flow": "flow_threshold",
    "exclude_borders": "exclude_borders",
}
_V1_CELL_MEASURES = ("area", "perim", "circ")  # now measured into the ObjectTable


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """A saved state dict in the current format (the input is not modified).

    v1 (no ``version``): ``ui_params`` held display names and flat keys, and
    each cell carried its own area/perimeter/circularity.
    """
    if data.get("version", 1) >= STATE_VERSION:
        return data
    out = dict(data)
    ui = dict(out.get("ui_params") or {})
    if ui and "values" not in ui:
        pipeline = str(ui.get("pipeline", ""))
        ui = {
            "pipeline": _V1_PIPELINES.get(pipeline, pipeline or None),
            "values": {new: ui[old] for old, new in _V1_KEYS.items() if old in ui},
        }
    out["ui_params"] = ui
    out["cells"] = [
        {k: v for k, v in c.items() if k not in _V1_CELL_MEASURES} for c in out.get("cells", [])
    ]
    out["version"] = STATE_VERSION
    return out


@dataclass
class CytoMetricsState(PluginState):
    """Mutable state for one CytoMetrics analysis session.

    Attributes:
        scale:              Imaging scale (microns per pixel); 0 means uncalibrated (pixels).
        scale_source:       Where the scale came from: "metadata (<reader>)", "manual" or "".
        cells:              List of detected cell objects (dict with points, metrics).
        cell_counter:       Running total of cells detected (used for IDs).
        channels_metadata:  List of channel information (name, path, color).
        ui_params:          ``{"pipeline": segmenter id, "values": Detect settings,
                            "measure": measurement options}``.
        gates:              Gates on the measurements (``analysis.gating.Gate`` dicts).
        batch_manifest:     Path of the batch run's ``batch.json`` (empty if none).
        version:            Format version; older saves are migrated by ``from_dict``.
    """

    version: int = STATE_VERSION
    scale: float = 0.0
    scale_source: str = ""
    cells: list[dict] = field(default_factory=list)
    cell_counter: int = 0
    channels_metadata: list[dict] = field(default_factory=list)
    ui_params: dict = field(default_factory=dict)
    gates: list[dict] = field(default_factory=list)
    batch_manifest: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CytoMetricsState:
        """Rebuild from a saved dict: migrate older formats, ignore unknown keys."""
        data = migrate(data)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            logger.warning("Ignoring unknown CytoMetrics state keys: %s", sorted(unknown))
        return cls(**{k: v for k, v in data.items() if k in known})
