"""CytoMetrics analysis state container."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from karcytics_sdk.plugin import PluginState

logger = logging.getLogger(__name__)

@dataclass
class CytoMetricsState(PluginState):
    """Mutable state for one CytoMetrics analysis session.
    
    Attributes:
        scale:              Imaging scale (microns per pixel).
        cells:              List of detected cell objects (dict with points, metrics).
        cell_counter:       Running total of cells detected (used for IDs).
        channels_metadata:  List of channel information (name, path, color).
        ui_params:          Configuration from the Smart Detect tab.
    """
    
    scale: float = 0.0
    cells: list[dict] = field(default_factory=list)
    cell_counter: int = 0
    channels_metadata: list[dict] = field(default_factory=list)
    ui_params: dict = field(default_factory=dict)
