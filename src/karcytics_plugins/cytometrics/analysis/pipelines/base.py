from abc import ABC, abstractmethod

from ..units import area_unit, effective_scale


class SegmentationPipeline(ABC):
    """Abstract Base Class for all CytoMetrics detection algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The display name in the UI dropdown."""
        pass

    @abstractmethod
    def get_parameters(self) -> dict:
        """Returns the UI inputs it requires (e.g., {'min_area': 'int', 'channel': 'list'})."""
        pass

    @abstractmethod
    def run(self, image_set, parameters: dict, scale: float) -> list[dict]:
        """Executes the algorithm.

        Returns a list of cell dictionaries: [{"points": [...], "area": X, "perim": Y, "circ": Z}, ...].
        """
        pass


def area_range(params: dict, scale: float) -> str:
    """The size filter as the user set it, in µm² (or px² when uncalibrated)."""
    factor = effective_scale(scale) ** 2
    lo = params.get("min_area_px", 0) * factor
    hi = params.get("max_area_px", 0) * factor
    return f"{lo:g}–{hi:g} {area_unit(scale)}"
