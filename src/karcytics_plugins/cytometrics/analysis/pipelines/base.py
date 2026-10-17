from abc import ABC, abstractmethod


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
    def run(self, image_stack, parameters: dict, scale: float) -> list[dict]:
        """Executes the algorithm.

        Returns a list of cell dictionaries: [{"points": [...], "area": X, "perim": Y, "circ": Z}, ...].
        """
        pass


def area_range_um2(params: dict, scale: float) -> str:
    """The size filter as the user set it, in µm²."""
    lo = params.get("min_area_px", 0) * scale**2
    hi = params.get("max_area_px", 0) * scale**2
    return f"{lo:g}–{hi:g} µm²"
