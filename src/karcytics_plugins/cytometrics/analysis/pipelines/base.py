from abc import ABC, abstractmethod
from typing import List, Tuple


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
    def run(self, image_stack, parameters: dict, scale: float) -> List[dict]:
        """
        Executes the algorithm.
        Returns a list of cell dictionaries: [{"points": [...], "area": X, "perim": Y, "circ": Z}, ...]
        """
        pass