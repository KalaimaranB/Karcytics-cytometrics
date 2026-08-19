"""CytoMetrics — Karcytics plugin entry point.

AI-assisted multi-channel cell morphology quantification.
"""

from typing import Any

__version__ = "0.2.2"
__plugin_id__ = "cytometrics"


def get_panel_class() -> type:
    """Standard Karcytics entry point.

    The core ``ModuleManager`` calls this function to obtain the class (not an
    instance) and then instantiates it into the central workspace container.
    """
    from .ui.main_panel import CytoMetricsPanel

    return CytoMetricsPanel


def cleanup() -> None:
    """Module-level cleanup."""


def shutdown() -> None:
    """Module-level shutdown: release global VRAM and AI models."""
    import sys

    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import gc

            gc.collect()


# Late import: PluginContext lives in karcytics_sdk, which depends on this
# package at runtime — importing it at module load would create a circular
# import.
from karcytics_sdk.plugin.context import PluginContext  # noqa: E402


def initialize(context: PluginContext) -> Any:
    """Plugin entry point."""
    logger = context.get("logger")
    logger.info("Initializing CytoMetrics with PluginContext")

    # Return the module itself so the core can call .get_panel_class(), .cleanup(), etc.
    import sys

    return sys.modules[__name__]
