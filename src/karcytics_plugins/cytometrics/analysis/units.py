"""Units for a calibrated (µm) or uncalibrated (px) analysis.

A scale of 0 means "not calibrated": measurements stay in pixels and are
labelled px / px², never silently presented as microns.
"""

from __future__ import annotations


def is_calibrated(scale: float) -> bool:
    return scale > 0


def effective_scale(scale: float) -> float:
    """The µm-per-px factor to measure with: 1 when uncalibrated (pixels)."""
    return scale if is_calibrated(scale) else 1.0


def length_unit(scale: float) -> str:
    return "µm" if is_calibrated(scale) else "px"


def area_unit(scale: float) -> str:
    return "µm²" if is_calibrated(scale) else "px²"


def ascii_unit(unit: str) -> str:
    """`unit` spelled for a column header: µm² → um2."""
    return unit.replace("µ", "u").replace("²", "2")
