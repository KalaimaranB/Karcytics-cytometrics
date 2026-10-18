"""Methods text and BibTeX written next to every export.

Scientists should be able to cite exactly what produced their numbers:
which segmentation method, with which settings, built on whose published
work. `write_methods_sidecars` writes ``<export>_methods.md`` and
``<export>_citations.bib`` beside an exported table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from karcytics_sdk.plugin.citations import MethodStep, methods_text, to_bibtex

from . import __version__
from . import citations as _registered  # noqa: F401  (registers CytoMetrics' entries)
from .analysis.units import is_calibrated

SOFTWARE = "Karcytics CytoMetrics"
SOFTWARE_URL = "https://github.com/KalaimaranB/Karcytics-cytometrics"


def _scale_origin(scale_source: str) -> str:
    if scale_source == "manual":
        return " (measured against a scale bar in the image)"
    if scale_source.startswith("metadata"):
        return " (read from the image metadata)"
    return ""


@dataclass(frozen=True)
class Measured:
    """What was measured, for the methods text."""

    channels: tuple[str, ...] = ()
    background: str = "none"
    texture: bool = False
    nucleus_channel: str | None = None


def build_steps(
    pipeline: Any,
    params: dict,
    scale: float,
    n_cells: int,
    *,
    n_manual: int = 0,
    scale_source: str = "",
    measured: Measured = Measured(),  # noqa: B008  (frozen, so a shared default is safe)
) -> list[MethodStep]:
    """The methods steps for one segmentation run, in reading order."""
    steps = [
        MethodStep(f"Pixel size was {scale:.4g} µm{_scale_origin(scale_source)}")
        if is_calibrated(scale)
        else MethodStep("Images were not spatially calibrated; sizes are reported in pixels")
    ]
    description, parameters = pipeline.describe(params, scale)
    steps.append(MethodStep(description, tuple(pipeline.citations), parameters))
    if n_manual:
        steps.append(MethodStep(f"{n_manual} cell(s) were outlined manually"))
    steps.append(_measurement_step(n_cells, measured.channels, measured.background))
    if measured.nucleus_channel:
        steps.append(
            MethodStep(
                f"Each cell was divided into nucleus (pixels above the Otsu threshold of the "
                f"'{measured.nucleus_channel}' channel) and cytoplasm, and nuclear and cytoplasmic "
                "areas and mean intensities and their ratios were measured",
                ("otsu1979",),
            )
        )
    if measured.texture:
        steps.append(
            MethodStep(
                "Haralick texture features (angular second moment, contrast, correlation, inverse "
                "difference moment, entropy and variance) were computed from each cell's grey-level "
                "co-occurrence matrix at a distance of 3 pixels, averaged over four directions, "
                "after scaling each channel to 8 bits",
                ("haralick1973", "skimage2014"),
            )
        )
    return steps


_BACKGROUND_TEXT = {
    "global": "after subtracting the median of all pixels outside the cells",
    "local": "after subtracting the median of a 5-pixel-wide ring 2 pixels outside each cell, "
    "excluding neighbouring cells",
}


def _measurement_step(n_cells: int, channels: tuple[str, ...], background: str) -> MethodStep:
    text = (
        f"For {n_cells} cell(s), area, perimeter, form factor (4π·area/perimeter²), solidity, "
        "eccentricity, ellipse axis lengths, minimum and maximum Feret diameters and equivalent "
        "diameter were measured from each cell's mask with scikit-image"
    )
    if channels:
        names = ", ".join(f"'{c}'" for c in channels)
        text += (
            f", and the mean, median, minimum, maximum, standard deviation and integrated pixel "
            f"intensity of each cell in {names} in native image units"
        )
        if background in _BACKGROUND_TEXT:
            text += f", also {_BACKGROUND_TEXT[background]}"
    text += "; measurement names follow CellProfiler"
    return MethodStep(text, ("skimage2014", "cellprofiler4"))


def gating_step(stats: Any) -> MethodStep:
    """Populations defined by manual gating, with their sizes (``gating.stats`` rows)."""
    parts = []
    for row in stats.iloc[1:].to_dict("records"):  # row 0 is all cells
        within = "" if row["Parent"] == "All cells" else f" within '{row['Parent']}'"
        parts.append(
            f"'{row['Population']}' ({row['Definition']}{within}; n = {row['Count']}, "
            f"{row['PctParent']:.1f}% of parent)"
        )
    return MethodStep(
        "Populations were defined by manual gating on the measurements: " + "; ".join(parts)
    )


def methods_markdown(steps: list[MethodStep]) -> str:
    text = methods_text(SOFTWARE, f"{__version__} ({SOFTWARE_URL})", steps)
    paragraph, _, references = text.partition("\n\nReferences\n")
    out = f"# Methods\n\n{paragraph}\n"
    if references:
        out += f"\n## References\n\n{references}\n"
    return out


def write_methods_sidecars(export_path: str | Path, steps: list[MethodStep]) -> tuple[Path, Path]:
    """Write ``<stem>_methods.md`` and ``<stem>_citations.bib`` next to `export_path`."""
    export_path = Path(export_path)
    methods_path = export_path.with_name(f"{export_path.stem}_methods.md")
    bib_path = export_path.with_name(f"{export_path.stem}_citations.bib")
    methods_path.write_text(methods_markdown(steps), encoding="utf-8")
    keys = [k for step in steps for k in step.citations]
    bib_path.write_text(to_bibtex(keys) if keys else "", encoding="utf-8")
    return methods_path, bib_path
