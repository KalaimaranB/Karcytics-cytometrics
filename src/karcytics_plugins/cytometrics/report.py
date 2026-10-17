"""Methods text and BibTeX written next to every export.

Scientists should be able to cite exactly what produced their numbers:
which segmentation method, with which settings, built on whose published
work. `write_methods_sidecars` writes ``<export>_methods.md`` and
``<export>_citations.bib`` beside an exported table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from karcytics_sdk.plugin.citations import MethodStep, methods_text, to_bibtex

from . import __version__
from . import citations as _registered  # noqa: F401  (registers CytoMetrics' entries)

SOFTWARE = "Karcytics CytoMetrics"
SOFTWARE_URL = "https://github.com/KalaimaranB/Karcytics-cytometrics"


def build_steps(
    pipeline: Any, params: dict, scale: float, n_cells: int, n_manual: int = 0
) -> list[MethodStep]:
    """The methods steps for one segmentation run, in reading order."""
    steps = [MethodStep(f"Pixel size was {scale:.4g} µm")] if scale > 0 else []
    description, parameters = pipeline.describe(params, scale)
    steps.append(MethodStep(description, tuple(pipeline.citations), parameters))
    if n_manual:
        steps.append(MethodStep(f"{n_manual} cell(s) were outlined manually"))
    steps.append(
        MethodStep(
            f"Area, perimeter, circularity (4π·area/perimeter²) and equivalent diameter were measured "
            f"for {n_cells} cell(s) from their outlines"
        )
    )
    return steps


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
