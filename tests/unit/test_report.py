"""Methods text and BibTeX sidecars: each pipeline credits its published methods."""

from __future__ import annotations

import pytest

from karcytics_plugins.cytometrics.analysis.pipelines.cellpose_pipeline import CellposePipeline
from karcytics_plugins.cytometrics.analysis.pipelines.otsu import OtsuPipeline
from karcytics_plugins.cytometrics.analysis.pipelines.watershed import WatershedPipeline
from karcytics_plugins.cytometrics.report import (
    build_steps,
    methods_markdown,
    write_methods_sidecars,
)

pytestmark = pytest.mark.unit

SCALE = 0.5
PARAMS = {
    "target_channel": "GFP",
    "seed_channel": "DAPI",
    "use_dual_channel": True,
    "min_area_px": 40,
    "max_area_px": 4000,
    "diameter": 60,
    "flow_threshold": 0.4,
    "exclude_borders": True,
}


@pytest.mark.parametrize(
    ("pipeline", "expected_refs"),
    [
        (OtsuPipeline(), ["Otsu, N. (1979)", "Bradski, G. (2000)"]),
        (WatershedPipeline(), ["Meyer, F. (1992)", "Otsu, N. (1979)", "Bradski, G. (2000)"]),
        (
            CellposePipeline(),
            [
                "Pachitariu, M., Rariden, M. & Stringer, C. (2025)",
                "Stringer, C.",
                "Paszke, A. et al. (2019)",
            ],
        ),
    ],
)
def test_every_pipeline_cites_its_methods_in_order(pipeline, expected_refs):
    md = methods_markdown(build_steps(pipeline, PARAMS, SCALE, n_cells=12))

    refs = md.split("## References\n\n")[1].strip().splitlines()
    assert len(refs) == len(expected_refs)
    for line, start in zip(refs, expected_refs, strict=True):
        assert start in line


def test_cellpose_methods_report_settings_in_microns():
    md = methods_markdown(build_steps(CellposePipeline(), PARAMS, SCALE, n_cells=12))

    assert "Pixel size was 0.5 µm." in md
    assert "Cellpose-SAM generalist model (cpsam), with the 'DAPI' channel as a second input" in md
    assert "cell diameter = 30 µm" in md
    assert "size filter = 10–1000 µm²" in md
    assert "for 12 cell(s)" in md


def test_watershed_names_its_seed_source():
    single = {**PARAMS, "use_dual_channel": False}
    md = methods_markdown(build_steps(WatershedPipeline(), single, SCALE, n_cells=3))

    assert "seeded from distance-transform peaks" in md


def test_manual_cells_are_mentioned():
    md = methods_markdown(build_steps(OtsuPipeline(), PARAMS, SCALE, n_cells=5, n_manual=2))

    assert "2 cell(s) were outlined manually." in md


def test_sidecars_are_written_next_to_the_export(tmp_path):
    export = tmp_path / "results.csv"
    export.write_text("Cell_ID\n")

    methods, bib = write_methods_sidecars(export, build_steps(CellposePipeline(), PARAMS, SCALE, 4))

    assert methods == tmp_path / "results_methods.md"
    assert methods.read_text(encoding="utf-8").startswith(
        "# Methods\n\nImages were analysed with Karcytics CytoMetrics"
    )
    bib_text = bib.read_text(encoding="utf-8")
    assert bib.name == "results_citations.bib"
    assert bib_text.count("@") == 3
    assert "doi = {10.1038/s41592-020-01018-x}" in bib_text


def test_uncalibrated_run_reports_pixels():
    md = methods_markdown(build_steps(CellposePipeline(), PARAMS, 0.0, n_cells=3))

    assert "not spatially calibrated; sizes are reported in pixels" in md
    assert "cell diameter = 60 px" in md
    assert "size filter = 40–4000 px²" in md


def test_scale_origin_is_stated():
    manual = methods_markdown(build_steps(OtsuPipeline(), PARAMS, SCALE, 1, scale_source="manual"))
    meta = methods_markdown(
        build_steps(OtsuPipeline(), PARAMS, SCALE, 1, scale_source="metadata (ome)")
    )

    assert "Pixel size was 0.5 µm (measured against a scale bar in the image)." in manual
    assert "Pixel size was 0.5 µm (read from the image metadata)." in meta
