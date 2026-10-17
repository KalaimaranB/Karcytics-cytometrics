"""End-to-end: image stack → CytoPipelineWorker → cell dicts.

Runs the worker synchronously (no TaskScheduler), with real classical
pipelines and a fake one, and — opt-in, marked `slow` — the real Cellpose-SAM
model when it is already cached on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import images as img
from fixtures.segmenters import FakeSegmenter, square_cell

from karcytics_plugins.cytometrics.analysis.pipelines.watershed import WatershedPipeline
from karcytics_plugins.cytometrics.ui.workers import CytoPipelineWorker

pytestmark = pytest.mark.functional

PARAMS = {"target_channel": "cells", "min_area_px": 50, "max_area_px": 100_000}
CELLPOSE_MODEL_DIR = Path.home() / ".cellpose" / "models"


def _run(pipeline, stack, params=PARAMS, scale=1.0) -> dict:
    worker = CytoPipelineWorker()
    worker.configure(pipeline, stack, params, scale)
    return worker.run()


def test_worker_passes_stack_params_and_scale_to_the_pipeline():
    fake = FakeSegmenter([square_cell(1, 1, 5)])
    stack = img.stack(("cells", img.isolated_discs()))

    result = _run(fake, stack, scale=0.5)

    assert result == {"result_cells": [square_cell(1, 1, 5)]}
    assert fake.calls == [(stack, PARAMS, 0.5)]


def test_worker_reports_pipeline_errors_instead_of_raising():
    result = _run(
        FakeSegmenter(error=RuntimeError("GPU fell over")), img.stack(("cells", img.blank()))
    )

    assert result == {"error": "GPU fell over"}


def test_worker_without_a_pipeline_reports_an_error():
    assert CytoPipelineWorker().run() == {"error": "No pipeline configured"}


def test_worker_restores_stderr_after_running():
    import sys

    before = sys.stderr
    _run(FakeSegmenter(), img.stack(("cells", img.blank())))

    assert sys.stderr is before


def test_watershed_through_the_worker_finds_every_isolated_disc():
    result = _run(WatershedPipeline(), img.stack(("cells", img.isolated_discs())))

    assert len(result["result_cells"]) == len(img.ISOLATED_CENTRES)


@pytest.mark.slow
@pytest.mark.skipif(
    not (CELLPOSE_MODEL_DIR / "cpsam").exists(),
    reason="Cellpose-SAM model not cached (~1.3 GB); never downloaded in tests",
)
def test_cellpose_sam_finds_isolated_discs():
    from karcytics_plugins.cytometrics.analysis.pipelines.cellpose_pipeline import CellposePipeline

    params = {**PARAMS, "diameter": 2 * img.ISOLATED_RADIUS, "exclude_borders": True}
    result = _run(CellposePipeline(), img.stack(("cells", img.isolated_discs())), params)

    assert len(result["result_cells"]) == len(img.ISOLATED_CENTRES)
