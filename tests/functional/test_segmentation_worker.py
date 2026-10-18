"""End-to-end: image stack → CytoPipelineWorker → cell dicts.

Runs the worker synchronously (no TaskScheduler), with real classical
pipelines and a fake one, and — opt-in, marked `slow` — the real Cellpose-SAM
model when it is already cached on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fixtures import images as img
from fixtures.segmenters import FakeSegmenter, square_labels

from karcytics_plugins.cytometrics.analysis.segment.watershed import WatershedSegmenter
from karcytics_plugins.cytometrics.ui.workers import CytoPipelineWorker

pytestmark = pytest.mark.functional

PARAMS = {"target_channel": "cells", "min_area_px": 50, "max_area_px": 100_000}
CELLPOSE_MODEL_DIR = Path.home() / ".cellpose" / "models"


def _run(pipeline, stack, params=PARAMS, scale=1.0) -> dict:
    worker = CytoPipelineWorker()
    worker.configure(pipeline, stack, params, scale)
    return worker.run()


def test_worker_segments_measures_and_scales():
    fake = FakeSegmenter(square_labels([(10, 10, 10), (40, 40, 12)]))
    stack = img.stack(("cells", img.blank()))

    result = _run(fake, stack, scale=0.5)

    assert fake.calls == [(stack, PARAMS)]
    assert [c["id"] for c in result["result_cells"]] == [1, 2]
    table = result["measurements"]
    assert table["AreaShape_Area"].tolist() == pytest.approx([100 * 0.25, 144 * 0.25])
    assert result["labels"].max() == 2


def test_worker_reports_progress_up_to_100():
    worker = CytoPipelineWorker()
    worker.configure(FakeSegmenter(), img.stack(("cells", img.blank())), PARAMS, 1.0)
    seen: list[int] = []
    worker.signals.analysis_progress.connect(seen.append)

    worker.run()

    assert seen[0] == 36 and seen[-1] == 100  # 45% of segmenting = 36% overall
    assert seen == sorted(seen)


def test_worker_reports_segmenter_errors_instead_of_raising():
    result = _run(
        FakeSegmenter(error=RuntimeError("GPU fell over")), img.stack(("cells", img.blank()))
    )

    assert result == {"error": "GPU fell over"}


def test_worker_without_a_pipeline_reports_an_error():
    assert CytoPipelineWorker().run() == {"error": "No pipeline configured"}


def test_watershed_through_the_worker_finds_every_isolated_disc():
    result = _run(WatershedSegmenter(), img.stack(("cells", img.isolated_discs())))

    assert len(result["result_cells"]) == len(img.ISOLATED_CENTRES)


@pytest.mark.slow
@pytest.mark.skipif(
    not (CELLPOSE_MODEL_DIR / "cpsam").exists(),
    reason="Cellpose-SAM model not cached (~1.3 GB); never downloaded in tests",
)
def test_cellpose_sam_finds_isolated_discs():
    from karcytics_plugins.cytometrics.analysis.segment.cellpose import CellposeSegmenter

    params = {**PARAMS, "diameter": 2 * img.ISOLATED_RADIUS, "exclude_borders": True}
    result = _run(CellposeSegmenter(), img.stack(("cells", img.isolated_discs())), params)

    assert len(result["result_cells"]) == len(img.ISOLATED_CENTRES)
    assert all(0.5 < c["confidence"] <= 1.0 for c in result["result_cells"])


@pytest.mark.slow
@pytest.mark.skipif(
    not (CELLPOSE_MODEL_DIR / "cpsam").exists(),
    reason="Cellpose-SAM model not cached (~1.3 GB); never downloaded in tests",
)
def test_cellpose_sam_reports_progress_while_the_network_runs():
    import numpy as np

    from karcytics_plugins.cytometrics.analysis.segment.cellpose import CellposeSegmenter

    seen: list[int] = []
    plane = np.tile(img.isolated_discs(), (3, 3))  # several batches of tiles
    CellposeSegmenter().segment(
        img.stack(("cells", plane)), {"target_channel": "cells", "batch_size": 1}, seen.append
    )

    network = [v for v in seen if 15 < v < 80]  # noqa: PLR2004
    assert len(network) >= 2  # noqa: PLR2004
    assert seen == sorted(seen) and seen[-1] == 85  # noqa: PLR2004
