"""Background-task wiring of the panel (SDK submit_task)."""

from __future__ import annotations

import pytest

from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui


@pytest.fixture
def panel(qtbot):
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    yield p
    p.cleanup()


def test_pipeline_loader_unlocks_the_detect_tab(panel, qtbot):
    qtbot.waitUntil(lambda: panel.tabs.tabText(1) == "2. Smart Detect", timeout=10_000)

    assert set(panel.pipelines) == {"otsu", "watershed", "cellpose"}
    assert panel.combo_pipeline.count() == 3


def test_pipeline_exception_is_reported_as_an_error_not_as_no_cells(panel, monkeypatch):
    shown = []
    monkeypatch.setattr(
        main_panel, "show_error", lambda _parent, title, msg: shown.append(("error", title, msg))
    )
    monkeypatch.setattr(
        main_panel, "show_info", lambda _parent, title, msg: shown.append(("info", title, msg))
    )

    panel._on_pipeline_result({"error": "CUDA out of memory"})

    assert shown == [("error", "Pipeline Error", "An error occurred:\nCUDA out of memory")]


def test_pipeline_cells_are_added_with_ids(panel):
    cells = [{"points": [[0, 0], [2, 0], [2, 2]], "area": 2.0, "perim": 6.8, "circ": 0.5}]

    panel._on_pipeline_result({"result_cells": cells})

    assert [c["id"] for c in panel.state.cells] == [1]
