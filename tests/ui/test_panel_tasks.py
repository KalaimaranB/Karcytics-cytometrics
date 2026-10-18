"""Background-task wiring of the panel (SDK submit_task)."""

from __future__ import annotations

import pytest

from karcytics_plugins.cytometrics.analysis.session import LastRun
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


def test_pipeline_cells_replace_the_old_ones_and_set_the_counter(panel):
    panel.state.cells = [{"id": 9, "points": [[0, 0], [2, 0], [2, 2]]}]
    cells = [{"id": i, "points": [[0, 0], [2, 0], [2, 2]]} for i in (1, 2)]

    panel._on_pipeline_result({"result_cells": cells})

    assert [c["id"] for c in panel.state.cells] == [1, 2]
    assert panel.state.cell_counter == 2


def test_detect_step_credits_the_selected_algorithm(panel, qtbot):
    qtbot.waitUntil(lambda: panel.tabs.tabText(1) == "2. Smart Detect", timeout=10_000)

    for key in ("otsu", "watershed", "cellpose"):
        panel.combo_pipeline.setCurrentIndex(panel.combo_pipeline.findData(key))
        visible = [k for k, label in panel._citation_labels.items() if not label.isHidden()]
        assert visible == [key]
    assert "Pachitariu et al., 2025" in panel._citation_labels["cellpose"].text()
    assert "Otsu, 1979" in panel._citation_labels["otsu"].text()


def test_csv_export_writes_methods_and_bibtex(panel, qtbot, tmp_path, monkeypatch):
    qtbot.waitUntil(lambda: panel.tabs.tabText(1) == "2. Smart Detect", timeout=10_000)
    export = tmp_path / "cells.csv"
    monkeypatch.setattr(main_panel, "get_save_path", lambda *a, **k: str(export))
    monkeypatch.setattr(main_panel, "show_info", lambda *a, **k: None)
    params = {"target_channel": "GFP", "min_area_px": 40, "max_area_px": 4000}
    monkeypatch.setattr(main_panel, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    panel.state.scale = 0.5
    panel.session.last_run = LastRun(panel.pipelines["otsu"], params)
    panel._on_pipeline_result({"result_cells": [{"id": 1, "points": [[0, 0], [2, 0], [2, 2]]}]})

    panel._on_export_csv()

    assert export.exists()
    assert "Otsu thresholding" in (tmp_path / "cells_methods.md").read_text(encoding="utf-8")
    assert "otsu1979" in (tmp_path / "cells_citations.bib").read_text(encoding="utf-8")
