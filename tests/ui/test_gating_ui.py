"""The Gate step: drawing gates on the plots, the population tree, colours, undo and export."""

from __future__ import annotations

import flowio
import pandas as pd
import pytest
from fixtures import images as img
from PIL import Image
from PySide6.QtCore import QPoint, Qt

from karcytics_plugins.cytometrics.analysis.gating import ALL, Gate
from karcytics_plugins.cytometrics.analysis.measure import measure_cells
from karcytics_plugins.cytometrics.analysis.session import LastRun
from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui


def _rect(cell_id, x, y, size):
    return {
        "id": cell_id,
        "points": [[x, y], [x + size - 1, y], [x + size - 1, y + size - 1], [x, y + size - 1]],
    }


@pytest.fixture
def panel(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(main_panel, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    p.resize(1200, 900)
    qtbot.waitUntil(lambda: bool(p.pipelines), timeout=10_000)
    p.set_assay("cells")
    path = tmp_path / "cells.png"
    Image.fromarray(img.isolated_discs()).save(path)
    p.channel_manager.ingest_path(str(path))
    # Areas 64, 100 and 144 px².
    cells = [_rect(1, 10, 10, 8), _rect(2, 40, 10, 10), _rect(3, 100, 100, 12)]
    p._on_pipeline_result(
        {"result_cells": cells, "measurements": measure_cells(cells, p.image_set, 0.0)}
    )
    p.session.last_run = LastRun(p.pipelines["otsu"], {"target_channel": "cells.png"})
    p.stepper.go_to("gate")
    yield p
    p.cleanup()


def _counts(panel) -> dict[str, int]:
    out = {}
    tree = panel.gate_step.tree

    def walk(item):
        out[item.text(0)] = int(item.text(1))
        for i in range(item.childCount()):
            walk(item.child(i))

    for i in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(i))
    return out


def _add_threshold(panel, value, above=True):
    panel.results.combo_hist_metric.setCurrentIndex(
        panel.results.combo_hist_metric.findData("AreaShape_Area")
    )
    panel.gate_step.tool_buttons["threshold"].click()
    panel.results.histogram.gate_drawn.emit("threshold", {"value": value, "above": above})


def test_threshold_gate_from_the_histogram(panel):
    _add_threshold(panel, 80)

    assert _counts(panel) == {ALL: 3, "Area+": 2}
    assert not panel.gate_step.tool_buttons["threshold"].isChecked()  # one gate per click
    assert panel.can_undo
    # Cells on the image are coloured by population right away.
    assert panel.combo_color_by.currentData() == "__population__"
    assert "Area+" in panel.review_step.lbl_color_legend.text()


def test_clicking_the_histogram_places_the_cut_off(panel, qtbot):
    hist = panel.results.histogram
    panel.results.setCurrentWidget(panel.results.histogram_page)
    panel.gate_step.tool_buttons["threshold"].click()
    hist.resize(500, 250)
    hist._plot_rect()
    x = int(hist.axis.to_screen(120.0))
    qtbot.mouseClick(hist, Qt.MouseButton.LeftButton, pos=QPoint(x, 100))

    gate = panel.session.gates[0]
    assert gate.kind == "threshold"
    assert gate.params["value"] == pytest.approx(120, abs=5)
    assert _counts(panel)["Area+"] == 1


def test_rectangle_gate_dragged_on_the_scatter(panel, qtbot):
    scatter = panel.results.scatter
    panel.results.combo_x.setCurrentIndex(panel.results.combo_x.findData("AreaShape_Area"))
    panel.results.combo_y.setCurrentIndex(panel.results.combo_y.findData("AreaShape_Perimeter"))
    panel.gate_step.tool_buttons["rect"].click()
    assert panel.results.currentWidget() is panel.results.scatter_page
    scatter.resize(500, 300)
    a = scatter._to_screen(90, 0).toPoint()
    b = scatter._to_screen(200, 1000).toPoint()
    a.setY(20)
    b.setY(260)
    qtbot.mousePress(scatter, Qt.MouseButton.LeftButton, pos=a)
    qtbot.mouseMove(scatter, b)
    qtbot.mouseRelease(scatter, Qt.MouseButton.LeftButton, pos=b)

    assert _counts(panel) == {ALL: 3, "Population": 2}
    assert panel.session.gates[0].kind == "rect"


def test_nested_gate_goes_inside_the_selected_population(panel):
    _add_threshold(panel, 80)  # cells 2 and 3; now selected
    panel.gate_step.tool_buttons["threshold"].click()
    # While drawing, the plots show the selected population's own cells.
    assert "Showing Area+: 2 cells" in panel.results.lbl_hist_population.text()
    panel.results.histogram.gate_drawn.emit("threshold", {"value": 120, "above": True})

    gates = {g.name: g for g in panel.session.gates}
    assert gates["Area+ 2"].parent == "Area+"
    assert _counts(panel)["Area+ 2"] == 1
    # Selected, a gate is shown over its parent's cells.
    assert "Showing Area+: 2 cells" in panel.results.lbl_hist_population.text()


def test_editing_the_cut_off_updates_counts_and_can_be_undone(panel):
    _add_threshold(panel, 80)
    panel.gate_step.combo_side.setCurrentIndex(1)  # below
    assert _counts(panel) == {ALL: 3, "Area−": 1}
    panel.gate_step.spin_value.setValue(120)
    assert _counts(panel)["Area−"] == 2

    panel.undo()
    assert _counts(panel)["Area−"] == 1
    panel.undo()
    assert _counts(panel) == {ALL: 3, "Area+": 2}


def test_combine_and_delete(panel, monkeypatch):
    _add_threshold(panel, 80)
    _add_threshold(panel, 120)  # inside Area+ (selected)
    panel.session.add_gate(Gate("Small", "not", params={"of": ["Area+"]}))
    panel.gating.refresh()
    assert _counts(panel)["Small"] == 1

    monkeypatch.setattr(main_panel, "ask_yes_no", lambda *a: True, raising=False)
    from karcytics_plugins.cytometrics.ui import gating as gating_ui

    monkeypatch.setattr(gating_ui, "ask_yes_no", lambda *a: True)
    panel.gating._population = "Area+"
    panel.gating._on_delete()
    assert _counts(panel) == {ALL: 3}  # its child and the NOT gate went with it


def test_gates_are_saved_and_restored(panel):
    _add_threshold(panel, 80)
    saved = panel.get_state().to_dict()

    panel.load_state({**saved, "gates": []})
    assert _counts(panel) == {ALL: 3}
    panel.load_state(saved)
    assert _counts(panel) == {ALL: 3, "Area+": 2}


def test_exports_carry_the_gates(panel, monkeypatch, tmp_path):
    _add_threshold(panel, 80)
    monkeypatch.setattr(main_panel, "show_info", lambda *a, **k: None)

    def save_as(name):
        monkeypatch.setattr(main_panel, "get_save_path", lambda *a, **k: str(tmp_path / name))

    save_as("cells.csv")
    panel._on_export_csv()
    table = pd.read_csv(tmp_path / "cells.csv")
    assert table["Gate_Area+"].tolist() == [False, True, True]

    save_as("cells.fcs")
    panel._on_export_fcs()
    fcs = flowio.FlowData(str(tmp_path / "cells.fcs"))
    assert fcs.event_count == 3
    assert "Gate_Area+" in fcs.pnn_labels
    assert "AreaShape_Area" in fcs.pnn_labels
    assert "Area (px²)" in fcs.pns_labels
    assert (tmp_path / "cells_methods.md").exists()

    save_as("populations.csv")
    panel._on_export_stats()
    stats = pd.read_csv(tmp_path / "populations.csv")
    assert stats["Count"].tolist() == [3, 2]

    methods = (tmp_path / "cells_methods.md").read_text()
    assert "manual gating" in methods and "Area+" in methods
