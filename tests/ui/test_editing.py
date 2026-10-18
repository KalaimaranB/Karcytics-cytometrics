"""Edit tools on the canvas, region runs and undo/redo."""

from __future__ import annotations

import pytest
from fixtures import images as img
from PIL import Image
from PySide6.QtCore import QPointF, Qt

from karcytics_plugins.cytometrics.analysis.measure import measure_cells
from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui


@pytest.fixture
def panel(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(main_panel, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    qtbot.waitUntil(lambda: bool(p.pipelines), timeout=10_000)
    p.set_assay("cells")
    path = tmp_path / "cells.png"
    Image.fromarray(img.isolated_discs()).save(path)
    p.channel_manager.ingest_path(str(path))
    cells = [_rect(1, 10, 10, 10, 10), _rect(2, 20, 10, 10, 10), _rect(3, 100, 100, 12, 12)]
    p._on_pipeline_result(
        {"result_cells": cells, "measurements": measure_cells(cells, p.image_set, 0.0)}
    )
    yield p
    p.cleanup()


def _rect(cell_id, x, y, w, h):
    return {
        "id": cell_id,
        "points": [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
    }


def _ids(panel):
    return sorted(c["id"] for c in panel.state.cells)


def test_click_selects_and_shift_click_adds(panel, qtbot):
    canvas = panel.canvas
    canvas.resize(600, 600)
    canvas.fit_to_view()
    viewport = canvas.viewport()

    def click(x, y, modifier=Qt.KeyboardModifier.NoModifier):
        qtbot.mouseClick(
            viewport, Qt.MouseButton.LeftButton, modifier, canvas.mapFromScene(QPointF(x, y))
        )

    click(15, 15)
    assert canvas.selection == [1]
    click(25, 15, Qt.KeyboardModifier.ShiftModifier)
    assert canvas.selection == [1, 2]
    assert panel.review_step.btn_merge.isEnabled()
    click(200, 200)
    assert canvas.selection == []


def test_merge_then_undo_and_redo(panel):
    panel.canvas.set_selection({1, 2}, emit=True)

    panel.review_step.btn_merge.click()

    assert _ids(panel) == [3, 4]
    assert panel.can_undo() and panel.undo_text() == "Undo Merge 2 cells"
    panel.undo()
    assert _ids(panel) == [1, 2, 3]
    panel.redo()
    assert _ids(panel) == [3, 4]


def test_undo_stays_on_the_current_step(panel):
    panel.stepper.go_to("review")
    panel.canvas.set_selection({3}, emit=True)
    panel.review_step.btn_delete.click()

    panel.undo()

    assert panel.stepper.current_id == "review"
    assert _ids(panel) == [1, 2, 3]


def test_split_with_a_line_across_a_cell(panel):
    panel.canvas.split_drawn.emit(((106, 90), (106, 120)))

    assert 3 not in _ids(panel) and len(_ids(panel)) == 4


def test_brush_needs_one_selected_cell(panel):
    messages = []
    panel.status_message.connect(messages.append)

    panel.canvas.brush_painted.emit(([(15, 15)], 3, False))
    assert "Select one cell" in messages[-1]

    panel.canvas.set_selection({3}, emit=True)
    panel.canvas.brush_painted.emit(([(100, 100), (111, 111)], 2, False))
    assert panel.state.cells[-1]["source"] in ("edited", "auto")
    assert panel.can_undo()


def test_box_re_detect_replaces_the_cells_in_the_box(panel, qtbot):
    panel.combo_pipeline.setCurrentIndex(panel.combo_pipeline.findData("otsu"))
    panel.params_form.set_display_values({"min_area_px": 50})
    panel.canvas.box_drawn.emit((0, 0, 90, 90))  # cells 1 and 2 and the first disc

    qtbot.waitUntil(
        lambda: any(c.get("source") == "auto" and c["id"] > 3 for c in panel.state.cells),
        timeout=10_000,
    )
    assert 1 not in _ids(panel) and 2 not in _ids(panel)  # centred in the box, replaced
    assert 3 in _ids(panel)
    assert panel.undo_text() == "Undo Re-detect region"


def test_preview_shows_dashed_outlines_and_keeps_nothing(panel, qtbot):
    panel.combo_pipeline.setCurrentIndex(panel.combo_pipeline.findData("otsu"))
    panel.params_form.set_display_values({"min_area_px": 50})
    before = _ids(panel)

    panel.detect_step.btn_preview.click()

    qtbot.waitUntil(lambda: len(panel.canvas.layer("preview").childItems()) > 0, timeout=10_000)
    assert _ids(panel) == before
    assert "Preview" in panel.detect_step.lbl_ai_status.text()


def test_tools_switch_canvas_modes(panel):
    panel.review_step.tool_buttons["SPLIT"].click()
    assert panel.canvas.mode == "SPLIT"
    panel.review_step.tool_buttons["PAN"].click()
    assert panel.canvas.mode == "PAN"


def test_inspector_shows_the_selected_cell(panel):
    panel.canvas.set_selection({3}, emit=True)

    inspector = panel.results.inspector
    assert inspector.title.text() == "Cell 3"
    names = [inspector.values.item(i, 0).text() for i in range(inspector.values.rowCount())]
    assert "Area (px²)" in names and "MaxFeretDiameter (px)" in names
    assert inspector.thumbnail.pixmap() is not None and not inspector.thumbnail.pixmap().isNull()

    panel.canvas.set_selection({1, 2}, emit=True)
    assert inspector.title.text() == "2 cells selected"


def test_flag_is_saved_shown_and_undoable(panel):
    panel.canvas.set_selection({3}, emit=True)

    panel.results.inspector.btn_flag.click()

    assert panel.session.cell(3)["flagged"] is True
    assert panel.measurements.set_index("ObjectNumber").loc[3, "Flagged"]
    assert next(i for i in panel.canvas._cell_items if i.cell_id == 3).flagged
    assert panel.undo_text() == "Undo Flag cell"
    panel.undo()
    assert "flagged" not in panel.session.cell(3)


def test_scatter_plots_cells_and_links_selection_both_ways(panel):
    scatter = panel.results.scatter
    panel.results.setCurrentIndex(2)
    scatter.resize(400, 300)
    panel._update_scatter()
    assert scatter.point_count == 3

    panel._on_scatter_clicked(2, False)
    assert panel.canvas.selection == [2]
    panel._on_scatter_clicked(3, True)
    assert panel.canvas.selection == [2, 3]

    panel.canvas.set_selection({1}, emit=True)
    assert scatter._selected == {1}


def test_scatter_point_hit_testing(qtbot):
    from karcytics_plugins.cytometrics.ui.widgets.scatter import ScatterPlot

    plot = ScatterPlot()
    qtbot.addWidget(plot)
    plot.resize(300, 200)
    plot.set_data([1, 2, 3], [1, 4, 9], [7, 8, 9], "x", "y")

    hit = plot.point_at(plot._to_screen(2, 4))

    assert hit == 8
    assert plot.point_at(plot._to_screen(2, 9)) is None
