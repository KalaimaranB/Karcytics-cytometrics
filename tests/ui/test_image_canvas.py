"""MultiChannelCanvas on top of the SDK ImageView: cell tools and modes."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QGraphicsView

from karcytics_plugins.cytometrics.ui.image_canvas import CellPolygonItem, MultiChannelCanvas

pytestmark = pytest.mark.ui

SQUARE = [[20, 20], [60, 20], [60, 60], [20, 60]]


@pytest.fixture
def canvas(qtbot):
    c = MultiChannelCanvas()
    qtbot.addWidget(c)
    c.resize(600, 600)
    c.set_array(np.zeros((100, 100, 3), np.uint8))
    return c


def _viewport_point(canvas, x, y) -> QPoint:
    return canvas.mapFromScene(x, y)


def test_pan_mode_pans_with_a_left_drag(canvas):
    canvas.set_mode("PAN")
    assert canvas.dragMode() == QGraphicsView.DragMode.ScrollHandDrag


def test_draw_mode_emits_the_outlined_polygon(canvas, qtbot):
    canvas.set_mode("DRAW")
    assert canvas.dragMode() == QGraphicsView.DragMode.NoDrag
    vp = canvas.viewport()
    points = [_viewport_point(canvas, x, y) for x, y in SQUARE]

    with qtbot.waitSignal(canvas.cell_drawn) as blocker:
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, pos=points[0])
        for p in points[1:]:
            QTest.mouseMove(vp, p)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=points[-1])

    assert len(blocker.args[0]) == len(SQUARE)
    assert blocker.args[0][0] == pytest.approx((20, 20), abs=1)


def test_right_click_on_a_cell_or_its_label_deletes_it(canvas, qtbot):
    canvas.draw_cells_from_state([{"id": 7, "points": SQUARE}])

    with qtbot.waitSignal(canvas.cell_deleted) as blocker:
        QTest.mouseClick(
            canvas.viewport(), Qt.MouseButton.RightButton, pos=_viewport_point(canvas, 25, 25)
        )

    assert blocker.args == [7]


def test_calibrate_mode_reports_the_line_then_returns_to_pan(canvas, qtbot):
    canvas.set_mode("CALIBRATE")
    vp = canvas.viewport()
    a, b = _viewport_point(canvas, 10, 50), _viewport_point(canvas, 90, 50)

    with qtbot.waitSignal(canvas.calibration_line_drawn) as blocker:
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, pos=a)
        QTest.mouseMove(vp, b)
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, pos=b)

    assert blocker.args[0] == pytest.approx(80, abs=1)
    assert canvas.mode == "PAN"
    assert canvas.dragMode() == QGraphicsView.DragMode.ScrollHandDrag


def test_redrawing_cells_replaces_the_old_outlines(canvas):
    canvas.draw_cells_from_state([{"id": 1, "points": SQUARE}, {"id": 2, "points": SQUARE}])
    canvas.draw_cells_from_state([{"id": 3, "points": SQUARE}])

    items = [i for i in canvas.scene().items() if isinstance(i, CellPolygonItem)]
    assert [i.cell_id for i in items] == [3]


def test_highlight_and_hide_ids(canvas):
    canvas.draw_cells_from_state([{"id": 1, "points": SQUARE}, {"id": 2, "points": SQUARE}])

    canvas.highlight_cell(2)
    canvas.set_show_ids(False)

    by_id = {i.cell_id: i for i in canvas._cell_items}
    assert by_id[2].zValue() > by_id[1].zValue()
    assert not by_id[1].text_item.isVisible()
