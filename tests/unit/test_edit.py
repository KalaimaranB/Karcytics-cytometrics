"""Cell edits: merge, split, brush and region re-detection."""

from __future__ import annotations

import numpy as np
import pytest

from karcytics_plugins.cytometrics.analysis import edit
from karcytics_plugins.cytometrics.analysis.model import Channel, ImageSet
from karcytics_plugins.cytometrics.analysis.session import AnalysisSession

pytestmark = pytest.mark.unit

SHAPE = (64, 64)


def _rect(cell_id, x, y, w, h, **extra):
    points = [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]]
    return {"id": cell_id, "points": points, **extra}


def _area(points) -> int:
    return int(edit.cell_mask({"points": points}, SHAPE).sum())


@pytest.fixture
def session():
    s = AnalysisSession()
    s.image_set.channels.append(Channel("c", np.zeros(SHAPE, np.uint8)))
    s.replace_cells(
        [_rect(1, 10, 10, 10, 10, confidence=0.9), _rect(2, 20, 10, 10, 10), _rect(3, 45, 45, 8, 8)]
    )
    return s


class TestGeometry:
    def test_merge_touching(self):
        outline = edit.merge([_rect(1, 10, 10, 10, 10), _rect(2, 20, 10, 10, 10)], SHAPE)

        assert outline is not None and _area(outline) == 200

    def test_merge_bridges_a_small_gap_but_not_a_large_one(self):
        near = edit.merge([_rect(1, 10, 10, 10, 10), _rect(2, 22, 10, 10, 10)], SHAPE)
        far = edit.merge([_rect(1, 10, 10, 10, 10), _rect(2, 40, 10, 10, 10)], SHAPE)

        assert near is not None
        assert far is None

    def test_split_along_a_line(self):
        pieces = edit.split(_rect(1, 10, 10, 20, 10), (20, 0), (20, 30), SHAPE)

        assert len(pieces) == 2
        assert sorted(_area(p) for p in pieces) == [90, 100]

    def test_a_diagonal_cut_separates_too(self):
        pieces = edit.split(_rect(1, 10, 10, 20, 20), (5, 5), (35, 35), SHAPE)

        assert len(pieces) == 2

    def test_a_line_that_misses_does_not_split(self):
        assert len(edit.split(_rect(1, 10, 10, 20, 10), (0, 40), (60, 40), SHAPE)) == 1

    def test_brush_adds_only_what_touches_the_cell(self):
        cell = _rect(1, 10, 10, 10, 10)

        grown = edit.paint(cell, [(19, 15), (25, 15)], 2, True, SHAPE)
        detached = edit.paint(cell, [(50, 50)], 3, True, SHAPE)

        assert len(grown) == 1 and _area(grown[0]) > 100
        assert len(detached) == 1 and _area(detached[0]) == 100

    def test_brush_erase_can_cut_a_cell_or_remove_it(self):
        cell = _rect(1, 10, 10, 20, 10)

        cut = edit.paint(cell, [(20, 5), (20, 25)], 1, False, SHAPE)
        gone = edit.paint(cell, [(10, 15), (30, 15)], 8, False, SHAPE)

        assert len(cut) == 2
        assert gone == []

    def test_crop_and_offset(self):
        images = ImageSet([Channel("c", np.arange(64 * 64).reshape(SHAPE))])

        sub = edit.crop(images, (60, 50, 10, 40))  # corners in any order, clipped

        assert sub.shape == (10, 50)
        assert sub.channels[0].data[0, 0] == 40 * 64 + 10
        moved = edit.offset_cells([_rect(1, 0, 0, 2, 2)], 10, 40)
        assert moved[0]["points"][0] == [10, 40]


class TestSession:
    def test_merge_replaces_the_cells_with_a_new_one(self, session):
        new_id = session.merge_cells({1, 2})

        assert new_id == 4
        assert sorted(c["id"] for c in session.cells) == [3, 4]
        assert session.cell(4)["source"] == "edited"

    def test_merging_apart_cells_changes_nothing(self, session):
        assert session.merge_cells({1, 3}) is None
        assert len(session.cells) == 3

    def test_split(self, session):
        assert session.split_cell(1, (15, 0), (15, 30)) == [4, 5]
        assert session.cell(1) is None

    def test_brush_keeps_the_id_and_drops_ai_confidence(self, session):
        assert session.paint_cell(1, [(19, 12)], 2, True) == [1]
        assert "confidence" not in session.cell(1)
        assert session.cell(1)["source"] == "edited"

    def test_replace_region(self, session):
        found = [{"points": [[11, 11], [14, 11], [14, 14]], "confidence": 0.7}]

        ids = session.replace_region((0, 0, 32, 32), found)

        assert ids == [4]
        assert sorted(c["id"] for c in session.cells) == [3, 4]
        assert session.cell(4)["confidence"] == 0.7
