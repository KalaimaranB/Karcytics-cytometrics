"""AnalysisSession: the panel's logic without Qt."""

from __future__ import annotations

import pytest
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis.measure import measure_cells
from karcytics_plugins.cytometrics.analysis.segment.otsu import OtsuSegmenter
from karcytics_plugins.cytometrics.analysis.session import AnalysisSession, LastRun
from karcytics_plugins.cytometrics.analysis.state import CytoMetricsState, migrate

pytestmark = pytest.mark.unit


def _cell(cell_id, x, conf=None):
    cell = {"id": cell_id, "points": [[x, 5], [x + 9, 5], [x + 9, 14], [x, 14]]}
    if conf is not None:
        cell["confidence"] = conf
    return cell


@pytest.fixture
def session(tmp_path):
    s = AnalysisSession()
    path = tmp_path / "cells.png"
    Image.fromarray(img.isolated_discs()).save(path)
    s.image_set.add_file(path, "cells.png")
    cells = [_cell(1, 0, 0.9), _cell(2, 20, 0.3), _cell(3, 40, 0.6)]
    s.replace_cells(cells, measure_cells(cells, s.image_set, 0.0))
    return s


def test_replace_cells_sets_the_counter(session):
    assert session.state.cell_counter == 3
    assert not session.needs_measuring()


def test_manual_cells_get_the_next_id_and_need_measuring(session):
    cell = session.add_manual_cell([(1, 1), (5, 1), (5, 5)])

    assert cell["id"] == 4 and cell["source"] == "manual"
    assert session.needs_measuring()


def test_deleting_drops_the_cell_and_its_row(session):
    session.delete_cells({2})

    assert [c["id"] for c in session.cells] == [1, 3]
    assert session.measurements["ObjectNumber"].tolist() == [1, 3]
    assert "units" in session.measurements.attrs
    assert not session.needs_measuring()


def test_uncertain_cells_least_confident_first(session):
    assert [c["id"] for c in session.uncertain_cells(0.75)] == [2, 3]


def test_calibration_rescales_and_records_the_source(session):
    area = session.measurements["AreaShape_Area"].iloc[0]

    session.set_scale(0.5, "manual")

    assert session.measurements["AreaShape_Area"].iloc[0] == pytest.approx(area / 4)
    assert session.state.scale_source == "manual"


def test_metadata_never_overrides_a_set_scale(session):
    session.set_scale(0.2, "manual")

    assert not session.apply_metadata_calibration()


def test_reload_reports_missing_files(tmp_path):
    session = AnalysisSession()
    present = tmp_path / "a.png"
    Image.fromarray(img.blank(16)).save(present)

    missing = session.reload_images(
        [
            {"name": "DAPI", "path": str(present), "color": "blue"},
            {"name": "gone", "path": str(tmp_path / "gone.png"), "color": "red"},
        ]
    )

    assert missing == [str(tmp_path / "gone.png")]
    assert [(c.name, c.color) for c in session.image_set.channels] == [("DAPI", "blue")]


def test_methods_need_a_run(session):
    assert session.methods_steps() is None

    session.last_run = LastRun(OtsuSegmenter(), {"target_channel": "cells.png"})

    steps = session.methods_steps()
    assert steps is not None and "Otsu" in steps[1].description


def test_export_stem_strips_ome_suffix():
    session = AnalysisSession()
    assert session.export_stem() == "cytometrics"


class TestMigration:
    V1 = {
        "scale": 0.5,
        "cells": [
            {"id": 1, "points": [], "area": 4.0, "perim": 8.0, "circ": 0.7, "source": "manual"}
        ],
        "ui_params": {
            "pipeline": "Watershed (Clump Splitting)",
            "min_area": 20.0,
            "use_dual": True,
        },
    }

    def test_v1_is_migrated(self):
        state = CytoMetricsState.from_dict(self.V1)

        assert state.version == 2
        assert state.cells == [{"id": 1, "points": [], "source": "manual"}]
        assert state.ui_params == {
            "pipeline": "watershed",
            "values": {"min_area_px": 20.0, "use_dual_channel": True},
        }

    def test_migration_does_not_modify_its_input(self):
        migrate(self.V1)

        assert "area" in self.V1["cells"][0]

    def test_current_format_passes_through(self):
        data = CytoMetricsState(scale=0.3).to_dict()

        assert migrate(data) is data
