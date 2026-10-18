"""ImageJ ROI and Cellpose _seg.npy exports."""

from __future__ import annotations

import numpy as np
import pytest
from roifile import roiread

from karcytics_plugins.cytometrics.analysis.export import (
    cells_to_labels,
    write_cellpose_seg,
    write_imagej_rois,
)

pytestmark = pytest.mark.unit


def _cell(cell_id, x, y, side=10, source=None):
    cell = {"id": cell_id, "points": [[x, y], [x + side, y], [x + side, y + side], [x, y + side]]}
    if source:
        cell["source"] = source
    return cell


CELLS = [_cell(3, 5, 5), _cell(7, 30, 30, source="manual")]


def test_cells_rasterise_to_sequential_labels():
    labels = cells_to_labels(CELLS, (50, 50))

    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[10, 10] == 1 and labels[35, 35] == 2
    assert (labels == 1).sum() == 11 * 11  # fillPoly includes the outline


def test_rois_round_trip_through_roifile(tmp_path):
    path = write_imagej_rois(tmp_path / "cells_rois.zip", CELLS)

    rois = roiread(path)

    assert [r.name for r in rois] == ["Cell 3", "Cell 7"]
    np.testing.assert_allclose(rois[1].coordinates(), CELLS[1]["points"])


def test_rois_overwrite_an_existing_zip(tmp_path):
    path = tmp_path / "cells_rois.zip"
    write_imagej_rois(path, CELLS)
    write_imagej_rois(path, CELLS[:1])

    assert len(roiread(path)) == 1


def test_cellpose_seg_has_the_keys_the_gui_reads(tmp_path):
    path = write_cellpose_seg(tmp_path / "img_seg.npy", CELLS, (50, 50), "/data/img.tif")

    data = np.load(path, allow_pickle=True).item()

    assert data["masks"].dtype == np.uint16 and data["masks"].max() == 2
    assert data["outlines"][5, 5] == 1 and data["outlines"][10, 10] == 0
    assert data["ismanual"].tolist() == [False, True]
    assert data["filename"] == "/data/img.tif"
