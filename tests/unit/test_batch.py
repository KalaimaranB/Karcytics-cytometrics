"""Batch runs: naming and grouping files, plate layouts, running, resuming and statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis import batch as bt
from karcytics_plugins.cytometrics.analysis.gating import Gate
from karcytics_plugins.cytometrics.analysis.segment.otsu import OtsuSegmenter

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "well"),
    [
        ("A01_DAPI.tif", "A01"),
        ("plate1_B3.tif", "B03"),
        ("scan-h12-field2.png", "H12"),
        ("LPS1.tif", None),  # "S1" is inside a word
        ("A1_B2.tif", None),  # two wells: ambiguous
        ("control.tif", None),
    ],
)
def test_parse_well(name, well):
    assert bt.parse_well(name) == well


def test_pattern_fields():
    fields = bt.parse_pattern("{well}_{condition}_r{replicate}", "B03_LPS 10ng_r2.tif")
    assert fields == {"Well": "B03", "Condition": "LPS 10ng", "Replicate": "2"}
    assert bt.parse_pattern("{well}_{condition}_r{replicate}", "nope.tif") is None


def test_layout_list_and_grid(tmp_path):
    listed = tmp_path / "list.csv"
    listed.write_text("Well,Condition,Dose\nA1,Control,0\nb02,LPS,10\n")
    assert bt.read_layout(listed) == {
        "A01": {"Condition": "Control", "Dose": "0"},
        "B02": {"Condition": "LPS", "Dose": "10"},
    }
    grid = tmp_path / "grid.csv"
    grid.write_text(",1,2,3\nA,Control,Control,LPS\nB,,LPS,\n")
    assert bt.read_layout(grid) == {
        "A01": {"Condition": "Control"},
        "A02": {"Condition": "Control"},
        "A03": {"Condition": "LPS"},
        "B02": {"Condition": "LPS"},
    }


def test_channel_tokens_and_grouping(tmp_path):
    assert bt.channel_tokens(["A01_DAPI.tif", "A01_GFP.tif"]) == ["DAPI", "GFP"]
    assert bt.channel_tokens(["img_c0.tif", "img_c1.tif"]) == ["c0", "c1"]
    assert bt.channel_tokens(["one.tif"]) == []

    names = ["B10_c1.tif", "B10_c0.tif", "A02_c0.tif", "A02_c1.tif", "A03_c0.tif", "notes.png"]
    groups, leftovers = bt.group_files([tmp_path / n for n in names], ["c0", "c1"])
    assert [(k, [f.name for f in files]) for k, files in groups] == [
        ("A02", ["A02_c0.tif", "A02_c1.tif"]),
        ("B10", ["B10_c0.tif", "B10_c1.tif"]),  # "c0" isn't found inside "B10"
    ]
    assert sorted(p.name for p in leftovers) == ["A03_c0.tif", "notes.png"]


def test_plate_shape_and_matrix():
    assert bt.plate_shape(["A01", "C04"]) == (3, 4)
    assert bt.plate_shape(["H12"]) == (8, 12)
    assert bt.plate_shape(["I01"]) == (16, 24)
    images = pd.DataFrame({"Well": ["A01", "A01", "B02"], "Count": [10, 20, 5]})
    grid = bt.plate_matrix(images, "Count", 2, 3)
    assert grid[0, 0] == 15 and grid[1, 1] == 5 and np.isnan(grid[0, 1])


def test_condition_summary_averages_replicates_first():
    images = pd.DataFrame(
        {
            "Condition": ["A", "A", "A", "B", "B"],
            "Replicate": ["1", "1", "2", "1", "2"],
            "Count": [10.0, 20.0, 30.0, 5.0, 7.0],
            "Status": ["done"] * 5,
        }
    )
    out = bt.condition_summary(images, "Count").set_index("Condition")
    # A: replicate means 15 and 30 → mean 22.5, n = 2 replicates.
    assert out.loc["A", "n"] == 2 and out.loc["A", "Mean"] == pytest.approx(22.5)
    assert out.loc["A", "SEM"] == pytest.approx(np.std([15, 30], ddof=1) / np.sqrt(2))
    assert out.loc["A", "Unit"] == "replicates"

    no_reps = bt.condition_summary(images.drop(columns="Replicate"), "Count").set_index("Condition")
    assert no_reps.loc["A", "n"] == 3 and no_reps.loc["A", "Unit"] == "images"


@pytest.fixture
def folder(tmp_path):
    """Three wells; B02 is unreadable."""
    for name in ("A01_ctrl.png", "A02_lps.png"):
        Image.fromarray(img.isolated_discs()).save(tmp_path / name)
    (tmp_path / "B02_lps.png").write_bytes(b"not an image")
    return tmp_path


def _recipe(**changes):
    base = {
        "segmenter": "otsu",
        "params": {"target_channel": "cells", "min_area_px": 50, "max_area_px": 1e9},
        "channel_names": ["cells"],
        "gates": [Gate("Big", "threshold", x="AreaShape_Area", params={"value": 0}).to_dict()],
    }
    return bt.Recipe(**{**base, **changes})


def test_batch_runs_saves_and_resumes(folder, tmp_path):
    recipe = _recipe()
    items, leftovers = bt.make_items(bt.find_images(folder), recipe, "{well}_{condition}")
    assert not leftovers
    assert [i.metadata for i in items] == [
        {"Well": "A01", "Condition": "ctrl"},
        {"Well": "A02", "Condition": "lps"},
        {"Well": "B02", "Condition": "lps"},
    ]
    batch = bt.Batch(str(tmp_path / "out"), recipe, items)

    bt.run_item(batch, items[0], OtsuSegmenter())
    batch.save()
    again = bt.Batch.load(batch.path)  # a stopped run picks up from the manifest
    assert [i.id for i in again.todo()] == ["A02_lps", "B02_lps"]

    for item in again.todo():
        bt.run_item(again, item, OtsuSegmenter())
    statuses = {i.id: i.status for i in again.items}
    assert statuses == {"A01_ctrl": "done", "A02_lps": "done", "B02_lps": "failed"}
    assert again.item("B02_lps").error

    n = len(img.ISOLATED_CENTRES)
    cells = bt.combined_cells(again)
    assert len(cells) == 2 * n
    assert list(cells.columns[:3]) == ["Image", "Well", "Condition"]
    assert cells["Gate_Big"].all()

    images = bt.image_summary(again, cells)
    assert images.set_index("Image").loc["A01_ctrl", "Count"] == n
    assert images.set_index("Image").loc["A01_ctrl", "Pct_Big"] == pytest.approx(100)
    assert np.isnan(images.set_index("Image").loc["B02_lps", "Count"])
    assert "Mean_AreaShape_Area" in images

    by_condition = bt.condition_summary(images, "Count").set_index("Condition")
    assert by_condition.loc["lps", "n"] == 1  # the failed image doesn't count


def test_items_need_the_recipe_channels(folder):
    recipe = _recipe(channel_names=["DAPI", "GFP"])
    item = bt.BatchItem("A01", [str(folder / "A01_ctrl.png")])
    with pytest.raises(ValueError, match="1 channel"):
        bt.load_item(item, recipe)
