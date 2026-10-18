"""Gates, populations and their statistics on a small hand-made object table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from karcytics_plugins.cytometrics.analysis import gating
from karcytics_plugins.cytometrics.analysis.gating import ALL, Gate


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ObjectNumber": [1, 2, 3, 4, 5, 6],
            "Area": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
            "GFP": [1.0, 100.0, 5.0, 200.0, 1000.0, 50.0],
        }
    )


def ids(df, mask) -> list[int]:
    return df.loc[mask, "ObjectNumber"].tolist()


def test_threshold_above_and_below(df):
    above = Gate("Big", "threshold", x="Area", params={"value": 25, "above": True})
    below = Gate("Small", "threshold", x="Area", params={"value": 25, "above": False})
    masks = gating.evaluate([above, below], df)

    assert ids(df, masks["Big"]) == [3, 4, 5]
    assert ids(df, masks["Small"]) == [1, 2]  # NaN is in neither


def test_range_and_rectangle(df):
    rng = Gate("Mid", "range", x="Area", params={"lo": 40, "hi": 20})  # order doesn't matter
    rect = Gate("Box", "rect", x="Area", y="GFP", params={"x0": 15, "x1": 45, "y0": 50, "y1": 250})
    masks = gating.evaluate([rng, rect], df)

    assert ids(df, masks["Mid"]) == [2, 3, 4]
    assert ids(df, masks["Box"]) == [2, 4]


def test_polygon_on_log_axis_tests_in_the_space_it_was_drawn(df):
    # The same triangle, drawn on a linear and on a log10 GFP axis. Its
    # sloping edge is straight in the space it was drawn in, so cell 2
    # (Area 20, GFP 100) is inside the linear one but outside the log one.
    points = [[15, 10], [55, 10], [55, 5000]]
    linear = Gate("Lin", "polygon", x="Area", y="GFP", params={"points": points})
    log = Gate("Log", "polygon", x="Area", y="GFP", y_log=True, params={"points": points})
    masks = gating.evaluate([linear, log], df)

    assert ids(df, masks["Lin"]) == [2, 4, 5]
    assert ids(df, masks["Log"]) == [4, 5]


def test_children_are_subsets_of_their_parent(df):
    big = Gate("Big", "threshold", x="Area", params={"value": 25})
    bright = Gate("Bright", "threshold", x="GFP", params={"value": 50}, parent="Big")
    masks = gating.evaluate([big, bright], df)

    assert ids(df, masks["Bright"]) == [4, 5]  # cell 2 is bright but not big


def test_boolean_gates(df):
    big = Gate("Big", "threshold", x="Area", params={"value": 25})
    bright = Gate("Bright", "threshold", x="GFP", params={"value": 50})
    both = Gate("Both", "and", params={"of": ["Big", "Bright"]})
    either = Gate("Either", "or", params={"of": ["Big", "Bright"]})
    dim = Gate("Dim", "not", params={"of": ["Bright"]})
    masks = gating.evaluate([both, either, dim, big, bright], df)  # any order

    assert ids(df, masks["Both"]) == [4, 5]
    assert ids(df, masks["Either"]) == [2, 3, 4, 5]
    assert ids(df, masks["Dim"]) == [1, 3, 6]


def test_cycles_and_missing_gates_are_errors(df):
    a = Gate("A", "and", params={"of": ["B"]})
    b = Gate("B", "and", params={"of": ["A"]})
    with pytest.raises(ValueError, match="itself"):
        gating.evaluate([a, b], df)
    with pytest.raises(ValueError, match="doesn't exist"):
        gating.evaluate([Gate("C", "not", params={"of": ["nope"]})], df)


def test_stats_counts_and_percentages(df):
    big = Gate("Big", "threshold", x="Area", params={"value": 25})
    bright = Gate("Bright", "threshold", x="GFP", params={"value": 50}, parent="Big")
    table = gating.stats([big, bright], df).set_index("Population")

    assert table.loc[ALL, "Count"] == 6
    assert table.loc["Big", "Count"] == 3
    assert table.loc["Big", "PctTotal"] == pytest.approx(50.0)
    assert table.loc["Bright", "PctParent"] == pytest.approx(200 / 3)
    assert table.loc["Bright", "Parent"] == "Big"


def test_deepest_population_colours_a_cell(df):
    big = Gate("Big", "threshold", x="Area", params={"value": 25}, color="#111111")
    bright = Gate(
        "Bright", "threshold", x="GFP", params={"value": 50}, parent="Big", color="#222222"
    )
    colors = gating.colors_by_population([bright, big], df)

    assert colors == {3: "#111111", 4: "#222222", 5: "#222222"}


def test_rename_and_remove_follow_references():
    gates = [
        Gate("Big", "threshold", x="Area"),
        Gate("Bright", "threshold", x="GFP", parent="Big"),
        Gate("Both", "and", params={"of": ["Big", "Bright"]}),
        Gate("Other", "threshold", x="GFP"),
    ]
    renamed = gating.rename(gates, "Big", "Large")
    assert [g.name for g in renamed][0] == "Large"
    assert renamed[1].parent == "Large"
    assert renamed[2].operands == ["Large", "Bright"]

    assert [g.name for g in gating.remove(gates, "Big")] == ["Other"]


def test_gate_columns_round_trip_and_names(df):
    gates = [Gate("Big", "threshold", x="Area", params={"value": 25})]
    out = gating.with_gate_columns(gates, df)
    assert out["Gate_Big"].tolist() == [False, False, True, True, True, False]

    again = [Gate.from_dict(g.to_dict()) for g in gates]
    assert again == gates
    assert gating.unique_name(gates, "Big") == "Big 2"
    assert gating.unique_name(gates, ALL) == f"{ALL} 2"
    assert gating.next_color(gates) != gates[0].color
