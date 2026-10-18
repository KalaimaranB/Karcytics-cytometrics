"""FCS 3.1 export, read back with FlowIO (the reader behind FlowKit, which flow uses)."""

from __future__ import annotations

import flowio
import numpy as np
import pandas as pd
import pytest

from karcytics_plugins.cytometrics.analysis.fcs import numeric_columns, write_fcs


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ObjectNumber": [1, 2, 3],
            "Source": ["auto", "manual", "auto"],
            "AreaShape_Area": [10.5, 200.25, 3000.0],
            "Intensity_MeanIntensity_GFP/FITC": [0.5, np.nan, 12.0],
            "Gate_Big": [False, True, True],
        }
    )


def test_numeric_columns_skip_text(df):
    assert numeric_columns(df) == [
        "ObjectNumber",
        "AreaShape_Area",
        "Intensity_MeanIntensity_GFP/FITC",
        "Gate_Big",
    ]


def test_flowio_reads_every_event_and_parameter(df, tmp_path):
    path = write_fcs(
        tmp_path / "cells.fcs",
        df,
        labels={"AreaShape_Area": "Area (µm²)"},
        extra={"$SRC": "plate/A1.tif"},
    )
    fcs = flowio.FlowData(str(path))

    assert fcs.event_count == 3
    assert fcs.channel_count == 4
    assert fcs.pnn_labels == numeric_columns(df)  # a "/" in a name survives escaping
    assert fcs.pns_labels[1] == "Area (µm²)"
    assert fcs.text["src"] == "plate/A1.tif"

    events = fcs.as_array(preprocess=False)
    np.testing.assert_allclose(events[:, 1], [10.5, 200.25, 3000.0])
    assert np.isnan(events[1, 2])
    assert events[:, 3].tolist() == [0.0, 1.0, 1.0]


def test_header_offsets_match_the_text_segment(df, tmp_path):
    raw = write_fcs(tmp_path / "cells.fcs", df).read_bytes()
    assert raw[:6] == b"FCS3.1"
    text_start, text_end, data_start, data_end = (
        int(raw[10 + 8 * i : 18 + 8 * i]) for i in range(4)
    )

    assert raw[text_start : text_start + 1] == b"/"
    assert data_start == text_end + 1
    assert data_end - data_start + 1 == 3 * 4 * 4  # events × parameters × float32
    assert data_end == len(raw) - 1


def test_no_events_still_writes_a_valid_file(df, tmp_path):
    path = write_fcs(tmp_path / "empty.fcs", df.iloc[0:0])
    assert flowio.FlowData(str(path)).event_count == 0
