"""The assays' own analyses, on synthetic tables and images."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from karcytics_sdk.plugin.citations import get

from karcytics_plugins.cytometrics import citations as _citations  # noqa: F401
from karcytics_plugins.cytometrics.analysis import gating
from karcytics_plugins.cytometrics.analysis.assays import REGISTRY, AssayContext

pytestmark = pytest.mark.unit


def test_every_assay_is_complete():
    for assay in REGISTRY.values():
        assert assay.steps[0] == "load" and "report" in assay.steps
        if assay.analysis is not None:
            assert "results" in assay.steps
            for key in (*assay.citations, *assay.analysis.citations):
                assert get(key).doi  # registered (raises KeyError if not)


# ── Cell cycle ───────────────────────────────────────────────────────────────


@pytest.fixture
def nuclei_table():
    rng = np.random.default_rng(0)
    g1 = rng.normal(1000, 50, 300)
    s = rng.uniform(1300, 1700, 100)
    g2 = rng.normal(2000, 80, 150)
    values = np.concatenate([g1, s, g2])
    return pd.DataFrame(
        {
            "ObjectNumber": np.arange(1, len(values) + 1),
            "Intensity_IntegratedIntensity_DAPI": values,
        }
    )


def test_cell_cycle_normalises_to_the_g1_peak_and_stages(nuclei_table):
    analysis = REGISTRY["nuclei"].analysis
    settings = {"dna_channel": "DAPI", "g1_peak": 0}
    derived = analysis.derive(nuclei_table, settings, AssayContext())

    assert analysis.g1_peak(nuclei_table, settings) == pytest.approx(1000, rel=0.05)
    assert derived["CellCycle_DNAContent"].iloc[:300].median() == pytest.approx(2.0, rel=0.05)
    results = {r.key: r.value for r in analysis.results(derived, [], settings, AssayContext())}
    assert results["Pct_G1"] == pytest.approx(300 / 550 * 100, abs=4)
    assert results["Pct_G2M"] == pytest.approx(150 / 550 * 100, abs=4)
    assert results["Pct_S"] == pytest.approx(100 / 550 * 100, abs=4)

    gates = analysis.suggested_gates(derived, settings)
    gated = {r.key: r.value for r in analysis.results(derived, gates, settings, AssayContext())}
    assert gated["Pct_G1"] == pytest.approx(results["Pct_G1"])
    stats = gating.stats(gates, derived).set_index("Population")
    assert stats["Count"].iloc[1:].sum() == 550  # every nucleus is in exactly one phase


def test_manual_g1_peak(nuclei_table):
    analysis = REGISTRY["nuclei"].analysis
    derived = analysis.derive(nuclei_table, {"g1_peak": 500}, AssayContext())
    assert derived["CellCycle_DNAContent"].iloc[:300].median() == pytest.approx(4.0, rel=0.05)
    assert "set to 500" in analysis.methods({"g1_peak": 500}, AssayContext())[0].description
