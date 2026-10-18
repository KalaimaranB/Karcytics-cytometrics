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


# ── Colonies ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lighter", [True, False])
def test_colonies_counts_inside_the_dish(lighter):
    from fixtures import images as img

    from karcytics_plugins.cytometrics.analysis.segment import run_segmentation
    from karcytics_plugins.cytometrics.analysis.segment.colonies import ColonySegmenter

    plane, n = img.plate_photo(lighter)
    params = {p.key: p.default for p in ColonySegmenter().parameters()}
    params.update(target_channel="plate", min_separation_px=6)
    result = run_segmentation(ColonySegmenter(), img.stack(("plate", plane)), params)

    assert result.extras["polarity"] == ("lighter" if lighter else "darker")
    x, y, r = result.extras["dish"]
    assert abs(x - 300) < 8 and abs(y - 300) < 8 and 225 < r < 262
    assert abs(len(result.cells) - n) <= 2


def test_cfu_per_ml_and_countable_range():
    analysis = REGISTRY["colonies"].analysis
    table = pd.DataFrame({"ObjectNumber": range(1, 151)})
    settings = {"dilution_factor": 1e6, "volume_ul": 100}
    results = {r.key: r for r in analysis.results(table, [], settings, AssayContext())}
    assert results["Colonies"].value == 150
    assert results["CFU_per_mL"].value == pytest.approx(150 * 1e6 / 0.1)
    assert not results["CFU_per_mL"].warning

    few = {r.key: r for r in analysis.results(table.iloc[:10], [], settings, AssayContext())}
    assert "too few" in few["CFU_per_mL"].warning
    overlay = analysis.overlays(settings, AssayContext(extras={"dish": [1, 2, 3]}))
    assert overlay[0].kind == "circle" and overlay[0].coords == (1, 2, 3)
