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


# ── Bacteria ─────────────────────────────────────────────────────────────────


def _capsule(length, width, bend=0.0):
    """A rod `length` × `width` px along x; `bend` bows it."""
    import cv2

    canvas = np.zeros((80, 160), np.uint8)
    xs = np.linspace(40, 40 + length - width, 200)
    for x in xs:
        t = (x - xs[0]) / (xs[-1] - xs[0]) - 0.5
        y = 40 + bend * (1 - (2 * t) ** 2)
        cv2.circle(canvas, (int(round(x)), int(round(y))), int(round(width / 2)), 1, -1)
    return canvas.astype(bool)


def test_rod_measurements():
    from karcytics_plugins.cytometrics.analysis.measure.rods import rod_shape

    length, width, curvature, constriction = rod_shape(_capsule(60, 12))
    assert length == pytest.approx(60, rel=0.08)
    assert width == pytest.approx(12, rel=0.15)
    assert curvature < 0.03 and constriction > 0.8

    *_, bent_curvature, _ = rod_shape(_capsule(60, 10, bend=15))
    assert bent_curvature > 0.05

    # A dividing cell: two daughter rods meeting end to end, pinched where they meet.
    dividing = _capsule(34, 14) | np.roll(_capsule(34, 14), 32, axis=1)
    *_, pinched = rod_shape(dividing)
    assert pinched < 0.7


def test_bacteria_results_and_dividing_gate():
    analysis = REGISTRY["bacteria"].analysis
    table = pd.DataFrame(
        {
            "Rod_Length": [3.0, 4.0, 5.0, 6.0],
            "Rod_Width": [1.0, 1.0, 1.1, 0.9],
            "Rod_Curvature": [0.0, 0.1, 0.0, 0.2],
            "Rod_Dividing": [0.0, 1.0, 0.0, 1.0],
        }
    )
    results = {r.key: r.value for r in analysis.results(table, [], {}, AssayContext(scale=0.1))}
    assert results["MedianLength"] == 4.5 and results["PctDividing"] == 50
    gate = analysis.suggested_gates(table, {})[0]
    assert gating.evaluate([gate], table)["Dividing"].tolist() == [False, True, False, True]


# ── Transfection ─────────────────────────────────────────────────────────────


def test_negative_control_sets_the_cutoff_and_scores_samples():
    analysis = REGISTRY["transfection"].analysis
    rng = np.random.default_rng(1)
    control = pd.DataFrame({"Intensity_MeanIntensity_GFP": rng.normal(100, 10, 1000)})
    settings = {"reporter_channel": "GFP", "statistic": "MeanIntensity", "percentile": 99}

    outcome = analysis.run_action("set_control", control, settings, AssayContext())
    cutoff = outcome.settings["cutoff"]
    assert cutoff == pytest.approx(100 + 2.33 * 10, rel=0.03)
    assert outcome.gates[0].name == "Positive" and outcome.gates[0].params["value"] == cutoff

    sample = pd.DataFrame(
        {"Intensity_MeanIntensity_GFP": np.r_[rng.normal(100, 10, 700), rng.normal(400, 50, 300)]}
    )
    settings.update(outcome.settings)
    results = {
        r.key: r.value for r in analysis.results(sample, outcome.gates, settings, AssayContext())
    }
    assert results["PctPositive"] == pytest.approx(30, abs=1.5)
    assert "negative-control cells" in analysis.methods(settings, AssayContext())[0].description


def test_no_cutoff_asks_for_a_control():
    analysis = REGISTRY["transfection"].analysis
    table = pd.DataFrame({"Intensity_MeanIntensity_GFP": [1.0, 2.0]})
    (value,) = analysis.results(table, [], {"reporter_channel": "GFP"}, AssayContext())
    assert "negative control" in value.warning


# ── Hemocytometer ────────────────────────────────────────────────────────────


def test_grid_period_from_the_lines():
    from karcytics_plugins.cytometrics.analysis.assays.hemocytometer import grid_period

    rng = np.random.default_rng(2)
    image = rng.normal(150, 8, (512, 640))
    for x in np.arange(7.3, 640, 42.6):
        image[:, int(round(x))] -= 60
    for y in np.arange(20.0, 512, 42.6):
        image[int(round(y)), :] -= 60
    assert grid_period(image) == pytest.approx(42.6, rel=0.02)
    assert grid_period(rng.normal(150, 8, (300, 300))) is None


def test_viability_counts_inside_the_area_and_converts_to_cells_per_ml():
    analysis = REGISTRY["hemocytometer"].analysis
    table = pd.DataFrame(
        {
            "Location_Center_X": [10, 20, 30, 40, 500],
            "Location_Center_Y": [10, 20, 30, 40, 500],
            "Intensity_MeanIntensity_red": [180, 170, 60, 175, 50],
            "Intensity_Background_red": [150, 150, 150, 150, 150],
        }
    )
    settings = {
        "stain_channel": "red",
        "darkness_cutoff": 0.35,
        "dilution_factor": 2,
        "depth_um": 100,
        "counting_area": [0, 0, 100, 100],
    }
    context = AssayContext(scale=10.0)  # 100 px = 1 mm
    derived = analysis.derive(table, settings, context)
    assert derived["Viability_Darkness"].round(2).tolist() == [0, 0, 0.6, 0, 0.67]
    results = {r.key: r.value for r in analysis.results(derived, [], settings, context)}
    assert (results["Counted"], results["Dead"]) == (4, 1)  # the cell at (500, 500) is outside
    assert results["Viability"] == 75
    # 1 mm² × 0.1 mm = 0.1 µL = 1e-4 mL; 3 live / 1e-4 × 2 = 60,000 /mL
    assert results["LivePerML"] == pytest.approx(60_000)

    uncalibrated = {r.key: r for r in analysis.results(derived, [], settings, AssayContext())}
    assert "calibrate" in uncalibrated["LivePerML"].warning
    box = analysis.run_action("box:counting_area", derived, settings, context, box=(90, 80, 10, 5))
    assert box.settings["counting_area"] == [10, 5, 90, 80]
