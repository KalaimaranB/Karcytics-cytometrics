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


def _sample(name, role, column, values, channel):
    from karcytics_plugins.cytometrics.analysis.experiment import Sample

    return Sample(name, role, pd.DataFrame({column: values}), fields={"reporter_channel": channel})


def test_negative_control_sets_the_cutoff_and_samples_are_compared():
    analysis = REGISTRY["transfection"].analysis
    rng = np.random.default_rng(1)
    settings = {"statistic": "MeanIntensity", "percentile": 99}
    # Each image's columns carry its own file name: the comparison follows each one's reporter.
    control = _sample(
        "Control",
        "control",
        "Intensity_MeanIntensity_ctrl_GFP",
        rng.normal(100, 10, 1000),
        "ctrl GFP",
    )
    values = np.r_[rng.normal(100, 10, 700), rng.normal(400, 50, 300)]
    sample = _sample("siRNA 1", "sample", "Intensity_MeanIntensity_s1_GFP", values, "s1 GFP")

    comparison = analysis.compare([control, sample], settings, AssayContext())
    cutoff = comparison.settings["cutoff"]
    assert cutoff == pytest.approx(100 + 2.33 * 10, rel=0.03)
    assert comparison.settings["control_cells"] == 1000
    by_key = {v.key: v for v in comparison.values}
    assert by_key["PctPositive_siRNA 1"].value == pytest.approx(30, abs=1.5)
    assert by_key["ControlPositive"].value == pytest.approx(1, abs=0.5)
    rows = comparison.table.set_index("Image")
    assert rows.loc["siRNA 1", "Cells"] == 1000
    assert rows.loc["siRNA 1", "Fold change (median)"] > 1
    assert comparison.plot.kind == "histogram" and comparison.plot.lines[0][0] == cutoff
    assert "pooled negative-control cells" in comparison.methods[0].description

    # The same cut-off scores the image open now, and suggests a Positive gate.
    table = pd.DataFrame({"Intensity_MeanIntensity_GFP": values})
    settings = {**settings, **comparison.settings, "reporter_channel": "GFP"}
    results = {r.key: r.value for r in analysis.results(table, [], settings, AssayContext())}
    assert results["PctPositive"] == pytest.approx(30, abs=1.5)
    (gate,) = analysis.suggested_gates(table, settings)
    assert gate.name == "Positive" and gate.params["value"] == cutoff


def test_experiment_samples_round_trip_through_the_saved_state():
    from karcytics_plugins.cytometrics.analysis.experiment import Sample

    sample = _sample("A", "sample", "Intensity_MeanIntensity_GFP", [1.0, np.nan, 3.0], "GFP")
    sample.table.attrs = {"units": {"Intensity_MeanIntensity_GFP": "a.u."}}
    back = Sample.from_dict(sample.to_dict())
    assert back.name == "A" and back.fields == {"reporter_channel": "GFP"}
    assert back.table["Intensity_MeanIntensity_GFP"].isna().tolist() == [False, True, False]
    assert back.table.attrs["units"]["Intensity_MeanIntensity_GFP"] == "a.u."


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


# ── Confluence and scratch wound ─────────────────────────────────────────────


def _texture(plane, detect):
    from fixtures import images as img

    from karcytics_plugins.cytometrics.analysis.segment import run_segmentation
    from karcytics_plugins.cytometrics.analysis.segment.texture import TextureSegmenter

    segmenter = TextureSegmenter()
    params = {p.key: p.default for p in segmenter.parameters()}
    params.update(target_channel="phase", detect=detect)
    return run_segmentation(segmenter, img.stack(("phase", plane)), params)


def _table(result, plane, scale=0.0, channel="phase"):
    from fixtures import images as img

    from karcytics_plugins.cytometrics.analysis.measure import MeasureOptions, measure_cells

    cells = [{**c, "id": c["label"]} for c in result.cells]
    return measure_cells(
        cells, img.stack((channel, plane)), scale, MeasureOptions(background="none")
    )


def test_confluence_is_the_covered_fraction():
    from fixtures import images as img

    result = _texture(img.scratch_photo(width=100), "covered")
    context = AssayContext(0.0, (300, 400), result.extras)
    values = {
        r.key: r.value
        for r in REGISTRY["confluence"].analysis.results(pd.DataFrame(), [], {}, context)
    }
    assert values["Confluence"] == pytest.approx(75, abs=4)  # 300 of 400 columns have cells


def _results(analysis, df, settings, context):
    return {r.key: r.value for r in analysis.results(df, [], settings, context)}


def _added(name, role, analysis, df, settings, context, fields=None):
    """A Sample as the panel adds it: the table and the image's own results."""
    from karcytics_plugins.cytometrics.analysis.experiment import Sample

    return Sample(
        name, role, df, fields=fields or {}, results=_results(analysis, df, settings, context)
    )


def test_wound_width_and_closure_against_the_start():
    from fixtures import images as img

    analysis = REGISTRY["wound"].analysis
    start = _texture(img.scratch_photo(width=100), "gap")
    start_df = _table(start, img.scratch_photo(width=100), scale=2.0)
    start_ctx = AssayContext(2.0, (300, 400), start.extras)
    assert _results(analysis, start_df, {}, start_ctx)["MeanWidth"] == pytest.approx(200, rel=0.1)

    later = _texture(img.scratch_photo(width=50, bridge=True, seed=1), "gap")
    later_df = _table(later, img.scratch_photo(width=50, bridge=True, seed=1), scale=2.0)
    later_ctx = AssayContext(2.0, (300, 400), later.extras)
    assert _results(analysis, later_df, {}, later_ctx)["MinWidth"] == 0  # cells bridged it

    samples = [
        _added("0 h", "start", analysis, start_df, {}, start_ctx),
        _added("24 h", "later", analysis, later_df, {}, later_ctx),
    ]
    comparison = analysis.compare(samples, {}, later_ctx)
    (closure,) = comparison.values
    assert closure.value == pytest.approx(100 * (1 - 50 * 280 / (100 * 300)), abs=6)
    settings = comparison.settings  # the start's area, for the image open now
    assert _results(analysis, later_df, settings, later_ctx)["Closure"] == pytest.approx(
        closure.value, abs=0.5
    )


def test_hemocytometer_pools_squares_and_flags_uneven_ones():
    from karcytics_plugins.cytometrics.analysis.experiment import Sample

    analysis = REGISTRY["hemocytometer"].analysis
    square = lambda n, dead: Sample(  # noqa: E731
        f"Square {n}", "square", pd.DataFrame(), results={"Counted": n, "Dead": dead, "Volume": 0.1}
    )
    even = analysis.compare(
        [square(50, 5), square(54, 6), square(46, 4)], {"dilution_factor": 2}, AssayContext()
    )
    values = {v.key: v for v in even.values}
    assert values["TotalPerML"].value == pytest.approx(150 / 0.3 * 1000 * 2)
    assert values["Viability"].value == pytest.approx(100 * 135 / 150)
    assert not values["TotalPerML"].warning
    uneven = analysis.compare([square(20, 0), square(90, 0), square(40, 0)], {}, AssayContext())
    assert "mix" in {v.key: v for v in uneven.values}["TotalPerML"].warning


def test_colony_series_uses_the_countable_plates():
    from karcytics_plugins.cytometrics.analysis.experiment import Sample

    analysis = REGISTRY["colonies"].analysis

    def plate(n, d):
        fields = {"dilution_factor": 10**d, "volume_ul": 100}
        return Sample(f"1e-{d}", "plate", pd.DataFrame(), fields=fields, results={"Colonies": n})

    plates = [plate(1200, 5), plate(150, 6), plate(130, 6), plate(14, 7)]
    comparison = analysis.compare(plates, {}, AssayContext())
    cfu = {v.key: v for v in comparison.values}["CFU_per_mL"]
    assert cfu.value == pytest.approx((150 + 130) / (2 * 0.1 / 1e6))  # the two 10⁻⁶ plates
    assert comparison.table["Used"].tolist() == ["", "✓", "✓", ""]
    assert not cfu.warning
    lonely = analysis.compare([plates[0]], {}, AssayContext())
    assert "no plate" in {v.key: v for v in lonely.values}["CFU_per_mL"].warning


def test_conditions_average_their_replicate_images():
    from karcytics_plugins.cytometrics.analysis.experiment import Sample

    analysis = REGISTRY["confluence"].analysis
    field = lambda name, c: Sample(name, "condition", pd.DataFrame(), results={"Confluence": c})  # noqa: E731
    comparison = analysis.compare(
        [field("Control", 80), field("Control", 90), field("Drug", 40)], {}, AssayContext()
    )
    by_key = {v.key: v for v in comparison.values}
    assert by_key["Confluence_Control"].value == 85
    assert "n = 2" in by_key["Confluence_Control"].help
    assert by_key["Confluence_Drug"].value == 40
    assert comparison.plot.kind == "bars" and len(comparison.plot.points) == 3


def test_cell_cycle_compares_treated_with_control(nuclei_table):
    analysis = REGISTRY["nuclei"].analysis
    settings = {"dna_channel": "DAPI", "g1_peak": 0}
    control = analysis.derive(nuclei_table, settings, AssayContext())
    treated = control.copy()
    treated["CellCycle_DNAContent"] = np.where(
        treated.index < 200, 4.0, treated["CellCycle_DNAContent"]
    )
    samples = [
        _added("Control", "control", analysis, control, settings, AssayContext()),
        _added("Nocodazole", "sample", analysis, treated, settings, AssayContext()),
    ]
    comparison = analysis.compare(samples, settings, AssayContext())
    g2 = {v.key: v for v in comparison.values}["G2/M_Nocodazole"]
    assert g2.value > 50 and "control" in g2.help
    assert [line[1] for line in comparison.plot.lines] == ["G1", "S", "G2/M", ">4N"]


# ── Spheroids ────────────────────────────────────────────────────────────────


def test_spheroid_keeps_the_largest_and_gives_its_volume():
    from fixtures import images as img

    from karcytics_plugins.cytometrics.analysis.segment import run_segmentation
    from karcytics_plugins.cytometrics.analysis.segment.spheroid import SpheroidSegmenter

    plane = img.spheroid_photo(radius=60)
    segmenter = SpheroidSegmenter()
    params = {p.key: p.default for p in segmenter.parameters()}
    params.update(target_channel="bf", min_area_px=0)
    result = run_segmentation(segmenter, img.stack(("bf", plane)), params)
    assert len(result.cells) == 1
    assert result.extras["polarity"] == "darker"

    df = _table(result, plane, scale=2.0, channel="bf")
    analysis = REGISTRY["spheroids"].analysis
    context = AssayContext(2.0, plane.shape, result.extras)
    values = {r.key: r for r in analysis.results(analysis.derive(df, {}, context), [], {}, context)}
    assert values["Diameter"].value == pytest.approx(240, rel=0.05)  # 120 px at 2 µm/px
    assert values["Volume"].value == pytest.approx(np.pi * 240**3 / 6, rel=0.15)
    assert values["Volume"].unit == "µm³"
    assert values["Circularity"].value > 0.85
    assert not values["Volume"].warning


def test_a_nearly_closed_wound_is_still_found():
    """Few open pixels: Otsu on the raw texture would split the cells' own texture instead."""
    from fixtures import images as img

    from karcytics_plugins.cytometrics.analysis.segment.texture import covered_mask

    covered = covered_mask(img.scratch_photo(width=20).astype(np.float32), 15)
    assert 100 * (~covered).mean() == pytest.approx(100 * 20 / 400, abs=3)  # raw Otsu: 38


def test_spheroid_in_a_darker_rimmed_well():
    """The well's lighting falls off towards its wall: it must not become the 'spheroid'."""
    from fixtures import images as img

    from karcytics_plugins.cytometrics.analysis.segment import run_segmentation
    from karcytics_plugins.cytometrics.analysis.segment.spheroid import SpheroidSegmenter

    plane = img.spheroid_photo(radius=50).astype(np.float32)
    yy, xx = np.mgrid[:400, :400]
    plane *= 1 - 0.5 * ((xx - 200) ** 2 + (yy - 200) ** 2) / 200**2  # dark corners
    segmenter = SpheroidSegmenter()
    params = {p.key: p.default for p in segmenter.parameters()}
    params.update(target_channel="bf")
    result = run_segmentation(segmenter, img.stack(("bf", plane)), params)
    assert len(result.cells) == 1
    area = np.count_nonzero(result.labels)
    assert area == pytest.approx(np.pi * 50**2, rel=0.15)
