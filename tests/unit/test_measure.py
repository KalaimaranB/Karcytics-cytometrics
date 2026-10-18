"""Measurements: shape from masks, intensities per channel, background, units."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from karcytics_plugins.cytometrics.analysis.measure import (
    MeasureOptions,
    channel_key,
    csv_headers,
    measure_cells,
    rescale,
    write_csv,
)
from karcytics_plugins.cytometrics.analysis.measure.shape import feret_diameters
from karcytics_plugins.cytometrics.analysis.model import Channel, ImageSet

pytestmark = pytest.mark.unit


def _rect(cell_id, x, y, w, h, source=None):
    cell = {
        "id": cell_id,
        "points": [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
    }
    if source:
        cell["source"] = source
    return cell


def _images(**planes) -> ImageSet:
    return ImageSet([Channel(name, data) for name, data in planes.items()])


def _plane(value=0, size=64, dtype=np.uint16):
    return np.full((size, size), value, dtype=dtype)


class TestShape:
    def test_rectangle(self):
        cells = [_rect(1, 10, 10, 20, 10)]

        row = measure_cells(cells, _images(c=_plane()), 0.0).iloc[0]

        assert row["AreaShape_Area"] == 200
        assert row["AreaShape_MinFeretDiameter"] == pytest.approx(10)
        assert row["AreaShape_MaxFeretDiameter"] == pytest.approx(math.hypot(10, 20))
        assert row["AreaShape_Solidity"] == pytest.approx(1.0)
        assert row["AreaShape_EquivalentDiameter"] == pytest.approx(2 * math.sqrt(200 / math.pi))
        assert row["AreaShape_MajorAxisLength"] > row["AreaShape_MinorAxisLength"]
        assert (row["Location_Center_X"], row["Location_Center_Y"]) == (19.5, 14.5)

    def test_disc_is_round(self):
        mask = np.zeros((64, 64), np.uint8)
        cv2.circle(mask, (32, 32), 15, 1, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell = {"id": 1, "points": contours[0][:, 0, :].tolist()}

        row = measure_cells([cell], _images(c=_plane()), 0.0).iloc[0]

        assert row["AreaShape_Area"] == mask.sum()
        assert row["AreaShape_Eccentricity"] == pytest.approx(0.0, abs=0.05)
        assert 0.9 < row["AreaShape_FormFactor"] < 1.0

    def test_calibration_scales_lengths_and_areas_only(self):
        cells = [_rect(1, 10, 10, 20, 10)]
        px = measure_cells(cells, _images(c=_plane(5)), 0.0).iloc[0]
        um = measure_cells(cells, _images(c=_plane(5)), 0.5).iloc[0]

        assert um["AreaShape_Area"] == pytest.approx(px["AreaShape_Area"] * 0.25)
        assert um["AreaShape_Perimeter"] == pytest.approx(px["AreaShape_Perimeter"] * 0.5)
        assert um["AreaShape_FormFactor"] == pytest.approx(px["AreaShape_FormFactor"])
        assert um["Location_Center_X"] == px["Location_Center_X"]  # always pixels
        assert um["Intensity_MeanIntensity_c"] == px["Intensity_MeanIntensity_c"]

    def test_feret_of_a_single_pixel(self):
        mask = np.zeros((5, 5), bool)
        mask[2, 2] = True

        assert feret_diameters(mask) == pytest.approx((1.0, math.sqrt(2)))


class TestIntensity:
    def test_statistics_in_native_sixteen_bit_units(self):
        plane = _plane(100)
        plane[10:20, 10:20] = 40000
        plane[10, 10] = 60000

        row = measure_cells([_rect(1, 10, 10, 10, 10)], _images(GFP=plane), 0.0).iloc[0]

        assert row["Intensity_MaxIntensity_GFP"] == 60000
        assert row["Intensity_MinIntensity_GFP"] == 40000
        assert row["Intensity_MedianIntensity_GFP"] == 40000
        assert row["Intensity_IntegratedIntensity_GFP"] == 99 * 40000 + 60000
        assert row["Intensity_MeanIntensity_GFP"] == pytest.approx((99 * 40000 + 60000) / 100)

    def test_every_channel_gets_its_own_columns(self):
        table = measure_cells(
            [_rect(1, 5, 5, 4, 4)], _images(**{"GFP (Ch 1)": _plane(1), "DAPI": _plane(2)}), 0.0
        )

        assert table["Intensity_MeanIntensity_GFP_Ch_1"].iloc[0] == 1
        assert table["Intensity_MeanIntensity_DAPI"].iloc[0] == 2

    def test_global_background_is_the_median_outside_cells(self):
        plane = _plane(10)
        plane[20:30, 20:30] = 110

        row = measure_cells(
            [_rect(1, 20, 20, 10, 10)], _images(c=plane), 0.0, MeasureOptions(background="global")
        ).iloc[0]

        assert row["Intensity_Background_c"] == 10
        assert row["Intensity_CorrectedMeanIntensity_c"] == 100
        assert row["Intensity_CorrectedIntegratedIntensity_c"] == 100 * 100

    def test_local_background_ring_ignores_neighbouring_cells(self):
        plane = _plane(10)
        plane[20:30, 20:30] = 110  # the cell
        plane[20:30, 33:43] = 5000  # a bright neighbour 3 px to the right
        cells = [_rect(1, 20, 20, 10, 10), _rect(2, 33, 20, 10, 10)]

        table = measure_cells(cells, _images(c=plane), 0.0, MeasureOptions(background="local"))

        assert table["Intensity_Background_c"].iloc[0] == 10

    def test_background_columns_only_when_asked(self):
        table = measure_cells([_rect(1, 5, 5, 4, 4)], _images(c=_plane()), 0.0)

        assert not any("Background" in c for c in table.columns)


class TestTable:
    def test_ids_source_and_confidence(self):
        cells = [_rect(4, 5, 5, 4, 4), {**_rect(9, 20, 20, 4, 4, "manual"), "confidence": 0.8}]

        table = measure_cells(cells, _images(c=_plane()), 0.0)

        assert table["ObjectNumber"].tolist() == [4, 9]
        assert table["Source"].tolist() == ["auto", "manual"]
        assert np.isnan(table["Confidence"].iloc[0]) and table["Confidence"].iloc[1] == 0.8

    def test_empty(self):
        table = measure_cells([], _images(c=_plane()), 0.0)

        assert len(table) == 0 and "AreaShape_Area" in table.columns

    def test_rescale_matches_measuring_at_the_new_scale(self):
        cells = [_rect(1, 10, 10, 20, 10)]
        images = _images(c=_plane(3))

        converted = rescale(measure_cells(cells, images, 0.0), 0.25)
        direct = measure_cells(cells, images, 0.25)

        for col in ("AreaShape_Area", "AreaShape_Perimeter", "AreaShape_MinFeretDiameter"):
            assert converted[col].iloc[0] == pytest.approx(direct[col].iloc[0])
        assert converted.attrs["scale"] == 0.25

    def test_csv_headers_carry_units(self, tmp_path):
        table = measure_cells([_rect(1, 10, 10, 4, 4)], _images(c=_plane()), 0.5)
        path = tmp_path / "t.csv"

        write_csv(table, str(path))

        header = path.read_text().splitlines()[0].split(",")
        assert header == csv_headers(table)
        assert "AreaShape_Area [um2]" in header
        assert "AreaShape_FormFactor" in header
        assert "Location_Center_X [px]" in header
        assert "Intensity_MeanIntensity_c [a.u.]" in header


@pytest.mark.parametrize(
    ("name", "key"),
    [("GFP", "GFP"), ("cells.ome.tif (DAPI)", "cells_ome_tif_DAPI"), ("µ", "channel")],
)
def test_channel_key(name, key):
    assert channel_key(name) == key


class TestTexture:
    def test_flat_and_noisy_patches(self):
        from karcytics_plugins.cytometrics.analysis.measure.texture import haralick

        mask = np.zeros((40, 40), bool)
        mask[5:35, 5:35] = True
        flat = haralick(np.full((40, 40), 100, np.uint8), mask, 3)
        noisy = haralick(np.random.default_rng(0).integers(0, 255, (40, 40), np.uint8), mask, 3)

        assert flat["Contrast"] == 0 and flat["AngularSecondMoment"] == 1
        assert noisy["Contrast"] > 1000 and noisy["Entropy"] > flat["Entropy"]

    def test_pixels_outside_the_object_are_ignored(self):
        from karcytics_plugins.cytometrics.analysis.measure.texture import haralick

        plane = np.full((30, 30), 100, np.uint8)
        plane[:, :5] = 255  # bright stripe outside the mask
        mask = np.zeros((30, 30), bool)
        mask[5:25, 10:25] = True

        assert haralick(plane, mask, 3)["Contrast"] == 0

    def test_object_thinner_than_the_distance_gives_nan(self):
        from karcytics_plugins.cytometrics.analysis.measure.texture import haralick

        mask = np.zeros((10, 10), bool)
        mask[5, 5] = True

        assert np.isnan(haralick(np.zeros((10, 10), np.uint8), mask, 3)["Contrast"])

    def test_columns_only_when_switched_on(self):
        cells = [_rect(1, 10, 10, 20, 20)]

        off = measure_cells(cells, _images(c=_plane(5)), 0.0)
        on = measure_cells(cells, _images(c=_plane(5)), 0.0, MeasureOptions(texture=True))

        assert not any(c.startswith("Texture_") for c in off.columns)
        assert on["Texture_Contrast_c_3"].iloc[0] == 0


class TestNucleus:
    def test_areas_and_translocation_ratio(self):
        dapi = _plane()
        dapi[16:24, 16:24] = 1000
        gfp = _plane(10)
        gfp[10:30, 10:30] = 100
        gfp[16:24, 16:24] = 200

        row = measure_cells(
            [_rect(1, 10, 10, 20, 20)],
            _images(DAPI=dapi, GFP=gfp),
            0.5,
            MeasureOptions(nucleus_channel="DAPI"),
        ).iloc[0]

        assert row["Nucleus_Area"] == 64 * 0.25
        assert row["Cytoplasm_Area"] == 336 * 0.25
        assert row["Nucleus_NCAreaRatio"] == pytest.approx(64 / 336)
        assert row["Intensity_NCRatio_GFP"] == pytest.approx(2.0)

    def test_missing_nucleus_channel_is_an_error(self):
        with pytest.raises(ValueError, match="'DAPI' is not loaded"):
            measure_cells(
                [_rect(1, 5, 5, 4, 4)],
                _images(c=_plane()),
                0.0,
                MeasureOptions(nucleus_channel="DAPI"),
            )
