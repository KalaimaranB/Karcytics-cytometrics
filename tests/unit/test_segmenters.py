"""Golden tests for the classical segmenters and the shared label-mask → cells step.

If a change moves a number here on purpose, re-derive it and update the
constant with the reason — don't loosen it into a wide range.

- Areas are polygon areas of the object outline, which under-count a filled
  disc by ~6% (662 px for a 707 px disc). Phase 4 measures from the label
  mask instead, and these area constants will move then.
- Watershed's footprint is deliberately generous (30% of the Otsu level after
  a 5 × 5 blur, to keep dim cell edges), so its discs measure 769–771 px.
  Phase 3 moved this from 620–626 px: the old OpenCV watershed drew a
  1-px boundary line inside every object.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures import images as img

from karcytics_plugins.cytometrics.analysis.segment import (
    REGISTRY,
    cells_from_labels,
    clear_border,
    run_segmentation,
)
from karcytics_plugins.cytometrics.analysis.segment.base import Param, relabel
from karcytics_plugins.cytometrics.analysis.segment.otsu import OtsuSegmenter
from karcytics_plugins.cytometrics.analysis.segment.watershed import WatershedSegmenter

pytestmark = pytest.mark.unit

PARAMS = {"target_channel": "cells", "min_area_px": 50, "max_area_px": 100_000}
KEEP_EDGES = {**PARAMS, "exclude_borders": False}

OTSU_DISC_AREA_PX = 662.0  # outline polygon area of a radius-15 filled disc
OTSU_DISC_CIRCULARITY = 0.910
OTSU_TOUCHING_PAIR_AREA_PX = 2792.0
OTSU_BORDER_DISC_AREA_PX = 471.0
WATERSHED_DISC_AREAS_PX = [770.0, 770.0, 769.0, 771.0]
WATERSHED_TOUCHING_AREAS_PX = [1530.0, 1493.0]


def _run(segmenter, stack, params=PARAMS, scale=1.0) -> list[dict]:
    return run_segmentation(segmenter, stack, params, scale).cells


def _areas(cells: list[dict]) -> list[float]:
    return [c["area"] for c in cells]


def test_registry_has_every_segmenter_once():
    assert set(REGISTRY) == {"otsu", "watershed", "cellpose"}
    for sid, cls in REGISTRY.items():
        seg = cls()
        assert seg.id == sid
        assert seg.name and seg.summary and seg.citations
        assert all(isinstance(p, Param) for p in seg.parameters())
        keys = [p.key for p in seg.parameters()]
        assert len(keys) == len(set(keys))
        assert {"target_channel", "exclude_borders", "min_area_px", "max_area_px"} <= set(keys)


class TestOtsu:
    def test_counts_each_isolated_disc_once(self):
        cells = _run(OtsuSegmenter(), img.stack(("cells", img.isolated_discs())))

        assert len(cells) == len(img.ISOLATED_CENTRES)
        assert _areas(cells) == pytest.approx([OTSU_DISC_AREA_PX] * 4)
        assert [c["circ"] for c in cells] == pytest.approx([OTSU_DISC_CIRCULARITY] * 4, abs=1e-3)

    def test_area_scales_with_square_of_calibration_and_perimeter_linearly(self):
        stack = img.stack(("cells", img.isolated_discs()))
        at_1 = _run(OtsuSegmenter(), stack, scale=1.0)
        at_half = _run(OtsuSegmenter(), stack, scale=0.5)

        assert _areas(at_half) == pytest.approx([a * 0.25 for a in _areas(at_1)])
        assert [c["perim"] for c in at_half] == pytest.approx([c["perim"] * 0.5 for c in at_1])
        assert [c["circ"] for c in at_half] == pytest.approx([c["circ"] for c in at_1])

    def test_merges_touching_cells_into_one_object(self):
        cells = _run(OtsuSegmenter(), img.stack(("cells", img.touching_pair())))

        assert _areas(cells) == pytest.approx([OTSU_TOUCHING_PAIR_AREA_PX])

    def test_keeps_or_drops_border_cells_as_asked(self):
        stack = img.stack(("cells", img.isolated_plus_border_disc()))

        kept = _run(OtsuSegmenter(), stack, KEEP_EDGES)
        dropped = _run(OtsuSegmenter(), stack, PARAMS)

        assert OTSU_BORDER_DISC_AREA_PX in _areas(kept)
        assert len(kept) == len(img.ISOLATED_CENTRES) + 1
        assert _areas(dropped) == pytest.approx([OTSU_DISC_AREA_PX] * 4)

    def test_min_area_filter_drops_objects_below_it(self):
        params = {**PARAMS, "min_area_px": OTSU_DISC_AREA_PX + 1}

        assert _run(OtsuSegmenter(), img.stack(("cells", img.isolated_discs())), params) == []

    def test_unknown_channel_is_an_error(self):
        with pytest.raises(ValueError, match="'nope' is not loaded"):
            _run(OtsuSegmenter(), img.stack(("cells", img.blank())), {"target_channel": "nope"})

    def test_circularity_matches_its_definition(self):
        cell = _run(OtsuSegmenter(), img.stack(("cells", img.isolated_discs())))[0]

        assert cell["circ"] == pytest.approx(4 * math.pi * cell["area"] / cell["perim"] ** 2)

    def test_sixteen_bit_input_gives_the_same_objects(self):
        deep = img.isolated_discs().astype(np.uint16) * 257

        cells = _run(OtsuSegmenter(), img.stack(("cells", deep)))

        assert _areas(cells) == pytest.approx([OTSU_DISC_AREA_PX] * 4)


class TestWatershed:
    def test_counts_each_isolated_disc_once(self):
        cells = _run(WatershedSegmenter(), img.stack(("cells", img.isolated_discs())))

        assert _areas(cells) == pytest.approx(WATERSHED_DISC_AREAS_PX)

    def test_drops_cells_touching_the_image_border(self):
        cells = _run(WatershedSegmenter(), img.stack(("cells", img.isolated_plus_border_disc())))

        assert _areas(cells) == pytest.approx(WATERSHED_DISC_AREAS_PX)

    def test_splits_a_touching_pair_without_a_seed_channel(self):
        """Regression: seeds used to be one blob at 0.5 × max distance, so this never split."""
        cells = _run(WatershedSegmenter(), img.stack(("cells", img.touching_pair())))

        assert _areas(cells) == pytest.approx(WATERSHED_TOUCHING_AREAS_PX)

    def test_nucleus_seed_channel_splits_a_touching_pair(self):
        stack = img.stack(("cells", img.touching_pair()), ("dapi", img.nuclei_for_touching_pair()))
        params = {**PARAMS, "use_dual_channel": True, "seed_channel": "dapi"}

        assert len(_run(WatershedSegmenter(), stack, params)) == len(img.TOUCHING_CENTRES)

    def test_large_separation_keeps_a_touching_pair_together(self):
        params = {**PARAMS, "min_separation_px": 60}

        cells = _run(WatershedSegmenter(), img.stack(("cells", img.touching_pair())), params)

        assert len(cells) == 1


class TestCellsFromLabels:
    def test_each_object_is_outlined_in_image_coordinates(self):
        labels = np.zeros((20, 30), np.int32)
        labels[5:10, 20:26] = 7

        _, (cell,) = cells_from_labels(labels, 1.0)

        xs = [p[0] for p in cell["points"]]
        ys = [p[1] for p in cell["points"]]
        assert (min(xs), max(xs), min(ys), max(ys)) == (20, 25, 5, 9)
        assert cell["label"] == 1
        assert cell["area"] == pytest.approx(5 * 4)

    def test_filtered_objects_leave_the_mask_and_the_rest_are_renumbered(self):
        labels = np.zeros((20, 20), np.int32)
        labels[1:3, 1:3] = 4  # tiny
        labels[8:16, 8:16] = 9

        out, cells = cells_from_labels(labels, 1.0, min_area_px=10)

        assert [c["label"] for c in cells] == [1]
        assert set(np.unique(out)) == {0, 1}
        assert (out[8:16, 8:16] == 1).all()

    def test_confidence_is_the_mean_probability_inside_the_object(self):
        labels = np.zeros((10, 10), np.int32)
        labels[2:6, 2:6] = 1
        probability = np.zeros((10, 10), np.float32)
        probability[2:6, 2:6] = 0.75

        _, (cell,) = cells_from_labels(labels, 1.0, probability=probability)

        assert cell["confidence"] == pytest.approx(0.75)

    def test_progress_reaches_100(self):
        labels = np.zeros((10, 10), np.int32)
        labels[1:4, 1:4], labels[5:9, 5:9] = 1, 2
        seen: list[int] = []

        cells_from_labels(labels, 1.0, progress=seen.append)

        assert seen[-1] == 100

    def test_clear_border_and_relabel(self):
        labels = np.zeros((10, 10), np.int32)
        labels[0:3, 4:6] = 3
        labels[4:7, 4:7] = 8

        assert set(np.unique(clear_border(labels))) == {0, 8}
        assert set(np.unique(relabel(clear_border(labels)))) == {0, 1}


def test_cellpose_eval_arguments_cover_the_exposed_options():
    from karcytics_plugins.cytometrics.analysis.segment.cellpose import eval_kwargs

    kwargs = eval_kwargs(
        {
            "diameter": 0,
            "cellprob_threshold": -1.5,
            "norm_percentile_low": 2,
            "norm_percentile_high": 98,
            "tile_norm_blocksize": 99.6,
            "augment": True,
            "niter": 0,
            "batch_size": 4,
        },
        ndim=3,
    )

    assert kwargs["channel_axis"] == 0
    assert kwargs["diameter"] is None and kwargs["niter"] is None
    assert kwargs["cellprob_threshold"] == -1.5
    assert kwargs["normalize"] == {
        "normalize": True,
        "percentile": [2, 98],
        "tile_norm_blocksize": 100,
    }
    assert kwargs["augment"] is True and kwargs["batch_size"] == 4
