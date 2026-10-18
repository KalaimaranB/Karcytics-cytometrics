"""Golden tests for the classical segmenters and the shared label-mask → cells step.

If a change moves a number here on purpose, re-derive it and update the
constant with the reason — don't loosen it into a wide range.

Areas are mask pixel counts measured by ``analysis.measure`` (Phase 4.1).
OpenCV draws a radius-15 disc with 709 px; Otsu after a 5 × 5 blur keeps 705.
Before 4.1 they were outline-polygon areas (662 px for the same disc).
Watershed's footprint is deliberately generous (30% of the Otsu level, to
keep dim cell edges), so its discs measure 816–818 px.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures import images as img

from karcytics_plugins.cytometrics.analysis.measure import measure_cells
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

OTSU_DISC_AREA_PX = 705.0
OTSU_DISC_FORM_FACTOR = 0.969
OTSU_TOUCHING_PAIR_AREA_PX = 2893.0
OTSU_BORDER_DISC_AREA_PX = 512.0
WATERSHED_DISC_AREAS_PX = [817.0, 817.0, 816.0, 818.0]
WATERSHED_TOUCHING_AREAS_PX = [1597.0, 1559.0]


def _run(segmenter, stack, params=PARAMS, scale=0.0) -> list[dict]:
    """Segment and measure; one dict per cell with the cell's columns."""
    cells = run_segmentation(segmenter, stack, params).cells
    for cell in cells:
        cell["id"] = cell["label"]
    return measure_cells(cells, stack, scale).to_dict("records")


def _areas(cells: list[dict]) -> list[float]:
    return [c["AreaShape_Area"] for c in cells]


def test_registry_has_every_segmenter_once():
    assert set(REGISTRY) == {"otsu", "watershed", "cellpose", "colonies", "texture"}
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
        assert [c["AreaShape_FormFactor"] for c in cells] == pytest.approx(
            [OTSU_DISC_FORM_FACTOR] * 4, abs=1e-3
        )

    def test_area_scales_with_square_of_calibration_and_perimeter_linearly(self):
        stack = img.stack(("cells", img.isolated_discs()))
        at_1 = _run(OtsuSegmenter(), stack, scale=1.0)
        at_half = _run(OtsuSegmenter(), stack, scale=0.5)

        assert _areas(at_half) == pytest.approx([a * 0.25 for a in _areas(at_1)])
        perim = "AreaShape_Perimeter"
        assert [c[perim] for c in at_half] == pytest.approx([c[perim] * 0.5 for c in at_1])
        ff = "AreaShape_FormFactor"
        assert [c[ff] for c in at_half] == pytest.approx([c[ff] for c in at_1])

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

    def test_area_filter_uses_the_same_pixel_area_that_is_reported(self):
        stack = img.stack(("cells", img.isolated_discs()))

        exactly = _run(OtsuSegmenter(), stack, {**PARAMS, "min_area_px": OTSU_DISC_AREA_PX})
        above = _run(OtsuSegmenter(), stack, {**PARAMS, "min_area_px": OTSU_DISC_AREA_PX + 1})

        assert len(exactly) == len(img.ISOLATED_CENTRES)
        assert above == []

    def test_unknown_channel_is_an_error(self):
        with pytest.raises(ValueError, match="'nope' is not loaded"):
            _run(OtsuSegmenter(), img.stack(("cells", img.blank())), {"target_channel": "nope"})

    def test_form_factor_matches_its_definition(self):
        cell = _run(OtsuSegmenter(), img.stack(("cells", img.isolated_discs())))[0]

        expected = 4 * math.pi * cell["AreaShape_Area"] / cell["AreaShape_Perimeter"] ** 2
        assert cell["AreaShape_FormFactor"] == pytest.approx(expected)

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

        _, (cell,) = cells_from_labels(labels)

        xs = [p[0] for p in cell["points"]]
        ys = [p[1] for p in cell["points"]]
        assert (min(xs), max(xs), min(ys), max(ys)) == (20, 25, 5, 9)
        assert cell["label"] == 1

    def test_filtered_objects_leave_the_mask_and_the_rest_are_renumbered(self):
        labels = np.zeros((20, 20), np.int32)
        labels[1:3, 1:3] = 4  # tiny
        labels[8:16, 8:16] = 9

        out, cells = cells_from_labels(labels, min_area_px=10)

        assert [c["label"] for c in cells] == [1]
        assert set(np.unique(out)) == {0, 1}
        assert (out[8:16, 8:16] == 1).all()

    def test_confidence_is_the_mean_probability_inside_the_object(self):
        labels = np.zeros((10, 10), np.int32)
        labels[2:6, 2:6] = 1
        probability = np.zeros((10, 10), np.float32)
        probability[2:6, 2:6] = 0.75

        _, (cell,) = cells_from_labels(labels, probability=probability)

        assert cell["confidence"] == pytest.approx(0.75)

    def test_progress_reaches_100(self):
        labels = np.zeros((10, 10), np.int32)
        labels[1:4, 1:4], labels[5:9, 5:9] = 1, 2
        seen: list[int] = []

        cells_from_labels(labels, progress=seen.append)

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


@pytest.mark.parametrize(
    ("shape", "params", "batches"),
    [
        # Counted from real Cellpose 4.2 runs on a 1080x1920 image.
        ((1080, 1920), {}, 8),
        ((1080, 1920), {"augment": True}, 18),
        ((1080, 1920), {"diameter": 60.0}, 2),
        ((100, 100), {}, 1),
    ],
)
def test_cellpose_batch_count_matches_its_tiling(shape, params, batches):
    from karcytics_plugins.cytometrics.analysis.segment.cellpose import (
        eval_kwargs,
        expected_batches,
    )

    assert expected_batches(shape, eval_kwargs(params, 2)) == batches


def test_batch_counter_reports_only_this_threads_passes():
    import threading

    from karcytics_plugins.cytometrics.analysis.segment.cellpose import _BatchCounter

    class Net:
        def __init__(self):
            self.hooks = []

        def register_forward_hook(self, hook):
            self.hooks.append(hook)
            net = self

            class Handle:
                def remove(self):
                    net.hooks.remove(hook)

            return Handle()

        def forward(self):
            for hook in list(self.hooks):
                hook(self, (), ())

    net, seen = Net(), []
    with _BatchCounter(net, seen.append):
        net.forward()
        other = threading.Thread(target=net.forward)
        other.start()
        other.join()
        net.forward()
    net.forward()

    assert seen == [1, 2]
    assert net.hooks == []
