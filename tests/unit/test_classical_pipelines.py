"""Golden tests for the classical (non-AI) segmentation pipelines.

These lock today's behaviour before the Segmenter/LabelMask refactor
(EXPANSION_PLAN.md Phases 2-4). If a change moves a number here on purpose,
re-derive it and update the constant with the reason — don't loosen it into
a wide range.

Known, intentional future changes:
- Areas are polygon areas of OpenCV contours, which under-count a filled
  disc by ~6% (662 px for a 707 px disc). Phase 4 measures from the label
  mask instead, and these area constants will move then.
"""

from __future__ import annotations

import math

import pytest
from fixtures import images as img

from karcytics_plugins.cytometrics.analysis.pipelines.otsu import OtsuPipeline
from karcytics_plugins.cytometrics.analysis.pipelines.watershed import WatershedPipeline

pytestmark = pytest.mark.unit

PARAMS = {"target_channel": "cells", "min_area_px": 50, "max_area_px": 100_000}

OTSU_DISC_AREA_PX = 662.0  # contour polygon area of a radius-15 filled disc
OTSU_DISC_CIRCULARITY = 0.910
OTSU_TOUCHING_PAIR_AREA_PX = 2792.0
OTSU_BORDER_DISC_AREA_PX = 471.0
WATERSHED_DISC_AREAS_PX = [626.0, 621.0, 622.0, 620.5]


def _areas(cells: list[dict]) -> list[float]:
    return [c["area"] for c in cells]


class TestOtsu:
    def test_counts_each_isolated_disc_once(self):
        cells = OtsuPipeline().run(img.stack(("cells", img.isolated_discs())), PARAMS, 1.0)

        assert len(cells) == len(img.ISOLATED_CENTRES)
        assert _areas(cells) == pytest.approx([OTSU_DISC_AREA_PX] * 4)
        assert [c["circ"] for c in cells] == pytest.approx([OTSU_DISC_CIRCULARITY] * 4, abs=1e-3)

    def test_area_scales_with_square_of_calibration_and_perimeter_linearly(self):
        stack = img.stack(("cells", img.isolated_discs()))
        at_1 = OtsuPipeline().run(stack, PARAMS, 1.0)
        at_half = OtsuPipeline().run(stack, PARAMS, 0.5)

        assert _areas(at_half) == pytest.approx([a * 0.25 for a in _areas(at_1)])
        assert [c["perim"] for c in at_half] == pytest.approx([c["perim"] * 0.5 for c in at_1])
        assert [c["circ"] for c in at_half] == pytest.approx([c["circ"] for c in at_1])

    def test_merges_touching_cells_into_one_object(self):
        cells = OtsuPipeline().run(img.stack(("cells", img.touching_pair())), PARAMS, 1.0)

        assert len(cells) == 1
        assert cells[0]["area"] == pytest.approx(OTSU_TOUCHING_PAIR_AREA_PX)

    def test_keeps_cells_cut_by_the_image_border(self):
        """Otsu has no border exclusion (Watershed and Cellpose do)."""
        cells = OtsuPipeline().run(
            img.stack(("cells", img.isolated_plus_border_disc())), PARAMS, 1.0
        )

        assert len(cells) == len(img.ISOLATED_CENTRES) + 1
        assert OTSU_BORDER_DISC_AREA_PX in _areas(cells)

    def test_min_area_filter_drops_objects_below_it(self):
        params = {**PARAMS, "min_area_px": OTSU_DISC_AREA_PX + 1}
        cells = OtsuPipeline().run(img.stack(("cells", img.isolated_discs())), params, 1.0)

        assert cells == []

    def test_unknown_channel_returns_no_cells(self):
        cells = OtsuPipeline().run(
            img.stack(("cells", img.isolated_discs())), {"target_channel": "nope"}, 1.0
        )

        assert cells == []

    def test_circularity_matches_its_definition(self):
        cell = OtsuPipeline().run(img.stack(("cells", img.isolated_discs())), PARAMS, 1.0)[0]

        assert cell["circ"] == pytest.approx(4 * math.pi * cell["area"] / cell["perim"] ** 2)


class TestWatershed:
    def test_counts_each_isolated_disc_once(self):
        cells = WatershedPipeline().run(img.stack(("cells", img.isolated_discs())), PARAMS, 1.0)

        assert _areas(cells) == pytest.approx(WATERSHED_DISC_AREAS_PX)

    def test_drops_cells_touching_the_image_border(self):
        cells = WatershedPipeline().run(
            img.stack(("cells", img.isolated_plus_border_disc())), PARAMS, 1.0
        )

        assert _areas(cells) == pytest.approx(WATERSHED_DISC_AREAS_PX)

    def test_nucleus_seed_channel_splits_a_touching_pair(self):
        stack = img.stack(("cells", img.touching_pair()), ("dapi", img.nuclei_for_touching_pair()))
        params = {**PARAMS, "use_dual_channel": True, "seed_channel": "dapi"}

        cells = WatershedPipeline().run(stack, params, 1.0)

        assert len(cells) == len(img.TOUCHING_CENTRES)

    @pytest.mark.xfail(
        strict=True,
        reason="Known bug: single-channel Watershed seeds at 0.5 x max distance, which leaves "
        "one connected seed for a two-cell clump, so it is never split. Fix in Phase 3.",
    )
    def test_splits_a_touching_pair_without_a_seed_channel(self):
        cells = WatershedPipeline().run(img.stack(("cells", img.touching_pair())), PARAMS, 1.0)

        assert len(cells) == len(img.TOUCHING_CENTRES)

    def test_unknown_channel_returns_no_cells(self):
        cells = WatershedPipeline().run(
            img.stack(("cells", img.isolated_discs())), {"target_channel": "nope"}, 1.0
        )

        assert cells == []
