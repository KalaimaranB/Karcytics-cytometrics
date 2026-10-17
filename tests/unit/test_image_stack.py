"""ImageStack loading and compositing, pinned before the Phase 2 ImageSet rewrite."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fixtures import images as img

from karcytics_plugins.cytometrics.analysis.image_stack import ImageStack

pytestmark = pytest.mark.unit


def _write(tmp_path, name: str, pixels: np.ndarray) -> str:
    path = str(tmp_path / name)
    assert cv2.imwrite(path, pixels)
    return path


def test_grayscale_file_becomes_one_channel_with_the_requested_colour(tmp_path):
    stack = ImageStack()
    added = stack.add_channel(_write(tmp_path, "g.png", img.isolated_discs()), "GFP", "green")

    assert added == [("GFP", "green")]
    assert stack.channels[0].data.dtype == np.uint8
    assert stack.channels[0].data.shape == (img.CANVAS, img.CANVAS)


def test_rgb_file_splits_into_blue_green_red_channels(tmp_path):
    bgr = np.dstack([img.blank(), img.isolated_discs(), img.touching_pair()])
    stack = ImageStack()

    added = stack.add_channel(_write(tmp_path, "rgb.png", bgr), "merge")

    assert added == [("merge (Ch 1)", "blue"), ("merge (Ch 2)", "green"), ("merge (Ch 3)", "red")]
    np.testing.assert_array_equal(stack.channels[1].data, bgr[:, :, 1])


def test_sixteen_bit_file_is_currently_truncated_to_eight_bit(tmp_path):
    """Pinned current (lossy) behaviour: Phase 2 keeps native bit depth and
    this test is replaced by one asserting uint16 survives.
    """
    img16 = np.full((8, 8), 51200, dtype=np.uint16)
    stack = ImageStack()

    stack.add_channel(_write(tmp_path, "deep.tif", img16), "deep")

    assert stack.channels[0].data.dtype == np.uint8
    assert int(stack.channels[0].data[0, 0]) == 51200 // 256


def test_unreadable_file_adds_nothing(tmp_path):
    stack = ImageStack()

    assert stack.add_channel(str(tmp_path / "missing.png"), "x") == []
    assert stack.channels == []


def test_composite_is_none_without_channels():
    assert ImageStack().get_composite() is None


def test_composite_tints_and_adds_visible_channels_only():
    stack = ImageStack()
    stack.channels = [
        img.FakeChannel("g", img.isolated_discs(), "green"),
        img.FakeChannel("r", img.isolated_discs(), "red"),
    ]

    both = stack.get_composite()
    stack.channels[1].visible = False
    green_only = stack.get_composite()

    assert both is not None and green_only is not None
    centre = img.ISOLATED_CENTRES[0][::-1]
    assert both[centre][1] > 0 and both[centre][2] > 0  # BGR: green and red both lit
    assert green_only[centre][2] == 0  # red hidden
    assert green_only[centre][0] == 0  # green tint has no blue component
