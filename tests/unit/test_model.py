"""ImageSet: native bit depth, channel naming/colours, metadata calibration, compositing."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis.model import Channel, ImageSet, color_for_name

pytestmark = pytest.mark.unit


def test_grayscale_png_becomes_one_channel_with_the_requested_colour(tmp_path):
    path = tmp_path / "g.png"
    Image.fromarray(img.isolated_discs()).save(path)
    images = ImageSet()

    added = images.add_file(path, "cells", "magenta")

    assert [(c.name, c.color, c.path) for c in added] == [("cells", "magenta", str(path))]
    assert added[0].data.dtype == np.uint8
    assert added[0].data.shape == (img.CANVAS, img.CANVAS)
    assert images.pixel_size is None


def test_stain_name_overrides_the_default_colour(tmp_path):
    path = tmp_path / "n.png"
    Image.fromarray(img.isolated_discs()).save(path)

    (channel,) = ImageSet().add_file(path, "DAPI_nuclei.png", "gray")

    assert channel.color == "blue"


@pytest.mark.parametrize("ext", ["png", "tif"])
def test_rgb_file_splits_into_red_green_blue(tmp_path, ext):
    rgb = np.dstack([img.touching_pair(), img.isolated_discs(), img.blank()])
    path = tmp_path / f"rgb.{ext}"
    Image.fromarray(rgb).save(path)
    images = ImageSet()

    added = images.add_file(path, "merge")

    assert [(c.name, c.color) for c in added] == [
        ("merge (Red)", "red"),
        ("merge (Green)", "green"),
        ("merge (Blue)", "blue"),
    ]
    np.testing.assert_array_equal(added[1].data, rgb[:, :, 1])


def test_sixteen_bit_tiff_keeps_its_native_values(tmp_path):
    img16 = np.full((8, 8), 51200, dtype=np.uint16)
    img16[0, 0] = 1000
    path = tmp_path / "deep.tif"
    tifffile.imwrite(path, img16)

    (channel,) = ImageSet().add_file(path, "deep")

    assert channel.data.dtype == np.uint16
    assert int(channel.data[1, 1]) == 51200
    assert channel.as_uint8().dtype == np.uint8
    assert int(channel.as_uint8()[0, 0]) == 0
    assert int(channel.as_uint8()[1, 1]) == 255


def test_ome_tiff_gives_pixel_size_and_named_channels(tmp_path):
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    path = tmp_path / "stack.ome.tif"
    tifffile.imwrite(
        path,
        data,
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.325,
            "PhysicalSizeY": 0.325,
            "Channel": {"Name": ["DAPI", "GFP"]},
        },
        ome=True,
    )
    images = ImageSet()

    added = images.add_file(path, "stack")

    assert [(c.name, c.color) for c in added] == [
        ("stack (DAPI)", "blue"),
        ("stack (GFP)", "green"),
    ]
    assert images.pixel_size is not None
    assert images.pixel_size.x_um == pytest.approx(0.325)
    assert images.pixel_size.source == "ome"


def test_first_calibrated_file_sets_the_pixel_size(tmp_path):
    plain, ome = tmp_path / "a.png", tmp_path / "b.ome.tif"
    Image.fromarray(img.blank(16)).save(plain)
    tifffile.imwrite(ome, np.zeros((16, 16), np.uint8), metadata={"PhysicalSizeX": 0.5}, ome=True)
    images = ImageSet()

    images.add_file(plain, "a")
    images.add_file(ome, "b")

    assert images.pixel_size is not None and images.pixel_size.x_um == pytest.approx(0.5)


def test_mismatched_size_is_rejected(tmp_path):
    small, big = tmp_path / "s.png", tmp_path / "b.png"
    Image.fromarray(img.blank(16)).save(small)
    Image.fromarray(img.blank(32)).save(big)
    images = ImageSet()
    images.add_file(small, "s")

    with pytest.raises(ValueError, match="32×32"):
        images.add_file(big, "b")
    assert len(images.channels) == 1


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageSet().add_file(tmp_path / "missing.png", "x")


def test_get_finds_a_channel_by_name():
    images = img.stack(("GFP", img.blank()), ("DAPI", img.blank()))

    assert images.get("DAPI") is images.channels[1]
    assert images.get("nope") is None


def test_flat_channel_scales_to_zeros():
    flat = Channel("f", np.full((4, 4), 7, dtype=np.uint16))

    assert not flat.as_uint8().any()


def test_composite_is_none_without_channels():
    assert ImageSet().composite() is None


def test_composite_tints_and_adds_visible_channels_only():
    images = ImageSet(
        [
            Channel("g", img.isolated_discs(), "green"),
            Channel("r", img.isolated_discs(), "red"),
        ]
    )

    both = images.composite()
    images.channels[1].visible = False
    green_only = images.composite()

    assert both is not None and green_only is not None
    centre = img.ISOLATED_CENTRES[0][::-1]
    assert both[centre][0] > 0 and both[centre][1] > 0  # RGB: red and green lit
    assert green_only[centre][0] == 0  # red hidden
    assert green_only[centre][2] == 0  # green tint has no blue component


@pytest.mark.parametrize(
    ("name", "color"),
    [
        ("DAPI", "blue"),
        ("Hoechst 33342", "blue"),
        ("EGFP", "green"),
        ("mCherry", "red"),
        ("Cy5", "magenta"),
        ("Far Red", "magenta"),
        ("Brightfield", None),
    ],
)
def test_colour_from_stain_name(name, color):
    assert color_for_name(name) == color


def test_display_range_scales_the_view_only():
    channel = Channel("c", np.array([[0, 100], [200, 4000]], dtype=np.uint16))

    channel.display_range = (0.0, 200.0)

    assert channel.display_uint8().tolist() == [[0, 128], [255, 255]]
    assert channel.as_uint8().max() == 255 and channel.data.max() == 4000


def test_auto_range_ignores_a_few_hot_pixels():
    data = np.full((100, 100), 100, dtype=np.uint16)
    data[:50] = 1000
    data[0, 0] = 65535

    lo, hi = Channel("c", data).auto_range()

    assert lo == 100 and hi == 1000


def test_full_range_is_the_dtype_range():
    assert Channel("c", np.zeros((2, 2), np.uint16)).full_range() == (0, 65535)
