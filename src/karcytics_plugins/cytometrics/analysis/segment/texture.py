"""Cell-covered area by texture: for confluence and scratch-wound images.

In phase contrast and bright field, cells are textured (edges, granules,
halos) and the empty dish is smooth. The local standard deviation of the
intensity (in a window about a cell wide) is high on cells and low on the
dish whatever the lighting, so thresholding it separates covered from open
areas. This is the approach of the common wound-healing tools (e.g. the
ImageJ "MRI Wound Healing Tool" uses a variance filter).

The threshold is Otsu's, on the logarithm of the texture: the dish's
texture is camera noise, many times lower than the cells', and on a log
scale the two stay well apart even when one of them is a small part of the
image (a nearly closed wound, a sparse dish), where Otsu on the raw values
splits the cells' own texture instead. A window over a cell's edge already
sees its texture, so the covered area is then shrunk by half a window.

Two outputs: the covered areas as objects (confluence), or the open gaps as
objects (the wound), leaving out gaps smaller than a few cells between
neighbours. For a wound, its width along its length is measured too.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..model import ImageSet
from ..units import area_range, effective_scale, length_unit
from .base import (
    EXCLUDE_BORDERS,
    MAX_AREA,
    MIN_AREA,
    TARGET_CHANNEL,
    Param,
    Progress,
    SegmentationResult,
    otsu_level,
    register,
    require_channel,
)

COVERED, GAP = "covered", "gap"
_NOISE_FLOOR = 1e-3  # added before the log (texture is on a 0–1 intensity scale)

OUTPUT = Param(
    "detect",
    "Detect",
    "choice",
    COVERED,
    choices=(("Areas covered by cells", COVERED), ("Open gaps (the wound)", GAP)),
    help="Covered areas give the confluence; gaps give the wound's area and width.",
)
WINDOW = Param(
    "window_px",
    "Texture window",
    "float",
    15.0,
    3.0,
    500.0,
    unit="length",
    help="Texture is measured over a square this wide: about one cell across.",
)
SENSITIVITY = Param(
    "sensitivity",
    "Sensitivity",
    "float",
    0.0,
    -0.9,
    0.9,
    step=0.1,
    decimals=2,
    help="Raise to count faint, flat cells as covered; lower to count less. 0: Otsu's threshold.",
)
MIN_GAP = Param(
    "min_gap_px",
    "Ignore gaps smaller than",
    "float",
    2000.0,
    0.0,
    1e9,
    unit="area",
    help="Open areas smaller than this (between neighbouring cells) count as covered.",
)
DIRECTION = Param(
    "wound_direction",
    "The wound runs",
    "choice",
    "auto",
    choices=(
        ("Found automatically", "auto"),
        ("Top to bottom", "vertical"),
        ("Left to right", "horizontal"),
    ),
    help="Its width is measured across this direction, on every line of the image.",
)


@register
class TextureSegmenter:
    id = "texture"
    name = "Covered area (texture)"
    summary = "Finds where cells are from the image's local texture: for confluence and wounds."
    citations = ("otsu1979", "opencv2000")
    for_assays = ("confluence", "wound")

    def parameters(self) -> list[Param]:
        return [
            TARGET_CHANNEL,
            OUTPUT,
            WINDOW,
            SENSITIVITY,
            MIN_GAP,
            DIRECTION,
            Param(
                EXCLUDE_BORDERS.key, EXCLUDE_BORDERS.label, "bool", False, help=EXCLUDE_BORDERS.help
            ),
            Param(MIN_AREA.key, MIN_AREA.label, "float", 0.0, 0, 1e9, unit="area"),
            Param(MAX_AREA.key, MAX_AREA.label, "float", 1e9, 0, 1e10, unit="area"),
        ]

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        size = effective_scale(scale)
        window = params.get("window_px", WINDOW.default) * size
        gap = params.get("min_gap_px", MIN_GAP.default) * size**2
        parameters: dict = {
            "texture window": f"{window:g} {length_unit(scale)}",
            "smallest gap kept": f"{gap:g} {length_unit(scale)}²",
            "size filter": area_range(params, scale),
        }
        if params.get("sensitivity"):
            parameters["threshold adjustment"] = f"{-params['sensitivity']:+.0%}"
        what = (
            "open gaps (the wound) were taken as the uncovered areas"
            if params.get("detect") == GAP
            else "covered areas were taken as the cell objects"
        )
        return (
            f"Cell-covered area in the '{params.get('target_channel')}' channel was found from "
            "the local standard deviation of intensity, thresholded by Otsu's method; gaps "
            f"smaller than the minimum gap size were counted as covered, and {what}",
            parameters,
        )

    def segment(
        self, image_set: ImageSet, params: dict, progress: Progress | None = None
    ) -> SegmentationResult:
        report = progress or (lambda _v: None)
        plane = require_channel(image_set, params.get("target_channel")).data
        covered = covered_mask(
            plane,
            float(params.get("window_px", WINDOW.default)),
            float(params.get("sensitivity", 0.0)),
            float(params.get("min_gap_px", MIN_GAP.default)),
        )
        report(70)
        extras: dict = {"coverage": float(covered.mean())}
        if params.get("detect", COVERED) == GAP:
            gap = ~covered
            labels, _ = ndimage.label(gap, structure=np.ones((3, 3), bool))
            extras["wound"] = wound_widths(gap, params.get("wound_direction", "auto"))
        else:
            labels, _ = ndimage.label(covered, structure=np.ones((3, 3), bool))
        report(95)
        return SegmentationResult(labels.astype(np.int32), extras=extras)


def local_std(plane: np.ndarray, window: float) -> np.ndarray:
    """The standard deviation of `plane` in a `window`-pixel square around every pixel."""
    image = plane.astype(np.float32)
    lo, hi = np.percentile(image, [0.5, 99.5])
    image = (image - lo) / max(float(hi - lo), 1e-6)
    size = (max(3, int(round(window))),) * 2
    mean = cv2.blur(image, size)
    mean_sq = cv2.blur(image * image, size)
    return np.sqrt(np.clip(mean_sq - mean * mean, 0, None))


def covered_mask(
    plane: np.ndarray, window: float, sensitivity: float = 0.0, min_gap: float = 0.0
) -> np.ndarray:
    """Where cells are: high local texture, small gaps filled in."""
    texture = np.log(local_std(plane, window) + _NOISE_FLOOR)
    values = texture[texture <= np.percentile(texture, 99.5)]  # a few bright specks skew Otsu
    level = np.exp(otsu_level(values)) * (1 - sensitivity)
    covered = (texture > np.log(level)).astype(np.uint8)
    half = int(window) // 2  # the window reaches this far past a cell's edge
    if half:
        square = np.ones((2 * half + 1, 2 * half + 1), np.uint8)
        covered = cv2.erode(covered, square, borderType=cv2.BORDER_REPLICATE)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    covered = cv2.morphologyEx(covered, cv2.MORPH_OPEN, kernel).astype(bool)
    if min_gap > 0:
        gaps, n = ndimage.label(~covered)
        if n:
            sizes = np.bincount(gaps.ravel())
            small = sizes < min_gap
            small[0] = False
            covered |= small[gaps]
    return covered


def wound_widths(gap: np.ndarray, direction: str = "auto") -> dict:
    """The wound's width on every line across it, in pixels.

    The wound runs top to bottom ("vertical") or left to right; its width
    on a line is the number of open pixels on it. Lines where the wound has
    closed have width 0, so the mean width is the open area ÷ the wound's
    length (the image's height or width).
    """
    if direction not in ("vertical", "horizontal"):
        ys, xs = np.nonzero(gap)
        direction = "vertical" if len(ys) and ys.var() >= xs.var() else "horizontal"
    widths = gap.sum(axis=1 if direction == "vertical" else 0).astype(float)
    if not widths.size:
        return {"direction": direction, "length_px": 0}
    return {
        "direction": direction,
        "length_px": int(widths.size),
        "mean_px": float(widths.mean()),
        "sd_px": float(widths.std(ddof=1)) if widths.size > 1 else 0.0,
        "min_px": float(widths.min()),
        "max_px": float(widths.max()),
        "closed_lines": float((widths == 0).mean()),
    }
