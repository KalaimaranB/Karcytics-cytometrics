"""Colonies on an agar plate: find the dish, flatten the agar, threshold, split touching colonies.

The approach follows OpenCFU (Geissmann 2013) in outline, reimplemented
here (no code reused): the plate is found with a Hough circle transform and
masked so the rim and the outside are ignored; uneven lighting is removed by
subtracting a heavily blurred copy of the plate; colonies are the pixels
that stand out from the flattened agar (Otsu's threshold); touching colonies
are split by watershed on the distance transform, seeded at its peaks.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import skew
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

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
    register,
    require_channel,
)

_WORK_PX = 600  # dish detection runs on a copy at most this many pixels across

FIND_DISH = Param(
    "find_dish",
    "Find the dish automatically",
    "bool",
    default=True,
    help="Look for the round plate and ignore everything outside it. Off: use the whole image.",
)
RIM = Param(
    "rim_percent",
    "Ignore rim",
    "float",
    5.0,
    0,
    40,
    decimals=1,
    help="Leave out this % of the dish radius at its edge, where the wall reflects light.",
    enabled_by="find_dish",
)
POLARITY = Param(
    "polarity",
    "Colonies are",
    "choice",
    "auto",
    choices=(
        ("Found automatically", "auto"),
        ("Lighter than the agar", "lighter"),
        ("Darker than the agar", "darker"),
    ),
    help="Auto: the side the brightness distribution leans to (colonies are the minority).",
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
    help="Raise to catch faint colonies, lower to ignore faint specks. 0: Otsu's threshold.",
)
SPLIT = Param(
    "split_touching",
    "Split touching colonies",
    "bool",
    default=True,
    help="Separate colonies that grew into each other, from their round shape.",
)
SEPARATION = Param(
    "min_separation_px",
    "Min. colony separation",
    "float",
    6.0,
    1.0,
    1000.0,
    unit="length",
    enabled_by="split_touching",
    help="Two colony centres closer than this count as one colony. Roughly a small colony's radius.",
)


@register
class ColonySegmenter:
    id = "colonies"
    name = "Colonies on a plate"
    summary = (
        "Finds the plate, evens out the lighting and counts colonies, splitting those that touch."
    )
    citations = ("opencfu2013", "otsu1979", "meyer1992", "opencv2000", "skimage2014")
    for_assays = ("colonies",)

    def parameters(self) -> list[Param]:
        return [
            TARGET_CHANNEL,
            FIND_DISH,
            RIM,
            POLARITY,
            SENSITIVITY,
            SPLIT,
            SEPARATION,
            Param(
                EXCLUDE_BORDERS.key, EXCLUDE_BORDERS.label, "bool", False, help=EXCLUDE_BORDERS.help
            ),
            Param(MIN_AREA.key, MIN_AREA.label, "float", 20.0, 0, 1e7, unit="area"),
            Param(MAX_AREA.key, MAX_AREA.label, "float", 1e6, 0, 1e9, unit="area"),
        ]

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        parameters: dict = {"size filter": area_range(params, scale)}
        if params.get("find_dish", True):
            parameters["rim ignored"] = f"{params.get('rim_percent', RIM.default):g}% of the radius"
        if params.get("split_touching", True):
            sep = params.get("min_separation_px", SEPARATION.default) * effective_scale(scale)
            parameters["minimum colony separation"] = f"{sep:g} {length_unit(scale)}"
        if params.get("sensitivity"):
            parameters["threshold adjustment"] = f"{-params['sensitivity']:+.0%}"
        dish = (
            "the plate was located by a Hough circle transform and its rim excluded; "
            if params.get("find_dish", True)
            else ""
        )
        split = (
            ", and touching colonies were split by distance-transform watershed"
            if params.get("split_touching", True)
            else ""
        )
        return (
            f"Colonies in the '{params.get('target_channel')}' channel were counted as in OpenCFU: "
            f"{dish}uneven illumination was removed by subtracting a blurred copy of the plate, "
            f"colonies were thresholded by Otsu's method{split}",
            parameters,
        )

    def segment(
        self, image_set: ImageSet, params: dict, progress: Progress | None = None
    ) -> SegmentationResult:
        report = progress or (lambda _v: None)
        plane = require_channel(image_set, params.get("target_channel")).data.astype(np.float32)
        h, w = plane.shape
        extras: dict = {}
        notes: list[str] = []
        if params.get("find_dish", True):
            dish = find_dish(plane)
            if dish is None:
                notes.append("No dish found: the whole image was used.")
        else:
            dish = None
        report(20)

        if dish is not None:
            x, y, r = dish
            r_inner = r * (1 - float(params.get("rim_percent", RIM.default)) / 100)
            extras["dish"] = [float(x), float(y), float(r_inner)]
            yy, xx = np.ogrid[:h, :w]
            inside = (xx - x) ** 2 + (yy - y) ** 2 <= r_inner**2
            sigma = max(3.0, r / 15)
        else:
            inside = np.ones((h, w), bool)
            sigma = max(3.0, min(h, w) / 30)

        # Flatten: subtract a blur of the plate, with the outside filled by the plate's median
        # so the dark surround doesn't bleed in.
        filled = np.where(inside, plane, float(np.median(plane[inside])))
        background = cv2.GaussianBlur(filled, (0, 0), sigma)
        flat = plane - background
        polarity = params.get("polarity", "auto")
        if polarity == "auto":
            polarity = "lighter" if skew(flat[inside].ravel()) >= 0 else "darker"
        signal = flat if polarity == "lighter" else -flat
        extras["polarity"] = polarity
        report(45)

        values = signal[inside]
        level = _otsu_level(values)
        level *= 1 - float(params.get("sensitivity", 0.0))
        mask = (signal > level) & inside
        mask = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        ).astype(bool)
        mask = ndimage.binary_fill_holes(mask)
        report(65)

        if params.get("split_touching", True):
            labels = split_touching(
                mask, float(params.get("min_separation_px", SEPARATION.default))
            )
        else:
            labels, _ = ndimage.label(mask, structure=np.ones((3, 3), bool))
        report(95)
        return SegmentationResult(labels.astype(np.int32), notes=notes, extras=extras)


def find_dish(plane: np.ndarray) -> tuple[float, float, float] | None:
    """The plate as (x, y, radius) in pixels: the largest circle the Hough transform finds."""
    h, w = plane.shape
    factor = min(1.0, _WORK_PX / max(h, w))
    small = cv2.resize(plane, (max(1, int(w * factor)), max(1, int(h * factor))))
    lo, hi = np.percentile(small, [1, 99])
    small8 = np.clip((small - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    small8 = cv2.medianBlur(small8, 5)
    side = min(small8.shape)
    circles = cv2.HoughCircles(
        small8,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=side,
        param1=80,
        param2=30,
        minRadius=int(side * 0.25),
        maxRadius=int(side * 0.6),
    )
    if circles is None:
        return None
    x, y, r = max(circles[0], key=lambda c: c[2])
    return float(x / factor), float(y / factor), float(r / factor)


def split_touching(mask: np.ndarray, separation: float) -> np.ndarray:
    """Label `mask`, splitting round objects that touch (watershed on the distance map)."""
    distance = ndimage.distance_transform_edt(mask)
    regions, _ = ndimage.label(mask, structure=np.ones((3, 3), bool))
    smooth = ndimage.gaussian_filter(distance, 1.0)
    peaks = peak_local_max(
        smooth, min_distance=max(1, int(round(separation))), labels=regions, exclude_border=False
    )
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-distance, markers, mask=mask)
    unseeded = mask & (labels == 0)
    if unseeded.any():
        extra, _ = ndimage.label(unseeded)
        labels = np.where(extra > 0, extra + labels.max(), labels)
    return labels


def _otsu_level(values: np.ndarray) -> float:
    """Otsu's threshold of float values (on a 256-bin histogram of their range)."""
    if not values.size:
        return 0.0
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return hi
    scaled = ((values - lo) / (hi - lo) * 255).astype(np.uint8)
    level, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return lo + float(level) / 255 * (hi - lo)
