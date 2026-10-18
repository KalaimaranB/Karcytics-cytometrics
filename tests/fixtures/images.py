"""Synthetic microscopy images with known ground truth.

Every fixture image is generated, never downloaded, so tests carry no data
license and the expected counts/areas are exact by construction. Shapes are
drawn with OpenCV so they match what the pipelines' own contour maths sees.
"""

from __future__ import annotations

import cv2
import numpy as np

from karcytics_plugins.cytometrics.analysis.model import Channel, ImageSet

CANVAS = 256
FOREGROUND = 200  # 8-bit intensity of a "cell"; background is 0 plus noise
NOISE_SEED = 1234


def blank(size: int = CANVAS) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


def draw_discs(
    centres: list[tuple[int, int]], radius: int, size: int = CANVAS, value: int = FOREGROUND
) -> np.ndarray:
    img = blank(size)
    for x, y in centres:
        cv2.circle(img, (x, y), radius, value, thickness=-1)
    return img


def add_noise(img: np.ndarray, sigma: float = 6.0, seed: int = NOISE_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# Four well-separated discs, radius 15, all far from the border.
ISOLATED_CENTRES = [(60, 60), (190, 60), (60, 190), (190, 190)]
ISOLATED_RADIUS = 15

# Two discs overlapping into one blob: the classic "clump" to split.
TOUCHING_CENTRES = [(110, 128), (146, 128)]
TOUCHING_RADIUS = 22

# One disc cut by the left image edge.
BORDER_CENTRE = (5, 128)


def isolated_discs() -> np.ndarray:
    return add_noise(draw_discs(ISOLATED_CENTRES, ISOLATED_RADIUS))


def touching_pair() -> np.ndarray:
    return add_noise(draw_discs(TOUCHING_CENTRES, TOUCHING_RADIUS))


def isolated_plus_border_disc() -> np.ndarray:
    return add_noise(draw_discs([*ISOLATED_CENTRES, BORDER_CENTRE], ISOLATED_RADIUS))


def nuclei_for_touching_pair() -> np.ndarray:
    """Small bright seeds at each touching cell's centre (a DAPI-like channel)."""
    return add_noise(draw_discs(TOUCHING_CENTRES, 7))


def stack(*channels: tuple[str, np.ndarray]) -> ImageSet:
    return ImageSet([Channel(name, data) for name, data in channels])
