"""The images being analysed: named, coloured channels in their native bit depth.

Files are read with the SDK's ``read_image``, so 16-bit and float data reach
the segmenters unchanged and the pixel size comes from the file's metadata
(OME, ImageJ, TIFF resolution, ND2) when it has one. Scaling to 8 bits
happens only where it is needed: for display, and for the classical
segmenters that run OpenCV's 8-bit operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from karcytics_sdk.contrib.imaging import ImageData, PixelSize, read_image

# RGB tint for each channel colour.
COLOR_MAPS: dict[str, tuple[float, float, float]] = {
    "gray": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}
CHANNEL_COLORS = ["gray", "magenta", "green", "blue", "red", "cyan", "yellow"]

# Common fluorophore/stain names → the colour microscopists expect to see them in.
_NAME_COLORS = [
    (("dapi", "hoechst", "blue"), "blue"),
    (("cy5", "647", "far red", "far-red", "magenta"), "magenta"),  # before "red"
    (("gfp", "fitc", "488", "green"), "green"),
    (("mcherry", "rfp", "tritc", "cy3", "568", "594", "red"), "red"),
]
_COLOR_CYCLE = ["green", "magenta", "blue", "red", "cyan", "yellow"]
_RGB_CHANNELS = 3
_ALPHA = "alpha"


def color_for_name(name: str) -> str | None:
    """The conventional colour for a stain named `name`, or None if unknown."""
    lowered = name.lower()
    for keys, color in _NAME_COLORS:
        if any(k in lowered for k in keys):
            return color
    return None


@dataclass
class Channel:
    """One stain/marker: a 2-D plane in its native dtype."""

    name: str
    data: np.ndarray
    color: str = "gray"
    path: str = ""
    visible: bool = True

    def as_uint8(self) -> np.ndarray:
        """The plane scaled to 0–255: unchanged for 8-bit data, min–max otherwise."""
        if self.data.dtype == np.uint8:
            return self.data
        plane = self.data.astype(np.float32)
        lo, hi = float(plane.min()), float(plane.max())
        if hi <= lo:
            return np.zeros(plane.shape, dtype=np.uint8)
        return np.rint((plane - lo) * (255.0 / (hi - lo))).astype(np.uint8)


@dataclass
class ImageSet:
    """The channels of one field of view, all the same height and width."""

    channels: list[Channel] = field(default_factory=list)
    pixel_size: PixelSize | None = None

    def get(self, name: str | None) -> Channel | None:
        return next((c for c in self.channels if c.name == name), None)

    @property
    def shape(self) -> tuple[int, int] | None:
        return self.channels[0].data.shape if self.channels else None  # type: ignore[return-value]

    def clear(self) -> None:
        self.channels.clear()
        self.pixel_size = None

    def add_file(self, path: str | Path, name: str, color: str = "gray") -> list[Channel]:
        """Read `path` and append one channel per image channel.

        A single-channel file becomes `name`, in `color` unless the name
        says otherwise ("DAPI" → blue). A multi-channel file becomes
        ``"name (<channel>)"`` for each channel, named from the file's
        metadata where it has names. Only the first time point and
        z-plane are used.

        Raises ``ValueError`` if the image's size differs from the channels
        already loaded, and the reader's errors if the file can't be read.
        """
        image = read_image(path)
        planes = _planes(image)
        if self.shape is not None and planes[0][1].shape != self.shape:
            h, w = self.shape
            ph, pw = planes[0][1].shape
            raise ValueError(
                f"'{Path(path).name}' is {pw}×{ph} px but the loaded channels are {w}×{h} px."
            )

        if len(planes) == 1:
            colors = [color_for_name(name) or color]
            names = [name]
        else:
            colors = _colors_for(image, [n for n, _ in planes])
            names = [f"{name} ({label})" for label, _ in planes]

        added = [
            Channel(n, data, c, str(path))
            for n, (_, data), c in zip(names, planes, colors, strict=True)
        ]
        self.channels.extend(added)
        if self.pixel_size is None:
            self.pixel_size = image.pixel_size
        return added

    def composite(self) -> np.ndarray | None:
        """Additive RGB blend of the visible channels, as uint8 (H, W, 3)."""
        if not self.channels:
            return None
        h, w = self.channels[0].data.shape
        out = np.zeros((h, w, 3), dtype=np.float32)
        for ch in self.channels:
            if not ch.visible:
                continue
            tint = np.asarray(COLOR_MAPS.get(ch.color, COLOR_MAPS["gray"]), dtype=np.float32)
            out += ch.as_uint8()[..., np.newaxis].astype(np.float32) * tint
        return np.clip(out, 0, 255).astype(np.uint8)


def _planes(image: ImageData) -> list[tuple[str, np.ndarray]]:
    """(label, plane) per channel at t=0, z=0; alpha is dropped."""
    defaults = _default_names(image)
    meaningful = image.channel_names != defaults
    out = []
    for i in range(image.sizes["C"]):
        label = image.channel_names[i] if meaningful else f"Ch {i + 1}"
        if label.lower() == _ALPHA:
            continue
        out.append((label, image.plane(c=i)))
    return out


def _default_names(image: ImageData) -> list[str]:
    """The names ``read_image`` gives channels when the file has none."""
    stem = image.path.stem
    n = image.sizes["C"]
    return [stem] if n == 1 else [f"{stem} (Ch {i + 1})" for i in range(n)]


def _colors_for(image: ImageData, labels: list[str]) -> list[str]:
    looks_rgb = len(labels) == _RGB_CHANNELS and image.dtype == np.uint8
    if looks_rgb and all(label.startswith("Ch ") for label in labels):
        return ["red", "green", "blue"]
    colors: list[str] = []
    for label in labels:
        unused = next((c for c in _COLOR_CYCLE if c not in colors), "gray")
        colors.append(color_for_name(label) or unused)
    return colors
