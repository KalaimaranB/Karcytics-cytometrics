"""Write cells in formats other tools open: ImageJ/Fiji ROIs and Cellpose ``_seg.npy``.

Both are written from the cells as they are now (after deletions and
manually drawn cells), not from the raw segmentation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from roifile import ImagejRoi, roiwrite
from skimage.segmentation import find_boundaries


def cells_to_labels(cells: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise the cell outlines into a label mask: the k-th cell gets label k (1-based).

    Where outlines overlap, the later cell wins.
    """
    labels = np.zeros(shape, dtype=np.int32)
    for k, cell in enumerate(cells, start=1):
        points = np.round(np.asarray(cell["points"], dtype=np.float64)).astype(np.int32)
        if len(points) >= 3:  # noqa: PLR2004  (a polygon)
            cv2.fillPoly(labels, [points], color=k)
    return labels


def write_imagej_rois(path: str | Path, cells: list[dict]) -> Path:
    """One polygon ROI per cell, named "Cell <id>", in a zip ImageJ/Fiji's ROI Manager opens."""
    path = Path(path)
    rois = [
        ImagejRoi.frompoints(np.asarray(c["points"], dtype=np.float32), name=f"Cell {c['id']}")
        for c in cells
        if len(c["points"]) >= 3  # noqa: PLR2004
    ]
    if path.exists():
        path.unlink()  # roiwrite appends to an existing zip
    roiwrite(path, rois, mode="w")
    return path


def write_cellpose_seg(
    path: str | Path, cells: list[dict], shape: tuple[int, int], image_path: str = ""
) -> Path:
    """A ``*_seg.npy`` the Cellpose GUI opens (and Cellpose can train on).

    Holds the label mask, its outlines, which cells were drawn by hand and
    the source image's path. Flows are left out: they're optional, and the
    GUI recomputes what it needs.
    """
    path = Path(path)
    masks = cells_to_labels(cells, shape)
    dtype = np.uint16 if len(cells) < 2**16 - 1 else np.uint32
    outlines = masks * find_boundaries(masks, mode="inner")
    data = {
        "masks": masks.astype(dtype),
        "outlines": outlines.astype(dtype),
        "ismanual": np.array([c.get("source") == "manual" for c in cells], dtype=bool),
        "filename": image_path,
        "chan_choose": [0, 0],
        "diameter": np.nan,
    }
    np.save(path, np.array(data, dtype=object), allow_pickle=True)  # 0-d: .item() gives the dict
    return path
