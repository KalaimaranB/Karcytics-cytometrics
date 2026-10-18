"""Synthetic sample images for trying every CytoMetrics assay by hand.

    uv run python tests/data/generate_samples.py

writes ``tests/data/samples/`` (one folder per assay) and ``truth.json`` with
what each image really contains (counts, % positive, wound width, …), so
results can be checked against it. Everything is drawn here from a fixed
seed: no real data, no licence to track (see ``LICENSES.md``).

The images imitate the real thing closely enough to exercise the
algorithms (blur, noise, uneven light, touching objects, debris), but they
are not real microscopy: use them to learn the workflow, not to validate.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
import tifffile

OUT = Path(__file__).parent / "samples"
SEED = 2026
truth: dict[str, dict] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────


def save_tif(path: Path, image: np.ndarray, um_per_px: float) -> None:
    """A 16-bit ImageJ TIFF carrying its pixel size (CytoMetrics calibrates from it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        np.clip(image, 0, 65535).astype(np.uint16),
        imagej=True,
        resolution=(1 / um_per_px, 1 / um_per_px),
        metadata={"unit": "micron"},
        compression="zlib",
    )


def save_photo(path: Path, image: np.ndarray) -> None:
    """An 8-bit RGB (or grey) photo, PNG or JPEG by the extension. `image` is RGB."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(image, 0, 255).astype(np.uint8)
    if data.ndim == 3:  # noqa: PLR2004
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if path.suffix == ".jpg" else []
    cv2.imwrite(str(path), data, params)


def scatter(rng, n, shape, min_dist, margin=0, touching=0.0, tries=20000):
    """`n` centres at least `min_dist` apart; a `touching` share placed against a neighbour."""
    h, w = shape
    points: list[tuple[float, float]] = []
    for _ in range(tries):
        if len(points) >= n:
            break
        if points and rng.random() < touching:
            px, py = points[rng.integers(len(points))]
            angle = rng.uniform(0, 2 * np.pi)
            x, y = px + 0.8 * min_dist * np.cos(angle), py + 0.8 * min_dist * np.sin(angle)
            close = 0.75 * min_dist
        else:
            x, y = rng.uniform(margin, w - margin), rng.uniform(margin, h - margin)
            close = min_dist
        if not (margin <= x < w - margin and margin <= y < h - margin):
            continue
        if all((x - a) ** 2 + (y - b) ** 2 >= close**2 for a, b in points):
            points.append((x, y))
    return points


def ellipse(image, centre, axes, angle, value):
    cv2.ellipse(
        image,
        (int(round(centre[0] * 16)), int(round(centre[1] * 16))),
        (int(round(axes[0] * 16)), int(round(axes[1] * 16))),
        angle,
        0,
        360,
        value,
        -1,
        cv2.LINE_AA,
        4,
    )


def camera(rng, signal, background=200.0, gain=1.0, read_noise=6.0, blur=1.2):
    """Optics and camera: blur, uneven illumination, offset, shot and read noise."""
    h, w = signal.shape
    yy, xx = np.mgrid[:h, :w]
    vignette = 1 - 0.25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2) * 4
    image = cv2.GaussianBlur(signal, (0, 0), blur) * vignette + background
    image = rng.poisson(np.clip(image / gain, 0, None)) * gain
    return image + rng.normal(0, read_noise, image.shape)


def texture(rng, shape, sigma=1.5, strength=1.0):
    noise = cv2.GaussianBlur(rng.normal(0, 1, shape).astype(np.float32), (0, 0), sigma)
    return noise / (noise.std() + 1e-9) * strength


# ── Fluorescence: cells, nuclei, transfection ────────────────────────────────


def fluorescent_cells(rng, shape, n, positive=None, touching=0.15, gfp_dim=(40, 120)):
    """DAPI and GFP planes of `n` cells; `positive` is the share expressing GFP (None: all)."""
    dapi = np.zeros(shape, np.float32)
    gfp = np.zeros(shape, np.float32)
    centres = scatter(rng, n, shape, 30, margin=20, touching=touching)
    expressing = 0
    for x, y in centres:
        a, b = rng.uniform(15, 22), rng.uniform(10, 15)
        angle = rng.uniform(0, 180)
        on = positive is None or rng.random() < positive
        expressing += on
        level = rng.lognormal(np.log(1800), 0.4) if on else rng.uniform(*gfp_dim)
        cell = np.zeros(shape, np.float32)
        ellipse(cell, (x, y), (a, b), angle, 1.0)
        gfp += cell * level * (1 + 0.15 * texture(rng, shape, 2.0))
        nx, ny = x + rng.normal(0, 2), y + rng.normal(0, 2)
        ellipse(dapi, (nx, ny), (a * 0.45, b * 0.55), angle, rng.normal(2500, 300))
    return dapi, gfp, len(centres), expressing


def make_cells(rng):
    dapi, gfp, n, _ = fluorescent_cells(rng, (768, 768), 80)
    save_tif(OUT / "cells" / "cells_DAPI.tif", camera(rng, dapi), 0.65)
    save_tif(OUT / "cells" / "cells_GFP.tif", camera(rng, gfp), 0.65)
    truth["cells"] = {"cells": n, "um_per_px": 0.65, "note": "about 15% placed touching"}


def make_nuclei(rng):
    shape = (768, 768)
    dapi = np.zeros(shape, np.float32)
    centres = scatter(rng, 240, shape, 26, margin=20)
    phases = {"Sub-G1": 0, "G1": 0, "S": 0, "G2/M": 0}
    for x, y in centres:
        roll = rng.random()
        if roll < 0.05:
            phase, dna = "Sub-G1", rng.uniform(0.6, 1.3)
        elif roll < 0.60:
            phase, dna = "G1", rng.normal(2.0, 0.08)
        elif roll < 0.80:
            phase, dna = "S", rng.uniform(2.3, 3.7)
        else:
            phase, dna = "G2/M", rng.normal(4.0, 0.15)
        phases[phase] += 1
        r = 8.5 * (dna / 2) ** (1 / 3)  # volume grows with DNA content
        a, b = r * rng.uniform(1.0, 1.2), r * rng.uniform(0.85, 1.0)
        mask = np.zeros(shape, np.float32)
        ellipse(mask, (x, y), (a, b), rng.uniform(0, 180), 1.0)
        dapi += mask * (dna * 600_000 / mask.sum()) * (1 + 0.1 * texture(rng, shape, 1.2))
    save_tif(OUT / "nuclei" / "nuclei_DAPI.tif", camera(rng, dapi, blur=1.0), 0.65)
    total = sum(phases.values())
    truth["nuclei"] = {
        "nuclei": total,
        "percent": {k: round(100 * v / total, 1) for k, v in phases.items()},
        "um_per_px": 0.65,
        "note": "integrated DAPI intensity is proportional to DNA content (G1 = 2N)",
    }


def make_transfection(rng):
    for name, share in (("control", 0.0), ("sample", 0.35)):
        dapi, gfp, n, on = fluorescent_cells(rng, (768, 768), 90, positive=share, touching=0.05)
        save_tif(OUT / "transfection" / f"{name}_DAPI.tif", camera(rng, dapi), 0.65)
        save_tif(OUT / "transfection" / f"{name}_GFP.tif", camera(rng, gfp), 0.65)
        truth[f"transfection_{name}"] = {
            "cells": n,
            "gfp_positive": on,
            "percent_positive": round(100 * on / n, 1),
        }


# ── Bright field and photos ──────────────────────────────────────────────────


def make_hemocytometer(rng):
    """A Neubauer chamber at 1 µm/px: 0.25 mm squares are 250 px; trypan blue fills dead cells."""
    h, w = 1000, 1000
    image = np.ones((h, w, 3), np.float32) * np.array([205, 203, 196])
    image *= 1 + 0.03 * texture(rng, (h, w), 30)[..., None]
    for v in range(0, 1001, 250):  # grid lines (triple lines on the big squares)
        for off in (-3, 0, 3):
            p = int(np.clip(v + off, 0, 999))
            image[:, p] *= 0.72
            image[p, :] *= 0.72
    centres = scatter(rng, 140, (h, w), 16, margin=8, touching=0.05)
    dead = 0
    for x, y in centres:
        r = rng.uniform(5.5, 8)
        is_dead = rng.random() < 0.2
        dead += is_dead
        ring = np.zeros((h, w), np.float32)
        cv2.circle(ring, (int(x), int(y)), int(r), 1.0, 2, cv2.LINE_AA)
        body = np.zeros((h, w), np.float32)
        cv2.circle(body, (int(x), int(y)), int(r) - 1, 1.0, -1, cv2.LINE_AA)
        if is_dead:  # blue: dark in red, a bit dark in green
            colour = np.array([70, 95, 150], np.float32)
            image = image * (1 - body[..., None]) + body[..., None] * colour
        else:  # bright, refractile centre
            image = image * (1 - body[..., None]) + body[..., None] * np.array([235, 235, 230])
        image *= 1 - 0.45 * ring[..., None]
    image = cv2.GaussianBlur(image, (0, 0), 0.8) + rng.normal(0, 4, image.shape)
    save_photo(OUT / "hemocytometer" / "trypan_blue_1um_per_px.png", image)
    truth["hemocytometer"] = {
        "cells": len(centres),
        "dead": dead,
        "viability_percent": round(100 * (1 - dead / len(centres)), 1),
        "grid_spacing_px": 250,
        "grid_spacing_mm": 0.25,
        "image_area_mm2": 1.0,
        "cells_per_ml_whole_image_dilution_2": round(len(centres) / 1e-4 * 2),
        "note": "whole image = 1 mm² × 0.1 mm depth = 0.1 µL",
    }


def plate(rng, n, size=900, spacing=22):
    """An agar plate photo with `n` cream colonies (some touching), lit from one side."""
    h = w = size
    yy, xx = np.mgrid[:h, :w]
    image = np.ones((h, w, 3), np.float32) * 22
    cx, cy, radius = w / 2 + rng.normal(0, 8), h / 2 + rng.normal(0, 8), size * 0.44
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    light = (1 + 0.25 * (xx - cx) / radius)[..., None]
    agar = np.array([196, 160, 98], np.float32) * light
    image[inside] = agar[inside]
    rim = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - radius) < 6
    image[rim] = 235
    centres = scatter(rng, n, (h, w), spacing, touching=0.12)
    kept = [(x, y) for x, y in centres if (x - cx) ** 2 + (y - cy) ** 2 < (radius * 0.88) ** 2]
    for x, y in kept:
        r = rng.uniform(4, 10)
        colony = np.zeros((h, w), np.float32)
        cv2.circle(colony, (int(x), int(y)), int(r), 1.0, -1, cv2.LINE_AA)
        colony = cv2.GaussianBlur(colony, (0, 0), 1.0)[..., None]
        image = image * (1 - colony) + colony * np.array([238, 226, 196]) * light
    image += rng.normal(0, 3, image.shape)
    return image, len(kept)


def make_colonies(rng):
    image, n = plate(rng, 160)
    save_photo(OUT / "colonies" / "plate_10e-6_100uL.jpg", image)
    dense, n_dense = plate(rng, 900, spacing=15)
    save_photo(OUT / "colonies" / "plate_dense.jpg", dense)
    truth["colonies"] = {
        "plate_10e-6_100uL.jpg": {
            "colonies": n,
            "cfu_per_ml": round(n * 1e6 / 0.1),
            "settings": "dilution 1000000, volume 100 µL",
        },
        "plate_dense.jpg": {"colonies": n_dense, "note": "over 300: CytoMetrics warns"},
    }


def make_bacteria(rng):
    """Phase contrast at 0.1 µm/px: dark rods with a bright halo, some dividing."""
    h, w = 768, 768
    signal = np.zeros((h, w), np.float32)
    centres = scatter(rng, 110, (h, w), 45, margin=30)
    dividing = 0
    lengths = []
    for x, y in centres:
        length, width = rng.uniform(20, 40), rng.uniform(8, 10)
        angle = rng.uniform(0, np.pi)
        bend = rng.normal(0, 0.15)
        divides = rng.random() < 0.15
        dividing += divides
        lengths.append((length + width) * 0.1)  # the line plus its rounded ends
        dx, dy = np.cos(angle), np.sin(angle)
        t = np.linspace(-0.5, 0.5, 12)
        pts = np.stack(
            [
                x + t * length * dx - bend * length * (t**2) * dy,
                y + t * length * dy + bend * length * (t**2) * dx,
            ],
            axis=1,
        )
        if divides:  # one cell pinched at the middle to about 60% of its width
            for half in (pts[:5], pts[7:]):
                cv2.polylines(signal, [half.astype(np.int32)], False, 1.0, int(width), cv2.LINE_AA)
            waist = pts[4:8].astype(np.int32)
            cv2.polylines(signal, [waist], False, 1.0, max(2, int(width * 0.6)), cv2.LINE_AA)
        else:
            cv2.polylines(signal, [pts.astype(np.int32)], False, 1.0, int(width), cv2.LINE_AA)
    body = cv2.GaussianBlur(signal, (0, 0), 1.0)
    halo = cv2.GaussianBlur(signal, (0, 0), 4.0) - body
    image = 1500 - 900 * body + 500 * np.clip(halo, 0, None) + 30 * texture(rng, (h, w), 20)
    image = image + rng.normal(0, 25, image.shape)
    save_tif(OUT / "bacteria" / "rods_phase_contrast.tif", image, 0.1)
    truth["bacteria"] = {
        "cells": len(centres),
        "dividing": dividing,
        "percent_dividing": round(100 * dividing / len(centres), 1),
        "median_length_um": round(float(np.median(lengths)), 2),
        "width_um": "0.8–1.0",
        "um_per_px": 0.1,
    }


def phase_cells(rng, covered: np.ndarray) -> np.ndarray:
    """Phase-contrast look: textured, haloed cells where `covered`, a smooth dish elsewhere."""
    h, w = covered.shape
    cells = covered.astype(np.float32)
    edges = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3)))
    inner = texture(rng, (h, w), 1.8, 22) + 18 * (texture(rng, (h, w), 5) > 1.2)
    image = (
        128
        + cells * inner
        - 25 * cells
        + 45 * cv2.GaussianBlur(edges.astype(np.float32), (0, 0), 1.5)
    )
    image += 8 * texture(rng, (h, w), 60)  # uneven light
    return image + rng.normal(0, 3, (h, w))


def make_confluence(rng):
    shape = (768, 768)
    for target in (30, 75):
        field = texture(rng, shape, 25)
        level = np.percentile(field, 100 - target)
        covered = field > level
        save_photo(OUT / "confluence" / f"confluence_{target}pct.png", phase_cells(rng, covered))
        truth[f"confluence_{target}pct"] = {"percent_covered": round(100 * covered.mean(), 1)}


def make_wound(rng):
    h, w = 768, 1024
    rows = np.arange(h)
    for hours, width in ((0, 380), (12, 220), (24, 70)):
        left = w / 2 - width / 2 + 25 * texture(rng, (h, 1), 12)[:, 0]
        right = w / 2 + width / 2 + 25 * texture(rng, (h, 1), 12)[:, 0]
        if hours == 24:  # cells have started to bridge the gap
            for start in (150, 520):
                right[start : start + 40] = left[start : start + 40]
        xx = np.arange(w)[None, :]
        open_ = (xx >= left[:, None]) & (xx < right[:, None])
        stragglers = texture(rng, (h, w), 3) > 2.2  # single cells that wandered in
        covered = ~open_ | stragglers
        save_photo(OUT / "wound" / f"A01_{hours:02d}h.png", phase_cells(rng, covered))
        open_area = int((~covered).sum())
        truth[f"wound_{hours:02d}h"] = {
            "open_area_px": open_area,
            "mean_width_px": round(open_area / h, 1),
        }
    start = truth["wound_00h"]["open_area_px"]
    for key in ("wound_12h", "wound_24h"):
        truth[key]["closure_percent"] = round(100 * (1 - truth[key]["open_area_px"] / start), 1)
    del rows


def make_spheroids(rng):
    """Bright-field wells at 2 µm/px: one dark spheroid growing over a week, plus debris."""
    h = w = 768
    yy, xx = np.mgrid[:h, :w]
    for day, radius in ((1, 60), (4, 105), (7, 150)):
        image = 180 - 40 * (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w / 2) ** 2)
        cx, cy = w / 2 + rng.normal(0, 30), h / 2 + rng.normal(0, 30)
        theta = np.arctan2(yy - cy, xx - cx)
        wobble = 1 + 0.04 * np.sin(5 * theta + rng.uniform(0, 6)) + 0.02 * np.sin(11 * theta)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (radius * wobble)
        body = np.clip(1 - dist, 0, None) ** 0.3  # denser towards the (necrotic) core
        image = image * (1 - 0.65 * body)
        for x, y in scatter(rng, 25, (h, w), 30, margin=10):
            if (x - cx) ** 2 + (y - cy) ** 2 > (radius * 1.4) ** 2:
                cv2.circle(image, (int(x), int(y)), int(rng.uniform(3, 6)), 90.0, -1, cv2.LINE_AA)
        image = image + 10 * texture(rng, (h, w), 1.5) * (body > 0) + rng.normal(0, 3, (h, w))
        save_photo(OUT / "spheroids" / f"spheroid_day{day}.png", image)
        truth[f"spheroid_day{day}"] = {
            "diameter_px": 2 * radius,
            "diameter_um_at_2um_per_px": 4 * radius,
            "volume_um3": round(math.pi * (4 * radius) ** 3 / 6),
        }


# ── A small plate for the Batch step ─────────────────────────────────────────


def make_batch(rng):
    """Six wells, one DAPI and one GFP file each; treated wells have more GFP+ cells."""
    folder = OUT / "batch_plate"
    rows = []
    for well, condition, replicate, share in (
        ("A01", "Control", 1, 0.05),
        ("A02", "Control", 2, 0.07),
        ("A03", "Control", 3, 0.04),
        ("B01", "Treated", 1, 0.40),
        ("B02", "Treated", 2, 0.46),
        ("B03", "Treated", 3, 0.38),
    ):
        dapi, gfp, n, on = fluorescent_cells(rng, (512, 512), 45, positive=share, touching=0.05)
        save_tif(folder / f"{well}_DAPI.tif", camera(rng, dapi), 0.65)
        save_tif(folder / f"{well}_GFP.tif", camera(rng, gfp), 0.65)
        rows.append({"Well": well, "Condition": condition, "Replicate": replicate})
        truth[f"batch_{well}"] = {"cells": n, "percent_positive": round(100 * on / n, 1)}
    with (folder / "layout.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Well", "Condition", "Replicate"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    for make in (
        make_cells,
        make_nuclei,
        make_transfection,
        make_hemocytometer,
        make_colonies,
        make_bacteria,
        make_confluence,
        make_wound,
        make_spheroids,
        make_batch,
    ):
        make(rng)
        print(f"✓ {make.__name__[5:]}")
    (OUT / "truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n")
    print(f"Written to {OUT}")


if __name__ == "__main__":
    main()
