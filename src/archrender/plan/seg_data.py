"""Synthetic training data for the plan segmentation model (ADR-M06, owner answer Q-4: synthetic
data only, no firm archive or third-party datasets).

A sample is a scanned synthetic sheet (any wall style, quality and resolution) with a per-pixel
label map rendered from the exact ground-truth plan in the scan's pixels:

    0 background · 1 wall · 2 door (the opening in the wall) · 3 window · 4 room interior

Tiles are cut at full resolution around the plan (the model, like the VLM, never sees a downscaled
sheet). Generation is CPU-only and deterministic per seed; ``archrender.plan.seg_train`` trains on
it (on the pod).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from shapely.geometry import LineString

from archrender.core.schemas.plan import PlanGraph, Segment
from archrender.scene.geometry import centerline_coords
from archrender.synth.plan import Variant, gt_plan, random_spec
from archrender.synth.raster import scan
from archrender.synth.sheets import WALL_STYLES, floor_plan_page

CLASSES = ("background", "wall", "door", "window", "room")
BACKGROUND, WALL, DOOR, WINDOW, ROOM = range(5)
VARIANTS: list[Variant] = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]


@dataclass
class Sample:
    image: NDArray[np.uint8]  # (H, W, 3) RGB
    label: NDArray[np.uint8]  # (H, W) class index
    meta: dict[str, object]


def _fill(
    mask: NDArray[np.uint8], ring: list[tuple[float, float]], to_px: NDArray[np.float64], value: int
) -> None:
    pts = np.asarray(ring, float) @ to_px[:, :2].T + to_px[:, 2]
    cv2.fillPoly(mask, [np.round(pts * 16).astype(np.int32)], value, cv2.LINE_8, 4)


def label_map(
    plan: PlanGraph, to_px: list[list[float]] | NDArray[np.float64], shape: tuple[int, ...]
) -> NDArray[np.uint8]:
    """Labels of ``plan`` (metres) drawn through the affine ``to_px`` into an image of ``shape``:
    rooms, then walls, then the openings cut into their walls."""
    m = np.asarray(to_px, float)[:2]
    out = np.zeros(shape[:2], np.uint8)
    for r in plan.rooms:
        _fill(out, [(p.x, p.y) for p in r.polygon], m, ROOM)
    for w in plan.walls:
        poly = LineString(centerline_coords(w)).buffer(
            w.thickness_m.value / 2, cap_style="square", join_style="mitre"
        )
        _fill(out, list(poly.exterior.coords), m, WALL)
    walls = {w.id: w for w in plan.walls}
    for o in plan.openings:
        w = walls[o.host_wall]
        if not isinstance(w.centerline, Segment):
            continue
        a = np.array([w.centerline.a.x, w.centerline.a.y])
        b = np.array([w.centerline.b.x, w.centerline.b.y])
        u = (b - a) / float(np.hypot(*(b - a)))
        n = np.array([-u[1], u[0]]) * (w.thickness_m.value / 2 + 0.01)
        c0 = a + u * (o.offset_m.value - o.width_m.value / 2)
        c1 = a + u * (o.offset_m.value + o.width_m.value / 2)
        quad = [tuple(c0 + n), tuple(c1 + n), tuple(c1 - n), tuple(c0 - n)]
        _fill(out, quad, m, WINDOW if o.type == "window" else DOOR)
    return out


def sheet_sample(seed: int, *, dpi: float | None = None, quality: str | None = None) -> Sample:
    """One scanned sheet and its full label map."""
    rng = np.random.default_rng(seed)
    spec = random_spec(rng, variant=VARIANTS[seed % len(VARIANTS)])
    style = str(rng.choice(WALL_STYLES))
    sheet = floor_plan_page(rng, spec=spec, wall_style=style)
    q = quality or str(rng.choice(["clean", "medium", "noisy"]))
    data, _, gt = scan(sheet.pdf, sheet.gt, rng, dpi=dpi, quality=q)
    image = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    label = label_map(gt_plan(spec), gt["plan_to_page_px"], image.shape)
    return Sample(image, label, {"seed": seed, "style": style, "quality": q, "dpi": gt.get("dpi")})


def tiles(sample: Sample, size: int = 512, *, stride: int | None = None) -> Iterator[Sample]:
    """Full-resolution tiles covering the drawing (tiles with no plan label are skipped)."""
    stride = stride or size // 2
    ys, xs = np.nonzero(sample.label)
    if not len(xs):
        return
    h, w = sample.label.shape
    x0, x1 = max(0, int(xs.min()) - size // 4), min(w, int(xs.max()) + size // 4)
    y0, y1 = max(0, int(ys.min()) - size // 4), min(h, int(ys.max()) + size // 4)
    for ty in range(y0, max(y0 + 1, y1 - size + 1), stride):
        for tx in range(x0, max(x0 + 1, x1 - size + 1), stride):
            lab = sample.label[ty : ty + size, tx : tx + size]
            if lab.shape != (size, size) or not lab.any():
                continue
            yield Sample(
                sample.image[ty : ty + size, tx : tx + size],
                lab,
                {**sample.meta, "tile": (tx, ty)},
            )


def class_frequencies(label: NDArray[np.uint8]) -> NDArray[np.float64]:
    counts = np.bincount(label.ravel(), minlength=len(CLASSES)).astype(np.float64)
    out: NDArray[np.float64] = counts / max(1.0, float(counts.sum()))
    return out
