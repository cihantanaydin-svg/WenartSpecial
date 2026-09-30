"""North arrow: CV-measured direction (ARCHITECTURE §S1).

The arrow is located from its letter ("K" for Kuzey, "N") and the nearest roughly circular symbol
of plausible size; the direction is measured on the ink inside the symbol (principal axis of the
arrow shape), with the letter disambiguating which end is north. Without a letter the arrow is not
guessed: the ``north_arrow_missing`` trigger lets the VLM point at it (hint only, ADR-S19).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.document import Word
from archrender.core.schemas.provenance import fact
from archrender.core.schemas.understanding import NorthArrow

LETTERS = {"K", "N"}


def _circles(ink: NDArray[np.uint8], px_per_mm: float) -> list[tuple[float, float, float]]:
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out: list[tuple[float, float, float]] = []
    if hierarchy is None:
        return out
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] != -1 or len(c) < 20:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if not (4 * px_per_mm <= r <= 30 * px_per_mm):
            continue
        # the outline must be close to that circle all around
        d = np.hypot(c[:, 0, 0] - cx, c[:, 0, 1] - cy)
        if np.percentile(np.abs(d - r), 90) > 0.08 * r:
            continue
        out.append((float(cx), float(cy), float(r)))
    return out


def measure_north(
    page_id: str, gray: NDArray[np.uint8], words: list[Word], dpi: float
) -> NorthArrow | None:
    px_per_mm = dpi / 25.4
    ink = (gray < 128).astype(np.uint8) * 255
    circles = _circles(ink, px_per_mm)
    best: tuple[float, tuple[float, float, float], Word] | None = None
    for w in words:
        if w.text.strip().upper() not in LETTERS:
            continue
        lx, ly = (w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2
        for cx, cy, r in circles:
            d = math.hypot(lx - cx, ly - cy)
            if r * 0.9 <= d <= r * 2.5 and (best is None or d / r < best[0]):
                best = (d / r, (cx, cy, r), w)
    if best is None:
        return None
    _, (cx, cy, r), letter = best
    lx, ly = (letter.x0 + letter.x1) / 2, (letter.y0 + letter.y1) / 2
    letter_dir = math.atan2(-(ly - cy), lx - cx)  # math convention: CCW from +x, y up
    # ink inside the circle, without the ring
    h, w_ = gray.shape
    yy, xx = np.mgrid[
        max(0, int(cy - r)) : min(h, int(cy + r) + 1),
        max(0, int(cx - r)) : min(w_, int(cx + r) + 1),
    ]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.85 * r) ** 2
    pts = np.stack([xx[inside], yy[inside]], -1)[ink[yy[inside], xx[inside]] > 0].astype(np.float64)
    if len(pts) < 20:
        return None
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    ax, ay = vt[0]
    axis = math.atan2(-ay, ax)
    # pick the axis direction that agrees with the letter
    if math.cos(axis - letter_dir) < 0:
        axis += math.pi
    if abs(math.degrees(math.atan2(math.sin(axis - letter_dir), math.cos(axis - letter_dir)))) > 25:
        axis = letter_dir  # the ink axis is unreliable (unusual symbol): fall back to the letter
        confidence = 0.5
    else:
        confidence = 0.9
    north_from_up = (math.degrees(axis) - 90.0) % 360.0
    return NorthArrow(
        page_id=page_id,
        bbox_px=(cx - r, cy - r, cx + r, cy + r),
        angle_deg=fact(
            round(north_from_up, 2),
            "raster_cv",
            confidence,
            note="principal axis of the arrow, letter decides the sense",
        ),
    )
