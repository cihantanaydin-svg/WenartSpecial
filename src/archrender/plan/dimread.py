"""Dimension text read along the dimension lines (raster plans).

Page OCR reads horizontal text; the dimension strings of a rotated or skewed plan run along their
dimension lines at any angle. For each straight stroke that may be a dimension line (not wall
body, long enough, no word already sitting on it), the bands just above and below it are cut out,
turned upright and read as one line with a digits whitelist; a reading counts only if the
dimension grammar parses it. The readings come back as ``Word``s (page pixels, with the line's
angle), so ``scale.dimension_pairs`` pairs them like any other word.
"""

from __future__ import annotations

import math
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.document import Word
from archrender.understand.text import parse_dimension

DIM_WHITELIST = "0123456789,.'\"- "
MAX_LINES = 80


class LineReader(Protocol):
    def read_line(self, img: NDArray[np.uint8], whitelist: str) -> str: ...


Segment = tuple[tuple[float, float], tuple[float, float]]


def _covered(seg: Segment, words: list[Word], text_px: float) -> bool:
    (x0, y0), (x1, y1) = seg
    a, b = np.array([x0, y0]), np.array([x1, y1])
    length = float(np.hypot(*(b - a)))
    u = (b - a) / max(length, 1e-9)
    for w in words:
        c = np.array([(w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2])
        along = float((c - a) @ u)
        perp = abs(float((c - a)[0] * u[1] - (c - a)[1] * u[0]))
        if 0.2 * length <= along <= 0.8 * length and perp <= 3 * text_px:
            return True
    return False


def _band(
    gray: NDArray[np.uint8],
    mid: NDArray[np.float64],
    u: NDArray[np.float64],
    up: NDArray[np.float64],
    length: float,
    lo: float,
    hi: float,
) -> NDArray[np.uint8]:
    """The band from ``lo`` to ``hi`` pixels on the ``up`` side of the line, upright."""
    w, h = max(8, int(length)), max(8, int(hi - lo))
    centre = mid + up * (lo + hi) / 2
    # destination (x right along u, y down = -up) ← source
    src = np.array(
        [
            centre - u * w / 2 + up * h / 2,
            centre + u * w / 2 + up * h / 2,
            centre - u * w / 2 - up * h / 2,
        ],
        np.float32,
    )
    dst = np.array([[0, 0], [w, 0], [0, h]], np.float32)
    m = cv2.getAffineTransform(src, dst)
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    scale = max(1.0, 32.0 / max(1.0, hi - lo) * 1.5)
    if scale > 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out = cv2.copyMakeBorder(out, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return np.asarray(out, np.uint8)


def read_dimension_lines(
    gray: NDArray[np.uint8],
    segments: list[Segment],
    words: list[Word],
    reader: LineReader,
    *,
    text_px: float,
    exclude: list[NDArray[np.bool_]] | None = None,
    max_lines: int = MAX_LINES,
) -> list[Word]:
    """Readings of the text along candidate dimension lines (see module docstring)."""
    h, w = gray.shape
    body = exclude[0] if exclude else None
    cands = []
    for seg in segments:
        (x0, y0), (x1, y1) = seg
        length = math.hypot(x1 - x0, y1 - y0)
        if length < 4 * text_px or length > 0.9 * max(h, w):
            continue
        mx, my = int((x0 + x1) / 2), int((y0 + y1) / 2)
        if body is not None and 0 <= my < h and 0 <= mx < w and body[my, mx]:
            continue
        if _covered(seg, words, text_px):
            continue
        cands.append((length, seg))
    cands.sort(key=lambda c: -c[0])
    out: list[Word] = []
    for length, seg in cands[:max_lines]:
        a, b = np.array(seg[0], np.float64), np.array(seg[1], np.float64)
        u = (b - a) / length
        if u[0] < -1e-9 or (abs(u[0]) <= 1e-9 and u[1] > 0):
            u = -u  # read left to right (upright text)
        up = np.array([u[1], -u[0]])  # 90° counter-clockwise on screen
        mid = (a + b) / 2
        best: tuple[str, float] | None = None
        for side in (1.0, -1.0):
            band = _band(gray, mid, u, up * side, 0.8 * length, 0.15 * text_px, 2.4 * text_px)
            text = reader.read_line(
                np.ascontiguousarray(np.stack([band] * 3, -1)), DIM_WHITELIST
            ).strip()
            dim = parse_dimension(text)
            if dim is not None and dim.metres and any(ch.isdigit() for ch in text):
                best = (text, side)
                break
        if best is None:
            continue
        text, side = best
        c = mid + up * side * 1.2 * text_px
        half = np.abs(u) * 0.3 * length + np.abs(up) * 0.6 * text_px
        angle = math.degrees(math.atan2(-u[1], u[0])) % 360.0
        out.append(
            Word(
                text=text,
                x0=float(c[0] - half[0]),
                y0=float(c[1] - half[1]),
                x1=float(c[0] + half[0]),
                y1=float(c[1] + half[1]),
                angle_deg=round(angle, 2),
                confidence=0.6,
                source="ocr:dimension-line",
            )
        )
    return out
