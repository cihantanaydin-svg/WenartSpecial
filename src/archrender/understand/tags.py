"""Door/window/room tags on plans.

Vector sheets and DXF: tags come from the text layer (exact). Raster sheets: tags usually sit in
circles or ellipses ("bubbles") that defeat general OCR, so bubbles are found as closed, ellipse-
shaped contours of plausible size, the ring stroke is erased, and the inside is read as a single
line with a tag whitelist at two magnifications. A reading is kept only if both magnifications agree
and it parses as a tag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.document import Word
from archrender.understand.text import TAG_KINDS, normalise_tag

WHITELIST = "ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789-"


class LineReader(Protocol):
    def read_line(self, img: NDArray[np.uint8], whitelist: str) -> str: ...


@dataclass(frozen=True)
class TagHit:
    tag: str
    kind: str | None  # door | window | None (room numbers and other codes)
    bbox: tuple[float, float, float, float]  # page px
    source: str  # text_layer | bubble_ocr
    confidence: float


def tags_from_words(words: list[Word]) -> list[TagHit]:
    hits = []
    for w in words:
        tag = normalise_tag(w.text)
        if tag is None:
            continue
        kind = TAG_KINDS.get(_prefix(tag))
        if kind is None:
            continue  # free text: only door/window codes; room numbers are linked via schedules
        hits.append(TagHit(tag, kind, (w.x0, w.y0, w.x1, w.y1), "text_layer", w.confidence))
    return hits


def _prefix(tag: str) -> str:
    out = ""
    for ch in tag:
        if ch.isdigit():
            break
        out += ch
    return out


def find_bubbles(
    gray: NDArray[np.uint8], dpi: float
) -> list[tuple[float, float, float, float, float]]:
    """Ellipse-shaped closed contours 2–6 mm across → (cx, cy, width, height, angle) in px."""
    px_per_mm = dpi / 25.4
    lo, hi = 2.0 * px_per_mm, 7.0 * px_per_mm
    ink = (gray < 128).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hierarchy is None:
        return out
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] != -1 or len(c) < 12:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if not (lo <= w <= hi * 1.4 and lo <= h <= hi):
            continue
        (cx, cy), (ew, eh), ang = cv2.fitEllipse(c)
        ell = np.pi * ew * eh / 4.0
        if ell <= 0 or abs(cv2.contourArea(c) - ell) / ell > 0.15:
            continue
        out.append((float(cx), float(cy), float(ew), float(eh), float(ang)))
    return out


def read_bubble_tags(gray: NDArray[np.uint8], dpi: float, reader: LineReader) -> list[TagHit]:
    hits = []
    for cx, cy, ew, eh, ang in find_bubbles(gray, dpi):
        half_w, half_h = max(ew, eh) / 2 + 2, max(ew, eh) / 2 + 2
        x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        x1, y1 = int(min(gray.shape[1], cx + half_w)), int(min(gray.shape[0], cy + half_h))
        crop = gray[y0:y1, x0:x1].copy()
        stroke = max(3, round(dpi / 45))
        cv2.ellipse(crop, ((cx - x0, cy - y0), (ew, eh), ang), 255, stroke)  # erase the ring
        readings = []
        for fx in (3, 4):
            up = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
            up = cv2.copyMakeBorder(up, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
            readings.append(
                normalise_tag(
                    reader.read_line(np.ascontiguousarray(np.stack([up] * 3, -1)), WHITELIST)
                )
            )
        if readings[0] is None or readings[0] != readings[1]:
            continue
        tag = readings[0]
        hits.append(
            TagHit(
                tag,
                TAG_KINDS.get(_prefix(tag)),
                (float(x0), float(y0), float(x1), float(y1)),
                "bubble_ocr",
                0.9,
            )
        )
    return hits
