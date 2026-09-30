"""Door/window/room tags on plans.

Vector sheets and DXF: tags come from the text layer (exact). Raster sheets: tags usually sit in
circles or ellipses ("bubbles") that defeat general OCR, so bubbles are found as ellipse-shaped
contours of plausible size in a locally thresholded ink mask; the lettering inside is isolated from
the ring and from swings or lines crossing it, and read as a single line with a tag whitelist at
three magnifications. A reading is kept only if it parses as a tag, at least two magnifications
give it and none gives another tag; failing that, digit look-alikes after a door/window prefix are
mapped (``PS`` → ``P5``) and the vote is repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.document import Word
from archrender.understand.text import TAG_KINDS, normalise_tag, tr_upper

WHITELIST = "ABCDEFGHIJKLMNOPRSTUVWXYZ0123456789-"
# letters OCR returns for digits in bold lettering (5 → S is the common one)
DIGIT_LOOKALIKES = str.maketrans(
    {"O": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
)


def bubble_reading_to_tag(reading: str) -> str | None:
    """A bubble holds a tag: a door/window prefix then a number. Letters in the number part that
    look like digits are read as those digits (only after a known prefix, only for bubbles)."""
    tag = normalise_tag(reading)
    if tag is not None:
        return tag
    t = tr_upper(reading).replace(" ", "").replace("-", "")
    for prefix in sorted(TAG_KINDS, key=len, reverse=True):
        rest = t[len(prefix) :]
        if t.startswith(prefix) and rest:
            digits = rest.translate(DIGIT_LOOKALIKES)
            if digits.isdigit():
                return normalise_tag(prefix + digits)
    return None


class LineReader(Protocol):
    def read_line(self, img: NDArray[np.uint8], whitelist: str) -> str: ...


@dataclass(frozen=True)
class TagHit:
    tag: str
    kind: str | None  # door | window | None (room numbers and other codes)
    bbox: tuple[float, float, float, float]  # page px
    source: str  # text_layer | bubble_ocr
    confidence: float


def tags_from_words(words: list[Word], *, door_window_only: bool = False) -> list[TagHit]:
    """Tag-like codes in the words: door/window tags (K1, P3 …) and other codes such as room
    numbers (Z01) that finish schedules refer to (kind None)."""
    hits = []
    for w in words:
        tag = normalise_tag(w.text)
        if tag is None:
            continue
        kind = TAG_KINDS.get(_prefix(tag))
        if kind is None and door_window_only:
            continue
        hits.append(TagHit(tag, kind, (w.x0, w.y0, w.x1, w.y1), "text_layer", w.confidence))
    return hits


def _prefix(tag: str) -> str:
    out = ""
    for ch in tag:
        if ch.isdigit():
            break
        out += ch
    return out


def ink_mask(gray: NDArray[np.uint8], contrast: int = 60) -> NDArray[np.uint8]:
    """Pixels clearly darker than the local paper (255) else 0. A fixed threshold loses thin,
    blurred strokes on grey or noisy scans; the paper level is estimated per neighbourhood."""
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.8)
    paper = cv2.medianBlur(cv2.dilate(smooth, np.ones((9, 9), np.uint8)), 21)
    return ((paper.astype(np.int16) - smooth) > contrast).astype(np.uint8) * 255


def find_bubbles(
    gray: NDArray[np.uint8], dpi: float
) -> list[tuple[float, float, float, float, float]]:
    """Ellipses 2–7 mm across → (cx, cy, width, height, angle) in px.

    Both boundaries of a ring are candidates: the outer one is lost when the bubble touches a
    door swing or a wall, the inner one (the hole the tag sits in) usually survives."""
    px_per_mm = dpi / 25.4
    lo, hi = 2.0 * px_per_mm, 7.0 * px_per_mm
    contours, hierarchy = cv2.findContours(ink_mask(gray), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out: list[tuple[float, float, float, float, float]] = []
    if hierarchy is None:
        return out
    for c in contours:
        if len(c) < 12:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if not (lo <= w <= hi * 1.4 and lo <= h <= hi):
            continue
        (cx, cy), (ew, eh), ang = cv2.fitEllipse(c)
        ell = np.pi * ew * eh / 4.0
        if ell <= 0 or abs(cv2.contourArea(c) - ell) / ell > 0.15:
            continue
        near = 0.3 * min(ew, eh)
        if any(abs(cx - o[0]) < near and abs(cy - o[1]) < near for o in out):
            continue  # the other boundary of a ring already found
        out.append((float(cx), float(cy), float(ew), float(eh), float(ang)))
    return out


def bubble_text(
    gray: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    bubble: tuple[float, float, float, float, float],
    dpi: float,
    open_px: int,
) -> NDArray[np.uint8] | None:
    """The characters inside a bubble on white, or None. Door swings and wall lines often cross a
    tag; an opening of ``open_px`` detaches thin strokes from bold lettering, then only compact
    components near the centre that do not reach the crop edge are kept (the ring, arcs and lines
    are long and leave the crop)."""
    cx, cy, ew, eh, _ = bubble
    r = max(ew, eh) / 2
    half = r * 1.15 + 4
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(gray.shape[1], cx + half)), int(min(gray.shape[0], cy + half))
    full = mask[y0:y1, x0:x1]
    m = (
        cv2.morphologyEx(
            full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        )
        if open_px > 1
        else full
    )
    n, labels, stats, centres = cv2.connectedComponentsWithStats(m, connectivity=8)
    px_per_mm = dpi / 25.4
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, w, h, _area = stats[i]
        if x == 0 or y == 0 or x + w >= m.shape[1] or y + h >= m.shape[0]:
            continue
        if h < 1.0 * px_per_mm or h > 0.75 * min(ew, eh) or w > 0.75 * max(ew, eh):
            continue  # specks; the ring and arcs
        if abs(centres[i][0] + x0 - cx) > 0.6 * r or abs(centres[i][1] + y0 - cy) > 0.6 * r:
            continue
        keep[labels == i] = 255
    if not keep.any():
        return None
    grow = max(3, open_px)
    keep = cv2.bitwise_and(
        cv2.dilate(keep, np.ones((grow, grow), np.uint8)),
        cv2.dilate(full, np.ones((3, 3), np.uint8)),
    )  # restore the stroke edges the opening ate, not the lines it removed
    ys, xs = np.nonzero(keep)
    pad = round(px_per_mm)
    out = np.where(keep > 0, gray[y0:y1, x0:x1], 255).astype(np.uint8)
    return out[
        max(0, ys.min() - pad) : ys.max() + pad + 1, max(0, xs.min() - pad) : xs.max() + pad + 1
    ]


def _vote(readings: list[str | None]) -> str | None:
    """A tag read the same at ≥ 2 magnifications with no other tag read at any."""
    seen = [r for r in readings if r]
    if not seen or len(set(seen)) != 1 or len(seen) < 2:
        return None
    return seen[0]


def read_bubble_tags(gray: NDArray[np.uint8], dpi: float, reader: LineReader) -> list[TagHit]:
    """Tags read inside bubbles. Each bubble is tried with a strong, a light and no opening (see
    ``bubble_text``) until one gives an unambiguous reading at three magnifications. Only if no
    variant does are digit look-alikes repaired (``bubble_reading_to_tag``) and the vote repeated,
    so a repaired misreading never outvotes clean readings."""
    mask = ink_mask(gray)
    strong = max(2, round(0.34 * dpi / 25.4))
    hits = []
    for bubble in find_bubbles(gray, dpi):
        tag = None
        raw_readings: list[list[str]] = []
        for open_px in dict.fromkeys((strong, strong - 1, 0)):
            crop = bubble_text(gray, mask, bubble, dpi, open_px)
            if crop is None:
                continue
            raw = []
            for fx in (2, 3, 4):
                up = cv2.resize(crop, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                up = cv2.copyMakeBorder(up, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
                raw.append(
                    reader.read_line(np.ascontiguousarray(np.stack([up] * 3, -1)), WHITELIST)
                )
            raw_readings.append(raw)
            tag = _vote([normalise_tag(r) for r in raw])
            if tag is not None:
                break
        if tag is None:
            for raw in raw_readings:
                tag = _vote([bubble_reading_to_tag(r) for r in raw])
                if tag is not None:
                    break
        if tag is None:
            continue
        cx, cy, ew, eh, _ = bubble
        hits.append(
            TagHit(
                tag,
                TAG_KINDS.get(_prefix(tag)),
                (cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2),
                "bubble_ocr",
                0.9,
            )
        )
    return hits
