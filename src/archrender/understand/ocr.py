"""Tiled OCR on full-resolution page rasters (ARCHITECTURE §S1).

Each tile is read at 0° and, for vertical dimension strings, rotated by 90°. Words are mapped back
to page pixels. A word touching an inner tile edge is dropped (the overlapping tile contains it
whole), and duplicates from overlaps and rotations are merged by IoU and equal text, keeping the
more confident reading. The VLM is never asked to read small text from a downscaled sheet.

A tile the engine cannot read (e.g. a timeout) fails the page unless the caller passes
``failures``: then the tile is recorded there and the other tiles are still read.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from archrender.core.errors import ArchRenderError
from archrender.core.schemas.document import Tile, Word
from archrender.models.roles import OcrEngine, OcrWord
from archrender.understand.text import fold

EDGE_PX = 3.0
MIN_CONFIDENCE = 0.30


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _unrotate(
    box: tuple[float, float, float, float], tile_h: int
) -> tuple[float, float, float, float]:
    """Box from a tile rotated 90° clockwise (np.rot90(k=-1)) back to the unrotated tile."""
    x0, y0, x1, y1 = box
    return (y0, tile_h - x1, y1, tile_h - x0)


def ocr_tiles(
    page: NDArray[np.uint8],
    tiles: list[Tile],
    engine: OcrEngine,
    *,
    langs: list[str],
    source: str,
    rotations: tuple[int, ...] = (0, 90),
    failures: list[str] | None = None,
) -> list[Word]:
    """OCR ``page`` (H, W, 3 uint8) tile by tile; returns merged words in page pixels."""
    h, w = page.shape[:2]
    found: list[tuple[OcrWord, float]] = []  # (word in page px, angle)
    for t in tiles:
        crop = np.ascontiguousarray(page[t.y : t.y + t.h, t.x : t.x + t.w])
        ch, cw = crop.shape[:2]
        for rot in rotations:
            img = crop if rot == 0 else np.ascontiguousarray(np.rot90(crop, k=-1))
            try:
                read = engine.read(img, langs)
            except ArchRenderError as e:
                if failures is None:
                    raise
                failures.append(f"tile x={t.x} y={t.y} {t.w}×{t.h} px, {rot}°: {e.message}")
                continue
            for wd in read:
                if wd.confidence < MIN_CONFIDENCE or not wd.text.strip():
                    continue
                box = wd.box if rot == 0 else _unrotate(wd.box, ch)
                bw, bh = box[2] - box[0], box[3] - box[1]
                # the rotated pass exists for vertical text; what it reads from horizontal text
                # (short, wide boxes once mapped back) is noise
                if rot != 0 and not (bh > 1.2 * bw and len(wd.text.strip()) > 1):
                    continue
                # drop words cut by an inner tile edge; the neighbouring tile has them whole
                if (box[0] <= EDGE_PX and t.x > 0) or (box[1] <= EDGE_PX and t.y > 0):
                    continue
                if (box[2] >= cw - EDGE_PX and t.x + cw < w) or (
                    box[3] >= ch - EDGE_PX and t.y + ch < h
                ):
                    continue
                page_box = (box[0] + t.x, box[1] + t.y, box[2] + t.x, box[3] + t.y)
                found.append((OcrWord(wd.text.strip(), page_box, wd.confidence), float(rot)))
    return [
        Word(
            text=o.text,
            x0=o.box[0],
            y0=o.box[1],
            x1=o.box[2],
            y1=o.box[3],
            angle_deg=angle,
            confidence=round(min(1.0, max(0.0, o.confidence)), 3),
            source=source,
        )
        for o, angle in _merge(found)
    ]


def _merge(found: list[tuple[OcrWord, float]]) -> list[tuple[OcrWord, float]]:
    """Greedy de-duplication: highest confidence first; drop overlapping readings of the same
    text, and weaker readings that overlap a stronger word by more than half of their area."""
    order = sorted(found, key=lambda x: -x[0].confidence)
    kept: list[tuple[OcrWord, float]] = []
    for cand, angle in order:
        cb = cand.box
        area = max(1e-9, (cb[2] - cb[0]) * (cb[3] - cb[1]))
        duplicate = False
        for k, _ in kept:
            kb = k.box
            ix = max(0.0, min(cb[2], kb[2]) - max(cb[0], kb[0]))
            iy = max(0.0, min(cb[3], kb[3]) - max(cb[1], kb[1]))
            if (fold(k.text) == fold(cand.text) and _iou(cb, kb) > 0.3) or ix * iy / area > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append((cand, angle))
    kept.sort(key=lambda x: (round(x[0].box[1] / 10), x[0].box[0]))
    return kept
