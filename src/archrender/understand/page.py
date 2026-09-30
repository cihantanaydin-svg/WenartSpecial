"""S1 per-page analysis shared by the pipeline stage, classifier training and the evaluation.

Words come from the PDF text layer, DXF text entities or tiled OCR (in that order of preference).
From the words and the raster: title block + scale, north arrow, tags (text + bubbles), and the
classification features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.document import Tile, Word
from archrender.core.schemas.provenance import Fact, Method
from archrender.core.schemas.understanding import NorthArrow, TitleBlock
from archrender.models.roles import OcrEngine
from archrender.understand.features import meta_features, text_features, visual_features
from archrender.understand.north import measure_north
from archrender.understand.ocr import ocr_tiles
from archrender.understand.tags import TagHit, read_bubble_tags, tags_from_words
from archrender.understand.titleblock import extract_title_block, scale_from_words


@dataclass
class PageAnalysis:
    page_id: str
    words: list[Word]
    words_method: Method
    title_block: TitleBlock | None
    scale: Fact[float] | None
    north: NorthArrow | None
    tags: list[TagHit]
    features: dict[str, float]
    notes: list[str] = field(default_factory=list)


def dxf_words(summary: dict[str, Any], height_px: int) -> list[Word]:
    """DXF TEXT/MTEXT/ATTRIB entities → words in the DXF preview raster's pixels."""
    s = summary.get("raster_scale_px_per_unit")
    ext = summary.get("extents")
    if not s or not ext:
        return []
    out = []
    for t in summary.get("texts", []):
        text = str(t["text"]).strip()
        if not text:
            continue
        x = (t["x"] - ext[0]) * s + 10
        y = height_px - 10 - (t["y"] - ext[1]) * s
        th = max(1.0, float(t.get("height") or 0) * s)
        width = max(th * 0.6 * len(text), 1.0)
        out.append(
            Word(
                text=text,
                x0=x,
                y0=y - th,
                x1=x + width,
                y1=y,
                angle_deg=float(t.get("rotation", 0.0)) % 360,
                source="dxf_text",
            )
        )
    return out


def analyse(
    page_id: str,
    rgb: NDArray[np.uint8],
    *,
    dpi: float,
    text_layer: list[Word] | None,
    tiles: list[Tile],
    ocr: OcrEngine | None,
    ocr_name: str,
    langs: list[str],
    page_meta: dict[str, Any],
    doc_meta: dict[str, Any],
    vector: dict[str, Any] | None,
    preview: NDArray[np.uint8] | None = None,
    dxf_summary: dict[str, Any] | None = None,
) -> PageAnalysis:
    notes: list[str] = []
    h, w = rgb.shape[:2]
    method: Method
    if text_layer:
        words, method = text_layer, "pdf_text"
    elif dxf_summary is not None:
        words, method = dxf_words(dxf_summary, h), "dxf_entity"
    elif ocr is not None and tiles:
        failed: list[str] = []
        words = ocr_tiles(rgb, tiles, ocr, langs=langs, source=f"ocr:{ocr_name}", failures=failed)
        method = "ocr"
        notes += [f"OCR incomplete, {f}" for f in failed]
    else:
        words, method = [], "ocr"
        notes.append("no text layer and no OCR engine: text features are empty")
    tb = extract_title_block(page_id, words, w, h, method)
    scale = tb.scale if tb and tb.scale else scale_from_words(words, method)
    gray = np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    north = measure_north(page_id, gray, words, dpi) if dpi else None
    if dxf_summary is not None and vector is None:
        # a DXF is a vector drawing: its entities play the part of a PDF's path objects
        vector = {
            "objects": {"path": int(dxf_summary.get("entity_count", 0))},
            "image_area_fraction": 0.0,
        }
    tags = tags_from_words(words)
    if method == "ocr" and ocr is not None and hasattr(ocr, "read_line") and dpi:
        seen = {t.tag for t in tags}
        tags += [t for t in read_bubble_tags(gray, dpi, ocr) if t.tag not in seen]  # type: ignore[arg-type]
    feats = {
        **visual_features(preview if preview is not None else rgb),
        **text_features(words, w, h, has_title_block=tb is not None),
        **meta_features(page_meta, doc_meta, vector),
    }
    return PageAnalysis(page_id, words, method, tb, scale, north, tags, feats, notes)
