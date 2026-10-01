"""Ingested documents and their pages (S0 output, S1 input)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from archrender.core.cas import CasRef
from archrender.core.schemas.common import Strict

PageKind = Literal[
    "pdf_page",  # one page of a PDF (vector content + raster + text layer)
    "image",  # a raster image (photo, scan, mood board, render)
    "dxf",  # a DXF drawing (also DWG converted by LibreDWG)
    "ifc",  # an IFC model (parsed in S2)
    "svg",
    "model_3dm",
    "docx",  # the body of a Word document
    "sheet",  # one worksheet of an XLSX workbook
    "slide",  # one PPTX slide
    "text",  # plain text / Markdown
]


class Tile(Strict):
    """Full-resolution raster tile (pixels of the page raster) for OCR and VLM reading."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class Word(Strict):
    """A word with its box in page pixels (raster frame, origin top-left, y down)."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    angle_deg: float = 0.0  # text direction, counter-clockwise from +x
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "pdf_text"  # pdf_text | dxf_text | ocr:<model> | office


class PageRef(Strict):
    id: str
    document_id: str
    index: int = Field(ge=0)
    kind: PageKind
    width_px: int | None = None
    height_px: int | None = None
    dpi: float | None = None  # raster resolution for PDF pages
    width_mm: float | None = None  # physical sheet size (PDF)
    height_mm: float | None = None
    raster: CasRef | None = None  # sRGB PNG
    words: CasRef | None = None  # JSON list[Word] (text layer / DXF text / office text)
    content: CasRef | None = None  # kind-specific JSON (vector stats, DXF summary, table cells …)
    vectors: CasRef | None = None  # drawing primitives for S2 (PDF paths, DXF entities, IFC model)
    tiles: list[Tile] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class SkippedEntry(Strict):
    """An archive entry or embedded object that was not ingested, and why."""

    name: str
    code: str
    message: str
    fix_hint: str = ""


class IntakeResult(Strict):
    document_id: str
    kind: str
    sha256: str
    deduplicated: bool
    pages: int
    children: list[str] = Field(default_factory=list)  # zip entries, embedded images, DWG→DXF
    skipped: list[SkippedEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
