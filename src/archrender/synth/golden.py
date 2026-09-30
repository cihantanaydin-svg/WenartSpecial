"""Golden projects (docs/PLAN.md), Phase-2 subset: the document sets S0/S1 must handle.

- **G1 "Daire"**: Turkish 3+1-style flat. Vector plan PDF (room numbers + door/window tags), the same
  plan as DXF (DUVAR/KAPI/PENCERE, centimetres), finish schedule XLSX (Mahal Listesi), a
  door/window schedule PDF, a Turkish brief (DOCX) and mood-board images.
- **G2 "Loft"**: scanned plan (noisy JPEG), ceiling plan PDF, door/window schedule PDF, DOCX brief in
  Turkish and English.

Non-Manhattan and arc walls, the phone photo of the printed plan (G2) and the IFC office (G3) come
with the Phase-3 plan generator.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import docx
import numpy as np
import openpyxl

from archrender.synth.dxf import layout_dxf
from archrender.synth.layout import Layout, random_layout
from archrender.synth.raster import photo, scan
from archrender.synth.sheets import (
    BRIEF_EN,
    BRIEF_TR,
    MATS_TR,
    floor_plan_page,
    moodboard_page,
    schedule_page,
    schedule_rows,
)
from archrender.understand.text import normalise_tag


@dataclass
class GoldenDoc:
    filename: str
    data: bytes
    expected_class: str | None  # None: container/derived, not classified as one page


@dataclass
class GoldenProject:
    name: str
    layout: Layout
    docs: list[GoldenDoc] = field(default_factory=list)
    schedule_tags: set[str] = field(default_factory=set)  # normalised tags every row should link to


def _xlsx(title: str, header: list[str], rows: list[list[str]]) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = title[:31]
    ws.append([title])
    ws.append(header)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _brief_docx(rng: np.random.Generator, bilingual: bool) -> bytes:
    d = docx.Document()
    d.add_heading("Tasarım Özeti", level=1)
    sentences = list(BRIEF_TR)
    rng.shuffle(sentences)
    for s in sentences[:5]:
        d.add_paragraph(s.format(mat=rng.choice(MATS_TR)))
    if bilingual:
        d.add_heading("Design brief (English)", level=1)
        for s in BRIEF_EN[:4]:
            d.add_paragraph(s.format(mat="oak"))
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def g1_daire(seed: int = 101) -> GoldenProject:
    rng = np.random.default_rng(seed)
    layout = random_layout(rng)
    plan = floor_plan_page(rng, layout, room_numbers=True)
    dw = schedule_page(rng, layout, kind="door_window")
    header, rows = schedule_rows(layout, "finish", "tr", rng)
    g = GoldenProject("G1 Daire", layout)
    g.docs = [
        GoldenDoc("Zemin Kat Planı.pdf", plan.pdf, "floor_plan"),
        GoldenDoc("Zemin Kat Planı.dxf", layout_dxf(layout, unit="cm"), "floor_plan"),
        GoldenDoc("Malzeme Listesi.xlsx", _xlsx("Mahal Listesi", header, rows), "schedule"),
        GoldenDoc("Doğrama Listesi.pdf", dw.pdf, "schedule"),
        GoldenDoc("Tasarım Özeti.docx", _brief_docx(rng, bilingual=False), None),
        GoldenDoc("Konsept Panosu.pdf", moodboard_page(rng).pdf, "moodboard"),
    ]
    for i in range(2):
        data, _ = photo(rng)
        g.docs.append(GoldenDoc(f"Referans {i + 1}.jpg", data, "photo"))
    g.schedule_tags = {
        normalise_tag(t) or t for t in [o.tag for o in layout.openings] + [r[0] for r in rows]
    }
    return g


def g2_loft(seed: int = 202) -> GoldenProject:
    rng = np.random.default_rng(seed)
    layout = random_layout(rng)
    plan = floor_plan_page(rng, layout)
    scan_bytes, _, _ = scan(plan.pdf, plan.gt, rng, dpi=300, quality="noisy")
    rcp = floor_plan_page(rng, layout, ceiling=True)
    dw = schedule_page(rng, layout, kind="door_window")
    g = GoldenProject("G2 Loft", layout)
    g.docs = [
        GoldenDoc("Zemin Kat Planı (tarama).jpg", scan_bytes, "floor_plan"),
        GoldenDoc("Tavan Planı.pdf", rcp.pdf, "ceiling_plan"),
        GoldenDoc("Doğrama Listesi.pdf", dw.pdf, "schedule"),
        GoldenDoc("Brief TR-EN.docx", _brief_docx(rng, bilingual=True), None),
    ]
    g.schedule_tags = {o.tag for o in layout.openings}
    return g
