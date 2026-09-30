"""Schedules (door/window "Doğrama Listesi", room finish "Mahal Listesi") and their links to plan tags.

Sources: XLSX sheets and DOCX tables (cells as given), and PDF pages (a table rebuilt from word
positions under a header row). Column meaning comes from a multilingual header dictionary matched
with :func:`fold`; the VLM is only a fallback for unknown headers (not used when every column
is recognised). Lengths are converted to metres using the unit in the header ("(cm)", "(mm)").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from archrender.core.schemas.document import Word
from archrender.core.schemas.understanding import Schedule, ScheduleRow
from archrender.understand.tags import TagHit
from archrender.understand.text import fold, normalise_tag, parse_number
from archrender.understand.titleblock import _Line, bands

HEADERS: dict[str, tuple[str, ...]] = {
    "tag": ("poz", "poz no", "no", "kod", "tag", "mark", "ref", "tip no", "kapi no", "pencere no"),
    "room_no": ("mahal no", "room no", "room number", "oda no"),
    "room_name": ("mahal adi", "mahal", "room name", "room", "oda"),
    "type": ("tip", "cins", "type", "tur"),
    "width": ("genislik", "en", "width", "w", "genislik cm", "genislik mm"),
    "height": ("yukseklik", "boy", "height", "h", "yukseklik cm", "yukseklik mm"),
    "sill": ("parapet", "denizlik", "sill", "parapet yuksekligi", "sill height"),
    "qty": ("adet", "miktar", "qty", "quantity", "sayi"),
    "material": ("malzeme", "material", "cinsi"),
    "floor": ("zemin", "floor", "doseme", "zemin kaplamasi"),
    "walls": ("duvar", "walls", "wall", "duvar kaplamasi"),
    "ceiling": ("tavan", "ceiling"),
    "skirting": ("supurgelik", "skirting", "baseboard"),
}
_INDEX = {v: k for k, vs in HEADERS.items() for v in vs}
LENGTH_FIELDS = ("width", "height", "sill")


def _header_key(text: str) -> tuple[str | None, str | None]:
    """Canonical field + unit from a header cell ('GENİŞLİK (cm)' → ('width', 'cm'))."""
    unit = None
    m = re.search(r"\((mm|cm|m)\)|\[(mm|cm|m)\]", text.lower())
    if m:
        unit = m.group(1) or m.group(2)
    key = fold(re.sub(r"[\(\[].*?[\)\]]", " ", text))
    return _INDEX.get(key), unit


def schedule_from_rows(
    schedule_id: str, page_id: str, rows: Sequence[Sequence[Any]]
) -> Schedule | None:
    """Find the header row (≥ 2 recognised columns), then read the rows below it."""
    for hi, row in enumerate(rows[:30]):
        cells = ["" if c is None else str(c) for c in row]
        keys = [_header_key(c) for c in cells]
        recognised = [k for k, _ in keys if k]
        if len(recognised) < 2:
            continue
        columns = {k: cells[j] for j, (k, _) in enumerate(keys) if k}
        units = {k: u for k, u in keys if k and u}
        out_rows: list[ScheduleRow] = []
        for ri, r in enumerate(rows[hi + 1 :], start=hi + 1):
            vals = ["" if c is None else str(c).strip() for c in r]
            if not any(vals):
                continue
            fields: dict[str, str | float | None] = {}
            for j, (k, _) in enumerate(keys):
                if not k or j >= len(vals):
                    continue
                v = vals[j]
                if k in LENGTH_FIELDS or k == "qty":
                    num = parse_number(v)
                    if k in LENGTH_FIELDS and num is not None:
                        num = (
                            num
                            * {"mm": 0.001, "cm": 0.01, "m": 1.0}[
                                units.get(k, "cm" if num > 10 else "m")
                            ]
                        )
                    fields[k] = num if num is not None else (None if v in ("", "-") else v)
                else:
                    fields[k] = v or None
            tag_src = fields.get("tag") or fields.get("room_no")
            tag = normalise_tag(str(tag_src)) if tag_src else None
            out_rows.append(
                ScheduleRow(
                    id=f"{schedule_id}_r{ri}",
                    tag=tag or (str(tag_src) if tag_src else None),
                    fields=fields,
                    source_page=page_id,
                    row_index=ri,
                )
            )
        kind = (
            "door_window"
            if {"width", "height"} & set(recognised)
            else "finish"
            if {"floor", "walls", "ceiling"} & set(recognised)
            else "unknown"
        )
        return Schedule(
            id=schedule_id,
            kind=kind,
            source_page=page_id,
            header=cells,
            columns=columns,
            rows=out_rows,
        )
    return None


def rows_from_words(words: list[Word]) -> list[list[str]]:
    """Rebuild a table from positioned words: columns start at the header cells' x positions."""
    lns = [_Line(b) for b in bands(words)]
    header_i = None
    cells_x: list[float] = []
    for i, ln in enumerate(lns):
        groups = _cells(ln)
        keys = [_header_key(" ".join(w.text for w in g))[0] for g in groups]
        if sum(k is not None for k in keys) >= 2:
            header_i = i
            cells_x = [g[0].x0 for g in groups]
            header = [" ".join(w.text for w in g) for g in groups]
            break
    if header_i is None:
        return []
    table = [header]
    h = lns[header_i].height
    last_y = lns[header_i].box[3]
    for ln in lns[header_i + 1 :]:
        if ln.box[1] - last_y > 4 * h:
            break  # the table ended
        row = [""] * len(cells_x)
        for w in ln.words:
            j = max((k for k, x in enumerate(cells_x) if w.x0 >= x - 0.5 * h), default=0)
            row[j] = (row[j] + " " + w.text).strip()
        table.append(row)
        last_y = ln.box[3]
    return table


def _cells(ln: Any) -> list[list[Word]]:
    """Split a header line into cells at gaps wider than ~1.5 character heights."""
    ws = sorted(ln.words, key=lambda w: w.x0)
    groups: list[list[Word]] = [[ws[0]]]
    for w in ws[1:]:
        prev = groups[-1][-1]
        if w.x0 - prev.x1 > 1.5 * (prev.y1 - prev.y0):
            groups.append([w])
        else:
            groups[-1].append(w)
    return groups


def link_rows(
    schedule: Schedule, hits_by_page: dict[str, list[TagHit]]
) -> tuple[Schedule, list[str]]:
    """Attach plan occurrences to rows by exact tag match; returns (schedule, unlinked row tags)."""
    unlinked = []
    rows = []
    for r in schedule.rows:
        links = [
            {"page_id": pid, "bbox_px": list(h.bbox), "text": h.tag, "source": h.source}
            for pid, hits in hits_by_page.items()
            for h in hits
            if r.tag and h.tag == r.tag
        ]
        if r.tag and not links:
            unlinked.append(r.tag)
        rows.append(r.model_copy(update={"links": links}))
    return schedule.model_copy(update={"rows": rows}), unlinked
