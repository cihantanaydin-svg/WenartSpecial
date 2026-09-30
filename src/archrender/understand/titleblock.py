"""Title block fields and the drawing scale from words (text layer, DXF text or OCR).

The title block is found from its field labels (Turkish and English, matched with
:func:`~archrender.understand.text.fold`), preferring the bottom-right of the sheet. A value is the
text directly below its label (stacked cells) or to its right on the same line. The scale becomes a
``Fact`` only when it appears next to a scale label; a bare "1/50" elsewhere is not trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from archrender.core.schemas.document import Word
from archrender.core.schemas.provenance import Fact, Method, fact
from archrender.core.schemas.understanding import TitleBlock
from archrender.understand.text import fold, is_scale_label, parse_scale

LABELS: dict[str, tuple[str, ...]] = {
    "project": ("proje", "proje adi", "project", "project name", "is", "isin adi"),
    "sheet_title": ("pafta adi", "pafta", "drawing title", "sheet title", "title", "cizim adi"),
    "sheet_no": ("pafta no", "sheet no", "drawing no", "sheet number", "dwg no", "pafta numarasi"),
    "scale": ("olcek", "scale"),
    "date": ("tarih", "date"),
    "drawn": ("cizen", "drawn by", "drawn", "cizim"),
    "level": ("kat", "level", "floor"),
}
_LABEL_INDEX = {v: k for k, vs in LABELS.items() for v in vs}


@dataclass
class _Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (
            min(w.x0 for w in self.words),
            min(w.y0 for w in self.words),
            max(w.x1 for w in self.words),
            max(w.y1 for w in self.words),
        )

    @property
    def height(self) -> float:
        return max(w.y1 - w.y0 for w in self.words)


def bands(words: list[Word]) -> list[list[Word]]:
    """Horizontal words clustered by text-line band (vertical centre), each band sorted by x."""
    horiz = [w for w in words if abs(((w.angle_deg + 45) % 180) - 45) < 10]
    horiz.sort(key=lambda w: (w.y0 + w.y1) / 2)
    out: list[list[Word]] = []
    centre = 0.0
    for w in horiz:
        c, h = (w.y0 + w.y1) / 2, w.y1 - w.y0
        if out and abs(c - centre) < 0.45 * max(h, out[-1][0].y1 - out[-1][0].y0):
            out[-1].append(w)
            centre = sum((x.y0 + x.y1) / 2 for x in out[-1]) / len(out[-1])
        else:
            out.append([w])
            centre = c
    return [sorted(b, key=lambda w: w.x0) for b in out]


def lines(words: list[Word]) -> list[_Line]:
    """Text lines: words of one band separated by less than ~one character height."""
    out: list[_Line] = []
    for band in bands(words):
        cur = [band[0]]
        for w in band[1:]:
            prev = cur[-1]
            h = max(w.y1 - w.y0, prev.y1 - prev.y0)
            if -0.2 * h <= w.x0 - prev.x1 <= 0.9 * h:
                cur.append(w)
            else:
                out.append(_Line(cur))
                cur = [w]
        out.append(_Line(cur))
    return out


def _label_of(text: str) -> str | None:
    key = fold(text).rstrip(" :")
    return _LABEL_INDEX.get(key)


def extract_title_block(
    page_id: str, words: list[Word], width_px: float, height_px: float, method: Method
) -> TitleBlock | None:
    lns = lines(words)
    labelled: list[tuple[str, _Line]] = []
    for ln in lns:
        key = _label_of(ln.text)
        if key is None:
            # "ÖLÇEK: 1/50" on one line → label + value together
            head = fold(ln.words[0].text).rstrip(" :")
            if _LABEL_INDEX.get(head) and len(ln.words) > 1:
                labelled.append((_LABEL_INDEX[head], _Line(ln.words[:1])))
            continue
        labelled.append((key, ln))
    if len(labelled) < 2:
        return None
    # prefer labels in the bottom-right region (standard position); fall back to all
    br = [
        (k, ln)
        for k, ln in labelled
        if ln.box[0] > 0.45 * width_px and ln.box[1] > 0.55 * height_px
    ]
    chosen = br if len(br) >= 2 else labelled
    fields: dict[str, Fact[str]] = {}
    used: list[_Line] = []
    for key, label in chosen:
        if key in fields:
            continue
        value = _value_for(label, lns, labelled)
        if value is None:
            continue
        used += [label, value]
        fields[key] = fact(
            value.text, method, 0.9 if method == "pdf_text" else 0.7, note=f"label '{label.text}'"
        )
    if len(fields) < 2:
        return None
    boxes = [ln.box for ln in used]
    bbox = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    scale: Fact[float] | None = None
    if "scale" in fields:
        den = parse_scale(fields["scale"].value)
        if den is not None:
            scale = fact(
                den,
                method,
                fields["scale"].confidence,
                note=f"title block '{fields['scale'].value}'",
            )
    return TitleBlock(
        page_id=page_id, bbox_px=bbox, fields=fields, scale=scale, level=fields.get("level")
    )


def _value_for(label: _Line, lns: list[_Line], labelled: list[tuple[str, _Line]]) -> _Line | None:
    lb = label.box
    label_ids = {id(ln) for _, ln in labelled}
    best: tuple[float, _Line] | None = None
    for ln in lns:
        if ln is label or id(ln) in label_ids:
            continue
        b = ln.box
        # below the label, left-aligned within the cell
        dy = b[1] - lb[3]
        if -0.2 * label.height <= dy <= 3.0 * label.height and abs(b[0] - lb[0]) < 4 * label.height:
            score = dy + abs(b[0] - lb[0]) * 0.2
            if best is None or score < best[0]:
                best = (score, ln)
        # to the right on the same line
        dx = b[0] - lb[2]
        if (
            abs((b[1] + b[3]) / 2 - (lb[1] + lb[3]) / 2) < 0.6 * label.height
            and 0 <= dx <= 12 * label.height
        ):
            score = dx * 0.5
            if best is None or score < best[0]:
                best = (score, ln)
    if best is None and label.words and len(label.words) == 1:
        return None
    return best[1] if best else None


def scale_from_words(words: list[Word], method: Method) -> Fact[float] | None:
    """A scale annotation anywhere on the sheet ("ÖLÇEK 1/50", "SCALE 1:100") next to its label."""
    for ln in lines(words):
        if is_scale_label(ln.text):
            den = parse_scale(ln.text)
            if den is not None:
                return fact(den, method, 0.8, note=f"'{ln.text}'")
    return None
