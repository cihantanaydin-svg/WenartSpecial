"""Source-specific extraction into primitives (DXF, vector PDF) or directly into a PlanGraph (IFC).

The sandboxed S0 tasks already parsed the files; this module reads their JSON (trusted data) and
converts it to plan metres, keeping the document → plan transform (``DocTransform``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from archrender.core.schemas.document import Word
from archrender.core.schemas.provenance import Fact, fact
from archrender.plan.prims import ArcPrim, DimPrim, Polyline, Prims, TextPrim

# order matters: pattern/hatch layers named after walls ("A-WALL-PATT") are not walls
LAYER_ROLES: list[tuple[str, re.Pattern[str]]] = [
    ("hatch", re.compile(r"patt|hatch|tarama|poch|tara$", re.I)),
    ("dims", re.compile(r"dim|[oö]l[cç][uü]|bema|cote|kot", re.I)),
    ("text", re.compile(r"anno|text|yaz[iı]|metin|label|etiket", re.I)),
    ("window", re.compile(r"glaz|pencere|window|fenster|fen[eê]tre|ventana|camekan", re.I)),
    ("door", re.compile(r"door|kap[iı]|t[uü]r\b|porte|puerta", re.I)),
    ("wall", re.compile(r"wall|duvar|mauer|wand|\bmur\b|muro|parete|perde", re.I)),
    ("furniture", re.compile(r"furn|mobil|e[sş]ya|m[oö]bel", re.I)),
]

UNIT_M = {1: 0.0254, 2: 0.3048, 4: 0.001, 5: 0.01, 6: 1.0, 14: 0.1}
UNIT_NAMES = {0.0254: "in", 0.3048: "ft", 0.001: "mm", 0.01: "cm", 1.0: "m", 0.1: "dm"}


def layer_role(name: str) -> str | None:
    if not name or name == "0":
        return None
    for role, rx in LAYER_ROLES:
        if rx.search(name):
            return role
    return None


@dataclass
class Extracted:
    prims: Prims
    doc_to_plan: list[list[float]]  # 2×3 affine: document units (or page px) → plan metres
    unit: Fact[float]  # metres per document unit (DXF) or per page pixel (PDF)
    notes: list[str] = field(default_factory=list)


def _unit_from_doors(radii: list[float]) -> float | None:
    """Metres per drawing unit that makes door swings 0.6–1.2 m (median), or None."""
    if not radii:
        return None
    med = float(np.median(radii))
    best = None
    for k in (1.0, 0.1, 0.01, 0.001, 0.0254, 0.3048):
        v = med * k
        if 0.55 <= v <= 1.25 and (best is None or abs(v - 0.85) < abs(med * best - 0.85)):
            best = k
    return best


def dxf_extract(entities: dict[str, Any], summary: dict[str, Any], doc_id: str) -> Extracted:
    notes: list[str] = []
    radii = [a["r"] for a in entities.get("arcs", []) if 60 <= (a["a1"] - a["a0"]) % 360 <= 120]
    insunits = int(summary.get("insunits") or 0)
    if insunits in UNIT_M:
        k = UNIT_M[insunits]
        unit = fact(
            k,
            "dxf_entity",
            1.0,
            source_doc=doc_id,
            note=f"$INSUNITS={insunits} ({UNIT_NAMES.get(k, '?')})",
        )
        guess = _unit_from_doors(radii)
        if guess is not None and abs(guess - k) / k > 0.01:
            notes.append(
                f"$INSUNITS says {UNIT_NAMES.get(k)} but door swings suggest {UNIT_NAMES.get(guess)}"
            )
    else:
        guess = _unit_from_doors(radii)
        if guess is None:
            guess = (
                0.001
                if (summary.get("extents") and max(abs(v) for v in summary["extents"]) > 2000)
                else 1.0
            )
            unit = fact(
                guess,
                "default",
                0.3,
                source_doc=doc_id,
                note="unitless DXF; no door swings to infer the unit",
            )
        else:
            unit = fact(
                guess,
                "derived",
                0.8,
                source_doc=doc_id,
                note=f"unitless DXF; door swing radii imply {UNIT_NAMES[guess]}",
            )
        notes.append(unit.provenance[0].note or "")
    k = unit.value
    # plan origin: the drawing's lower-left corner rounded down to whole metres
    xs = [p[0] for ln in entities.get("lines", []) for p in ln["pts"]]
    ys = [p[1] for ln in entities.get("lines", []) for p in ln["pts"]]
    ox = math.floor(min(xs) * k) if xs else 0.0
    oy = math.floor(min(ys) * k) if ys else 0.0

    def tx(x: float, y: float) -> tuple[float, float]:
        return x * k - ox, y * k - oy

    prims = Prims(method="dxf_entity")
    for i, ln in enumerate(entities.get("lines", [])):
        pts = np.array([tx(*p) for p in ln["pts"]], np.float64)
        if len(pts) < 2:
            continue
        prims.polylines.append(
            Polyline(
                pts,
                closed=bool(ln.get("closed")),
                curve=bool(ln.get("curve")),
                role=layer_role(ln.get("layer", "")),
                group=ln.get("layer", "") or f"e{i}",
            )
        )
    for a in entities.get("arcs", []):
        cx, cy = tx(*a["c"])
        prims.arcs.append(
            ArcPrim(cx, cy, a["r"] * k, a["a0"], a["a1"], layer_role(a.get("layer", "")))
        )
    for t in summary.get("texts", []):
        x, y = tx(t["x"], t["y"])
        prims.texts.append(
            TextPrim(
                t["text"],
                x,
                y,
                float(t.get("height") or 0) * k,
                float(t.get("rotation") or 0),
                "dxf_text",
            )
        )
    for d in entities.get("dims", []):
        if d.get("p1") and d.get("p2"):
            value = d["measurement"] * k if d.get("measurement") else None
            prims.dims.append(DimPrim(tx(*d["p1"]), tx(*d["p2"]), value, d.get("text", "")))
    prims.notes = notes
    return Extracted(prims, [[k, 0.0, -ox], [0.0, k, -oy]], unit, notes)


def pdf_extract(paths: dict[str, Any], words: list[Word], m_per_px: float) -> Extracted:
    """Page pixels (y down) → plan metres (y up) at ``m_per_px``; origin at the drawing's corner."""
    all_x = [q[0] for p in paths.get("paths", []) for s in p["sub"] for q in s]
    all_y = [q[1] for p in paths.get("paths", []) for s in p["sub"] for q in s]
    ox = math.floor(min(all_x) * m_per_px) if all_x else 0.0
    oy = math.floor(-max(all_y) * m_per_px) if all_y else 0.0

    def tx(x: float, y: float) -> tuple[float, float]:
        return x * m_per_px - ox, -y * m_per_px - oy

    prims = Prims(method="pdf_vector")
    for i, p in enumerate(paths.get("paths", [])):
        for sub, closed, curve in zip(p["sub"], p["closed"], p["curve"], strict=True):
            pts = np.array([tx(*q) for q in sub], np.float64)
            if len(pts) < 2:
                continue
            if closed and len(pts) > 2 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            prims.polylines.append(
                Polyline(
                    pts,
                    closed=bool(closed),
                    curve=bool(curve),
                    width=(p.get("w") or 0.0) * m_per_px,
                    fill=bool(p.get("fill")),
                    clip=bool(p.get("clip")),
                    group=f"path{i}",
                )
            )
    for w in words:
        x, y = tx((w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2)
        h = (w.y1 - w.y0) * m_per_px if w.angle_deg in (0.0, 180.0) else (w.x1 - w.x0) * m_per_px
        prims.texts.append(TextPrim(w.text, x, y, h, w.angle_deg, "pdf_text"))
    unit = fact(m_per_px, "derived", 0.9, note="metres per page pixel from the scale estimate")
    return Extracted(prims, [[m_per_px, 0.0, -ox], [0.0, -m_per_px, -oy]], unit, [])
