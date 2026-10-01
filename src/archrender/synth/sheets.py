"""Synthetic drawing sheets (vector PDF) with exact ground truth.

Every string drawn through :meth:`Sheet.text` is recorded as ground-truth words (boxes in page
pixels at ``GT_DPI``, the intake raster resolution), with an optional role (``title:scale``,
``tag:K1``, ``dim``, …). Page classes follow :data:`archrender.core.schemas.understanding.PAGE_CLASSES`.
Everything is generated from a seed; nothing is copied from real projects.
"""

from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as rl_canvas
from shapely.geometry.polygon import orient

from archrender.assets import font_path
from archrender.synth.layout import Layout, random_layout
from archrender.synth.plan import PlanSpec, from_layout, wall_solid

GT_DPI = 300.0
MM = 72.0 / 25.4
SHEETS_MM = {"A4": (210.0, 297.0), "A3": (420.0, 297.0), "A2": (594.0, 420.0), "A1": (841.0, 594.0)}
FONTS = {"sans": "DejaVu", "bold": "DejaVu-Bold", "mono": "DejaVu-Mono"}
_registered = False


def _register_fonts() -> None:
    global _registered
    if not _registered:
        pdfmetrics.registerFont(TTFont("DejaVu", str(font_path("sans"))))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(font_path("sans-bold"))))
        pdfmetrics.registerFont(TTFont("DejaVu-Mono", str(font_path("mono"))))
        _registered = True


def fmt_m(v: float, style: str) -> str:
    """Dimension text: '3,50' (Turkish decimal comma), '3.50', '350' (centimetres) or
    feet-inches to the nearest inch ('12'-6"')."""
    if style == "cm":
        return f"{round(v * 100):d}"
    if style == "ftin":
        inches = round(v / 0.0254)
        return f"{inches // 12}'-{inches % 12}\""
    s = f"{v:.2f}"
    return s.replace(".", ",") if style == "comma" else s


@dataclass
class Sheet:
    """A reportlab page plus its ground truth."""

    c: rl_canvas.Canvas
    w_mm: float
    h_mm: float
    words: list[dict[str, Any]] = field(default_factory=list)
    gt: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> float:
        return GT_DPI / 72.0

    def px(self, x_pt: float, y_pt: float) -> tuple[float, float]:
        return x_pt * self.k, (self.h_mm * MM - y_pt) * self.k

    def bbox_px(self, pts: list[tuple[float, float]]) -> list[float]:
        p = [self.px(x, y) for x, y in pts]
        return [
            min(a for a, _ in p),
            min(b for _, b in p),
            max(a for a, _ in p),
            max(b for _, b in p),
        ]

    def text(
        self,
        x: float,
        y: float,
        s: str,
        size: float,
        *,
        font: str = "sans",
        angle: float = 0.0,
        anchor: str = "left",
        role: str | None = None,
        record: bool = True,
    ) -> list[float]:
        """Draw ``s`` with its baseline at (x, y) points; returns the bbox (page px) of the string."""
        name = FONTS[font]
        total = pdfmetrics.stringWidth(s, name, size)
        asc, desc = pdfmetrics.getAscentDescent(name, size)
        dx0 = {"left": 0.0, "center": -total / 2, "right": -total}[anchor]
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))

        def tr(u: float, v: float) -> tuple[float, float]:
            return x + u * ca - v * sa, y + u * sa + v * ca

        self.c.saveState()
        self.c.translate(x, y)
        self.c.rotate(angle)
        self.c.setFont(name, size)
        self.c.drawString(dx0, 0, s)
        self.c.restoreState()
        if record:
            offset = 0.0
            for part in s.split(" "):
                if part:
                    ww = pdfmetrics.stringWidth(part, name, size)
                    u0 = dx0 + offset
                    box = self.bbox_px(
                        [tr(u0, desc), tr(u0 + ww, desc), tr(u0 + ww, asc), tr(u0, asc)]
                    )
                    self.words.append(
                        {
                            "text": part,
                            "bbox": box,
                            "angle_deg": round(angle % 360, 1),
                            "role": role,
                        }
                    )
                offset += pdfmetrics.stringWidth(part + " ", name, size)
        return self.bbox_px(
            [tr(dx0, desc), tr(dx0 + total, desc), tr(dx0 + total, asc), tr(dx0, asc)]
        )

    def frame(self, margin_mm: float = 10.0) -> None:
        c = self.c
        c.setLineWidth(1.2)
        c.rect(
            margin_mm * MM,
            margin_mm * MM,
            (self.w_mm - 2 * margin_mm) * MM,
            (self.h_mm - 2 * margin_mm) * MM,
        )


# ---------------------------------------------------------------------------------------------
# title block, north arrow, scale bar
# ---------------------------------------------------------------------------------------------
TB_LABELS = {
    "tr": {
        "project": "PROJE",
        "sheet_title": "PAFTA ADI",
        "sheet_no": "PAFTA NO",
        "scale": "ÖLÇEK",
        "date": "TARİH",
        "drawn": "ÇİZEN",
        "level": "KAT",
    },
    "en": {
        "project": "PROJECT",
        "sheet_title": "DRAWING TITLE",
        "sheet_no": "SHEET NO",
        "scale": "SCALE",
        "date": "DATE",
        "drawn": "DRAWN BY",
        "level": "LEVEL",
    },
}
PROJECTS = [
    "Kadıköy Konut Projesi",
    "Moda Daire Tadilatı",
    "Çamlıca Villa",
    "Ege Evleri",
    "Şişli Ofis Katı",
]
FIRMS = ["ÖRNEK MİMARLIK", "ÇİZGİ TASARIM STÜDYOSU", "KUZEY YAPI ATÖLYESİ"]


IMPERIAL_SCALES = {48: '1/4" = 1\'-0"', 96: '1/8" = 1\'-0"', 24: '1/2" = 1\'-0"'}


def scale_text(scale: int, rng: np.random.Generator) -> str:
    if scale in IMPERIAL_SCALES:
        return IMPERIAL_SCALES[scale]
    return str(rng.choice([f"1/{scale}", f"1:{scale}", f"M 1:{scale}"]))


def title_block(sh: Sheet, rng: np.random.Generator, lang: str, fields: dict[str, str]) -> None:
    """Bottom-right title block; ground truth goes to ``sh.gt['title_block']``."""
    c = sh.c
    w, h = 175.0, 58.0
    x0 = (sh.w_mm - 10 - w) * MM
    y0 = 10 * MM
    c.setLineWidth(0.8)
    c.rect(x0, y0, w * MM, h * MM)
    labels = TB_LABELS[lang]
    firm = str(rng.choice(FIRMS))
    sh.text(x0 + 4 * MM, y0 + (h - 8) * MM, firm, 11, font="bold", role="title:firm")
    rows = [("project", "sheet_title"), ("sheet_no", "scale"), ("date", "drawn"), ("level", None)]
    row_h = 10.0
    for i, (left, right) in enumerate(rows):
        yy = y0 + (h - 14 - (i + 1) * row_h) * MM
        c.line(x0, yy + row_h * MM, x0 + w * MM, yy + row_h * MM)
        for col, key in enumerate((left, right)):
            if key is None or key not in fields:
                continue
            xx = x0 + (4 + col * 88) * MM
            sh.text(xx, yy + 6.2 * MM, labels[key], 5.5, role=f"label:{key}")
            sh.text(xx, yy + 1.8 * MM, fields[key], 8.5, font="bold", role=f"title:{key}")
        c.line(x0 + 86 * MM, yy, x0 + 86 * MM, yy + row_h * MM)
    sh.gt["title_block"] = {
        "bbox": sh.bbox_px([(x0, y0), (x0 + w * MM, y0 + h * MM)]),
        "fields": dict(fields),
        "lang": lang,
    }


def north_arrow(sh: Sheet, x: float, y: float, r: float, angle: float, lang: str) -> None:
    """Circle with an arrow pointing north at ``angle`` (deg CCW from sheet-up), letter at the tip."""
    c = sh.c
    c.setLineWidth(0.8)
    c.circle(x, y, r)
    a = math.radians(90 + angle)
    tip = (x + r * 0.95 * math.cos(a), y + r * 0.95 * math.sin(a))
    tail = (x - r * 0.7 * math.cos(a), y - r * 0.7 * math.sin(a))
    left = (x + r * 0.35 * math.cos(a + math.pi / 2), y + r * 0.35 * math.sin(a + math.pi / 2))
    right = (x + r * 0.35 * math.cos(a - math.pi / 2), y + r * 0.35 * math.sin(a - math.pi / 2))
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(*left)
    p.lineTo(*tail)
    p.close()
    c.drawPath(p, stroke=1, fill=1)
    p = c.beginPath()
    p.moveTo(*tip)
    p.lineTo(*right)
    p.lineTo(*tail)
    p.close()
    c.drawPath(p, stroke=1, fill=0)
    lx, ly = x + (r + 9) * math.cos(a), y + (r + 9) * math.sin(a)
    sh.text(
        lx,
        ly - 4,
        "K" if lang == "tr" else "N",
        11,
        font="bold",
        anchor="center",
        role="north_letter",
    )
    sh.gt["north"] = {
        "angle_deg": round(angle % 360.0, 2),
        "bbox": sh.bbox_px([(x - r, y - r), (x + r, y + r)]),
    }


def scale_bar(sh: Sheet, x: float, y: float, scale: int) -> None:
    c = sh.c
    seg_m = 1.0 if scale <= 100 else 10.0
    seg_pt = seg_m * 1000 / scale * MM
    for i in range(5):
        c.rect(x + i * seg_pt, y, seg_pt, 2 * MM, stroke=1, fill=i % 2)
    for i in range(0, 6, 5):
        sh.text(x + i * seg_pt, y + 3 * MM, f"{i * seg_m:g}", 6, anchor="center", role="scalebar")
    sh.text(x + 5 * seg_pt + 2 * MM, y + 0.4 * MM, "m", 6, role="scalebar")
    sh.gt["scale_bar"] = {"segment_m": seg_m, "segment_px": seg_pt * sh.k}


# ---------------------------------------------------------------------------------------------
# plans
# ---------------------------------------------------------------------------------------------
@dataclass
class PlanFrame:
    ox: float  # sheet points of plan (0, 0)
    oy: float
    scale: int

    def p(self, x: float, y: float) -> tuple[float, float]:
        f = 1000.0 / self.scale * MM
        return self.ox + x * f, self.oy + y * f

    def d(self, v: float) -> float:
        return v * 1000.0 / self.scale * MM

    def to_page_px(self, sh: Sheet) -> list[list[float]]:
        """Affine plan metres → page pixels at GT_DPI (y down): [[a, b, c], [d, e, f]]."""
        f = 1000.0 / self.scale * MM * sh.k
        return [[f, 0.0, self.ox * sh.k], [0.0, -f, (sh.h_mm * MM - self.oy) * sh.k]]


def _fit_plan(
    sheet_mm: tuple[float, float], extent_m: tuple[float, float], imperial: bool = False
) -> tuple[int, float] | None:
    avail_w, avail_h = sheet_mm[0] - 20 - 10, sheet_mm[1] - 20 - 70
    for s in (48, 96) if imperial else (50, 75, 100):
        w, h = extent_m[0] * 1000 / s, extent_m[1] * 1000 / s
        if w <= avail_w and h <= avail_h:
            return s, 0.0
    return None


def _rings(geom: Any) -> list[list[tuple[float, float]]]:
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for p in polys:
        if p.is_empty:
            continue
        p = orient(p)
        out.append(list(p.exterior.coords)[:-1])
        out += [list(h.coords)[:-1] for h in p.interiors]
    return out


def _path(sh: Sheet, fr: PlanFrame, rings: list[list[tuple[float, float]]]) -> Any:
    p = sh.c.beginPath()
    for ring in rings:
        p.moveTo(*fr.p(*ring[0]))
        for q in ring[1:]:
            p.lineTo(*fr.p(*q))
        p.close()
    return p


WALL_STYLES = ["solid", "solid", "grey", "hatch", "outline"]


def draw_walls(sh: Sheet, fr: PlanFrame, spec: PlanSpec, style: str) -> None:
    """Wall bodies with the opening gaps cut, as one cleaned outline (no lines across junctions)."""
    c = sh.c
    rings = _rings(wall_solid(spec, cut_openings=True))
    c.setLineWidth(0.6)
    if style in ("solid", "grey"):
        c.setFillGray(0.0 if style == "solid" else 0.55)
        c.drawPath(_path(sh, fr, rings), stroke=1, fill=1)
        c.setFillGray(0.0)
        return
    c.drawPath(_path(sh, fr, rings), stroke=1, fill=0)
    if style == "hatch":  # 45° poché lines, clipped to the walls
        c.saveState()
        c.clipPath(_path(sh, fr, rings), stroke=0, fill=0)
        c.setLineWidth(0.2)
        x0, y0, x1, y1 = spec.extent()
        a0, b0 = fr.p(x0 - 1, y0 - 1)
        a1, b1 = fr.p(x1 + 1, y1 + 1)
        step = 1.3 * MM
        u = a0 - (b1 - b0)
        while u < a1:
            c.line(u, b0, u + (b1 - b0), b1)
            u += step
        c.restoreState()


def draw_openings(sh: Sheet, fr: PlanFrame, spec: PlanSpec, *, swings: bool) -> None:
    c = sh.c
    for o in spec.openings:
        w = spec.walls[o.wall]
        ux, uy = w.direction(o.t)
        nx, ny = -uy * o.swing, ux * o.swing
        cx, cy = w.point(o.t)
        t = w.thickness / 2
        if o.kind == "window":
            c.setLineWidth(0.35)
            for off in (-t, 0.0, t):
                a = (cx - ux * o.width / 2 - uy * off, cy - uy * o.width / 2 + ux * off)
                b = (cx + ux * o.width / 2 - uy * off, cy + uy * o.width / 2 + ux * off)
                c.line(*fr.p(*a), *fr.p(*b))
            continue
        # door: jambs, leaf and swing arc; the hinge is at the wall-start side of the gap
        s = -1.0 if o.hinge == "start" else 1.0
        hinge = (cx + s * ux * o.width / 2, cy + s * uy * o.width / 2)
        c.setLineWidth(0.3)
        for k in (-1, 1):
            j = (cx + k * ux * o.width / 2, cy + k * uy * o.width / 2)
            c.line(*fr.p(j[0] - nx * t, j[1] - ny * t), *fr.p(j[0] + nx * t, j[1] + ny * t))
        if not swings:
            continue
        leaf_end = (hinge[0] + nx * o.width, hinge[1] + ny * o.width)
        c.setLineWidth(0.6)
        c.line(*fr.p(*hinge), *fr.p(*leaf_end))
        c.setLineWidth(0.25)
        closed = (-s * ux, -s * uy)  # from the hinge along the wall to the other jamb
        a0 = math.degrees(math.atan2(closed[1], closed[0]))
        a1 = math.degrees(math.atan2(ny, nx))
        extent = (a1 - a0 + 180) % 360 - 180
        hx, hy = fr.p(*hinge)
        r = fr.d(o.width)
        c.arc(hx - r, hy - r, hx + r, hy + r, a0, extent)


def draw_tags(sh: Sheet, fr: PlanFrame, spec: PlanSpec, rng: np.random.Generator) -> None:
    c = sh.c
    tags = []
    xs = [p[0] for w in spec.walls for p in w.polyline()]
    ys = [p[1] for w in spec.walls for p in w.polyline()]
    mid = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    for o in spec.openings:
        w = spec.walls[o.wall]
        ux, uy = w.direction(o.t)
        nx, ny = -uy, ux
        cx, cy = w.point(o.t)
        off = 0.75 if o.kind == "window" else -0.55
        inward = math.hypot(cx + nx - mid[0], cy + ny - mid[1]) < math.hypot(
            cx - mid[0], cy - mid[1]
        )
        if w.exterior and o.kind == "window" and inward:  # tag on the outward side of the facade
            nx, ny = -nx, -ny
        tx, ty = fr.p(cx + nx * off, cy + ny * off)
        r = 3.2 * MM
        c.setLineWidth(0.4)
        if o.kind == "window":
            c.ellipse(tx - r * 1.25, ty - r * 0.8, tx + r * 1.25, ty + r * 0.8)
        else:
            c.circle(tx, ty, r)
        box = sh.text(tx, ty - 2.2, o.tag, 6.5, font="bold", anchor="center", role=f"tag:{o.tag}")
        tags.append({"tag": o.tag, "kind": o.kind, "bbox": box})
    sh.gt["tags"] = tags


def draw_room_labels(
    sh: Sheet,
    fr: PlanFrame,
    spec: PlanSpec,
    rng: np.random.Generator,
    lang: str,
    *,
    ceiling: bool = False,
    room_numbers: bool = False,
) -> None:
    rooms = []
    area_style = rng.choice(["comma", "dot"]) if lang == "en" else "comma"
    for i, r in enumerate(spec.rooms):
        x, y = fr.p(*r.label)
        name = r.name.upper() if rng.random() < 0.4 else r.name
        if room_numbers:  # matches the finish schedule's "MAHAL NO" (Z01, Z02, …)
            sh.text(x, y + 13, room_number(i), 7.5, anchor="center", role=f"tag:{room_number(i)}")
        sh.text(x, y + 2, name, 8.5, font="bold", anchor="center", role="room_name")
        area = round(spec.faces[r.id].area, 2)  # net floor area, wall face to wall face
        if ceiling:
            label = "+" + fmt_m(r.ceiling, "comma" if lang == "tr" else "dot")
            sh.text(x, y - 9, label, 7, anchor="center", role="ceiling_height")
        else:
            unit = "m²" if rng.random() < 0.7 else "m2"
            sh.text(
                x,
                y - 9,
                f"{fmt_m(area, str(area_style))} {unit}",
                7,
                anchor="center",
                role="room_area",
            )
        rooms.append({"id": r.id, "name": r.name, "area_m2": area, "ceiling_m": r.ceiling})
    sh.gt["rooms"] = rooms


def room_number(i: int) -> str:
    return f"Z{i + 1:02d}"


def _dim_line(
    sh: Sheet,
    a: tuple[float, float],
    b: tuple[float, float],
    offset: tuple[float, float],
    value: float,
    style: str,
    fr: PlanFrame,
) -> None:
    """Aligned dimension from plan point a to b, drawn at ``offset`` (plan metres) from them."""
    c = sh.c
    c.setLineWidth(0.25)
    ox, oy = offset
    pa = (a[0] + ox, a[1] + oy)
    pb = (b[0] + ox, b[1] + oy)
    c.line(*fr.p(*a), *fr.p(pa[0] + ox * 0.15, pa[1] + oy * 0.15))
    c.line(*fr.p(*b), *fr.p(pb[0] + ox * 0.15, pb[1] + oy * 0.15))
    c.line(*fr.p(*pa), *fr.p(*pb))
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    for q in (pa, pb):  # 45° ticks
        qx, qy = fr.p(*q)
        tx, ty = (ux - uy) / math.sqrt(2) * 1.7 * MM, (uy + ux) / math.sqrt(2) * 1.7 * MM
        c.line(qx - tx, qy - ty, qx + tx, qy + ty)
    angle = math.degrees(math.atan2(uy, ux))
    if angle <= -90.0 or angle > 90.0:  # keep text readable
        angle = (angle + 180.0) % 360.0
        if angle > 180:
            angle -= 360
    rad = math.radians(angle)
    mx, my = fr.p((pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2)
    tx, ty = mx - math.sin(rad) * 1.2 * MM, my + math.cos(rad) * 1.2 * MM
    text = fmt_m(value, style)
    box = sh.text(tx, ty, text, 6.5, angle=angle, anchor="center", role="dim")
    sh.gt.setdefault("dimensions", []).append(
        {
            "text": text,
            "value_m": round(value, 3),
            "bbox": box,
            "angle_deg": round(angle % 360, 1),
            "a_m": [round(pa[0], 4), round(pa[1], 4)],
            "b_m": [round(pb[0], 4), round(pb[1], 4)],
        }
    )


def draw_dimensions(sh: Sheet, fr: PlanFrame, spec: PlanSpec, style: str) -> None:
    """Exterior dimension chains along both lattice axes (room splits + overall length), on the
    sides whose facade is straight. Values are lattice lengths, i.e. true lengths along e1/e2."""
    layout = spec.layout
    if layout is None:
        return
    width, depth = layout.width, layout.depth
    arc_walls = {i for i, w in enumerate(spec.walls[:4]) if w.is_arc}
    # e1 chain: bottom (wall 0) unless it is the arc, else top (wall 2); e2 chain: left (3) / right (1)
    use_top = 0 in arc_walls
    use_right = 3 in arc_walls
    y_line = depth if use_top else 0.0
    xs = sorted(
        {0.0, width}
        | {r.x0 for r in layout.rooms if abs((r.y1 if use_top else r.y0) - y_line) < 1e-6}
        | {r.x1 for r in layout.rooms if abs((r.y1 if use_top else r.y0) - y_line) < 1e-6}
    )
    n1 = (-spec.e1[1], spec.e1[0])  # left normal of e1 (towards the plan)
    sgn = 1.0 if use_top else -1.0  # outward
    for k, lst in ((1.4, list(itertools.pairwise(xs))), (2.1, [(0.0, width)])):
        for a, b in lst:
            pa, pb = spec.lattice(a, y_line), spec.lattice(b, y_line)
            base_a = (pa[0] + n1[0] * sgn * 0.4, pa[1] + n1[1] * sgn * 0.4)
            base_b = (pb[0] + n1[0] * sgn * 0.4, pb[1] + n1[1] * sgn * 0.4)
            _dim_line(sh, base_a, base_b, (n1[0] * sgn * k, n1[1] * sgn * k), b - a, style, fr)
    x_line = width if use_right else 0.0
    ys = sorted(
        {0.0, depth}
        | {r.y0 for r in layout.rooms if abs((r.x1 if use_right else r.x0) - x_line) < 1e-6}
        | {r.y1 for r in layout.rooms if abs((r.x1 if use_right else r.x0) - x_line) < 1e-6}
    )
    n2 = (-spec.e2[1], spec.e2[0])  # left normal of e2: away from the plan on the left side
    sgn = -1.0 if use_right else 1.0
    for k, lst in ((1.4, list(itertools.pairwise(ys))), (2.1, [(0.0, depth)])):
        for a, b in lst:
            pa, pb = spec.lattice(x_line, a), spec.lattice(x_line, b)
            base_a = (pa[0] + n2[0] * sgn * 0.4, pa[1] + n2[1] * sgn * 0.4)
            base_b = (pb[0] + n2[0] * sgn * 0.4, pb[1] + n2[1] * sgn * 0.4)
            _dim_line(sh, base_a, base_b, (n2[0] * sgn * k, n2[1] * sgn * k), b - a, style, fr)


def draw_fixtures(sh: Sheet, fr: PlanFrame, spec: PlanSpec, rng: np.random.Generator) -> None:
    c = sh.c
    c.setLineWidth(0.3)
    if spec.layout is None:
        return
    for r in spec.layout.rooms:
        nx = max(1, int((r.x1 - r.x0) / 1.4))
        ny = max(1, int((r.y1 - r.y0) / 1.4))
        for i in range(nx):
            for j in range(ny):
                x = r.x0 + (i + 0.5) * (r.x1 - r.x0) / nx
                y = r.y0 + (j + 0.5) * (r.y1 - r.y0) / ny
                px, py = fr.p(*spec.lattice(x, y))
                rr = 1.4 * MM
                c.circle(px, py, rr)
                c.line(px - rr, py - rr, px + rr, py + rr)
                c.line(px - rr, py + rr, px + rr, py - rr)


def draw_islands(sh: Sheet, fr: PlanFrame, spec: PlanSpec, lang: str) -> None:
    c = sh.c
    c.setLineWidth(0.25)
    for isl in spec.islands:
        p = c.beginPath()
        p.moveTo(*fr.p(*isl.polygon[0]))
        for q in isl.polygon[1:]:
            p.lineTo(*fr.p(*q))
        p.close()
        c.drawPath(p, stroke=1, fill=0)
        cx = sum(q[0] for q in isl.polygon) / 4
        cy = sum(q[1] for q in isl.polygon) / 4
        x, y = fr.p(cx, cy)
        sh.text(
            x, y - 2, "ADA" if lang == "tr" else "ISLAND", 5.5, anchor="center", role="furniture"
        )
    for r in spec.rooms:
        if r.double_height:
            x, y = fr.p(*r.label)
            label = "ÇİFT YÜKSEKLİK" if lang == "tr" else "DOUBLE HEIGHT"
            sh.text(x, y - 19, label, 6, anchor="center", role="double_height")


STAMP_TEXT = {"tr": ("ONAYLANDI", "UYGULAMAYA"), "en": ("APPROVED", "FOR CONSTRUCTION")}


def draw_stamp(sh: Sheet, x: float, y: float, rng: np.random.Generator, lang: str) -> None:
    """A red office stamp (often over the drawing in scans)."""
    c = sh.c
    c.saveState()
    c.setStrokeColorRGB(0.75, 0.1, 0.12)
    c.setFillColorRGB(0.75, 0.1, 0.12)
    c.setStrokeAlpha(0.8)
    c.setFillAlpha(0.8)
    r = 15 * MM
    angle = float(rng.uniform(-25, 25))
    c.setLineWidth(1.4)
    c.circle(x, y, r)
    c.setLineWidth(0.6)
    c.circle(x, y, r - 2 * MM)
    top, bottom = STAMP_TEXT[lang]
    sh.text(x, y + 1.5 * MM, top, 9, font="bold", angle=angle, anchor="center", role="stamp")
    sh.text(x, y - 4 * MM, bottom, 5.5, angle=angle, anchor="center", role="stamp")
    c.restoreState()
    sh.gt["stamp"] = {"bbox": sh.bbox_px([(x - r, y - r), (x + r, y + r)])}


# ---------------------------------------------------------------------------------------------
# page classes
# ---------------------------------------------------------------------------------------------
@dataclass
class Page:
    klass: str
    pdf: bytes
    gt: dict[str, Any]


def _new(size: str, landscape: bool = True) -> tuple[Sheet, io.BytesIO]:
    _register_fonts()
    w, h = SHEETS_MM[size]
    if (landscape and h > w) or (not landscape and w > h):
        w, h = h, w
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(w * MM, h * MM), invariant=1)  # byte-reproducible
    c.setAuthor("")
    c.setCreator("archrender.synth")
    return Sheet(c, w, h), buf


def _finish(sh: Sheet, buf: io.BytesIO, klass: str) -> Page:
    sh.c.showPage()
    sh.c.save()
    sh.gt.update(
        {
            "class": klass,
            "page_mm": [sh.w_mm, sh.h_mm],
            "gt_dpi": GT_DPI,
            "words": sh.words,
        }
    )
    return Page(klass, buf.getvalue(), sh.gt)


def _common_fields(
    rng: np.random.Generator,
    lang: str,
    title: str,
    sheet_no: str,
    scale: int | None,
    level: str | None,
) -> dict[str, str]:
    day, month = int(rng.integers(1, 29)), int(rng.integers(1, 13))
    f = {
        "project": str(rng.choice(PROJECTS)),
        "sheet_title": title,
        "sheet_no": sheet_no,
        "date": f"{day:02d}.{month:02d}.2026" if lang == "tr" else f"2026-{month:02d}-{day:02d}",
        "drawn": str(rng.choice(["A.Y.", "M.K.", "S.Ö.", "E.Ş."])),
    }
    if scale is not None:
        f["scale"] = scale_text(scale, rng)
    if level is not None:
        f["level"] = level
    return f


def floor_plan_page(
    rng: np.random.Generator,
    layout: Layout | None = None,
    *,
    lang: str = "tr",
    ceiling: bool = False,
    room_numbers: bool = False,
    spec: PlanSpec | None = None,
    wall_style: str | None = None,
    stamp: bool | None = None,
    imperial: bool = False,
) -> Page:
    """Floor plan (or reflected ceiling plan) sheet of ``spec`` (default: ``layout`` drawn
    orthogonally, or a random layout). Ground truth: words, tags, rooms, dimensions, scale, north
    and ``plan_to_page_px`` (plan metres → page pixels at GT_DPI)."""
    if spec is None:
        spec = from_layout(layout or random_layout(rng, english=lang == "en"))
    x0, y0, x1, y1 = spec.extent()
    extent = (x1 - x0 + 6.0, y1 - y0 + 6.0)
    for size in ("A3", "A2", "A1"):
        fit = _fit_plan(SHEETS_MM[size], extent, imperial)
        if fit:
            break
    assert fit is not None
    scale = fit[0]
    sh, buf = _new(size)
    sh.frame()
    fr = PlanFrame(0, 0, scale)
    fr.ox = (sh.w_mm / 2) * MM - fr.d((x0 + x1) / 2) - 10 * MM
    fr.oy = (sh.h_mm / 2 + 20) * MM - fr.d((y0 + y1) / 2)
    if ceiling:
        style = "outline"
        draw_walls(sh, fr, spec, style)
        draw_openings(sh, fr, spec, swings=False)
        draw_fixtures(sh, fr, spec, rng)
        draw_room_labels(sh, fr, spec, rng, lang, ceiling=True)
        title = "TAVAN PLANI" if lang == "tr" else "REFLECTED CEILING PLAN"
    else:
        style = wall_style or str(rng.choice(WALL_STYLES))
        draw_walls(sh, fr, spec, style)
        draw_openings(sh, fr, spec, swings=True)
        draw_room_labels(sh, fr, spec, rng, lang, room_numbers=room_numbers)
        draw_islands(sh, fr, spec, lang)
        draw_tags(sh, fr, spec, rng)
        dim_style = (
            "ftin" if imperial else "comma" if lang == "tr" else str(rng.choice(["dot", "cm"]))
        )
        draw_dimensions(sh, fr, spec, dim_style)
        title = "ZEMİN KAT PLANI" if lang == "tr" else "GROUND FLOOR PLAN"
    level = "Zemin Kat" if lang == "tr" else "Ground Floor"
    # heading above the plan and any dimension chain on its top side (≤ 2.5 m outside the walls)
    sh.text(
        fr.ox + fr.d(x0),
        fr.oy + fr.d(y1 + 2.6) + 4 * MM,
        title,
        14,
        font="bold",
        role="sheet_heading",
    )
    angle = float(rng.choice([0.0, 0.0, float(rng.uniform(-180, 180))]))
    north_arrow(sh, (sh.w_mm - 35) * MM, (sh.h_mm - 35) * MM, 9 * MM, angle, lang)
    if not imperial and rng.random() < 0.6:
        scale_bar(sh, 20 * MM, 18 * MM, scale)
    title_block(
        sh,
        rng,
        lang,
        _common_fields(
            rng,
            lang,
            title,
            f"{'M' if not ceiling else 'T'}-{int(rng.integers(100, 300))}",
            scale,
            level,
        ),
    )
    if stamp if stamp is not None else (not ceiling and rng.random() < 0.3):
        sx = fr.ox + fr.d(float(rng.uniform(x0, x1)))
        sy = fr.oy + fr.d(float(rng.uniform(y0, y1)))
        draw_stamp(sh, sx, sy, rng, lang)
    sh.gt["scale"] = scale
    sh.gt["plan_to_page_px"] = fr.to_page_px(sh)
    sh.gt["plan_variant"] = spec.variant
    sh.gt["wall_style"] = style
    sh.gt["layout"] = {
        "openings": [
            {
                "tag": o.tag,
                "kind": o.kind,
                "width_m": o.width,
                "height_m": o.height,
                "sill_m": o.sill,
            }
            for o in spec.openings
        ],
    }
    return _finish(sh, buf, "ceiling_plan" if ceiling else "floor_plan")


def _hatch_rect(
    sh: Sheet, x: float, y: float, w: float, h: float, pattern: str, spacing: float = 2.5 * MM
) -> None:
    c = sh.c
    c.saveState()
    p = c.beginPath()
    p.rect(x, y, w, h)
    c.clipPath(p, stroke=1, fill=0)
    c.setLineWidth(0.25)
    if pattern == "diag":
        s = -h
        while s < w:
            c.line(x + s, y, x + s + h, y + h)
            s += spacing
    elif pattern == "cross":
        s = -h
        while s < w:
            c.line(x + s, y, x + s + h, y + h)
            c.line(x + s, y + h, x + s + h, y)
            s += spacing * 1.5
    elif pattern == "dots":
        yy = y
        while yy < y + h:
            xx = x + (spacing / 2 if int((yy - y) / spacing) % 2 else 0)
            while xx < x + w:
                c.circle(xx, yy, 0.25, stroke=0, fill=1)
                xx += spacing
            yy += spacing
    elif pattern == "zigzag":
        yy = y
        while yy < y + h:
            p = c.beginPath()
            p.moveTo(x, yy)
            xx, up = x, True
            while xx < x + w:
                xx += spacing
                p.lineTo(xx, yy + (spacing if up else 0))
                up = not up
            c.drawPath(p, stroke=1, fill=0)
            yy += spacing * 1.6
    c.restoreState()


def _level_marker(sh: Sheet, x: float, y: float, value: float, lang: str) -> None:
    c = sh.c
    c.setLineWidth(0.4)
    p = c.beginPath()
    p.moveTo(x, y)
    p.lineTo(x - 2 * MM, y + 2.5 * MM)
    p.lineTo(x + 2 * MM, y + 2.5 * MM)
    p.close()
    c.drawPath(p, stroke=1, fill=1)
    c.line(x - 8 * MM, y, x + 25 * MM, y)
    txt = (
        "±0,00"
        if abs(value) < 1e-9
        else f"{'+' if value > 0 else '-'}{fmt_m(abs(value), 'comma' if lang == 'tr' else 'dot')}"
    )
    sh.text(x + 3 * MM, y + 1 * MM, txt, 7, role="level")


def section_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    sh, buf = _new("A3")
    sh.frame()
    scale = int(rng.choice([50, 100]))
    fr = PlanFrame(0, 0, scale)
    floors = int(rng.integers(1, 4))
    span = float(rng.uniform(8, 14))
    fh = 3.0
    fr.ox = (sh.w_mm / 2) * MM - fr.d(span) / 2
    fr.oy = 95 * MM
    c = sh.c
    # soil
    _hatch_rect(sh, fr.ox - 20 * MM, fr.oy - 12 * MM, fr.d(span) + 40 * MM, 12 * MM, "cross")
    c.setLineWidth(1.0)
    c.line(fr.ox - 20 * MM, fr.oy, fr.ox + fr.d(span) + 20 * MM, fr.oy)
    for f in range(floors + 1):
        y = f * fh
        c.setFillGray(0.0)
        c.rect(*fr.p(-0.25, y - 0.2), fr.d(span + 0.5), fr.d(0.2), stroke=1, fill=1)
        _level_marker(sh, *fr.p(span + 0.8, y), y, lang)
    for f in range(floors):
        y = f * fh
        for x in (-0.25, span, float(rng.uniform(2.5, span - 2.5))):
            wt = 0.25 if x in (-0.25, span) else 0.1
            c.rect(*fr.p(x, y), fr.d(wt), fr.d(fh - 0.2), stroke=1, fill=1)
        # window openings drawn as gaps with glazing lines in the facades
        c.setFillGray(1.0)
        for x in (-0.25, span):
            c.rect(*fr.p(x, y + 0.9), fr.d(0.25), fr.d(1.4), stroke=1, fill=1)
        c.setFillGray(0.0)
    if rng.random() < 0.5:  # pitched roof
        top = floors * fh
        p = c.beginPath()
        p.moveTo(*fr.p(-0.6, top))
        p.lineTo(*fr.p(span / 2, top + 2.2))
        p.lineTo(*fr.p(span + 0.6, top))
        c.drawPath(p, stroke=1, fill=0)
    title = "A-A KESİTİ" if lang == "tr" else "SECTION A-A"
    sh.text(fr.ox, 75 * MM, title, 14, font="bold", role="sheet_heading")
    title_block(
        sh,
        rng,
        lang,
        _common_fields(rng, lang, title, f"K-{int(rng.integers(1, 20))}", scale, None),
    )
    sh.gt["scale"] = scale
    return _finish(sh, buf, "section")


def elevation_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    sh, buf = _new("A3")
    sh.frame()
    scale = int(rng.choice([50, 100]))
    fr = PlanFrame(0, 0, scale)
    floors = int(rng.integers(1, 4))
    span = float(rng.uniform(8, 16))
    fh = 3.0
    fr.ox = (sh.w_mm / 2) * MM - fr.d(span) / 2
    fr.oy = 95 * MM
    c = sh.c
    c.setLineWidth(1.2)
    c.line(fr.ox - 25 * MM, fr.oy, fr.ox + fr.d(span) + 25 * MM, fr.oy)
    c.setLineWidth(0.7)
    c.rect(*fr.p(0, 0), fr.d(span), fr.d(floors * fh + 0.4), stroke=1, fill=0)
    n_win = max(2, int(span / 2.2))
    for f in range(floors):
        for i in range(n_win):
            x = (i + 0.5) * span / n_win - 0.6
            y = f * fh + 0.9
            c.setLineWidth(0.5)
            c.rect(*fr.p(x, y), fr.d(1.2), fr.d(1.4), stroke=1, fill=0)
            c.setLineWidth(0.25)
            c.line(*fr.p(x + 0.6, y), *fr.p(x + 0.6, y + 1.4))
            c.line(*fr.p(x, y + 0.9), *fr.p(x + 1.2, y + 0.9))
        _level_marker(sh, *fr.p(span + 1.2, f * fh), f * fh, lang)
    if rng.random() < 0.6:
        top = floors * fh + 0.4
        p = c.beginPath()
        p.moveTo(*fr.p(-0.5, top))
        p.lineTo(*fr.p(span / 2, top + 2.5))
        p.lineTo(*fr.p(span + 0.5, top))
        c.drawPath(p, stroke=1, fill=0)
    side = rng.choice(
        ["GÜNEY", "KUZEY", "DOĞU", "BATI"] if lang == "tr" else ["SOUTH", "NORTH", "EAST", "WEST"]
    )
    title = f"{side} GÖRÜNÜŞÜ" if lang == "tr" else f"{side} ELEVATION"
    sh.text(fr.ox, 75 * MM, title, 14, font="bold", role="sheet_heading")
    title_block(
        sh,
        rng,
        lang,
        _common_fields(rng, lang, title, f"G-{int(rng.integers(1, 20))}", scale, None),
    )
    sh.gt["scale"] = scale
    return _finish(sh, buf, "elevation")


DETAIL_LAYERS_TR = [
    ("2 cm iç sıva", "dots", 20),
    ("19 cm yatay delikli tuğla", "diag", 190),
    ("5 cm XPS ısı yalıtımı", "zigzag", 50),
    ("0,8 cm dış cephe sıvası", "dots", 8),
    ("20 cm betonarme perde", "cross", 200),
    ("3 cm şap", "dots", 30),
]
DETAIL_LAYERS_EN = [
    ("20 mm plaster", "dots", 20),
    ("190 mm clay block", "diag", 190),
    ("50 mm XPS insulation", "zigzag", 50),
    ("8 mm render", "dots", 8),
    ("200 mm reinforced concrete", "cross", 200),
    ("30 mm screed", "dots", 30),
]


def detail_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    sh, buf = _new("A3")
    sh.frame()
    scale = int(rng.choice([5, 10, 20]))
    c = sh.c
    layers = list(DETAIL_LAYERS_TR if lang == "tr" else DETAIL_LAYERS_EN)
    rng.shuffle(layers)
    layers = layers[: int(rng.integers(3, 6))]
    x = 60 * MM
    y0, height = 70 * MM, 150 * MM
    c.setLineWidth(0.7)
    for i, (name, pat, thick_mm) in enumerate(layers):
        w = thick_mm / scale * MM * 1.0
        w = max(w, 6 * MM)
        c.rect(x, y0, w, height, stroke=1, fill=0)
        _hatch_rect(sh, x, y0, w, height, pat)
        # leader to the note
        ly = y0 + height - (i + 1) * 18 * MM
        c.setLineWidth(0.3)
        c.line(x + w / 2, ly, 200 * MM, ly)
        c.circle(x + w / 2, ly, 0.8, stroke=0, fill=1)
        sh.text(202 * MM, ly - 2, name, 8, role="note")
        x += w
    title = "DETAY D-1" if lang == "tr" else "DETAIL D-1"
    sh.text(60 * MM, 55 * MM, title, 14, font="bold", role="sheet_heading")
    title_block(
        sh,
        rng,
        lang,
        _common_fields(rng, lang, title, f"D-{int(rng.integers(1, 20))}", scale, None),
    )
    sh.gt["scale"] = scale
    return _finish(sh, buf, "detail")


def site_plan_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    sh, buf = _new("A3")
    sh.frame()
    scale = int(rng.choice([200, 500]))
    fr = PlanFrame(0, 0, scale)
    c = sh.c
    pw, pd = float(rng.uniform(25, 45)), float(rng.uniform(20, 35))
    if scale == 200:
        pw, pd = pw * 0.8, pd * 0.8
    fr.ox = (sh.w_mm / 2 - 20) * MM - fr.d(pw) / 2
    fr.oy = (sh.h_mm / 2 + 25) * MM - fr.d(pd) / 2
    jitter = [(float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))) for _ in range(4)]
    parcel = [
        (0 + jitter[0][0], 0 + jitter[0][1]),
        (pw + jitter[1][0], jitter[1][1]),
        (pw + jitter[2][0], pd + jitter[2][1]),
        (jitter[3][0], pd + jitter[3][1]),
    ]
    c.setLineWidth(0.9)
    c.setDash(6, 2)
    p = c.beginPath()
    p.moveTo(*fr.p(*parcel[0]))
    for q in parcel[1:]:
        p.lineTo(*fr.p(*q))
    p.close()
    c.drawPath(p, stroke=1, fill=0)
    c.setDash()
    bw, bd = pw * 0.45, pd * 0.45
    bx, by = pw * 0.3, pd * 0.3
    c.rect(*fr.p(bx, by), fr.d(bw), fr.d(bd), stroke=1, fill=0)
    _hatch_rect(sh, *fr.p(bx, by), fr.d(bw), fr.d(bd), "diag", 1.5 * MM)
    # road along the bottom
    c.setLineWidth(0.6)
    c.line(*fr.p(-8, -3), *fr.p(pw + 8, -3))
    c.line(*fr.p(-8, -10), *fr.p(pw + 8, -10))
    street = rng.choice(
        ["GÜL SOKAK", "LALE CADDESİ", "ÇINAR SOKAK"]
        if lang == "tr"
        else ["ELM STREET", "OAK AVENUE"]
    )
    sh.text(*fr.p(pw / 2, -7.5), str(street), 8, anchor="center", role="street")
    for _ in range(int(rng.integers(4, 10))):
        tx, ty = float(rng.uniform(1, pw - 1)), float(rng.uniform(1, pd - 1))
        if bx - 1 < tx < bx + bw + 1 and by - 1 < ty < by + bd + 1:
            continue
        c.setLineWidth(0.3)
        c.circle(*fr.p(tx, ty), fr.d(float(rng.uniform(1.0, 2.0))))
    ada = (
        f"ADA {int(rng.integers(100, 999))} PARSEL {int(rng.integers(1, 40))}"
        if lang == "tr"
        else f"LOT {int(rng.integers(1, 99))}"
    )
    sh.text(*fr.p(pw * 0.05, pd * 0.9), ada, 8, role="parcel")
    angle = float(rng.uniform(-180, 180))
    north_arrow(sh, (sh.w_mm - 35) * MM, (sh.h_mm - 35) * MM, 9 * MM, angle, lang)
    title = "VAZİYET PLANI" if lang == "tr" else "SITE PLAN"
    sh.text(25 * MM, (sh.h_mm - 25) * MM, title, 14, font="bold", role="sheet_heading")
    title_block(
        sh, rng, lang, _common_fields(rng, lang, title, f"V-{int(rng.integers(1, 9))}", scale, None)
    )
    sh.gt["scale"] = scale
    return _finish(sh, buf, "site_plan")


DW_HEADERS = {
    "tr": ["POZ", "TİP", "GENİŞLİK (cm)", "YÜKSEKLİK (cm)", "PARAPET (cm)", "ADET", "MALZEME"],
    "en": ["TAG", "TYPE", "WIDTH (mm)", "HEIGHT (mm)", "SILL (mm)", "QTY", "MATERIAL"],
}
FINISH_HEADERS = {
    "tr": ["MAHAL NO", "MAHAL ADI", "ZEMİN", "DUVAR", "TAVAN", "SÜPÜRGELİK"],
    "en": ["ROOM NO", "ROOM NAME", "FLOOR", "WALLS", "CEILING", "SKIRTING"],
}
FLOORS_TR = ["Meşe parke", "Porselen seramik 60x60", "Laminat parke", "Doğal taş"]
WALLS_TR = ["Saten boya RAL 9010", "Seramik kaplama", "Duvar kağıdı", "Plastik boya"]
CEILS_TR = ["Alçıpan asma tavan", "Saten boya", "Nem dayanımlı alçıpan"]
FLOORS_EN = ["Oak parquet", "Porcelain tile 600x600", "Laminate", "Natural stone"]
WALLS_EN = ["Satin paint RAL 9010", "Ceramic tiles", "Wallpaper", "Emulsion paint"]
CEILS_EN = ["Gypsum board ceiling", "Satin paint", "Moisture-resistant gypsum"]


def schedule_rows(
    layout: Layout, kind: str, lang: str, rng: np.random.Generator
) -> tuple[list[str], list[list[str]]]:
    if kind == "door_window":
        header = DW_HEADERS[lang]
        rows = []
        for o in sorted(layout.openings, key=lambda o: (o.kind, int(o.tag[1:]))):
            mult = 100 if lang == "tr" else 1000
            typ = (
                ("Kanat kapı" if o.kind == "door" else "Pencere")
                if lang == "tr"
                else ("Hinged door" if o.kind == "door" else "Window")
            )
            mat = str(
                rng.choice(
                    ["Ahşap", "PVC", "Alüminyum"]
                    if lang == "tr"
                    else ["Timber", "PVC", "Aluminium"]
                )
            )
            rows.append(
                [
                    o.tag,
                    typ,
                    f"{round(o.width * mult)}",
                    f"{round(o.height * mult)}",
                    f"{round(o.sill * mult)}" if o.kind == "window" else "-",
                    "1",
                    mat,
                ]
            )
        return header, rows
    header = FINISH_HEADERS[lang]
    floors, walls, ceils = (
        (FLOORS_TR, WALLS_TR, CEILS_TR) if lang == "tr" else (FLOORS_EN, WALLS_EN, CEILS_EN)
    )
    rows = []
    for i, r in enumerate(layout.rooms):
        rows.append(
            [
                room_number(i),
                r.name,
                str(rng.choice(floors)),
                str(rng.choice(walls)),
                str(rng.choice(ceils)),
                str(
                    rng.choice(
                        ["Ahşap", "Seramik", "-"] if lang == "tr" else ["Timber", "Tile", "-"]
                    )
                ),
            ]
        )
    return header, rows


def schedule_page(
    rng: np.random.Generator,
    layout: Layout | None = None,
    *,
    lang: str = "tr",
    kind: str | None = None,
) -> Page:
    layout = layout or random_layout(rng, english=lang == "en")
    kind = kind or str(rng.choice(["door_window", "finish"]))
    header, rows = schedule_rows(layout, kind, lang, rng)
    sh, buf = _new("A3")
    sh.frame()
    c = sh.c
    title = {
        ("door_window", "tr"): "DOĞRAMA LİSTESİ",
        ("finish", "tr"): "MAHAL LİSTESİ",
        ("door_window", "en"): "DOOR AND WINDOW SCHEDULE",
        ("finish", "en"): "ROOM FINISH SCHEDULE",
    }[(kind, lang)]
    sh.text(25 * MM, (sh.h_mm - 28) * MM, title, 14, font="bold", role="sheet_heading")
    widths = [22, 34, 34, 36, 30, 16, 36] if kind == "door_window" else [24, 40, 52, 52, 52, 30]
    x0 = 25 * MM
    y = (sh.h_mm - 40) * MM
    rh = 8 * MM
    total_w = sum(widths) * MM
    c.setLineWidth(0.8)
    c.line(x0, y, x0 + total_w, y)
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w * MM)
    for j, htext in enumerate(header):
        sh.text(
            xs[j] + 1.5 * MM, y - rh + 2.6 * MM, htext, 6.5, font="bold", role="schedule_header"
        )
    y -= rh
    c.line(x0, y, x0 + total_w, y)
    cells = []
    for row in rows:
        for j, v in enumerate(row):
            sh.text(xs[j] + 1.5 * MM, y - rh + 2.6 * MM, v, 7, role="schedule_cell")
        cells.append(row)
        y -= rh
        c.setLineWidth(0.3)
        c.line(x0, y, x0 + total_w, y)
    c.setLineWidth(0.3)
    for xx in xs:
        c.line(xx, (sh.h_mm - 40) * MM, xx, y)
    title_block(
        sh, rng, lang, _common_fields(rng, lang, title, f"L-{int(rng.integers(1, 9))}", None, None)
    )
    sh.gt["schedule"] = {"kind": kind, "header": header, "rows": cells}
    return _finish(sh, buf, "schedule")


BRIEF_TR = [
    "Salonun doğal ışık alması ve {mat} zemin ile sıcak bir atmosfer oluşturması isteniyor.",
    "Mutfakta ada tezgâh, mat beyaz dolap kapakları ve {mat} tezgâh kullanılacaktır.",
    "Yatak odalarında gömme dolaplar ve yumuşak tonlarda duvar boyası tercih edilmektedir.",
    "Banyoda {mat} kaplama ve asma lavabo düşünülmektedir; zemin kaymaz seramik olmalıdır.",
    "Genel renk paleti toprak tonları, bej ve antrasit vurgular üzerine kuruludur.",
    "Aydınlatmada gizli LED şeritler ve 3000 K sıcak beyaz armatürler kullanılacaktır.",
    "Mevcut duvarlar korunacak, yalnızca mutfak ile salon arasındaki bölme kaldırılacaktır.",
]
BRIEF_EN = [
    "The living room should feel warm, with {mat} flooring and plenty of daylight.",
    "The kitchen gets an island, matte white fronts and a {mat} worktop.",
    "Bedrooms have built-in wardrobes and soft wall colours.",
    "The bathroom uses {mat} cladding and a wall-hung basin; the floor must be non-slip.",
    "The overall palette is earthy: beige, sand and anthracite accents.",
    "Lighting uses concealed LED strips and 3000 K warm-white fittings.",
]
MATS_TR = ["meşe", "traverten", "mermer", "ceviz", "beton görünümlü"]
MATS_EN = ["oak", "travertine", "marble", "walnut", "concrete-look"]


def text_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    sh, buf = _new("A4", landscape=False)
    heading = str(
        rng.choice(
            ["TASARIM ÖZETİ", "TEKNİK ŞARTNAME", "PROJE NOTLARI"]
            if lang == "tr"
            else ["DESIGN BRIEF", "SPECIFICATION", "PROJECT NOTES"]
        )
    )
    y = (sh.h_mm - 30) * MM
    sh.text(25 * MM, y, heading, 16, font="bold", role="heading")
    y -= 14 * MM
    sentences = list(BRIEF_TR if lang == "tr" else BRIEF_EN)
    mats = MATS_TR if lang == "tr" else MATS_EN
    for _ in range(int(rng.integers(3, 6))):
        rng.shuffle(sentences)
        para = " ".join(
            s.format(mat=rng.choice(mats)) for s in sentences[: int(rng.integers(2, 4))]
        )
        line, words = "", para.split(" ")
        for w in words:
            cand = (line + " " + w).strip()
            if pdfmetrics.stringWidth(cand, FONTS["sans"], 10) > (sh.w_mm - 50) * MM:
                sh.text(25 * MM, y, line, 10, role="body")
                y -= 5.2 * MM
                line = w
            else:
                line = cand
        if line:
            sh.text(25 * MM, y, line, 10, role="body")
            y -= 9 * MM
    sh.text(
        sh.w_mm / 2 * MM,
        15 * MM,
        "Sayfa 1/1" if lang == "tr" else "Page 1 of 1",
        8,
        anchor="center",
        role="footer",
    )
    return _finish(sh, buf, "text_document")


def _swatch_image(rng: np.random.Generator, size: int = 256) -> Image.Image:
    """Texture-like patch (wood grain / stone / fabric) from noise; stands in for material photos."""
    base = rng.uniform(40, 220, 3)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    kind = rng.integers(3)
    if kind == 0:  # wood grain
        grain = np.sin(xx / rng.uniform(3, 8) + 3 * np.sin(yy / rng.uniform(20, 60)))
    elif kind == 1:  # stone
        grain = rng.normal(0, 1, (size, size)).astype(np.float32)
        grain = (
            np.asarray(
                Image.fromarray(((grain + 3) * 40).clip(0, 255).astype(np.uint8))
                .resize((size // 8, size // 8))
                .resize((size, size), Image.Resampling.BICUBIC),
                np.float32,
            )
            / 40
            - 3
        )
    else:  # fabric
        grain = np.sin(xx / 2.0) * np.sin(yy / 2.0)
    img = np.clip(base[None, None, :] + grain[..., None] * rng.uniform(8, 30), 0, 255).astype(
        np.uint8
    )
    return Image.fromarray(img, "RGB")


def moodboard_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    sh, buf = _new("A3")
    c = sh.c
    title = str(
        rng.choice(
            ["KONSEPT PANOSU", "MOOD BOARD", "MALZEME PALETİ"]
            if lang == "tr"
            else ["MOOD BOARD", "MATERIAL PALETTE", "CONCEPT"]
        )
    )
    sh.text(20 * MM, (sh.h_mm - 22) * MM, title, 18, font="bold", role="heading")
    labels_tr = [
        "RAL 9010",
        "NCS S 1005-Y20R",
        "Meşe",
        "Traverten",
        "Antrasit",
        "Keten",
        "Pirinç",
        "Terrakota",
    ]
    labels_en = [
        "RAL 9010",
        "NCS S 1005-Y20R",
        "Oak",
        "Travertine",
        "Anthracite",
        "Linen",
        "Brass",
        "Terracotta",
    ]
    labels = labels_tr if lang == "tr" else labels_en
    cols, rows = int(rng.integers(3, 5)), 2
    cw, chh = (sh.w_mm - 40) / cols, (sh.h_mm - 60) / rows
    for r in range(rows):
        for q in range(cols):
            x = (20 + q * cw) * MM
            y = (sh.h_mm - 40 - (r + 1) * chh) * MM
            w, h = (cw - 6) * MM, (chh - 14) * MM
            if rng.random() < 0.5:
                col = rng.uniform(0.1, 0.95, 3)
                c.setFillColorRGB(*col)
                c.rect(x, y + 10 * MM, w, h, stroke=0, fill=1)
            else:
                c.drawImage(ImageReader(_swatch_image(rng)), x, y + 10 * MM, w, h)
            c.setFillColorRGB(0, 0, 0)
            sh.text(x, y + 4 * MM, str(rng.choice(labels)), 9, role="swatch_label")
    return _finish(sh, buf, "moodboard")


def cover_page(rng: np.random.Generator, *, lang: str = "tr") -> Page:
    """'other': a cover sheet with the drawing list (not a drawing, not a brief)."""
    sh, buf = _new("A3")
    sh.frame()
    c = sh.c
    firm = str(rng.choice(FIRMS))
    c.circle(60 * MM, (sh.h_mm - 60) * MM, 22 * MM, stroke=1, fill=0)
    sh.text(
        60 * MM,
        (sh.h_mm - 62) * MM,
        firm.split()[0][:3],
        20,
        font="bold",
        anchor="center",
        role="logo",
    )
    sh.text(
        sh.w_mm / 2 * MM,
        (sh.h_mm / 2 + 20) * MM,
        str(rng.choice(PROJECTS)).upper(),
        26,
        font="bold",
        anchor="center",
        role="heading",
    )
    sh.text(
        sh.w_mm / 2 * MM,
        (sh.h_mm / 2 + 5) * MM,
        "UYGULAMA PROJESİ" if lang == "tr" else "CONSTRUCTION DOCUMENTS",
        14,
        anchor="center",
        role="heading",
    )
    y = (sh.h_mm / 2 - 25) * MM
    sh.text(
        sh.w_mm / 2 * MM - 60 * MM,
        y,
        "PAFTA LİSTESİ" if lang == "tr" else "DRAWING LIST",
        10,
        font="bold",
        role="list",
    )
    for i in range(int(rng.integers(4, 9))):
        y -= 6 * MM
        sh.text(
            sh.w_mm / 2 * MM - 60 * MM,
            y,
            f"M-{101 + i}  "
            + str(
                rng.choice(["Zemin Kat Planı", "Kesitler", "Görünüşler", "Detaylar", "Tavan Planı"])
            ),
            9,
            role="list",
        )
    return _finish(sh, buf, "other")


GENERATORS = {
    "floor_plan": lambda rng, lang: floor_plan_page(rng, lang=lang),
    "ceiling_plan": lambda rng, lang: floor_plan_page(rng, lang=lang, ceiling=True),
    "section": lambda rng, lang: section_page(rng, lang=lang),
    "elevation": lambda rng, lang: elevation_page(rng, lang=lang),
    "detail": lambda rng, lang: detail_page(rng, lang=lang),
    "site_plan": lambda rng, lang: site_plan_page(rng, lang=lang),
    "schedule": lambda rng, lang: schedule_page(rng, lang=lang),
    "text_document": lambda rng, lang: text_page(rng, lang=lang),
    "moodboard": lambda rng, lang: moodboard_page(rng, lang=lang),
    "other": lambda rng, lang: cover_page(rng, lang=lang),
}
