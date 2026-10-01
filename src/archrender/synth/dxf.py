"""Synthetic plans as DXF, with the plan→drawing transform as ground truth.

Variants seen in practice:
- layer names: Turkish (DUVAR, KAPI, PENCERE, YAZI, OLCU, TARAMA), AIA/NCS (A-WALL, A-DOOR,
  A-GLAZ, A-ANNO-TEXT, A-ANNO-DIMS, A-WALL-PATT), or everything on layer "0" (no layer semantics:
  only geometric evidence is left);
- units: metres, centimetres or millimetres via ``$INSUNITS``, or unitless (``$INSUNITS = 0``,
  the unit has to be inferred);
- origin: at 0, slightly offset, or far away (site coordinates);
- walls as cleaned closed outlines, optionally with an ANSI31 or solid HATCH; doors as jamb lines,
  a leaf LINE and an ARC; windows as three LINEs; room names, numbers and net areas as TEXT; tag
  bubbles as CIRCLE + TEXT; aligned DIMENSION entities on the exterior chains.
"""

from __future__ import annotations

import io
import itertools
import math
from typing import Any, Literal

import numpy as np
from ezdxf.enums import TextEntityAlignment
from ezdxf.filemanagement import new as new_dxf

from archrender.synth.layout import Layout
from archrender.synth.plan import PlanSpec, from_layout, wall_solid
from archrender.synth.sheets import _rings, fmt_m, room_number

UNITS = {"m": (6, 1.0), "cm": (5, 100.0), "mm": (4, 1000.0)}
LayerScheme = Literal["tr", "aia", "zero"]
LAYERS: dict[str, dict[str, str]] = {
    "tr": {
        "wall": "DUVAR",
        "door": "KAPI",
        "window": "PENCERE",
        "text": "YAZI",
        "dims": "OLCU",
        "hatch": "TARAMA",
    },
    "aia": {
        "wall": "A-WALL",
        "door": "A-DOOR",
        "window": "A-GLAZ",
        "text": "A-ANNO-TEXT",
        "dims": "A-ANNO-DIMS",
        "hatch": "A-WALL-PATT",
    },
    "zero": dict.fromkeys(("wall", "door", "window", "text", "dims", "hatch"), "0"),
}


def plan_dxf(
    spec: PlanSpec,
    *,
    unit: str = "cm",
    layers: LayerScheme = "tr",
    unitless: bool = False,
    origin: tuple[float, float] = (0.0, 0.0),
    hatch: Literal["none", "ansi31", "solid"] = "none",
    dims: bool = True,
    room_numbers: bool = True,
    lang: str = "tr",
) -> tuple[bytes, dict[str, Any]]:
    """DXF of ``spec`` drawn at ``unit`` per metre with the plan origin at ``origin`` (metres).
    Returns (bytes, ground truth with ``plan_to_doc`` = [[k, 0, tx], [0, k, ty]])."""
    code, k = UNITS[unit]
    doc = new_dxf("R2010", units=0 if unitless else code)
    doc.header["$MEASUREMENT"] = 1
    names = LAYERS[layers]
    colors = {"wall": 7, "door": 3, "window": 5, "text": 2, "dims": 1, "hatch": 8}
    for role, name in names.items():
        if name != "0" and name not in doc.layers:
            doc.layers.add(name, color=colors[role])
    msp = doc.modelspace()
    ox, oy = origin

    def p(x: float, y: float) -> tuple[float, float]:
        return ((x + ox) * k, (y + oy) * k)

    rings = _rings(wall_solid(spec, cut_openings=True))
    for ring in rings:
        msp.add_lwpolyline([p(*q) for q in ring], close=True, dxfattribs={"layer": names["wall"]})
    if hatch != "none":
        h = msp.add_hatch(
            color=8 if hatch == "ansi31" else 250, dxfattribs={"layer": names["hatch"]}
        )
        if hatch == "ansi31":
            h.set_pattern_fill("ANSI31", scale=0.02 * k)
        for ring in rings:
            h.paths.add_polyline_path([p(*q) for q in ring], is_closed=True)

    for o in spec.openings:
        w = spec.walls[o.wall]
        ux, uy = w.direction(o.t)
        cx, cy = w.point(o.t)
        t = w.thickness / 2
        if o.kind == "window":
            for off in (-t, 0.0, t):
                a = (cx - ux * o.width / 2 - uy * off, cy - uy * o.width / 2 + ux * off)
                b = (cx + ux * o.width / 2 - uy * off, cy + uy * o.width / 2 + ux * off)
                msp.add_line(p(*a), p(*b), dxfattribs={"layer": names["window"]})
            continue
        nx, ny = -uy * o.swing, ux * o.swing
        s = -1.0 if o.hinge == "start" else 1.0
        hinge = (cx + s * ux * o.width / 2, cy + s * uy * o.width / 2)
        for j in (-1, 1):
            q = (cx + j * ux * o.width / 2, cy + j * uy * o.width / 2)
            msp.add_line(
                p(q[0] - nx * t, q[1] - ny * t),
                p(q[0] + nx * t, q[1] + ny * t),
                dxfattribs={"layer": names["door"]},
            )
        leaf = (hinge[0] + nx * o.width, hinge[1] + ny * o.width)
        msp.add_line(p(*hinge), p(*leaf), dxfattribs={"layer": names["door"]})
        a_closed = math.degrees(math.atan2(-s * uy, -s * ux))
        a_open = math.degrees(math.atan2(ny, nx))
        start, end = (a_closed, a_open) if (a_open - a_closed) % 360 <= 180 else (a_open, a_closed)
        msp.add_arc(p(*hinge), o.width * k, start, end, dxfattribs={"layer": names["door"]})
        # tag bubble
        tx, ty = cx - 0.55 * -uy, cy - 0.55 * ux
        msp.add_circle(p(tx, ty), 0.16 * k, dxfattribs={"layer": names["text"]})
        msp.add_text(o.tag, height=0.12 * k, dxfattribs={"layer": names["text"]}).set_placement(
            p(tx, ty), align=TextEntityAlignment.MIDDLE_CENTER
        )
    for o in spec.openings:
        if o.kind != "window":
            continue
        w = spec.walls[o.wall]
        ux, uy = w.direction(o.t)
        cx, cy = w.point(o.t)
        tx, ty = cx + uy * 0.75, cy - ux * 0.75  # right normal = outside of the CCW facade
        msp.add_ellipse(
            p(tx, ty),
            major_axis=(0.2 * k, 0.0),
            ratio=0.64,
            dxfattribs={"layer": names["text"]},
        )
        msp.add_text(o.tag, height=0.12 * k, dxfattribs={"layer": names["text"]}).set_placement(
            p(tx, ty), align=TextEntityAlignment.MIDDLE_CENTER
        )

    style = "comma" if lang == "tr" else "dot"
    for i, r in enumerate(spec.rooms):
        x, y = r.label
        msp.add_text(r.name, height=0.25 * k, dxfattribs={"layer": names["text"]}).set_placement(
            p(x, y + 0.05), align=TextEntityAlignment.BOTTOM_CENTER
        )
        area = spec.faces[r.id].area
        msp.add_text(
            f"{fmt_m(area, style)} m²", height=0.16 * k, dxfattribs={"layer": names["text"]}
        ).set_placement(p(x, y - 0.3), align=TextEntityAlignment.BOTTOM_CENTER)
        if room_numbers:
            msp.add_text(
                room_number(i), height=0.18 * k, dxfattribs={"layer": names["text"]}
            ).set_placement(p(x, y + 0.45), align=TextEntityAlignment.BOTTOM_CENTER)

    dim_values: list[dict[str, Any]] = []
    if dims and spec.layout is not None:
        layout = spec.layout
        override = {
            "dimtxt": 0.15 * k,
            "dimasz": 0.1 * k,
            "dimexe": 0.08 * k,
            "dimexo": 0.05 * k,
            "dimdec": 0 if unit in ("cm", "mm") else 2,
            "dimtsz": 0.05 * k,  # oblique ticks, as on architectural drawings
        }
        # chains on straight facades: along e1 at the bottom (or top), along e2 at the left (or right)
        arc_bottom, arc_left = spec.walls[0].is_arc, spec.walls[3].is_arc
        y_line = layout.depth if arc_bottom else 0.0
        x_line = layout.width if arc_left else 0.0
        xs = sorted(
            {0.0, layout.width}
            | {r.x0 for r in layout.rooms if abs((r.y1 if arc_bottom else r.y0) - y_line) < 1e-6}
            | {r.x1 for r in layout.rooms if abs((r.y1 if arc_bottom else r.y0) - y_line) < 1e-6}
        )
        ys = sorted(
            {0.0, layout.depth}
            | {r.y0 for r in layout.rooms if abs((r.x1 if arc_left else r.x0) - x_line) < 1e-6}
            | {r.y1 for r in layout.rooms if abs((r.x1 if arc_left else r.x0) - x_line) < 1e-6}
        )
        chains = [
            ([spec.lattice(x, y_line) for x in xs], xs, 1.0 if arc_bottom else -1.0),
            ([spec.lattice(x_line, y) for y in ys], ys, -1.0 if arc_left else 1.0),
        ]
        for pts, coords, side in chains:
            for (pa, ca), (pb, cb) in itertools.pairwise(zip(pts, coords, strict=True)):
                # aligned dimension: positive distance offsets to the left of pa→pb
                dim = msp.add_aligned_dim(
                    p(*pa),
                    p(*pb),
                    distance=side * 1.6 * k,
                    override=override,
                    dxfattribs={"layer": names["dims"]},
                )
                dim.render()
                dim_values.append({"value_m": round(cb - ca, 3), "a": pa, "b": pb})
    buf = io.StringIO()
    doc.write(buf)
    gt = {
        "plan_to_doc": [[k, 0.0, ox * k], [0.0, k, oy * k]],
        "unit": unit,
        "unitless": unitless,
        "layers": layers,
        "hatch": hatch,
        "dimensions": dim_values,
    }
    return buf.getvalue().encode("utf-8"), gt


def random_plan_dxf(
    spec: PlanSpec, rng: np.random.Generator, *, lang: str = "tr"
) -> tuple[bytes, dict[str, Any]]:
    unit = str(rng.choice(["cm", "cm", "mm", "m"]))
    origin_kind = rng.random()
    origin = (
        (0.0, 0.0)
        if origin_kind < 0.5
        else (float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20)))
        if origin_kind < 0.8
        else (float(rng.uniform(4.0e5, 5.0e5)), float(rng.uniform(4.4e6, 4.6e6)))
    )
    return plan_dxf(
        spec,
        unit=unit,
        layers=str(rng.choice(["tr", "tr", "aia", "zero"])),  # type: ignore[arg-type]
        unitless=bool(rng.random() < 0.15),
        origin=origin,
        hatch=str(rng.choice(["none", "none", "ansi31", "solid"])),  # type: ignore[arg-type]
        dims=bool(rng.random() < 0.8),
        lang=lang,
    )


def layout_dxf(layout: Layout, *, unit: str = "cm", room_numbers: bool = True) -> bytes:
    """Orthogonal layout as DXF with Turkish layers (golden project G1)."""
    data, _ = plan_dxf(from_layout(layout), unit=unit, room_numbers=room_numbers)
    return data
