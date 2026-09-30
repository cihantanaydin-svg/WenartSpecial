"""Synthetic layouts as DXF (Turkish CAD conventions): layers DUVAR (walls), KAPI (doors),
PENCERE (windows), YAZI (room names and tags); drawing units m, cm or mm via ``$INSUNITS``."""

from __future__ import annotations

import io
import math

from ezdxf.filemanagement import new as new_dxf

from archrender.synth.layout import Layout, opening_point
from archrender.synth.sheets import _wall_pieces, _wall_rect, room_number

UNITS = {"m": (6, 1.0), "cm": (5, 100.0), "mm": (4, 1000.0)}


def layout_dxf(layout: Layout, *, unit: str = "cm", room_numbers: bool = True) -> bytes:
    code, k = UNITS[unit]
    doc = new_dxf("R2010", units=code)
    doc.header["$MEASUREMENT"] = 1
    for name, color in (("DUVAR", 7), ("KAPI", 3), ("PENCERE", 5), ("YAZI", 2)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()

    def p(x: float, y: float) -> tuple[float, float]:
        return (x * k, y * k)

    for wi in range(len(layout.walls)):
        for a, b in _wall_pieces(layout, wi):
            msp.add_lwpolyline(
                [p(*q) for q in _wall_rect(layout, wi, a, b)],
                close=True,
                dxfattribs={"layer": "DUVAR"},
            )
    for o in layout.openings:
        w = layout.walls[o.wall]
        ux, uy = (w.bx - w.ax) / w.length, (w.by - w.ay) / w.length
        cx, cy = opening_point(layout, o)
        if o.kind == "window":
            for off in (-w.thickness / 2, 0.0, w.thickness / 2):
                start = p(cx - ux * o.width / 2 - uy * off, cy - uy * o.width / 2 + ux * off)
                end = p(cx + ux * o.width / 2 - uy * off, cy + uy * o.width / 2 + ux * off)
                msp.add_line(start, end, dxfattribs={"layer": "PENCERE"})
        else:
            nx, ny = -uy * o.swing, ux * o.swing
            hinge = (cx - ux * o.width / 2, cy - uy * o.width / 2)
            leaf = (hinge[0] + nx * o.width, hinge[1] + ny * o.width)
            msp.add_line(p(*hinge), p(*leaf), dxfattribs={"layer": "KAPI"})
            a0 = math.degrees(math.atan2(uy, ux))
            a1 = math.degrees(math.atan2(ny, nx))
            arc_from, arc_to = (a0, a1) if (a1 - a0) % 360 <= 180 else (a1, a0)
            msp.add_arc(p(*hinge), o.width * k, arc_from, arc_to, dxfattribs={"layer": "KAPI"})
        msp.add_text(o.tag, height=0.15 * k, dxfattribs={"layer": "YAZI"}).set_placement(
            p(cx - 0.3 * uy, cy + 0.3 * ux)
        )
    for i, r in enumerate(layout.rooms):
        cx, cy = r.center
        msp.add_text(r.name, height=0.25 * k, dxfattribs={"layer": "YAZI"}).set_placement(
            p(cx - 0.8, cy)
        )
        if room_numbers:
            msp.add_text(
                room_number(i), height=0.18 * k, dxfattribs={"layer": "YAZI"}
            ).set_placement(p(cx - 0.3, cy + 0.4))
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")
