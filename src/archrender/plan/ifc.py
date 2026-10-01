"""IFC → PlanGraph (authoritative geometry; ARCHITECTURE §S2 step 1).

Reads the JSON written by the sandboxed S0 ``ifc`` task (IfcOpenShell): storeys become levels;
walls keep their axis (centerline; a polyline axis that fits a circle becomes an arc) and the
thickness of their material layer set (or of their body); openings are hosted on the wall they
void, with the position of their body centre projected on the host's centerline, the filling's
overall width/height, the sill from their elevation, and the swing from the filling's placement
(local +y = the side the leaf opens to); spaces give the rooms (footprint, number, name, net floor
area, height).
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import (
    Arc,
    Level,
    Opening,
    OpeningType,
    PlanGraph,
    Room,
    Segment,
    Wall,
)
from archrender.core.schemas.provenance import Assumption, Fact, fact
from archrender.plan.builder import DEFAULTS, room_type
from archrender.plan.prims import circle_fit


def _p(x: float, y: float) -> Point2:
    return Point2(x=round(x, 6), y=round(y, 6))


def _centerline(axis: list[list[float]]) -> list[Segment | Arc]:
    pts = np.array(axis, np.float64)
    if len(pts) < 2:
        return []
    if len(pts) == 2:
        return [Segment(a=_p(*pts[0]), b=_p(*pts[1]))]
    cx, cy, r, rms = circle_fit(pts)
    if rms < 0.005 and r < 500:
        ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
        total = float(((np.diff(ang) + 180) % 360 - 180).sum())
        a0 = float(ang[0]) % 360
        if total > 0:
            return [
                Arc(center=_p(cx, cy), radius=round(r, 6), start_deg=a0, end_deg=(a0 + total) % 360)
            ]
        return [
            Arc(center=_p(cx, cy), radius=round(r, 6), start_deg=(a0 + total) % 360, end_deg=a0)
        ]
    return [Segment(a=_p(*pts[i]), b=_p(*pts[i + 1])) for i in range(len(pts) - 1)]


def _line(c: Segment | Arc) -> LineString:
    if isinstance(c, Segment):
        return LineString([(c.a.x, c.a.y), (c.b.x, c.b.y)])
    sweep = (c.end_deg - c.start_deg) % 360 or 360
    n = max(8, int(sweep / 2))
    return LineString(
        [
            (
                c.center.x + c.radius * math.cos(math.radians(c.start_deg + sweep * k / n)),
                c.center.y + c.radius * math.sin(math.radians(c.start_deg + sweep * k / n)),
            )
            for k in range(n + 1)
        ]
    )


def plan_from_ifc(
    model: dict[str, Any], *, project: str, version: str, source_doc: str | None = None
) -> PlanGraph:
    def f(value: Any, conf: float = 0.98, note: str | None = None) -> Fact[Any]:
        return fact(value, "ifc", conf, source_doc=source_doc, note=note)

    def default(key: str) -> Fact[float]:
        return fact(DEFAULTS[key], "default", 0.5, note=f"default/{key}")

    storeys = sorted(model.get("storeys", []), key=lambda s: s["elevation_m"]) or [
        {"id": None, "name": "Zemin Kat", "elevation_m": 0.0}
    ]
    level_of = {s["id"]: f"L{i}" for i, s in enumerate(storeys)}
    elev = {f"L{i}": s["elevation_m"] for i, s in enumerate(storeys)}
    levels = [
        Level(
            id=f"L{i}",
            name=s["name"] or f"Kat {i}",
            elevation_m=s["elevation_m"],
            floor_to_floor_m=f(round(storeys[i + 1]["elevation_m"] - s["elevation_m"], 4))
            if i + 1 < len(storeys)
            else None,
        )
        for i, s in enumerate(storeys)
    ]
    walls: list[Wall] = []
    wall_ids: dict[str, list[str]] = {}
    for w in model.get("walls", []):
        level = level_of.get(w.get("storey"), "L0")
        axis = w.get("axis") or []
        if not axis and w.get("footprint"):
            axis = _axis_from_footprint(w["footprint"])
        t = w.get("thickness_m")
        if not t and w.get("footprint"):
            t = _thickness_from_footprint(w["footprint"])
        if not axis or not t:
            continue
        for k, cl in enumerate(_centerline(axis)):
            wid = f"W{len(walls) + 1}"
            wall_ids.setdefault(w["id"], []).append(wid)
            walls.append(
                Wall(
                    id=wid,
                    level=level,
                    centerline=cl,
                    thickness_m=f(round(float(t), 4)),
                    height_m=f(round(float(w["height_m"]), 4))
                    if w.get("height_m")
                    else default("wall_height_m"),
                    kind="exterior" if w.get("external") else "unknown",
                )
            )
            del k
    by_id = {w.id: w for w in walls}
    openings: list[Opening] = []
    for o in model.get("openings", []):
        hosts = [by_id[i] for i in wall_ids.get(o["host"], [])]
        if not hosts:
            continue
        c = Point(*o["center"])
        host = min(hosts, key=lambda w: _line(w.centerline).distance(c))
        line = _line(host.centerline)
        offset = float(line.project(c))
        kind: OpeningType = (
            "door" if o["kind"] == "door" else "window" if o["kind"] == "window" else "opening"
        )
        width = o.get("width_m") or 0.9
        height = o.get("height_m")
        sill = max(0.0, (o.get("z_m") or 0.0) - elev.get(host.level, 0.0))
        hinge: Literal["start", "end"] | None = None
        side: Literal["pos", "neg"] | None = None
        if kind == "door":
            d = line.interpolate(min(line.length, offset + 0.01))
            q = line.interpolate(max(0.0, offset - 0.01))
            u = np.array([d.x - q.x, d.y - q.y])
            u /= max(float(np.hypot(*u)), 1e-12)
            n = np.array([-u[1], u[0]])
            side = "pos" if float(np.dot(n, o["y_dir"])) > 0 else "neg"
            along = float(np.dot(u, o["x_dir"])) > 0
            left = "LEFT" in str(o.get("operation") or "")
            hinge = "start" if left == along else "end"
        openings.append(
            Opening(
                id=f"O{len(openings) + 1}",
                host_wall=host.id,
                offset_m=f(round(offset, 4)),
                width_m=f(round(float(width), 4)),
                height_m=f(round(float(height), 4))
                if height
                else default("door_height_m" if kind != "window" else "window_height_m"),
                sill_m=f(round(sill, 4)),
                type=kind,
                hinge=hinge,
                swing_side=side,
                tag=o.get("tag"),
            )
        )
    rooms: list[Room] = []
    for s in model.get("spaces", []):
        ring = s.get("footprint") or []
        if len(ring) > 1 and ring[0] == ring[-1]:
            ring = ring[:-1]
        if len(ring) < 3:
            continue
        poly = Polygon(ring)
        if not poly.is_valid or poly.area <= 0:
            continue
        if not poly.exterior.is_ccw:
            ring = ring[::-1]
        name = s.get("name") or s.get("number") or f"Mahal {len(rooms) + 1}"
        rooms.append(
            Room(
                id=f"R{len(rooms) + 1}",
                level=level_of.get(s.get("storey"), "L0"),
                polygon=[_p(*q) for q in ring],
                name=f(name),
                number=f(s["number"]) if s.get("number") and s.get("number") != name else None,
                type=fact(room_type(name), "derived", 0.9),
                area_label_m2=f(round(float(s["net_area_m2"]), 3))
                if s.get("net_area_m2")
                else None,
                ceiling_height_m=f(round(float(s["height_m"]), 4))
                if s.get("height_m")
                else default("ceiling_height_m"),
            )
        )
    assumptions = [
        Assumption(
            key=f"default/{k}",
            value=v,
            reason="not in the IFC model; architectural default",
            stage="S2",
        )
        for k, v in DEFAULTS.items()
    ]
    return PlanGraph(
        version=version,
        project=project,
        levels=levels,
        north_angle_deg=fact(
            0.0, "default", 0.3, note="IfcGeometricRepresentationContext.TrueNorth not read yet"
        ),
        walls=walls,
        openings=openings,
        rooms=rooms,
        assumptions=assumptions,
        source="ifc",
    )


def _axis_from_footprint(fp: list[list[float]]) -> list[list[float]]:
    poly = Polygon(fp).minimum_rotated_rectangle
    pts = list(poly.exterior.coords)[:4]
    e = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    lens = [math.dist(a, b) for a, b in e]
    i = int(np.argmax(lens))
    a, b = e[i]
    c, d = e[(i + 2) % 4]
    return [[(a[0] + d[0]) / 2, (a[1] + d[1]) / 2], [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2]]


def _thickness_from_footprint(fp: list[list[float]]) -> float:
    poly = Polygon(fp).minimum_rotated_rectangle
    pts = list(poly.exterior.coords)[:4]
    return min(math.dist(pts[0], pts[1]), math.dist(pts[1], pts[2]))
