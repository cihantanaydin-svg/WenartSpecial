"""Defects injected into correct plans, for the validator detection rate (Phase-3 acceptance:
every injected defect is detected). Used by the tests and by ``make eval``.

- ``deleted_wall``: the longest straight interior wall removed → the rooms along it are open;
- ``opening_outside_host``: an opening pushed past the end of its wall;
- ``overlapping_rooms``: a room polygon grown 0.4 m into its neighbours;
- ``area_label_off``: every room's area label 5 % off;
- ``unreachable_room``: an inner room's doors removed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import LineString, Point, Polygon

from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import PlanGraph, Segment
from archrender.core.schemas.provenance import fact
from archrender.plan.validate import validate_plan


@dataclass
class Injected:
    defect: str
    plan: PlanGraph
    code: str  # the issue code that must appear
    element_ids: set[str]  # … naming at least these elements


def found(plan: PlanGraph) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for i in validate_plan(plan):
        out.setdefault(i.code, set()).update(i.element_ids)
    return out


def detected(inj: Injected) -> bool:
    return bool(inj.element_ids) and inj.element_ids <= found(inj.plan).get(inj.code, set())


def room_poly(plan: PlanGraph, room_id: str) -> Polygon:
    r = next(r for r in plan.rooms if r.id == room_id)
    return Polygon([(p.x, p.y) for p in r.polygon])


def opening_point(plan: PlanGraph, opening_id: str) -> Point:
    o = next(o for o in plan.openings if o.id == opening_id)
    w = plan.wall(o.host_wall)
    assert isinstance(w.centerline, Segment)
    a, b = w.centerline.a, w.centerline.b
    f = o.offset_m.value / w.centerline.length()
    return Point(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f)


def rooms_by_door(plan: PlanGraph) -> dict[str, list[str]]:
    """Room → its doors and openings (not windows), by proximity of the opening's centre."""
    out: dict[str, list[str]] = {r.id: [] for r in plan.rooms}
    for o in plan.openings:
        if o.type == "window" or not isinstance(plan.wall(o.host_wall).centerline, Segment):
            continue
        p = opening_point(plan, o.id)
        t = plan.wall(o.host_wall).thickness_m.value
        for r in plan.rooms:
            if room_poly(plan, r.id).distance(p) <= t:
                out[r.id].append(o.id)
    return out


def deleted_wall(plan: PlanGraph) -> Injected:
    plan = plan.model_copy(deep=True)
    interior = [w for w in plan.walls if w.kind == "interior" and isinstance(w.centerline, Segment)]
    w = max(interior, key=lambda x: x.centerline.length())
    plan.walls = [x for x in plan.walls if x.id != w.id]
    plan.openings = [o for o in plan.openings if o.host_wall != w.id]
    cl = LineString([(w.centerline.a.x, w.centerline.a.y), (w.centerline.b.x, w.centerline.b.y)])  # type: ignore[union-attr]
    band = cl.buffer(w.thickness_m.value / 2 + 0.02, cap_style="flat")
    # rooms with a stretch of boundary along the deleted wall (not just touching its end)
    beside = {
        r.id for r in plan.rooms if room_poly(plan, r.id).exterior.intersection(band).length > 0.2
    }
    return Injected("deleted_wall", plan, "ROOM_OPEN", beside)


def opening_outside_host(plan: PlanGraph) -> Injected:
    plan = plan.model_copy(deep=True)
    o = plan.openings[0]
    length = plan.wall(o.host_wall).centerline.length()
    o.offset_m = fact(length - o.width_m.value / 4, "user", 1.0)
    return Injected("opening_outside_host", plan, "OPENING_OUTSIDE_HOST", {o.id})


def overlapping_rooms(plan: PlanGraph) -> Injected:
    plan = plan.model_copy(deep=True)
    r = plan.rooms[0]
    grown = room_poly(plan, r.id).buffer(0.4, join_style="mitre")
    r.polygon = [Point2(x=x, y=y) for x, y in list(grown.exterior.coords)[:-1]]
    return Injected("overlapping_rooms", plan, "ROOMS_OVERLAP", {r.id})


def area_label_off(plan: PlanGraph, factor: float = 0.05) -> Injected:
    plan = plan.model_copy(deep=True)
    for k, r in enumerate(plan.rooms):
        assert r.area_label_m2 is not None
        f = 1 + factor if k % 2 == 0 else 1 - factor
        r.area_label_m2 = fact(r.area_label_m2.value * f, "ocr", 0.9)
    return Injected("area_label_off", plan, "ROOM_AREA_MISMATCH", {r.id for r in plan.rooms})


def unreachable_room(plan: PlanGraph) -> Injected:
    plan = plan.model_copy(deep=True)
    doors = rooms_by_door(plan)

    def exterior(oid: str) -> bool:
        o = next(o for o in plan.openings if o.id == oid)
        return plan.wall(o.host_wall).kind == "exterior"

    entry_rooms = {rid for rid, ds in doors.items() if any(exterior(d) for d in ds)}
    target = next(rid for rid in doors if rid not in entry_rooms and doors[rid])
    plan.openings = [o for o in plan.openings if o.id not in set(doors[target])]
    return Injected("unreachable_room", plan, "ROOM_UNREACHABLE", {target})


DEFECTS: list[Callable[[PlanGraph], Injected]] = [
    deleted_wall,
    opening_outside_host,
    overlapping_rooms,
    area_label_off,
    unreachable_room,
]
