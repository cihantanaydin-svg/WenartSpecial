"""Rooms re-derived from the walls after a Gate A change to the wall network.

Rooms are the holes of the wall solids (as the PlanBuilder derives them). Each new room keeps the
identity, name, number, labels and heights of the old room it overlaps most; a room split in two
keeps them in the larger part, the other part gets a default name (an assumption the user sees).
"""

from __future__ import annotations

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import PlanGraph, Room
from archrender.core.schemas.provenance import fact
from archrender.scene.geometry import centerline_coords

MIN_ROOM_M2 = 0.5


def _poly(r: Room) -> Polygon:
    return Polygon([(p.x, p.y) for p in r.polygon], [[(p.x, p.y) for p in h] for h in r.holes])


def rederive_rooms(plan: PlanGraph) -> PlanGraph:
    out = plan.model_copy(deep=True)
    rooms: list[Room] = []
    used: set[str] = set()
    next_n = 1 + max((int(r.id[1:]) for r in plan.rooms if r.id[1:].isdigit()), default=0)
    for level in plan.levels:
        walls = [w for w in plan.walls if w.level == level.id]
        solids = [
            LineString(centerline_coords(w)).buffer(
                w.thickness_m.value / 2, cap_style="flat", join_style="mitre"
            )
            for w in walls
        ]
        body = unary_union(solids) if solids else Polygon()
        parts = list(body.geoms) if hasattr(body, "geoms") else [body]
        holes = [Polygon(h) for p in parts if not p.is_empty for h in p.interiors]
        holes = sorted((h for h in holes if h.area >= MIN_ROOM_M2), key=lambda h: -h.area)
        old = [(r, _poly(r)) for r in plan.rooms if r.level == level.id]
        for h in holes:
            ring = list(h.exterior.coords)[:-1]
            if not Polygon(ring).exterior.is_ccw:
                ring.reverse()
            polygon = [Point2(x=round(x, 6), y=round(y, 6)) for x, y in ring]
            match = max(
                ((r, h.intersection(p).area) for r, p in old if r.id not in used),
                key=lambda rp: rp[1],
                default=None,
            )
            if match is not None and match[1] >= 0.5 * min(h.area, _poly(match[0]).area):
                r = match[0]
                used.add(r.id)
                rooms.append(r.model_copy(update={"polygon": polygon, "holes": []}))
                continue
            rid = f"R{next_n}"
            next_n += 1
            rooms.append(
                Room(
                    id=rid,
                    level=level.id,
                    polygon=polygon,
                    name=fact(
                        f"Mahal {rid[1:]}", "default", 0.3, note="new room after a wall edit"
                    ),
                    type=fact("other", "default", 0.3),
                    ceiling_height_m=fact(2.70, "default", 0.5, note="default/ceiling_height_m"),
                )
            )
    out.rooms = rooms
    return PlanGraph.model_validate(out.model_dump())
