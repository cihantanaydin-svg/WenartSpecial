"""Mock plan extractor (cpu_test profile and the Phase-1 walking skeleton).

Produces a single "Salon" room with one door and two windows. Every fact is marked
``method="mock"`` so the plan can never be mistaken for an extracted one. The real IFC / DXF /
vector-PDF / raster extractors replace this in Phase 3.
"""

from __future__ import annotations

from archrender.core.hashing import sha256_json
from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import Level, Opening, PlanGraph, Room, RoomType, Segment, Wall
from archrender.core.schemas.provenance import Fact, fact

MOCK_CONF = 1.0


def _f(value: float) -> Fact[float]:
    return fact(value, "mock", MOCK_CONF, note="mock plan")


def _s(value: str) -> Fact[str]:
    return fact(value, "mock", MOCK_CONF, note="mock plan")


def _room_type(value: RoomType) -> Fact[RoomType]:
    return fact(value, "mock", MOCK_CONF, note="mock plan")


def mock_plan(project_id: str, doc_hashes: list[str], *, width: float = 5.0, depth: float = 4.0) -> PlanGraph:
    """Rectangular room, interior ``width × depth`` m, 0.2 m walls, 2.7 m ceiling."""
    t = 0.2
    h = 2.7
    # centerlines of the four walls (interior faces at 0..width, 0..depth)
    x0, y0, x1, y1 = -t / 2, -t / 2, width + t / 2, depth + t / 2
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        walls.append(
            Wall(
                id=f"W{i + 1}",
                level="L0",
                centerline=Segment(a=Point2(x=a[0], y=a[1]), b=Point2(x=b[0], y=b[1])),
                thickness_m=_f(t),
                height_m=_f(h),
                kind="exterior",
            )
        )
    south_len = x1 - x0
    openings = [
        Opening(
            id="D1",
            host_wall="W1",
            offset_m=_f(south_len * 0.25),
            width_m=_f(0.9),
            height_m=_f(2.1),
            sill_m=_f(0.0),
            type="door",
            hinge="start",
            swing="left",
            swing_side="pos",
            tag="K-01",
        ),
        Opening(
            id="P1",
            host_wall="W3",
            offset_m=_f(south_len * 0.5),
            width_m=_f(1.6),
            height_m=_f(1.4),
            sill_m=_f(0.9),
            type="window",
            tag="P-01",
        ),
        Opening(
            id="P2",
            host_wall="W2",
            offset_m=_f((y1 - y0) * 0.5),
            width_m=_f(1.2),
            height_m=_f(1.4),
            sill_m=_f(0.9),
            type="window",
            tag="P-02",
        ),
    ]
    room = Room(
        id="R1",
        level="L0",
        polygon=[
            Point2(x=0.0, y=0.0),
            Point2(x=width, y=0.0),
            Point2(x=width, y=depth),
            Point2(x=0.0, y=depth),
        ],
        name=_s("Salon"),
        number=_s("01"),
        type=_room_type("living"),
        area_label_m2=_f(round(width * depth, 2)),
        ceiling_height_m=_f(h),
    )
    version = "plan_" + sha256_json({"docs": sorted(doc_hashes), "w": width, "d": depth})[:16]
    return PlanGraph(
        version=version,
        project=project_id,
        levels=[Level(id="L0", name="Zemin Kat", elevation_m=0.0)],
        north_angle_deg=_f(0.0),
        walls=walls,
        openings=openings,
        rooms=[room],
        source="mock",
    )
