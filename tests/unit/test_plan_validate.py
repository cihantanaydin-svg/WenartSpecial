"""Plan validators: every injected defect is detected (Phase-3 acceptance: 100 %), and the clean
synthetic plans raise nothing.

Defects: an open room (a wall deleted), an opening outside its host wall, overlapping rooms, an
area label more than 3 % off, an unreachable room (its doors removed).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import PlanGraph, Segment
from archrender.core.schemas.provenance import fact
from archrender.plan.validate import validate_plan
from archrender.synth.plan import Variant, gt_plan, random_spec

VARIANTS: list[Variant] = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]
CASES = list(itertools.product(range(6), VARIANTS))


def _plan(seed: int, variant: Variant) -> PlanGraph:
    return gt_plan(random_spec(np.random.default_rng(seed), variant=variant))


def _codes(plan: PlanGraph) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for i in validate_plan(plan):
        out.setdefault(i.code, set()).update(i.element_ids)
    return out


def _poly(plan: PlanGraph, room_id: str) -> Polygon:
    r = next(r for r in plan.rooms if r.id == room_id)
    return Polygon([(p.x, p.y) for p in r.polygon])


def _opening_point(plan: PlanGraph, opening_id: str) -> Point:
    o = next(o for o in plan.openings if o.id == opening_id)
    w = plan.wall(o.host_wall)
    assert isinstance(w.centerline, Segment)
    a, b = w.centerline.a, w.centerline.b
    f = o.offset_m.value / w.centerline.length()
    return Point(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f)


def _rooms_by_door(plan: PlanGraph) -> dict[str, list[str]]:
    """Room → its doors and openings (not windows), by proximity of the opening's centre."""
    out: dict[str, list[str]] = {r.id: [] for r in plan.rooms}
    for o in plan.openings:
        if o.type == "window" or not isinstance(plan.wall(o.host_wall).centerline, Segment):
            continue
        p = _opening_point(plan, o.id)
        t = plan.wall(o.host_wall).thickness_m.value
        for r in plan.rooms:
            if _poly(plan, r.id).distance(p) <= t:
                out[r.id].append(o.id)
    return out


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_clean_plans_raise_no_issue(seed: int, variant: Variant) -> None:
    assert validate_plan(_plan(seed, variant)) == []


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_a_deleted_wall_leaves_open_rooms(seed: int, variant: Variant) -> None:
    plan = _plan(seed, variant)
    interior = [w for w in plan.walls if w.kind == "interior" and isinstance(w.centerline, Segment)]
    w = max(interior, key=lambda x: x.centerline.length())
    plan.walls = [x for x in plan.walls if x.id != w.id]
    plan.openings = [o for o in plan.openings if o.host_wall != w.id]
    cl = LineString([(w.centerline.a.x, w.centerline.a.y), (w.centerline.b.x, w.centerline.b.y)])
    band = cl.buffer(w.thickness_m.value / 2 + 0.02, cap_style="flat")
    # rooms with a stretch of boundary along the deleted wall (not just touching its end)
    beside = {
        r.id for r in plan.rooms if _poly(plan, r.id).exterior.intersection(band).length > 0.2
    }
    found = _codes(plan).get("ROOM_OPEN", set())
    assert beside and beside <= found


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_an_opening_beyond_its_wall_is_flagged(seed: int, variant: Variant) -> None:
    plan = _plan(seed, variant)
    o = plan.openings[0]
    length = plan.wall(o.host_wall).centerline.length()
    o.offset_m = fact(length - o.width_m.value / 4, "user", 1.0)
    assert o.id in _codes(plan).get("OPENING_OUTSIDE_HOST", set())


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_overlapping_rooms_are_flagged(seed: int, variant: Variant) -> None:
    plan = _plan(seed, variant)
    r = plan.rooms[0]
    grown = _poly(plan, r.id).buffer(0.4, join_style="mitre")
    r.polygon = [Point2(x=x, y=y) for x, y in list(grown.exterior.coords)[:-1]]
    assert r.id in _codes(plan).get("ROOMS_OVERLAP", set())


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_area_labels_off_by_more_than_3_percent_are_flagged(seed: int, variant: Variant) -> None:
    plan = _plan(seed, variant)
    for k, r in enumerate(plan.rooms):
        assert r.area_label_m2 is not None
        factor = 1.05 if k % 2 == 0 else 0.95
        r.area_label_m2 = fact(r.area_label_m2.value * factor, "ocr", 0.9)
    found = _codes(plan).get("ROOM_AREA_MISMATCH", set())
    assert found == {r.id for r in plan.rooms}
    # within tolerance: nothing
    plan = _plan(seed, variant)
    for r in plan.rooms:
        assert r.area_label_m2 is not None
        r.area_label_m2 = fact(r.area_label_m2.value * 1.02, "ocr", 0.9)
    assert "ROOM_AREA_MISMATCH" not in _codes(plan)


@pytest.mark.parametrize(("seed", "variant"), CASES)
def test_a_room_without_doors_is_unreachable(seed: int, variant: Variant) -> None:
    plan = _plan(seed, variant)
    doors = _rooms_by_door(plan)
    entry_rooms = {rid for rid, ds in doors.items() if any(_is_exterior(plan, d) for d in ds)}
    target = next(rid for rid in doors if rid not in entry_rooms and doors[rid])
    plan.openings = [o for o in plan.openings if o.id not in set(doors[target])]
    assert target in _codes(plan).get("ROOM_UNREACHABLE", set())


def _is_exterior(plan: PlanGraph, opening_id: str) -> bool:
    o = next(o for o in plan.openings if o.id == opening_id)
    return plan.wall(o.host_wall).kind == "exterior"
