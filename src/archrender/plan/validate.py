"""Deterministic PlanGraph validators. Each issue has a code, severity, location and fix hint."""

from __future__ import annotations

import itertools
from collections import deque

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from archrender.core.schemas.common import Point2, Severity
from archrender.core.schemas.plan import PlanGraph, Segment, ValidationIssue
from archrender.scene.geometry import centerline_coords, room_polygon

AREA_TOLERANCE = 0.03
MIN_ROOM_AREA_M2 = 1.5
DOOR_WIDTH_RANGE = (0.6, 2.4)
WINDOW_WIDTH_RANGE = (0.3, 6.0)
WALL_THICKNESS_RANGE = (0.05, 1.0)
MIN_OPENING_GAP_M = 0.05


def _pt(x: float, y: float) -> Point2:
    return Point2(x=x, y=y)


def validate_plan(plan: PlanGraph) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues += _rooms(plan)
    issues += _enclosure(plan)
    issues += _walls(plan)
    issues += _openings(plan)
    issues += _reachability(plan)
    return issues


def blocking(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return [i for i in issues if i.severity in (Severity.ERROR, Severity.BLOCKER)]


def _rooms(plan: PlanGraph) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    polys: dict[str, Polygon] = {}
    for r in plan.rooms:
        poly = Polygon([(p.x, p.y) for p in r.polygon], [[(p.x, p.y) for p in h] for h in r.holes])
        c = (
            poly.representative_point()
            if not poly.is_empty
            else Point(r.polygon[0].x, r.polygon[0].y)
        )
        if not poly.is_valid:
            out.append(
                ValidationIssue(
                    code="ROOM_INVALID_POLYGON",
                    severity=Severity.ERROR,
                    message=f"Room {r.name.value} ({r.id}) boundary is self-intersecting or not closed.",
                    fix_hint="Redraw the room boundary so it follows the inner wall faces.",
                    element_ids=[r.id],
                    location=_pt(c.x, c.y),
                )
            )
            continue
        polys[r.id] = poly
        if poly.area < MIN_ROOM_AREA_M2:
            out.append(
                ValidationIssue(
                    code="ROOM_TOO_SMALL",
                    severity=Severity.WARNING,
                    message=f"Room {r.name.value} is only {poly.area:.2f} m².",
                    fix_hint="Check the scale or merge the room with its neighbour.",
                    element_ids=[r.id],
                    location=_pt(c.x, c.y),
                )
            )
        if r.area_label_m2 is not None:
            label = r.area_label_m2.value
            if label > 0 and abs(poly.area - label) / label > AREA_TOLERANCE:
                out.append(
                    ValidationIssue(
                        code="ROOM_AREA_MISMATCH",
                        severity=Severity.ERROR,
                        message=(
                            f"Room {r.name.value}: computed area {poly.area:.2f} m² differs from the "
                            f"label {label:.2f} m² by {100 * abs(poly.area - label) / label:.1f}% (> 3%)."
                        ),
                        fix_hint="Verify the drawing scale (Gate A calibration) or the room boundary.",
                        element_ids=[r.id],
                        location=_pt(c.x, c.y),
                    )
                )
    ids = list(polys)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            inter = polys[a].intersection(polys[b]).area
            if inter > 1e-4:
                c = polys[a].intersection(polys[b]).representative_point()
                out.append(
                    ValidationIssue(
                        code="ROOMS_OVERLAP",
                        severity=Severity.ERROR,
                        message=f"Rooms {a} and {b} overlap by {inter:.3f} m².",
                        fix_hint="Adjust the shared boundary so rooms meet at the wall centre or face.",
                        element_ids=[a, b],
                        location=_pt(c.x, c.y),
                    )
                )
    return out


OPEN_TOLERANCE_M = 0.03
MIN_OPEN_RUN_M = 0.10


def _enclosure(plan: PlanGraph) -> list[ValidationIssue]:
    """A room's boundary must run along walls (openings are in walls): a part of it with no wall
    is an open room (a deleted or missing wall, a leak into the neighbour)."""
    by_level: dict[str, BaseGeometry] = {}
    for lvl in {w.level for w in plan.walls}:
        bodies = [
            LineString(centerline_coords(w)).buffer(
                w.thickness_m.value / 2 + OPEN_TOLERANCE_M, cap_style="flat"
            )
            for w in plan.walls
            if w.level == lvl
        ]
        # flat caps leave the corner squares of L-junctions out: close them
        by_level[lvl] = unary_union(bodies).buffer(OPEN_TOLERANCE_M).buffer(-OPEN_TOLERANCE_M)
    out: list[ValidationIssue] = []
    for r in plan.rooms:
        body = by_level.get(r.level)
        ring = LineString([(p.x, p.y) for p in [*r.polygon, r.polygon[0]]])
        bare = ring if body is None else ring.difference(body)
        if bare.is_empty or bare.length < MIN_OPEN_RUN_M:
            continue
        parts = list(getattr(bare, "geoms", [bare]))
        longest = max(parts, key=lambda g: g.length)
        c = longest.interpolate(0.5, normalized=True)
        out.append(
            ValidationIssue(
                code="ROOM_OPEN",
                severity=Severity.ERROR,
                message=(
                    f"Room {r.name.value} ({r.id}) is not enclosed: {bare.length:.2f} m of its "
                    "boundary has no wall."
                ),
                fix_hint="Draw the missing wall (Gate A), or merge the room with its neighbour.",
                element_ids=[r.id],
                location=_pt(c.x, c.y),
            )
        )
    return out


def _walls(plan: PlanGraph) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    lo, hi = WALL_THICKNESS_RANGE
    for w in plan.walls:
        t = w.thickness_m.value
        cl = centerline_coords(w)
        mid = cl[len(cl) // 2]
        if not lo <= t <= hi:
            out.append(
                ValidationIssue(
                    code="WALL_THICKNESS_IMPLAUSIBLE",
                    severity=Severity.ERROR,
                    message=f"Wall {w.id} thickness {t:.3f} m is outside {lo}–{hi} m.",
                    fix_hint="Check the scale or the wall pairing at Gate A.",
                    element_ids=[w.id],
                    location=_pt(*mid),
                )
            )
        if w.centerline.length() < 0.05:
            out.append(
                ValidationIssue(
                    code="WALL_DEGENERATE",
                    severity=Severity.ERROR,
                    message=f"Wall {w.id} is shorter than 5 cm.",
                    fix_hint="Delete the wall or merge it with its neighbour.",
                    element_ids=[w.id],
                    location=_pt(*mid),
                )
            )
    return out


def _openings(plan: PlanGraph) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    by_wall: dict[str, list[tuple[float, float, str]]] = {}
    for o in plan.openings:
        w = plan.wall(o.host_wall)
        length = w.centerline.length()
        off, width = o.offset_m.value, o.width_m.value
        loc = None
        if isinstance(w.centerline, Segment):
            a, b = w.centerline.a, w.centerline.b
            f = off / length if length else 0.0
            loc = _pt(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f)
        if off - width / 2 < -1e-6 or off + width / 2 > length + 1e-6:
            out.append(
                ValidationIssue(
                    code="OPENING_OUTSIDE_HOST",
                    severity=Severity.ERROR,
                    message=f"Opening {o.id} ({width:.2f} m at {off:.2f} m) exceeds wall {w.id} ({length:.2f} m).",
                    fix_hint="Move or resize the opening, or re-host it on the correct wall.",
                    element_ids=[o.id, w.id],
                    location=loc,
                )
            )
        lo, hi = WINDOW_WIDTH_RANGE if o.type == "window" else DOOR_WIDTH_RANGE
        if not lo <= width <= hi:
            out.append(
                ValidationIssue(
                    code="OPENING_WIDTH_IMPLAUSIBLE",
                    severity=Severity.WARNING,
                    message=f"{o.type} {o.id} width {width:.2f} m is outside {lo}–{hi} m.",
                    fix_hint="Check the scale or the schedule entry for this tag.",
                    element_ids=[o.id],
                    location=loc,
                )
            )
        if o.sill_m.value + o.height_m.value > w.height_m.value + 1e-6:
            out.append(
                ValidationIssue(
                    code="OPENING_ABOVE_WALL",
                    severity=Severity.ERROR,
                    message=f"Opening {o.id} head is above the wall height.",
                    fix_hint="Correct the sill/head height or the wall height.",
                    element_ids=[o.id, w.id],
                    location=loc,
                )
            )
        by_wall.setdefault(w.id, []).append((off - width / 2, off + width / 2, o.id))
    for wall_id, spans in by_wall.items():
        spans.sort()
        for (_a0, a1, aid), (b0, _b1, bid) in itertools.pairwise(spans):
            if b0 - a1 < MIN_OPENING_GAP_M:
                out.append(
                    ValidationIssue(
                        code="OPENINGS_OVERLAP",
                        severity=Severity.ERROR,
                        message=f"Openings {aid} and {bid} on wall {wall_id} overlap or touch.",
                        fix_hint="Separate them by at least 5 cm or merge them into one opening.",
                        element_ids=[aid, bid],
                    )
                )
    return out


def _reachability(plan: PlanGraph) -> list[ValidationIssue]:
    """Every room must be reachable through doors/openings from a room with an exterior door."""
    if len(plan.rooms) <= 1:
        return []
    polys = {r.id: room_polygon(plan, r.id) for r in plan.rooms}
    graph: dict[str, set[str]] = {rid: set() for rid in polys}
    entries: set[str] = set()
    for o in plan.openings:
        if o.type == "window":
            continue
        w = plan.wall(o.host_wall)
        if not isinstance(w.centerline, Segment):
            continue
        a, b = w.centerline.a, w.centerline.b
        f = o.offset_m.value / max(w.centerline.length(), 1e-9)
        p = Point(a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f)
        near = [rid for rid, poly in polys.items() if poly.distance(p) <= w.thickness_m.value]
        if len(near) == 1:
            entries.add(near[0])
        for i, r1 in enumerate(near):
            for r2 in near[i + 1 :]:
                graph[r1].add(r2)
                graph[r2].add(r1)
    if not entries:
        entries = {next(iter(polys))}
    seen: set[str] = set()
    todo = deque(entries)
    while todo:
        r = todo.popleft()
        if r in seen:
            continue
        seen.add(r)
        todo.extend(graph[r] - seen)
    out = []
    for rid in polys:
        if rid not in seen:
            c = polys[rid].representative_point()
            out.append(
                ValidationIssue(
                    code="ROOM_UNREACHABLE",
                    severity=Severity.ERROR,
                    message=f"Room {rid} cannot be reached through any door or opening.",
                    fix_hint="Add the missing door/opening, or check that doors are hosted on the shared wall.",
                    element_ids=[rid],
                    location=_pt(c.x, c.y),
                )
            )
    return out
