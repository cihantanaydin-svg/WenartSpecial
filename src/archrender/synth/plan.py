"""Synthetic plans with exact ground truth (Phase 3).

A :class:`PlanSpec` is the drawing-independent geometry of one storey in plan metres (Y up):
walls as centerlines (segments, or one circular arc) with thickness, hosted openings, and rooms.
It is built from a :class:`~archrender.synth.layout.Layout` (an orthogonal room split) through a
*lattice embedding*: ``P(s, t) = s·e1 + t·e2`` with unit vectors ``e1``/``e2``. Rotating both gives a
plan that is not aligned with the sheet; an angle between them other than 90° gives a
non-Manhattan plan (parallelogram rooms). Lengths along the lattice axes are preserved, so the
layout's wall lengths, opening offsets and dimension values stay exact. Optionally one facade wall
becomes a circular arc bulging outward (its openings are moved to other walls or dropped).

Every renderer (vector PDF sheet, DXF, raster scan, phone photo, IFC) draws from the same spec, and
:func:`gt_plan` converts it to the ground-truth :class:`PlanGraph`: rooms are the net floor areas
between wall faces (the convention for room polygons and area labels).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import Arc as PArc
from archrender.core.schemas.plan import Level, Opening, PlanGraph, Room, RoomType, Segment, Wall
from archrender.core.schemas.provenance import Fact, fact
from archrender.synth.layout import Layout, random_layout

Variant = Literal["manhattan", "rotated", "skewed", "arc", "skewed_arc"]
ARC_SEGMENTS = 48
STOREY_HEIGHT = 2.80
WET_CEILING = 2.40
WET_ROOMS = {"Banyo", "WC", "Bathroom"}

ROOM_TYPES: dict[str, RoomType] = {
    "Salon": "living",
    "Living Room": "living",
    "Mutfak": "kitchen",
    "Kitchen": "kitchen",
    "Yatak Odası": "bedroom",
    "Bedroom": "bedroom",
    "Banyo": "bathroom",
    "Bathroom": "bathroom",
    "WC": "wc",
    "Antre": "entrance",
    "Entrance": "entrance",
    "Çalışma Odası": "office",
    "Study": "office",
    "Kiler": "storage",
    "Storage": "storage",
    "Toplantı Odası": "meeting",
    "Meeting Room": "meeting",
    "Ofis": "office",
    "Office": "office",
    "Koridor": "corridor",
    "Corridor": "corridor",
}


@dataclass
class SWall:
    a: tuple[float, float]
    b: tuple[float, float]
    thickness: float
    exterior: bool
    sagitta: float = 0.0  # > 0: circular arc bulging to the right of a→b (outward on a CCW ring;
    # such an arc runs counter-clockwise around its centre)

    @property
    def is_arc(self) -> bool:
        return abs(self.sagitta) > 1e-9

    @property
    def chord(self) -> float:
        return math.hypot(self.b[0] - self.a[0], self.b[1] - self.a[1])

    def arc_params(self) -> tuple[tuple[float, float], float, float, float]:
        """(centre, radius, start angle, sweep) in radians; sweep > 0 = counter-clockwise."""
        c = self.chord
        s = self.sagitta
        r = (c * c / 4 + s * s) / (2 * s)
        ux, uy = (self.b[0] - self.a[0]) / c, (self.b[1] - self.a[1]) / c
        mx, my = (self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2
        # right normal of a→b is (uy, -ux); the bulge apex is at mid + s·right, the centre at
        # apex − r·right
        d = r - s
        cx, cy = mx - uy * d, my + ux * d
        a0 = math.atan2(self.a[1] - cy, self.a[0] - cx)
        a1 = math.atan2(self.b[1] - cy, self.b[0] - cx)
        sweep = (a1 - a0) % (2 * math.pi)  # counter-clockwise from a through the apex to b
        return (cx, cy), r, a0, sweep

    @property
    def length(self) -> float:
        if not self.is_arc:
            return self.chord
        _, r, _, sweep = self.arc_params()
        return abs(sweep) * r

    def point(self, t: float) -> tuple[float, float]:
        """Point at arc length ``t`` from ``a``."""
        if not self.is_arc:
            f = t / self.chord
            return self.a[0] + (self.b[0] - self.a[0]) * f, self.a[1] + (self.b[1] - self.a[1]) * f
        (cx, cy), r, a0, sweep = self.arc_params()
        ang = a0 + math.copysign(t / r, sweep)
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    def direction(self, t: float) -> tuple[float, float]:
        if not self.is_arc:
            c = self.chord
            return (self.b[0] - self.a[0]) / c, (self.b[1] - self.a[1]) / c
        _, r, a0, sweep = self.arc_params()
        ang = a0 + math.copysign(t / r, sweep)
        sgn = 1.0 if sweep > 0 else -1.0
        return -math.sin(ang) * sgn, math.cos(ang) * sgn

    def normal(self, t: float) -> tuple[float, float]:
        """Left normal of the direction at ``t`` (the 'pos' side)."""
        ux, uy = self.direction(t)
        return -uy, ux

    def polyline(self) -> list[tuple[float, float]]:
        if not self.is_arc:
            return [self.a, self.b]
        n = ARC_SEGMENTS
        return [self.point(self.length * i / n) for i in range(n + 1)]


@dataclass
class SOpening:
    tag: str
    kind: Literal["door", "window"]
    wall: int
    t: float  # arc length from the wall start to the opening centre
    width: float
    swing: int = 1  # door leaf opens towards the wall's left normal (+1) or right (−1)
    hinge: Literal["start", "end"] = "start"
    sill: float = 0.0
    height: float = 2.1


@dataclass
class SRoom:
    id: str
    name: str
    number: str
    label: tuple[float, float]  # a point inside the room (labels are drawn here)
    cell: list[tuple[float, float]]  # the lattice cell (wall centerlines) as a plan polygon
    ceiling: float
    double_height: bool = False


@dataclass
class Island:
    """Free-standing furniture outline (a kitchen island): drawn with thin lines, never a wall."""

    room: str
    polygon: list[tuple[float, float]]


@dataclass
class PlanSpec:
    walls: list[SWall]
    openings: list[SOpening]
    rooms: list[SRoom]
    variant: Variant
    e1: tuple[float, float] = (1.0, 0.0)
    e2: tuple[float, float] = (0.0, 1.0)
    storey_height: float = STOREY_HEIGHT
    level_name: str = "Zemin Kat"
    layout: Layout | None = None  # the orthogonal source layout (lattice coordinates)
    faces: dict[str, Polygon] = field(default_factory=dict)  # room id → net floor polygon
    islands: list[Island] = field(default_factory=list)

    def lattice(self, s: float, t: float) -> tuple[float, float]:
        return s * self.e1[0] + t * self.e2[0], s * self.e1[1] + t * self.e2[1]

    def opening_point(self, o: SOpening) -> tuple[float, float]:
        return self.walls[o.wall].point(o.t)

    def extent(self) -> tuple[float, float, float, float]:
        pts = [p for w in self.walls for p in w.polyline()]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------------------------
def from_layout(
    layout: Layout,
    *,
    variant: Variant = "manhattan",
    rng: np.random.Generator | None = None,
    level_name: str = "Zemin Kat",
) -> PlanSpec:
    rng = rng or np.random.default_rng(0)
    theta = 0.0
    phi = math.pi / 2
    if variant == "rotated":
        theta = math.radians(float(rng.choice([-1, 1])) * float(rng.uniform(8, 35)))
    elif variant in ("skewed", "skewed_arc"):
        theta = math.radians(float(rng.uniform(-20, 20)))
        phi = math.radians(
            float(rng.choice([float(rng.uniform(65, 80)), float(rng.uniform(100, 115))]))
        )
    e1 = (math.cos(theta), math.sin(theta))
    e2 = (math.cos(theta + phi), math.sin(theta + phi))
    spec = PlanSpec([], [], [], variant, e1, e2, level_name=level_name, layout=layout)
    for w in layout.walls:
        spec.walls.append(
            SWall(spec.lattice(w.ax, w.ay), spec.lattice(w.bx, w.by), w.thickness, w.exterior)
        )
    for o in layout.openings:
        spec.openings.append(
            SOpening(
                o.tag,
                "door" if o.kind == "door" else "window",
                o.wall,
                o.t,
                o.width,
                swing=o.swing,
                hinge="start",
                sill=o.sill,
                height=o.height,
            )
        )
    for i, r in enumerate(layout.rooms):
        ceiling = WET_CEILING if r.name in WET_ROOMS else STOREY_HEIGHT
        cell = [
            spec.lattice(x, y) for x, y in ((r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1))
        ]
        spec.rooms.append(
            SRoom(r.id, r.name, f"Z{i + 1:02d}", spec.lattice(*r.center), cell, ceiling)
        )
    if variant in ("arc", "skewed_arc"):
        _make_arc_wall(spec, rng)
    spec.faces = room_faces(spec)
    return spec


def _make_arc_wall(spec: PlanSpec, rng: np.random.Generator) -> None:
    """Turn the facade wall without the entrance door into an outward arc (no openings on it)."""
    entrance = [o.wall for o in spec.openings if o.kind == "door" and spec.walls[o.wall].exterior]
    candidates = [i for i in range(4) if i not in entrance]
    wi = int(rng.choice(candidates))
    w = spec.walls[wi]
    a, b = w.a, w.b
    w.sagitta = round(float(rng.uniform(0.12, 0.22)) * w.chord, 2)
    spec.openings = [o for o in spec.openings if o.wall != wi]
    arc = LineString(w.polyline())
    # interior walls that ended on the old straight facade now run on to the arc
    ux, uy = (b[0] - a[0]) / w.chord, (b[1] - a[1]) / w.chord
    for other in spec.walls:
        if other.exterior:
            continue
        for end in ("a", "b"):
            p = getattr(other, end)
            q = other.b if end == "a" else other.a
            if abs((p[0] - a[0]) * uy - (p[1] - a[1]) * ux) > 1e-6:
                continue  # not on the facade line
            dx, dy = p[0] - q[0], p[1] - q[1]
            n = math.hypot(dx, dy)
            far = (p[0] + dx / n * w.chord, p[1] + dy / n * w.chord)
            hit = LineString([p, far]).intersection(arc)
            if not hit.is_empty:
                setattr(other, end, (hit.x, hit.y) if hit.geom_type == "Point" else p)
    # room labels and cells stay; faces grow into the bay


def add_loft_features(spec: PlanSpec) -> None:
    """G2 'Loft': the living room is double height and the kitchen gets an island."""
    for r in spec.rooms:
        if spec.layout and r.name in ("Salon", "Living Room"):
            r.double_height = True
            r.ceiling = round(2 * spec.storey_height, 2)
        if spec.layout and r.name in ("Mutfak", "Kitchen"):
            lr = next(x for x in spec.layout.rooms if x.id == r.id)
            cx, cy = lr.center
            w, d = min(2.0, (lr.x1 - lr.x0) * 0.45), 0.9
            corners = [
                (cx - w / 2, cy - d / 2),
                (cx + w / 2, cy - d / 2),
                (cx + w / 2, cy + d / 2),
                (cx - w / 2, cy + d / 2),
            ]
            spec.islands.append(Island(r.id, [spec.lattice(x, y) for x, y in corners]))


def random_spec(
    rng: np.random.Generator, *, variant: Variant | None = None, english: bool = False
) -> PlanSpec:
    v: Variant = variant or str(  # type: ignore[assignment]
        rng.choice(["manhattan", "manhattan", "rotated", "skewed", "arc"])
    )
    return from_layout(
        random_layout(rng, english=english),
        variant=v,
        rng=rng,
        level_name="Ground Floor" if english else "Zemin Kat",
    )


# ---------------------------------------------------------------------------------------------
# solids and faces
# ---------------------------------------------------------------------------------------------
def _exterior_ring(spec: PlanSpec) -> list[tuple[float, float]]:
    ring: list[tuple[float, float]] = []
    for w in spec.walls:
        if w.exterior:
            pts = w.polyline()
            ring += pts[:-1]
    return ring


def wall_solid(spec: PlanSpec, *, cut_openings: bool = False) -> BaseGeometry:
    """Union of the wall bodies (mitred exterior corners). ``cut_openings`` removes the opening
    gaps (for drawing); without it the walls are closed (for room faces)."""
    ext = [w for w in spec.walls if w.exterior]
    t_ext = ext[0].thickness if ext else 0.25
    ring = Polygon(_exterior_ring(spec))
    parts: list[BaseGeometry] = [
        ring.buffer(t_ext / 2, join_style="mitre").difference(
            ring.buffer(-t_ext / 2, join_style="mitre")
        )
    ]
    for w in spec.walls:
        if w.exterior:
            continue
        # interior walls run centerline to centerline; their flat ends sit inside the host walls
        parts.append(LineString(w.polyline()).buffer(w.thickness / 2, cap_style="flat"))
    solid = unary_union(parts)
    if cut_openings:
        solid = solid.difference(unary_union([opening_cut(spec, o) for o in spec.openings]))
    return solid


def opening_cut(spec: PlanSpec, o: SOpening, extra: float = 0.02) -> Polygon:
    """Rectangle (in the wall band) covering an opening's gap."""
    w = spec.walls[o.wall]
    cx, cy = w.point(o.t)
    ux, uy = w.direction(o.t)
    nx, ny = -uy, ux
    h = w.thickness / 2 + extra
    d = o.width / 2
    return Polygon(
        [
            (cx - ux * d + nx * h, cy - uy * d + ny * h),
            (cx + ux * d + nx * h, cy + uy * d + ny * h),
            (cx + ux * d - nx * h, cy + uy * d - ny * h),
            (cx - ux * d - nx * h, cy - uy * d - ny * h),
        ]
    )


def room_faces(spec: PlanSpec) -> dict[str, Polygon]:
    """Net floor polygon of every room: the holes of the closed wall solid, matched by label."""
    solid = wall_solid(spec)
    polys = list(solid.geoms) if hasattr(solid, "geoms") else [solid]
    holes = [Polygon(h) for p in polys for h in p.interiors]
    faces: dict[str, Polygon] = {}
    for r in spec.rooms:
        hit = [h for h in holes if h.contains(Point(r.label))]
        if len(hit) != 1:
            raise ValueError(f"room {r.id} has {len(hit)} faces")
        faces[r.id] = hit[0].normalize()
    return faces


def net_area(spec: PlanSpec, room_id: str) -> float:
    return float(spec.faces[room_id].area)


# ---------------------------------------------------------------------------------------------
# ground truth PlanGraph
# ---------------------------------------------------------------------------------------------
def _gt[T](value: T) -> Fact[T]:
    return fact(value, "derived", 1.0, note="synthetic ground truth")


def _p(x: float, y: float) -> Point2:
    return Point2(x=round(x, 6), y=round(y, 6))


def gt_plan(spec: PlanSpec, project: str = "synthetic", version: str = "gt") -> PlanGraph:
    walls: list[Wall] = []
    for i, w in enumerate(spec.walls):
        centerline: Segment | PArc
        if w.is_arc:
            (cx, cy), radius, a0, sweep = w.arc_params()
            centerline = PArc(
                center=_p(cx, cy),
                radius=round(radius, 6),
                start_deg=round(math.degrees(a0) % 360, 6),
                end_deg=round(math.degrees(a0 + sweep) % 360, 6),
            )
        else:
            centerline = Segment(a=_p(*w.a), b=_p(*w.b))
        walls.append(
            Wall(
                id=f"W{i + 1}",
                level="L0",
                centerline=centerline,
                thickness_m=_gt(w.thickness),
                height_m=_gt(spec.storey_height),
                kind="exterior" if w.exterior else "interior",
            )
        )
    openings: list[Opening] = []
    for o in spec.openings:
        offset = o.t
        swing_side: Literal["pos", "neg"] | None = None
        hinge: Literal["start", "end"] | None = None
        if o.kind == "door":
            swing_side = "pos" if o.swing > 0 else "neg"
            hinge = o.hinge
        openings.append(
            Opening(
                id=f"O{len(openings) + 1}",
                host_wall=f"W{o.wall + 1}",
                offset_m=_gt(offset),
                width_m=_gt(o.width),
                height_m=_gt(o.height),
                sill_m=_gt(o.sill),
                type="door" if o.kind == "door" else "window",
                hinge=hinge,
                swing=("left" if (o.swing > 0) == (o.hinge == "start") else "right")
                if o.kind == "door"
                else None,
                swing_side=swing_side,
                tag=o.tag,
            )
        )
    rooms: list[Room] = []
    for r in spec.rooms:
        face = spec.faces[r.id]
        ring = list(face.exterior.coords)[:-1]
        if Polygon(ring).exterior.is_ccw is False:
            ring.reverse()
        rooms.append(
            Room(
                id=r.id,
                level="L0",
                polygon=[_p(x, y) for x, y in ring],
                name=_gt(r.name),
                number=_gt(r.number),
                type=_gt(ROOM_TYPES.get(r.name, "other")),
                area_label_m2=_gt(round(face.area, 2)),
                ceiling_height_m=_gt(r.ceiling),
                double_height=r.double_height,
            )
        )
    return PlanGraph(
        version=version,
        project=project,
        levels=[Level(id="L0", name=spec.level_name, elevation_m=0.0)],
        north_angle_deg=_gt(0.0),
        walls=walls,
        openings=openings,
        rooms=rooms,
        source="user",
    )
