"""Deterministic architectural geometry (ADR-S04): walls, openings, slabs.

Walls are built by construction: each wall's 2D footprint is the buffered centerline. Footprints are
partitioned so no two walls own the same area, which avoids coincident faces at junctions. They are
extruded with manifold3d, and openings are subtracted as prisms with manifold3d's guaranteed-manifold
booleans, so reveals, sills and heads are capped. Every returned mesh is checked for
watertightness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection, Error, FillRule, Manifold
from numpy.typing import NDArray
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.plan import Arc, Opening, PlanGraph, Segment, Wall

Float = NDArray[np.float64]
Int = NDArray[np.int64]
EPS = 1e-4


@dataclass(frozen=True)
class TriMesh:
    vertices: Float  # (N, 3)
    faces: Int  # (M, 3)

    @property
    def triangles(self) -> int:
        return int(self.faces.shape[0])


def centerline_coords(wall: Wall, arc_segments: int = 48) -> list[tuple[float, float]]:
    cl = wall.centerline
    if isinstance(cl, Segment):
        return [(cl.a.x, cl.a.y), (cl.b.x, cl.b.y)]
    sweep = (cl.end_deg - cl.start_deg) % 360.0 or 360.0
    n = max(4, int(math.ceil(arc_segments * sweep / 360.0)))
    return [
        (
            cl.center.x + cl.radius * math.cos(math.radians(cl.start_deg + sweep * i / n)),
            cl.center.y + cl.radius * math.sin(math.radians(cl.start_deg + sweep * i / n)),
        )
        for i in range(n + 1)
    ]


def wall_footprints(plan: PlanGraph) -> dict[str, Polygon]:
    """Non-overlapping footprints whose union is the full wall network.

    Each centerline is buffered by half its thickness with square caps, which fills L-corners.
    Overlaps go to the earlier wall, so every point of the union belongs to exactly one wall.
    """
    owned: dict[str, Polygon] = {}
    taken: BaseGeometry = Polygon()
    for wall in plan.walls:
        t = wall.thickness_m.value
        if t <= 0:
            raise ArchRenderError(
                ErrorCode.PLAN_INVALID,
                f"Wall {wall.id} has non-positive thickness {t}.",
                "Correct the wall thickness at Gate A.",
            )
        buf = LineString(centerline_coords(wall)).buffer(
            t / 2.0, cap_style="square", join_style="mitre", mitre_limit=4.0
        )
        piece = buf.difference(taken) if not taken.is_empty else buf
        piece = _largest_polygon(piece)
        if piece is None or piece.area < 1e-6:
            raise ArchRenderError(
                ErrorCode.PLAN_INVALID,
                f"Wall {wall.id} is completely covered by other walls (duplicate wall?).",
                "Remove the duplicate wall at Gate A.",
                context={"wall": wall.id},
            )
        owned[wall.id] = piece
        taken = unary_union([taken, buf])
    return owned


def _largest_polygon(geom: BaseGeometry) -> Polygon | None:
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        polys = [g for g in geom.geoms if g.area > 1e-6]
        if not polys:
            return None
        if len(polys) > 1:
            # A wall piece split in two means a sliver: keep the main body, the sliver is owned
            # by the union through the neighbouring wall. Guard against real splits.
            polys.sort(key=lambda p: p.area, reverse=True)
            if polys[1].area > 0.01 * polys[0].area:
                raise ArchRenderError(
                    ErrorCode.PLAN_INVALID,
                    "A wall footprint is split by another wall (crossing walls without a junction).",
                    "Split crossing walls at their intersection at Gate A.",
                )
        return polys[0]
    return None


def polygon_to_cross_section(poly: Polygon) -> CrossSection:
    """manifold3d wants counter-clockwise outer rings and clockwise holes (shapely ``orient``)."""
    poly = orient(poly, sign=1.0)
    rings = [list(poly.exterior.coords)[:-1]] + [list(r.coords)[:-1] for r in poly.interiors]
    return CrossSection([[(float(x), float(y)) for x, y in ring] for ring in rings], FillRule.EvenOdd)


def extrude(poly: Polygon, z0: float, z1: float) -> Manifold:
    if z1 <= z0:
        raise ArchRenderError(ErrorCode.SCENE_INVALID, "Extrusion height must be positive.", "Check heights.")
    return Manifold.extrude(polygon_to_cross_section(poly), z1 - z0).translate((0.0, 0.0, z0))


def wall_frame(wall: Wall) -> tuple[np.ndarray, np.ndarray, float]:
    """Start point, unit direction and length of a straight wall."""
    cl = wall.centerline
    if not isinstance(cl, Segment):
        raise ArchRenderError(
            ErrorCode.SCENE_INVALID,
            f"Openings on curved wall {wall.id} are not supported by the scene compiler yet.",
            "Straighten the wall or move the opening to a straight wall at Gate A.",
        )
    a = np.array([cl.a.x, cl.a.y])
    b = np.array([cl.b.x, cl.b.y])
    length = float(np.linalg.norm(b - a))
    if length < EPS:
        raise ArchRenderError(ErrorCode.PLAN_INVALID, f"Wall {wall.id} has zero length.", "Delete it.")
    return a, (b - a) / length, length


def oriented_box(
    center_xy: np.ndarray, direction: np.ndarray, along: float, across: float, z0: float, z1: float
) -> Manifold:
    """Box of size ``along × across × (z1-z0)`` centred on ``center_xy``, rotated to ``direction``."""
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    box = Manifold.cube((along, across, z1 - z0), center=True)
    return box.rotate((0.0, 0.0, angle)).translate(
        (float(center_xy[0]), float(center_xy[1]), (z0 + z1) / 2.0)
    )


def opening_cutter(wall: Wall, opening: Opening, extra_depth: float = 0.02) -> Manifold:
    a, d, length = wall_frame(wall)
    off = opening.offset_m.value
    w = opening.width_m.value
    if off - w / 2 < -EPS or off + w / 2 > length + EPS:
        raise ArchRenderError(
            ErrorCode.PLAN_INVALID,
            f"Opening {opening.id} extends beyond its host wall {wall.id}.",
            "Move or resize the opening at Gate A.",
        )
    center = a + d * off
    sill = opening.sill_m.value
    return oriented_box(
        center, d, w, wall.thickness_m.value + 2 * extra_depth, sill, sill + opening.height_m.value
    )


def check_manifold(m: Manifold, what: str) -> None:
    if m.status() != Error.NoError or m.is_empty():
        raise ArchRenderError(
            ErrorCode.SCENE_NOT_WATERTIGHT,
            f"Geometry for {what} is not a valid closed solid ({m.status()}).",
            "Check the plan for self-intersecting or zero-thickness elements at Gate A.",
        )


def to_trimesh(m: Manifold) -> TriMesh:
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return TriMesh(v, f)


def is_watertight(mesh: TriMesh) -> bool:
    """Every undirected edge is shared by exactly two triangles with opposite orientation."""
    f = mesh.faces
    if f.size == 0:
        return False
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    directed = {tuple(e) for e in edges.tolist()}
    if len(directed) != len(edges):
        return False  # a directed edge used twice → non-manifold or flipped
    return all((b, a) in directed for a, b in directed)


def box_uv(mesh: TriMesh) -> tuple[Float, Int, Float]:
    """Split vertices per triangle corner and assign world-space box UVs (metres).

    The dominant axis of each face normal picks the projection plane, so textures keep their
    real-world scale on every surface. Returns ``(vertices, faces, uv)``.
    """
    v = mesh.vertices[mesh.faces].reshape(-1, 3)
    tri = v.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    axis = np.argmax(np.abs(n), axis=1)
    uv = np.empty((tri.shape[0], 3, 2))
    for ax, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        sel = axis == ax
        uv[sel] = tri[sel][:, :, [i, j]]
    faces = np.arange(v.shape[0], dtype=np.int64).reshape(-1, 3)
    return v, faces, uv.reshape(-1, 2)


def slab(poly: Polygon, z0: float, z1: float, what: str) -> TriMesh:
    m = extrude(poly, z0, z1)
    check_manifold(m, what)
    return to_trimesh(m)


def room_polygon(plan: PlanGraph, room_id: str) -> Polygon:
    r = plan.room(room_id)
    poly = Polygon([(p.x, p.y) for p in r.polygon], [[(p.x, p.y) for p in h] for h in r.holes])
    if not poly.is_valid or poly.area <= 0:
        raise ArchRenderError(
            ErrorCode.PLAN_INVALID,
            f"Room {room_id} polygon is invalid (self-intersecting or empty).",
            "Redraw the room boundary at Gate A.",
        )
    return poly


def arc_or_segment_length(cl: Segment | Arc) -> float:
    return cl.length()
