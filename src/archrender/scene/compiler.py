"""S5 scene compiler: PlanGraph + Section + DesignBrief → SceneSpec + watertight meshes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from manifold3d import Manifold
from shapely.geometry import Polygon
from shapely.ops import unary_union

from archrender.core.assumptions import AssumptionRegister
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.hashing import sha256_bytes
from archrender.core.schemas.brief import DesignBrief
from archrender.core.schemas.plan import Opening, PlanGraph, Wall
from archrender.core.schemas.scene import (
    AssetLicense,
    MaterialSpec,
    MeshFile,
    ObjectCategory,
    RenderSettings,
    SceneObject,
    SceneSpec,
    SunSpec,
    WorldSpec,
)
from archrender.core.schemas.section import Section
from archrender.scene.geometry import (
    TriMesh,
    box_uv,
    check_manifold,
    extrude,
    is_watertight,
    opening_cutter,
    oriented_box,
    room_polygon,
    slab,
    to_trimesh,
    wall_footprints,
    wall_frame,
)
from archrender.scene.materials import MaterialLibrary, hex_to_linear
from archrender.scene.meshio import mesh_npz

SLAB_THICKNESS_M = 0.12
FRAME_PROFILE_M = 0.05
WINDOW_FRAME_DEPTH_M = 0.07
DOOR_FRAME_DEPTH_M = 0.08
DOOR_LEAF_THICKNESS_M = 0.04
GLASS_THICKNESS_M = 0.006


@dataclass
class CompiledMeshes:
    spec: SceneSpec
    mesh_blobs: dict[str, bytes] = field(default_factory=dict)  # package-relative path → npz bytes


def section_rooms(plan: PlanGraph, section: Section) -> list[str]:
    if section.kind == "rooms":
        known = {r.id for r in plan.rooms}
        missing = [r for r in section.room_ids if r not in known]
        if missing:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Section references unknown rooms: {', '.join(missing)}.",
                "Pick rooms from the approved plan.",
            )
        return list(section.room_ids)
    if section.kind == "polygon" and section.polygon:
        sel = Polygon([(p.x, p.y) for p in section.polygon])
        return [r.id for r in plan.rooms if room_polygon(plan, r.id).intersects(sel)]
    raise ArchRenderError(
        ErrorCode.VALIDATION,
        "Cut-line sections are compiled in Phase 4 (sectional perspectives).",
        "Select rooms or draw a polygon for this section.",
    )


class SceneCompiler:
    def __init__(self, library: MaterialLibrary) -> None:
        self.library = library

    def compile(
        self,
        plan: PlanGraph,
        section: Section,
        brief: DesignBrief,
        render: RenderSettings,
        register: AssumptionRegister,
        *,
        scene_id: str,
    ) -> CompiledMeshes:
        rooms = section_rooms(plan, section)
        footprint = unary_union([room_polygon(plan, r) for r in rooms])
        pieces = wall_footprints(plan)
        walls = [w for w in plan.walls if pieces[w.id].intersects(footprint.buffer(0.6))]
        openings = [o for o in plan.openings if o.host_wall in {w.id for w in walls}]

        blobs: dict[str, bytes] = {}
        objects: list[SceneObject] = []
        material_ids: list[str] = []

        def material_for(surface: str, fallback: str) -> str:
            sm = brief.surface(surface) or brief.surface(fallback)
            if sm is None:
                raise ArchRenderError(
                    ErrorCode.SCENE_INVALID,
                    f"The design brief has no material for '{surface}'.",
                    "Assign a material for this surface at Gate B.",
                )
            self.library.get(sm.material_id)
            if sm.material_id not in material_ids:
                material_ids.append(sm.material_id)
            return sm.material_id

        def add(obj_id: str, category: ObjectCategory, mesh: TriMesh, material: str, ref: str | None) -> None:
            if mesh.triangles == 0:
                raise ArchRenderError(
                    ErrorCode.SCENE_INVALID, f"Object {obj_id} has no geometry.", "Check the plan element."
                )
            watertight = is_watertight(mesh)
            v, f, uv = box_uv(mesh)
            blob = mesh_npz(v, f, uv)
            path = f"meshes/{obj_id}.npz"
            blobs[path] = blob
            objects.append(
                SceneObject(
                    id=obj_id,
                    category=category,
                    mesh=MeshFile(
                        path=path, sha256=sha256_bytes(blob), triangles=mesh.triangles, watertight=watertight
                    ),
                    material=material,
                    pass_index=len(objects) + 1,
                    element_ref=ref,
                )
            )

        # walls with openings subtracted
        cutters = {o.id: opening_cutter(plan.wall(o.host_wall), o) for o in openings}
        for w in walls:
            base = w.base_offset_m
            solid = extrude(pieces[w.id], base, base + w.height_m.value)
            for oid, cutter in cutters.items():
                bmin, bmax = _bbox(cutter)
                if pieces[w.id].intersects(Polygon.from_bounds(bmin[0], bmin[1], bmax[0], bmax[1])):
                    solid = solid - cutter
            check_manifold(solid, f"wall {w.id}")
            add(f"wall_{w.id}", "wall", to_trimesh(solid), material_for(f"wall:{w.id}", "walls"), w.id)

        # opening infill: frames, leaves, glazing
        for o in openings:
            host = plan.wall(o.host_wall)
            for suffix, category, mesh in _opening_parts(host, o):
                surface = {"opening_frame": "opening_frame", "door_leaf": "door_leaf", "glass": "glass"}[
                    category
                ]
                add(f"{o.id}_{suffix}", category, mesh, material_for(surface, surface), o.id)

        # floor / ceiling slabs per room (extend under the walls to avoid light leaks)
        for rid in rooms:
            room = plan.room(rid)
            poly = room_polygon(plan, rid)
            adjacent = [w.thickness_m.value for w in walls if pieces[w.id].distance(poly) < 1e-3]
            grow = (max(adjacent) / 2.0) if adjacent else 0.0
            ext = poly.buffer(grow, join_style="mitre") if grow > 0 else poly
            h = room.ceiling_height_m.value
            add(f"floor_{rid}", "floor", slab(ext, -SLAB_THICKNESS_M, 0.0, f"floor {rid}"), material_for(f"floor:{rid}", "floor"), rid)
            add(
                f"ceiling_{rid}",
                "ceiling",
                slab(ext, h, h + SLAB_THICKNESS_M, f"ceiling {rid}"),
                material_for(f"ceiling:{rid}", "ceiling"),
                rid,
            )

        materials = []
        for i, mid in enumerate(material_ids, start=1):
            lib = self.library.get(mid)
            materials.append(
                MaterialSpec(
                    id=mid,
                    base_color_linear=hex_to_linear(lib.color),
                    roughness=lib.roughness,
                    metallic=lib.metallic,
                    transmission=lib.transmission,
                    ior=lib.ior,
                    texture_size_m=lib.texture_size_m,
                    pass_index=i,
                    source=lib.source,
                    license=lib.license,
                )
            )
        sun_az = float(register.use("sun_azimuth_deg", 225.0, "Phase-1 skeleton: fixed sun (SW); pvlib ephemeris arrives in Phase 4."))
        sun_el = float(register.use("sun_elevation_deg", 35.0, "Phase-1 skeleton: fixed sun elevation."))
        spec = SceneSpec(
            scene_id=scene_id,
            north_angle_deg=plan.north_angle_deg.value,
            objects=objects,
            materials=materials,
            sun=SunSpec(azimuth_deg=sun_az, elevation_deg=sun_el, color_k=5500.0, strength=3.0),
            world=WorldSpec(kind="color", color_linear=(0.55, 0.68, 0.9), strength=1.0),
            cameras=[],
            render=render,
            assets=[AssetLicense(id=m.id, source=m.source, license=m.license) for m in materials],
        )
        return CompiledMeshes(spec=spec, mesh_blobs=blobs)


def _bbox(m: Manifold) -> tuple[np.ndarray, np.ndarray]:
    bb = m.bounding_box()
    return np.array(bb[:3]), np.array(bb[3:])


def _ring(
    center: np.ndarray, d: np.ndarray, w: float, depth: float, z0: float, z1: float, profile: float, open_bottom: bool
) -> TriMesh:
    outer = oriented_box(center, d, w, depth, z0, z1)
    inner_z0 = z0 - 0.01 if open_bottom else z0 + profile
    inner = oriented_box(center, d, w - 2 * profile, depth + 0.02, inner_z0, z1 - profile)
    ring = outer - inner
    check_manifold(ring, "opening frame")
    return to_trimesh(ring)


def _opening_parts(wall: Wall, o: Opening) -> list[tuple[str, ObjectCategory, TriMesh]]:
    a, d, _ = wall_frame(wall)
    c = a + d * o.offset_m.value
    w, h, sill = o.width_m.value, o.height_m.value, o.sill_m.value
    parts: list[tuple[str, ObjectCategory, TriMesh]] = []
    if o.type == "window":
        parts.append(
            ("frame", "opening_frame", _ring(c, d, w, WINDOW_FRAME_DEPTH_M, sill, sill + h, FRAME_PROFILE_M, False))
        )
        glass = oriented_box(
            c,
            d,
            w - 2 * FRAME_PROFILE_M + 0.01,
            GLASS_THICKNESS_M,
            sill + FRAME_PROFILE_M - 0.005,
            sill + h - FRAME_PROFILE_M + 0.005,
        )
        check_manifold(glass, "glass")
        parts.append(("glass", "glass", to_trimesh(glass)))
    elif o.type in ("door", "double_door", "french_door", "sliding_door"):
        profile = 0.04
        parts.append(("frame", "opening_frame", _ring(c, d, w, DOOR_FRAME_DEPTH_M, 0.0, h, profile, True)))
        leaf = oriented_box(c, d, w - 2 * profile - 0.004, DOOR_LEAF_THICKNESS_M, 0.005, h - profile - 0.002)
        check_manifold(leaf, "door leaf")
        parts.append(("leaf", "door_leaf", to_trimesh(leaf)))
    return parts
