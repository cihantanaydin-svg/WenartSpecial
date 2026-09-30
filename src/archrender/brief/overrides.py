"""User edits to the design brief: a library material per surface (the Gate B edit).

Overrides are applied on top of the cached S4 output, so changing a material re-runs only S5
onwards (scene, render, refine, QA, bundle); plan and brief stay cached. A surface is a whole
category (``floor``, ``walls``, ``ceiling``, ``door_leaf``, ``opening_frame``, ``glass``,
``skirting``) or one element (``wall:<wall id>``, ``floor:<room id>``, ``ceiling:<room id>``).
"""

from __future__ import annotations

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.brief import DesignBrief, SurfaceMaterial
from archrender.core.schemas.plan import PlanGraph
from archrender.core.schemas.provenance import Assumption
from archrender.scene.materials import MaterialLibrary

SURFACE_CATEGORY = {
    "floor": "floor",
    "walls": "wall",
    "ceiling": "ceiling",
    "door_leaf": "door_leaf",
    "opening_frame": "opening_frame",
    "glass": "glass",
    "skirting": "skirting",
}
ELEMENT_CATEGORY = {"wall": "wall", "floor": "floor", "ceiling": "ceiling"}
# default-brief assumptions that an explicit choice for the whole category replaces
_REPLACES_ASSUMPTION = {
    "floor": "floor_material",
    "walls": "wall_material",
    "ceiling": "ceiling_material",
}


def surface_category(surface: str) -> str:
    if surface in SURFACE_CATEGORY:
        return SURFACE_CATEGORY[surface]
    prefix, sep, element = surface.partition(":")
    if sep and element and prefix in ELEMENT_CATEGORY:
        return ELEMENT_CATEGORY[prefix]
    raise ArchRenderError(
        ErrorCode.VALIDATION,
        f"Unknown surface {surface!r}.",
        f"Use one of {', '.join(SURFACE_CATEGORY)}, or wall:<wall id>, floor:<room id>, "
        "ceiling:<room id>.",
    )


def validate_material_overrides(overrides: dict[str, str], library: MaterialLibrary) -> None:
    """Check surface names and material/category compatibility (no plan needed)."""
    for surface, material_id in overrides.items():
        category = surface_category(surface)
        if material_id not in library.ids():
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Material {material_id!r} is not in the library.",
                f"Pick a {category} material: {', '.join(library.ids(category))}.",
            )
        material = library.get(material_id)
        if material.category != category:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Material {material_id!r} is a {material.category} material and cannot be used "
                f"on {surface!r}.",
                f"Pick a {category} material: {', '.join(library.ids(category))}.",
            )


def check_elements_exist(overrides: dict[str, str], plan: PlanGraph) -> None:
    """Element surfaces (``wall:W3``) must name a wall/room of the approved plan."""
    walls = {w.id for w in plan.walls}
    rooms = {r.id for r in plan.rooms}
    for surface in overrides:
        prefix, sep, element = surface.partition(":")
        if not sep:
            continue
        known = walls if prefix == "wall" else rooms
        if element not in known:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Surface {surface!r} names no {prefix} of the plan.",
                f"Known {'walls' if prefix == 'wall' else 'rooms'}: {', '.join(sorted(known))}.",
            )


def apply_material_overrides(
    brief: DesignBrief, assumptions: list[Assumption], overrides: dict[str, str]
) -> tuple[DesignBrief, list[Assumption]]:
    if not overrides:
        return brief, assumptions
    surfaces = [s for s in brief.surfaces if s.surface not in overrides]
    surfaces += [SurfaceMaterial(surface=k, material_id=v) for k, v in sorted(overrides.items())]
    replaced = {_REPLACES_ASSUMPTION[k] for k in overrides if k in _REPLACES_ASSUMPTION}
    return (
        brief.model_copy(update={"surfaces": surfaces}),
        [a for a in assumptions if a.key not in replaced],
    )
