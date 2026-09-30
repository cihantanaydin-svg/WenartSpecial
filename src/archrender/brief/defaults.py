"""Default DesignBrief (used when no brief documents exist yet; Phase 5 adds extraction).

Every default is recorded in the assumption register and every fact is ``method="default"``, so the
QA report shows that the look was not taken from the client's documents.
"""

from __future__ import annotations

from datetime import datetime

from archrender.core.assumptions import AssumptionRegister
from archrender.core.schemas.brief import DesignBrief, Lighting, SurfaceMaterial
from archrender.core.schemas.provenance import fact


def default_brief(section_id: str, register: AssumptionRegister) -> DesignBrief:
    cct = float(register.default("lighting_cct_k"))
    when = datetime.fromisoformat(str(register.default("sun_datetime_local")))
    floor = register.use(
        "floor_material", "oak_floor_natural", "No finish schedule or brief: neutral oak floor."
    )
    walls = register.use("wall_material", "paint_warm_white", "No finish schedule: warm white paint.")
    ceiling = register.use(
        "ceiling_material", "paint_ceiling_white", "No finish schedule: white ceiling paint."
    )
    style = register.use("style", "contemporary", "No brief document: neutral contemporary style.")
    return DesignBrief(
        section_id=section_id,
        style=fact(style, "default", 1.0),
        surfaces=[
            SurfaceMaterial(surface="floor", material_id=floor),
            SurfaceMaterial(surface="walls", material_id=walls),
            SurfaceMaterial(surface="ceiling", material_id=ceiling),
            SurfaceMaterial(surface="door_leaf", material_id="door_oak_veneer"),
            SurfaceMaterial(surface="opening_frame", material_id="frame_aluminium_anthracite"),
            SurfaceMaterial(surface="glass", material_id="glass_clear"),
        ],
        lighting=Lighting(
            cct_k=fact(cct, "default", 1.0),
            datetime_local=fact(when, "default", 1.0),
        ),
        decor_density="none",
    )
