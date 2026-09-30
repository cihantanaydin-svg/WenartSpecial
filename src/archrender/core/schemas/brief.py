"""DesignBrief: what the section should look like, with provenance for every decision."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from archrender.core.schemas.common import Strict
from archrender.core.schemas.provenance import Conflict, Fact


class LabColor(Strict):
    L: float = Field(ge=0, le=100)
    a: float
    b: float


class WeightedColor(Strict):
    lab: LabColor
    weight: float = Field(ge=0, le=1)
    name: str | None = None


class SurfaceMaterial(Strict):
    surface: str  # floor | ceiling | wall:<id> | walls | joinery | counter | backsplash | metal
    material_id: str  # asset-library material id
    color_lab: LabColor | None = None
    derived: bool = False  # tinted/scaled from the closest library match
    paint_code: str | None = None
    paint_code_approximation: bool = False


class FurnitureItem(Strict):
    type: str
    style: str | None = None
    material: str | None = None
    approx_size_m: tuple[float, float, float] | None = None
    must_have: bool = False
    room_id: str | None = None


class Lighting(Strict):
    cct_k: Fact[float]
    mood: Literal["bright", "neutral", "warm", "evening", "dramatic"] = "neutral"
    datetime_local: Fact[datetime]
    timezone: str = "Europe/Istanbul"
    fixtures_from_rcp: bool = False


class DesignBrief(Strict):
    section_id: str
    style: Fact[str]
    palette: list[WeightedColor] = Field(default_factory=list)
    surfaces: list[SurfaceMaterial] = Field(default_factory=list)
    furniture: list[FurnitureItem] = Field(default_factory=list)
    lighting: Lighting
    decor_density: Literal["none", "low", "medium", "high"] = "low"
    keep: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    contradictions: list[Conflict] = Field(default_factory=list)

    def surface(self, name: str) -> SurfaceMaterial | None:
        for s in self.surfaces:
            if s.surface == name:
                return s
        return None
