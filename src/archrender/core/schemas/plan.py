"""PlanGraph: the verified vector plan (metres, Z-up, plan XY in the level frame)."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, model_validator

from archrender.core.schemas.common import Point2, Severity, Strict
from archrender.core.schemas.provenance import Assumption, Conflict, Fact

RoomType = Literal[
    "living",
    "kitchen",
    "dining",
    "bedroom",
    "bathroom",
    "wc",
    "entrance",
    "corridor",
    "balcony",
    "storage",
    "office",
    "meeting",
    "utility",
    "other",
]

OpeningType = Literal[
    "door",
    "double_door",
    "sliding_door",
    "french_door",
    "window",
    "opening",
    "pass",
]


class Segment(Strict):
    kind: Literal["segment"] = "segment"
    a: Point2
    b: Point2

    def length(self) -> float:
        return math.hypot(self.b.x - self.a.x, self.b.y - self.a.y)


class Arc(Strict):
    kind: Literal["arc"] = "arc"
    center: Point2
    radius: float = Field(gt=0)
    start_deg: float
    end_deg: float  # counter-clockwise from start

    def length(self) -> float:
        sweep = (self.end_deg - self.start_deg) % 360.0 or 360.0
        return self.radius * math.radians(sweep)


class Level(Strict):
    id: str
    name: str
    elevation_m: float = 0.0
    floor_to_floor_m: Fact[float] | None = None


class DocTransform(Strict):
    """Affine ``[[a, b, tx], [c, d, ty]]`` mapping document units to plan metres."""

    doc_id: str
    page: int | None = None
    matrix: tuple[tuple[float, float, float], tuple[float, float, float]]
    residual_m: float = 0.0
    method: str


class Wall(Strict):
    id: str
    level: str
    centerline: Segment | Arc = Field(discriminator="kind")
    thickness_m: Fact[float]
    height_m: Fact[float]
    base_offset_m: float = 0.0
    kind: Literal["exterior", "interior", "partition", "curtain", "unknown"] = "unknown"


class Opening(Strict):
    id: str
    host_wall: str
    offset_m: Fact[float]  # along the centerline, start → opening centre
    width_m: Fact[float]
    height_m: Fact[float]
    sill_m: Fact[float]
    type: OpeningType
    hinge: Literal["start", "end"] | None = None
    swing: Literal["left", "right", "none"] | None = None
    swing_side: Literal["pos", "neg"] | None = None
    tag: str | None = None


class Room(Strict):
    id: str
    level: str
    polygon: list[Point2] = Field(min_length=3)  # counter-clockwise outer ring
    holes: list[list[Point2]] = Field(default_factory=list)
    name: Fact[str]
    number: Fact[str] | None = None
    type: Fact[RoomType]
    area_label_m2: Fact[float] | None = None
    ceiling_height_m: Fact[float]
    double_height: bool = False


class Column(Strict):
    id: str
    level: str
    footprint: list[Point2] = Field(min_length=3)
    height_m: Fact[float]


class ValidationIssue(Strict):
    code: str
    severity: Severity
    message: str
    fix_hint: str
    element_ids: list[str] = Field(default_factory=list)
    location: Point2 | None = None


class PlanGraph(Strict):
    version: str
    project: str
    levels: list[Level] = Field(min_length=1)
    north_angle_deg: Fact[float]
    doc_transforms: list[DocTransform] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    rooms: list[Room] = Field(default_factory=list)
    columns: list[Column] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    source: Literal["ifc", "dxf", "pdf_vector", "raster", "mock", "user"]

    @model_validator(mode="after")
    def _check_refs(self) -> PlanGraph:
        level_ids = {lv.id for lv in self.levels}
        wall_ids = {w.id for w in self.walls}
        ids: set[str] = set()
        elements: list[Wall | Opening | Room | Column] = [
            *self.walls,
            *self.openings,
            *self.rooms,
            *self.columns,
        ]
        for el in elements:
            if el.id in ids:
                raise ValueError(f"duplicate element id {el.id!r}")
            ids.add(el.id)
        for w in self.walls:
            if w.level not in level_ids:
                raise ValueError(f"wall {w.id} references unknown level {w.level!r}")
        for o in self.openings:
            if o.host_wall not in wall_ids:
                raise ValueError(f"opening {o.id} references unknown wall {o.host_wall!r}")
        for r in self.rooms:
            if r.level not in level_ids:
                raise ValueError(f"room {r.id} references unknown level {r.level!r}")
        return self

    def wall(self, wall_id: str) -> Wall:
        for w in self.walls:
            if w.id == wall_id:
                return w
        raise KeyError(wall_id)

    def room(self, room_id: str) -> Room:
        for r in self.rooms:
            if r.id == room_id:
                return r
        raise KeyError(room_id)
