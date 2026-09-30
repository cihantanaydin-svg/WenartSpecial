"""Section selection and deliverable requests."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from archrender.core.schemas.common import Point2, Strict


class CutLine(Strict):
    p0: Point2
    p1: Point2
    view_dir: Literal["left", "right"] = "left"  # side of p0→p1 the viewer looks toward
    depth_m: float = Field(default=8.0, gt=0)


class Deliverables(Strict):
    views: int = Field(default=3, ge=1, le=12)
    width: int = Field(default=3840, ge=64, le=8192)
    height: int = Field(default=2160, ge=36, le=8192)
    cutaway: bool = True
    section_perspective: bool = False
    panorama: bool = False
    blend_file: bool = False
    exr: bool = False
    people: Literal[False] = False  # out of scope for v1


class Section(Strict):
    id: str
    level: str
    kind: Literal["rooms", "polygon", "cutline"]
    room_ids: list[str] = Field(default_factory=list)
    polygon: list[Point2] | None = None
    cut: CutLine | None = None
    mode: Literal["faithful", "concept"] = "faithful"
    deliverables: Deliverables = Field(default_factory=Deliverables)

    @model_validator(mode="after")
    def _check_kind(self) -> Section:
        if self.kind == "rooms" and not self.room_ids:
            raise ValueError("kind='rooms' needs at least one room id")
        if self.kind == "polygon" and (not self.polygon or len(self.polygon) < 3):
            raise ValueError("kind='polygon' needs a polygon with ≥ 3 points")
        if self.kind == "cutline" and self.cut is None:
            raise ValueError("kind='cutline' needs a cut line")
        return self
