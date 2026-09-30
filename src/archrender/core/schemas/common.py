"""Shared primitive types. Geometry is metres, right-handed, Z-up; plan XY in the level frame."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Id = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Point2(Frozen):
    x: float
    y: float


class Vec3(Frozen):
    x: float
    y: float
    z: float

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


class BBox(Frozen):
    """Axis-aligned box ``[x0, y0, x1, y1]`` (pixels or metres depending on context)."""

    x0: float
    y0: float
    x1: float
    y1: float


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


class Unit(StrEnum):
    M = "m"
    CM = "cm"
    MM = "mm"
    FT = "ft"
    IN = "in"
    M2 = "m2"
    DEG = "deg"
    KELVIN = "K"
    LUMEN = "lm"


class ModelRef(Frozen):
    """Exact identity of a model used to produce a fact or an image."""

    role: str
    name: str
    repo: str
    revision: str | None = None
    weights_sha256: str | None = None
    mock: bool = False
