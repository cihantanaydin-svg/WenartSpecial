"""Drawing primitives in plan metres (Y up): the common input of the PlanBuilder.

Each extractor (DXF entities, PDF paths, raster vectorisation) converts its source into these, with
the document → plan transform recorded separately (``DocTransform``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.provenance import Method

Role = str  # "wall" | "door" | "window" | "text" | "dims" | "hatch" | "furniture" | "other"


@dataclass
class Polyline:
    pts: NDArray[np.float64]  # (N, 2)
    closed: bool = False
    curve: bool = False  # contains flattened curves
    width: float | None = None  # stroke width in metres, when known
    fill: bool = False
    role: Role | None = None  # from layer names, when the source has meaningful layers
    clip: bool = False  # drawn under a clipping path (hatch/pattern lines)
    group: str = ""  # rings of one drawing object share a group (fill parity)

    def segments(self) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
        p = self.pts
        n = len(p)
        out = [(p[i], p[i + 1]) for i in range(n - 1)]
        if self.closed and n > 2:
            out.append((p[-1], p[0]))
        return out


@dataclass
class ArcPrim:
    cx: float
    cy: float
    r: float
    a0: float  # degrees; the arc runs counter-clockwise from a0 to a1
    a1: float
    role: Role | None = None

    @property
    def sweep(self) -> float:
        return (self.a1 - self.a0) % 360 or 360.0

    def point(self, f: float) -> tuple[float, float]:
        a = math.radians(self.a0 + self.sweep * f)
        return self.cx + self.r * math.cos(a), self.cy + self.r * math.sin(a)


@dataclass
class TextPrim:
    text: str
    x: float
    y: float
    height: float = 0.0
    angle: float = 0.0  # degrees CCW
    source: str = ""


@dataclass
class DimPrim:
    p1: tuple[float, float]  # measured points (metres)
    p2: tuple[float, float]
    value: float | None  # written value in metres when known
    text: str = ""


@dataclass
class Prims:
    method: Method
    polylines: list[Polyline] = field(default_factory=list)
    arcs: list[ArcPrim] = field(default_factory=list)
    texts: list[TextPrim] = field(default_factory=list)
    dims: list[DimPrim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    resolution: float = 0.0  # positional uncertainty in metres (one pixel for rasters; 0 = exact)


def circle_fit(pts: NDArray[np.float64]) -> tuple[float, float, float, float]:
    """Least-squares circle (Kåsa) → (cx, cy, r, rms residual)."""
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    rms = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
    return float(cx), float(cy), r, rms


def arc_from_points(pts: NDArray[np.float64], max_rms: float) -> ArcPrim | None:
    """Circular arc through an ordered point run (e.g. a flattened Bézier), or None."""
    if len(pts) < 4:
        return None
    cx, cy, r, rms = circle_fit(pts)
    if r <= 0 or rms > max_rms or rms > 0.05 * r:
        return None
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    steps = (np.diff(ang) + 180) % 360 - 180
    total = float(steps.sum())
    if abs(total) < 5:
        return None
    if total > 0:
        return ArcPrim(cx, cy, r, float(ang[0]) % 360, float(ang[0] + total) % 360)
    return ArcPrim(cx, cy, r, float(ang[0] + total) % 360, float(ang[0]) % 360)
