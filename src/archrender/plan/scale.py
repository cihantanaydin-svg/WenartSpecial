"""Scale estimators and their reconciliation (ARCHITECTURE §S2 step 6).

Every estimator gives metres per document unit (per page pixel for PDFs and rasters, per drawing
unit for DXF) with a relative standard deviation:

(a) **stated scale** × page units (exact for CAD PDFs: 1 px = 25.4/dpi mm × N);
(b) **dimension strings** paired with the dimension lines they annotate, fitted through the origin
    by RANSAC over each string's unit readings (``350`` may be 3.50 m or 0.35 m);
(c) **scale bars** (segments of known length);
(d) the **door-width prior** (swing radius ≈ 0.85 m), only as a last resort.

The fused value is the inverse-variance weighted median of the precise estimators that agree. Any
two precise estimators (σ ≤ 1 %) disagreeing by more than 1.5 % raise ``PLAN_SCALE_CONFLICT``:
both are shown at Gate A, the better-documented one (the smaller σ) is proposed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from archrender.core.schemas.document import Word
from archrender.understand.text import parse_dimension

CONFLICT_REL = 0.015
PRECISE_SIGMA = 0.01
DOOR_RADIUS_M = 0.85


@dataclass(frozen=True)
class ScaleEstimate:
    value: float  # metres per document unit
    rel_sigma: float
    method: str  # stated | dimensions | scale_bar | door_prior | units
    n: int = 1
    detail: str = ""


@dataclass
class ScaleResult:
    estimate: ScaleEstimate | None
    estimates: list[ScaleEstimate]
    conflicts: list[tuple[ScaleEstimate, ScaleEstimate, float]] = field(default_factory=list)

    @property
    def conflicted(self) -> bool:
        return bool(self.conflicts)


def stated(n: float, unit_m: float, detail: str = "") -> ScaleEstimate:
    """``1:n`` drawing with ``unit_m`` metres per document unit at 1:1 (e.g. 0.0254/dpi)."""
    return ScaleEstimate(n * unit_m, 0.0005, "stated", 1, detail or f"1:{n:g}")


def from_dimensions(pairs: list[tuple[float, tuple[float, ...]]]) -> ScaleEstimate | None:
    """(measured length in document units, candidate values in metres) → RANSAC through 0."""
    pairs = [(m, vs) for m, vs in pairs if m > 0 and vs]
    if not pairs:
        return None
    ratios = sorted({v / m for m, vs in pairs for v in vs if v > 0})
    best: tuple[int, float, list[float]] | None = None
    for r in ratios:
        inl = []
        for m, vs in pairs:
            hits = [v / m for v in vs if v > 0 and abs(v / m - r) / r <= 0.01]
            if hits:
                inl.append(min(hits, key=lambda x: abs(x - r)))
        spread = float(np.std(inl) / np.mean(inl)) if len(inl) > 1 else 1.0
        if best is None or len(inl) > best[0] or (len(inl) == best[0] and spread < best[1]):
            best = (len(inl), spread, inl)
    assert best is not None
    n, _, inl = best
    value = float(np.median(inl))
    mad = float(np.median(np.abs(np.array(inl) - value))) / value if n > 1 else 0.03
    sigma = max(0.002, 1.4826 * mad / math.sqrt(n)) if n > 1 else 0.03
    return ScaleEstimate(
        value, sigma, "dimensions", n, f"{n} of {len(pairs)} dimension strings agree"
    )


def from_door_radii(radii: list[float]) -> ScaleEstimate | None:
    radii = [r for r in radii if r > 0]
    if len(radii) < 2:
        return None
    med = float(np.median(radii))
    return ScaleEstimate(
        DOOR_RADIUS_M / med,
        0.12,
        "door_prior",
        len(radii),
        f"median door swing radius {med:.4g} units ≈ {DOOR_RADIUS_M} m",
    )


def fuse(estimates: list[ScaleEstimate]) -> ScaleResult:
    ests = [e for e in estimates if e is not None and e.value > 0]
    if not ests:
        return ScaleResult(None, [])
    precise = [e for e in ests if e.rel_sigma <= PRECISE_SIGMA]
    conflicts = []
    for i, a in enumerate(precise):
        for b in precise[i + 1 :]:
            rel = abs(a.value - b.value) / min(a.value, b.value)
            if rel > CONFLICT_REL:
                conflicts.append((a, b, rel))
    if not precise:
        best = min(ests, key=lambda e: e.rel_sigma)
        return ScaleResult(best, ests, [])
    if conflicts:
        # propose the best-documented estimator; the user decides at Gate A
        best = min(precise, key=lambda e: e.rel_sigma)
        return ScaleResult(best, ests, conflicts)
    w = np.array([1.0 / e.rel_sigma**2 for e in precise])
    v = np.array([e.value for e in precise])
    order = np.argsort(v)
    cum = np.cumsum(w[order])
    med = float(v[order][int(np.searchsorted(cum, cum[-1] / 2))])
    sigma = float(1.0 / math.sqrt(w.sum()))
    methods = "+".join(sorted({e.method for e in precise}))
    return ScaleResult(ScaleEstimate(med, sigma, methods, sum(e.n for e in precise)), ests, [])


# ---------------------------------------------------------------------------------------------
# PDF/raster dimension strings ↔ dimension lines
# ---------------------------------------------------------------------------------------------
def dimension_pairs(
    words: list[Word],
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    max_gap_px: float,
) -> list[tuple[float, tuple[float, ...]]]:
    """Each dimension-like word with the nearest parallel line it sits on (page px)."""
    if not segments:
        return []
    a = np.array([s[0] for s in segments], np.float64)
    b = np.array([s[1] for s in segments], np.float64)
    d = b - a
    lens = np.hypot(d[:, 0], d[:, 1])
    ok = lens > 1e-6
    u = np.zeros_like(d)
    u[ok] = d[ok] / lens[ok, None]
    out = []
    for w in words:
        dim = parse_dimension(w.text)
        if dim is None or not dim.metres or not any(ch.isdigit() for ch in w.text):
            continue
        ang = math.radians(w.angle_deg)
        tu = np.array([math.cos(ang), -math.sin(ang)])  # text direction in page px (y down)
        c = np.array([(w.x0 + w.x1) / 2, (w.y0 + w.y1) / 2])
        par = np.abs(u @ tu) > math.cos(math.radians(2.0))
        best = None
        for k in np.nonzero(par & ok)[0]:
            rel = c - a[k]
            along = float(rel @ u[k])
            perp = abs(float(rel[0] * u[k][1] - rel[1] * u[k][0]))
            if perp > max_gap_px or not 0.25 * lens[k] <= along <= 0.75 * lens[k]:
                continue
            if best is None or perp < best[0]:
                best = (perp, float(lens[k]))
        if best is not None:
            out.append((best[1], tuple(dim.metres)))
    return out


def open_segments(
    paths: dict[str, Any], *, max_width_px: float | None = None
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Straight segments of open, unclipped sub-paths (dimension lines, ticks, symbols)."""
    out = []
    for p in paths.get("paths", []):
        if p.get("clip") or (max_width_px is not None and (p.get("w") or 0) > max_width_px):
            continue
        for sub, closed, curve in zip(p["sub"], p["closed"], p["curve"], strict=True):
            if closed or curve:
                continue
            for q0, q1 in itertools.pairwise(sub):
                out.append((tuple(q0), tuple(q1)))
    return out
