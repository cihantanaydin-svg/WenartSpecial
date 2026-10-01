"""PlanBuilder: drawing primitives (plan metres) → PlanGraph. Shared by the DXF, vector-PDF and
raster paths, so they share validators (ARCHITECTURE §S2 step 4).

1. **Wall evidence.** Segments on wall layers when the source has meaningful layers; otherwise the
   edges of closed outlines (CAD wall bodies are closed polylines / filled paths), and as a last
   resort every straight segment.
2. **Face pairs.** Two evidence segments facing each other (Δθ < 1°, separation 4–65 cm, overlap ≥
   20 cm or half the shorter one, the nearest on that side — all of them when the opposite face is
   split into pieces) bound a wall if the strip between them is inside the drawn wall body (fill
   parity of the rings of the same drawing object). The pair gives a centerline piece and the
   thickness.
3. **Arc walls.** Chains of short pieces that turn consistently and fit a circle become one arc
   wall (non-Manhattan and curved walls are first-class; nothing is snapped to 90°).
4. **Merge.** Collinear pieces of equal thickness are joined; the holes between them (≤ 3 m) are
   kept as gaps.
5. **Junctions.** Wall ends are extended or trimmed to the centerline of the wall they run into
   (L, T and X junctions).
6. **Openings.** A gap mostly covered by walls running into it is a junction (also two staggered
   T's from both sides). Otherwise it is a door when an arc of
   radius ≈ gap width is centred at one of its ends (hinge + swing side), a window when ≥ 2 thin
   lines run along the gap inside the wall band, else a pass-through opening.
7. **Rooms.** The holes of the closed wall body (openings closed) are the rooms (net floor areas),
   labelled by the texts inside: name, number, area label.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from archrender.core.schemas.common import Point2, Unit
from archrender.core.schemas.plan import (
    Arc,
    Level,
    Opening,
    OpeningType,
    PlanGraph,
    Room,
    RoomType,
    Segment,
    Wall,
)
from archrender.core.schemas.provenance import Assumption, Fact, Method, fact
from archrender.plan.prims import ArcPrim, Prims, TextPrim, arc_from_points, circle_fit
from archrender.understand.text import TAG_KINDS, fold, normalise_tag, parse_number

ANGLE_TOL = math.radians(1.0)
MIN_T, MAX_T = 0.04, 0.65
MAX_GAP = 3.0
DEFAULTS = {
    "wall_height_m": 2.70,
    "ceiling_height_m": 2.70,
    "door_height_m": 2.10,
    "window_sill_m": 0.90,
    "window_height_m": 1.20,
}


@dataclass
class Piece:
    """A centerline piece from one face pair."""

    a: NDArray[np.float64]
    b: NDArray[np.float64]
    t: float

    @property
    def length(self) -> float:
        return float(np.hypot(*(self.b - self.a)))

    @property
    def u(self) -> NDArray[np.float64]:
        return (self.b - self.a) / max(self.length, 1e-12)


@dataclass
class BWall:
    """A wall being built: straight (origin + direction + interval) or arc."""

    t: float
    support: float
    p0: NDArray[np.float64] = field(default_factory=lambda: np.zeros(2))
    u: NDArray[np.float64] = field(default_factory=lambda: np.array([1.0, 0.0]))
    s0: float = 0.0
    s1: float = 0.0
    gaps: list[tuple[float, float]] = field(default_factory=list)
    arc: tuple[float, float, float, float, float] | None = None  # cx, cy, r, a0, a1 (deg CCW)

    @property
    def n(self) -> NDArray[np.float64]:
        return np.array([-self.u[1], self.u[0]])

    def at(self, s: float) -> NDArray[np.float64]:
        return self.p0 + self.u * s

    @property
    def a(self) -> NDArray[np.float64]:
        return self.at(self.s0)

    @property
    def b(self) -> NDArray[np.float64]:
        return self.at(self.s1)

    def line(self) -> LineString:
        if self.arc is None:
            return LineString([tuple(self.a), tuple(self.b)])
        cx, cy, r, a0, a1 = self.arc
        sweep = (a1 - a0) % 360 or 360
        n = max(8, int(sweep / 2))
        return LineString(
            [
                (
                    cx + r * math.cos(math.radians(a0 + sweep * k / n)),
                    cy + r * math.sin(math.radians(a0 + sweep * k / n)),
                )
                for k in range(n + 1)
            ]
        )


@dataclass
class BuildResult:
    plan: PlanGraph
    notes: list[str]
    pieces: int
    unexplained_gaps: int


# ---------------------------------------------------------------------------------------------
# 1–2: evidence and face pairs
# ---------------------------------------------------------------------------------------------
@dataclass
class _Segs:
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    group: list[str]

    def __len__(self) -> int:
        return len(self.a)


def _evidence(prims: Prims) -> tuple[_Segs, dict[str, list[Polygon]], str]:
    has_roles = any(p.role == "wall" for p in prims.polylines)
    rings: dict[str, list[Polygon]] = {}
    if has_roles:
        chosen = [p for p in prims.polylines if p.role == "wall"]
        mode = "layers"
    else:
        chosen = [
            p
            for p in prims.polylines
            if p.closed and not p.curve and not p.clip and len(p.pts) >= 3
        ]
        mode = "closed outlines"
        if not chosen:
            chosen = [p for p in prims.polylines if not p.curve and not p.clip]
            mode = "all lines"
    a_list, b_list, groups = [], [], []
    for p in chosen:
        if p.closed and len(p.pts) >= 3:
            poly = Polygon(p.pts)
            if poly.is_valid and poly.area > 1e-6:
                rings.setdefault(p.group, []).append(poly)
        for a, b in _merge_collinear(p.segments()):
            if float(np.hypot(*(b - a))) >= 0.02:
                a_list.append(a)
                b_list.append(b)
                groups.append(p.group)
    segs = _Segs(np.array(a_list).reshape(-1, 2), np.array(b_list).reshape(-1, 2), groups)
    return segs, rings, mode


def _merge_collinear(
    segs: list[tuple[NDArray[np.float64], NDArray[np.float64]]],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    out: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for a, b in segs:
        if out:
            pa, pb = out[-1]
            if np.allclose(pb, a, atol=1e-6):
                d1, d2 = pb - pa, b - a
                n1, n2 = np.hypot(*d1), np.hypot(*d2)
                if (
                    n1 > 1e-9
                    and n2 > 1e-9
                    and abs(d1[0] * d2[1] - d1[1] * d2[0]) / (n1 * n2) < 0.003
                    and np.dot(d1, d2) > 0
                ):
                    out[-1] = (pa, b)
                    continue
        out.append((a, b))
    return out


def _inside(rings: dict[str, list[Polygon]], group: str, p: NDArray[np.float64]) -> bool | None:
    """Odd number of rings of the drawing object containing p → inside its body; None = no rings."""
    polys = rings.get(group)
    if not polys:
        return None
    pt = Point(float(p[0]), float(p[1]))
    return sum(1 for g in polys if g.contains(pt)) % 2 == 1


def _pairs(segs: _Segs, rings: dict[str, list[Polygon]]) -> list[Piece]:
    n = len(segs)
    if n == 0:
        return []
    d = segs.b - segs.a
    length = np.hypot(d[:, 0], d[:, 1])
    u = d / length[:, None]
    nrm = np.column_stack([-u[:, 1], u[:, 0]])
    ang = np.arctan2(u[:, 1], u[:, 0]) % math.pi
    mid = (segs.a + segs.b) / 2
    order = np.argsort(ang)
    ang_sorted = ang[order]
    best: dict[tuple[int, int], list[tuple[float, int]]] = {}  # (i, side) → [(distance, j)]
    for i in range(n):
        lo = np.searchsorted(ang_sorted, ang[i] - ANGLE_TOL)
        hi = np.searchsorted(ang_sorted, ang[i] + ANGLE_TOL, side="right")
        cand = list(order[lo:hi])
        if ang[i] < ANGLE_TOL:  # wrap around 0/π
            cand += list(order[np.searchsorted(ang_sorted, math.pi - ANGLE_TOL + ang[i]) :])
        if ang[i] > math.pi - ANGLE_TOL:
            cand += list(
                order[: np.searchsorted(ang_sorted, ang[i] + ANGLE_TOL - math.pi, side="right")]
            )
        for j in cand:
            if j == i:
                continue
            off = float(np.dot(nrm[i], mid[j] - segs.a[i]))
            dist = abs(off)
            if not MIN_T <= dist <= MAX_T:
                continue
            # the other segment must be parallel along its whole length
            e0 = float(np.dot(nrm[i], segs.a[j] - segs.a[i]))
            e1 = float(np.dot(nrm[i], segs.b[j] - segs.a[i]))
            if abs(e0 - e1) > 0.01 + 0.01 * length[j]:
                continue
            t0 = float(np.dot(u[i], segs.a[j] - segs.a[i]))
            t1 = float(np.dot(u[i], segs.b[j] - segs.a[i]))
            lo_t, hi_t = max(0.0, min(t0, t1)), min(length[i], max(t0, t1))
            overlap = hi_t - lo_t
            if overlap < 0.03 or (overlap < 0.5 * min(length[i], length[j]) and overlap < 0.2):
                continue
            best.setdefault((i, 1 if off > 0 else -1), []).append((dist, j))
    pieces: list[Piece] = []
    seen: set[tuple[int, int]] = set()
    partners = []
    for (i, _side), cands in best.items():
        dmin = min(c[0] for c in cands)
        # every partner at the nearest distance: the opposite face is often split into pieces
        partners += [(i, j, dist) for dist, j in cands if dist <= dmin + 0.005]
    for i, j, dist in partners:
        k = (min(i, j), max(i, j))
        if k in seen:
            continue
        seen.add(k)
        t0 = float(np.dot(u[i], segs.a[j] - segs.a[i]))
        t1 = float(np.dot(u[i], segs.b[j] - segs.a[i]))
        lo_t, hi_t = max(0.0, min(t0, t1)), min(length[i], max(t0, t1))
        off = float(np.dot(nrm[i], mid[j] - segs.a[i]))
        pa = segs.a[i] + u[i] * lo_t + nrm[i] * off / 2
        pb = segs.a[i] + u[i] * hi_t + nrm[i] * off / 2
        if float(np.hypot(*(pb - pa))) < 0.5 * dist:
            continue  # a cap or jamb facing a far face, not a wall
        # the strip between the faces must be wall body (when the source has filled/closed outlines)
        if any(
            _inside(rings, segs.group[i], pa + (pb - pa) * f) is False for f in (0.25, 0.5, 0.75)
        ):
            continue
        pieces.append(Piece(pa, pb, dist))
    return pieces


# ---------------------------------------------------------------------------------------------
# 3–4: arcs and merging
# ---------------------------------------------------------------------------------------------
def _arc_walls(pieces: list[Piece]) -> tuple[list[BWall], list[Piece]]:
    """Chains of short pieces turning consistently → circle fit → arc walls. Chains may jump the
    small gaps that junctions leave on one face; pieces lying on a found circle join it, and arcs
    on the same circle are merged."""
    short = [i for i, p in enumerate(pieces) if p.length < 0.8]
    if len(short) < 4:
        return [], pieces
    adj: dict[int, set[int]] = {i: set() for i in short}
    for x, i in enumerate(short):
        for j in short[x + 1 :]:
            pi, pj = pieces[i], pieces[j]
            touch = min(float(np.hypot(*(ea - eb))) for ea in (pi.a, pi.b) for eb in (pj.a, pj.b))
            if touch > 0.15:
                continue
            cosang = abs(float(np.dot(pi.u, pj.u)))
            if 0.0 < math.degrees(math.acos(min(1.0, cosang))) <= 15.0 and abs(pi.t - pj.t) < 0.03:
                adj[i].add(j)
                adj[j].add(i)
    used: set[int] = set()
    circles: list[tuple[float, float, float, float, list[int]]] = []  # cx, cy, r, t, members
    for start_i in short:
        if start_i in used or not adj[start_i]:
            continue
        comp, stack = [], [start_i]
        seen: set[int] = set()
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            comp.append(k)
            stack.extend(adj[k] - seen)
        if len(comp) < 4:
            continue
        pts = np.array([q for k in comp for q in (pieces[k].a, pieces[k].b)])
        cx, cy, r, rms = circle_fit(pts)
        if rms > 0.02 or r > 60:
            continue
        used.update(comp)
        circles.append((cx, cy, r, float(np.median([pieces[k].t for k in comp])), comp))
    # pieces lying on a circle (tangent, both ends on it) belong to that arc wall
    for i, p in enumerate(pieces):
        if i in used:
            continue
        for cx, cy, r, t, comp in circles:
            c = np.array([cx, cy])
            on = all(abs(float(np.hypot(*(q - c))) - r) < 0.02 for q in (p.a, p.b))
            radial = (p.a + p.b) / 2 - c
            tangent = abs(float(np.dot(p.u, radial / max(float(np.hypot(*radial)), 1e-9)))) < 0.1
            if on and tangent and abs(p.t - t) < 0.03:
                comp.append(i)
                used.add(i)
                break
    # merge arcs on the same circle
    merged: list[tuple[float, float, float, float, list[int]]] = []
    for cx, cy, r, t, comp in circles:
        for k, (mx, my, mr, mt, mcomp) in enumerate(merged):
            if math.hypot(cx - mx, cy - my) < 0.05 and abs(r - mr) < 0.03 and abs(t - mt) < 0.03:
                merged[k] = (mx, my, mr, mt, mcomp + comp)
                break
        else:
            merged.append((cx, cy, r, t, list(comp)))
    walls: list[BWall] = []
    for _, _, _, _, comp in merged:
        pts = np.array([q for k in comp for q in (pieces[k].a, pieces[k].b)])
        cx, cy, r, _ = circle_fit(pts)
        ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360
        srt = np.sort(ang)
        gaps = np.diff(np.concatenate([srt, [srt[0] + 360]]))
        k = int(np.argmax(gaps))
        a0 = float(srt[(k + 1) % len(srt)])
        a1 = float(srt[k])
        if (a1 - a0) % 360 < 10:
            used.difference_update(comp)
            continue
        t = float(np.median([pieces[k2].t for k2 in comp]))
        walls.append(
            BWall(
                t=t, support=float(sum(pieces[k2].length for k2 in comp)), arc=(cx, cy, r, a0, a1)
            )
        )
    rest = [p for i, p in enumerate(pieces) if i not in used]
    return walls, rest


def _merge(pieces: list[Piece]) -> list[BWall]:
    """Collinear pieces of equal thickness → walls with gaps."""
    clusters: list[list[Piece]] = []
    keys: list[tuple[NDArray[np.float64], float, float]] = []  # (u, offset, t)
    for p in sorted(pieces, key=lambda q: -q.length):
        u = p.u
        if u[0] < 0 or (abs(u[0]) < 1e-9 and u[1] < 0):
            u = -u
        nrm = np.array([-u[1], u[0]])
        off = float(np.dot(nrm, p.a))
        for k, (ku, koff, kt) in enumerate(keys):
            if (
                abs(float(ku[0] * u[1] - ku[1] * u[0])) < math.sin(math.radians(0.8))
                and abs(float(np.dot(np.array([-ku[1], ku[0]]), p.a)) - koff) < 0.025
                and abs(kt - p.t) < 0.025
            ):
                clusters[k].append(p)
                break
        else:
            keys.append((u, off, p.t))
            clusters.append([p])
    walls: list[BWall] = []
    for (u, off, _), group in zip(keys, clusters, strict=True):
        nrm = np.array([-u[1], u[0]])
        # refine the line by length-weighted averaging
        wts = np.array([q.length for q in group])
        offs = np.array([float(np.dot(nrm, (q.a + q.b) / 2)) for q in group])
        off = float(np.average(offs, weights=wts))
        t = float(np.average([q.t for q in group], weights=wts))
        p0 = nrm * off
        iv = sorted(
            (
                min(float(np.dot(u, q.a)), float(np.dot(u, q.b))),
                max(float(np.dot(u, q.a)), float(np.dot(u, q.b))),
            )
            for q in group
        )
        runs: list[list[tuple[float, float]]] = [[iv[0]]]
        for s0, s1 in iv[1:]:
            last_end = max(e for _, e in runs[-1])
            if s0 - last_end <= MAX_GAP:
                runs[-1].append((s0, s1))
            else:
                runs.append([(s0, s1)])
        for run in runs:
            run.sort()
            gaps: list[tuple[float, float]] = []
            end = run[0][1]
            for s0, s1 in run[1:]:
                if s0 - end > 0.02:
                    gaps.append((end, s0))
                end = max(end, s1)
            support = sum(s1 - s0 for s0, s1 in run)
            walls.append(
                BWall(
                    t=t, support=support, p0=p0.copy(), u=u.copy(), s0=run[0][0], s1=end, gaps=gaps
                )
            )
    return walls


# ---------------------------------------------------------------------------------------------
# 5: junctions
# ---------------------------------------------------------------------------------------------
def _line_hit(
    p: NDArray[np.float64], d: NDArray[np.float64], w: BWall
) -> tuple[float, float] | None:
    """Ray p + λd against wall w's centerline → (λ, position along w) or None."""
    if w.arc is not None:
        cx, cy, r, a0, a1 = w.arc
        f = p - np.array([cx, cy])
        b = float(np.dot(f, d))
        c = float(np.dot(f, f)) - r * r
        disc = b * b - c
        if disc < 0:
            return None
        best = None
        for lam in (-b - math.sqrt(disc), -b + math.sqrt(disc)):
            q = p + d * lam
            ang = math.degrees(math.atan2(q[1] - cy, q[0] - cx)) % 360
            on_arc = (ang - a0) % 360 <= (a1 - a0) % 360 + 2.0 or (a0 - ang) % 360 < 2.0
            if on_arc and (best is None or abs(lam) < abs(best[0])):
                best = (lam, ang)
        return best
    den = d[0] * w.u[1] - d[1] * w.u[0]
    if abs(den) < 1e-6:
        return None
    q = w.p0 - p
    lam = (q[0] * w.u[1] - q[1] * w.u[0]) / den
    s = float(np.dot(p + d * lam - w.p0, w.u))
    return float(lam), s


def _close_junctions(walls: list[BWall]) -> None:
    straight = [w for w in walls if w.arc is None]
    for w in straight:
        for end in (0, 1):
            p = w.a if end == 0 else w.b
            d = -w.u if end == 0 else w.u
            best: tuple[float, float] | None = None
            for o in walls:
                if o is w:
                    continue
                hit = _line_hit(p, d, o)
                if hit is None:
                    continue
                lam, s = hit
                sin = (
                    1.0
                    if o.arc is not None
                    else max(0.3, abs(float(d[0] * o.u[1] - d[1] * o.u[0])))
                )
                # oblique junctions leave both ends further from the corner than square ones
                reach = max(w.t, o.t) / 2 / sin + 0.25
                if not -o.t / 2 / sin - 0.02 <= lam <= reach:
                    continue
                slack = (o.t + w.t) / sin + 0.1
                if o.arc is None and not o.s0 - slack <= s <= o.s1 + slack:
                    continue
                if best is None or abs(lam) < abs(best[0]):
                    best = (lam, s)
            if best is not None:
                if end == 0:
                    w.s0 -= best[0]
                else:
                    w.s1 += best[0]
    # extend straight walls through their partner's end (L corners): handled above; arc ends next
    for w in walls:
        if w.arc is None:
            continue
        cx, cy, r, a0, a1 = w.arc
        for end in (0, 1):
            ang = a0 if end == 0 else a1
            p = np.array(
                [cx + r * math.cos(math.radians(ang)), cy + r * math.sin(math.radians(ang))]
            )
            best_ang = None
            best_d = 1e9
            for o in straight:
                # intersect the circle with o's line near p
                f = o.p0 - np.array([cx, cy])
                b = float(np.dot(f, o.u))
                c = float(np.dot(f, f)) - r * r
                disc = b * b - c
                if disc < 0:
                    continue
                for lam in (-b - math.sqrt(disc), -b + math.sqrt(disc)):
                    q = o.p0 + o.u * lam
                    if not o.s0 - o.t - 0.3 <= lam <= o.s1 + o.t + 0.3:
                        continue
                    dist = float(np.hypot(*(q - p)))
                    if dist < best_d and dist <= max(w.t, o.t) + 0.3:
                        best_d = dist
                        best_ang = math.degrees(math.atan2(q[1] - cy, q[0] - cx)) % 360
            if best_ang is not None:
                if end == 0:
                    a0 = best_ang
                else:
                    a1 = best_ang
        w.arc = (cx, cy, r, a0, a1)


# ---------------------------------------------------------------------------------------------
# 6: openings
# ---------------------------------------------------------------------------------------------
@dataclass
class _Opening:
    wall: int
    s: float  # centre, along the wall from its start
    width: float
    kind: OpeningType
    hinge: Literal["start", "end"] | None = None
    swing_side: Literal["pos", "neg"] | None = None
    confidence: float = 0.8


def _symbol_segments(prims: Prims) -> list[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    out = []
    for p in prims.polylines:
        if p.role == "wall" or p.clip or p.closed:
            continue
        if p.curve:
            continue
        out += p.segments()
    return out


def _door_arcs(prims: Prims) -> list[ArcPrim]:
    arcs = [a for a in prims.arcs if 5 < a.sweep < 200]
    for p in prims.polylines:
        if p.curve and not p.closed and len(p.pts) >= 4:
            arc = arc_from_points(p.pts, max_rms=0.01)
            if arc is not None:
                arcs.append(arc)
    return arcs


def _junction_cover(w: BWall, g0: float, g1: float, walls: list[BWall]) -> float:
    """Length of the gap [g0, g1] on wall w covered by the bands of walls that run into it."""
    gap_line = LineString([tuple(w.at(g0 - 0.02)), tuple(w.at(g1 + 0.02))])
    spans = []
    for o in walls:
        if o is w or o.line().distance(gap_line) > w.t / 2 + 0.02:
            continue
        if o.arc is not None:
            continue
        sin = abs(float(o.u[0] * w.u[1] - o.u[1] * w.u[0]))
        if sin < 0.2:
            continue
        hit = _line_hit(o.p0, o.u, w)
        if hit is None:
            continue
        lam, s_on_w = hit
        if not o.s0 - o.t - 0.05 <= lam <= o.s1 + o.t + 0.05:
            continue
        half = o.t / 2 / sin
        spans.append((max(g0, s_on_w - half), min(g1, s_on_w + half)))
    spans = sorted(sp for sp in spans if sp[1] > sp[0])
    covered, end = 0.0, g0
    for a, b in spans:
        a = max(a, end)
        if b > a:
            covered += b - a
            end = b
    return covered


def _openings(walls: list[BWall], prims: Prims) -> tuple[list[_Opening], int]:
    arcs = _door_arcs(prims)
    syms = _symbol_segments(prims)
    sym_a = np.array([a for a, _ in syms]).reshape(-1, 2)
    sym_b = np.array([b for _, b in syms]).reshape(-1, 2)
    out: list[_Opening] = []
    unexplained = 0
    for wi, w in enumerate(walls):
        if w.arc is not None:
            continue
        for g0, g1 in w.gaps:
            # junction: walls running into the gap cover it (several when T's are staggered); what
            # is left between them is narrower than any door or window
            covered = _junction_cover(w, g0, g1, walls)
            if covered > 0 and (g1 - g0) - covered < 0.5:
                continue
            width = g1 - g0
            if width < 0.25:
                unexplained += 1
                continue
            mid = (g0 + g1) / 2
            ends = (w.at(g0), w.at(g1))
            kind: OpeningType = "opening"
            hinge: Literal["start", "end"] | None = None
            side: Literal["pos", "neg"] | None = None
            conf = 0.6
            # door: an arc of radius ≈ width centred at a gap end
            for arc in arcs:
                if abs(arc.r - width) > max(0.06, 0.15 * width):
                    continue
                c = np.array([arc.cx, arc.cy])
                dists = [float(np.hypot(*(c - e))) for e in ends]
                k = int(np.argmin(dists))
                if dists[k] > max(0.12, w.t):
                    continue
                kind = "door"
                hinge = "start" if k == 0 else "end"
                mx, my = arc.point(0.5)
                side = "pos" if float(np.dot(w.n, np.array([mx, my]) - w.at(mid))) > 0 else "neg"
                conf = 0.9
                break
            if kind == "opening" and len(sym_a):
                # window: ≥ 2 thin lines along the gap inside the wall band
                d = sym_b - sym_a
                lens = np.hypot(d[:, 0], d[:, 1])
                ok = lens > 0.5 * width
                if ok.any():
                    uu = d[ok] / lens[ok, None]
                    par = np.abs(uu @ w.u) > math.cos(math.radians(2))
                    am = sym_a[ok][par]
                    bm = sym_b[ok][par]
                    count = 0
                    for a, b in zip(am, bm, strict=True):
                        off_a = float(np.dot(w.n, a - w.p0))
                        off_b = float(np.dot(w.n, b - w.p0))
                        if max(abs(off_a), abs(off_b)) > w.t / 2 + 0.03:
                            continue
                        s_a, s_b = sorted(
                            (float(np.dot(w.u, a - w.p0)), float(np.dot(w.u, b - w.p0)))
                        )
                        cover = min(s_b, g1) - max(s_a, g0)
                        if cover >= 0.8 * width:
                            count += 1
                    if count >= 2:
                        kind = "window"
                        conf = 0.85
            out.append(_Opening(wi, mid, width, kind, hinge, side, conf))
    return out, unexplained


# ---------------------------------------------------------------------------------------------
# 7: rooms and labels
# ---------------------------------------------------------------------------------------------
ROOM_TYPES: dict[str, RoomType] = {
    "salon": "living",
    "oturma odasi": "living",
    "living room": "living",
    "living": "living",
    "mutfak": "kitchen",
    "kitchen": "kitchen",
    "yemek odasi": "dining",
    "dining": "dining",
    "yatak odasi": "bedroom",
    "ebeveyn yatak odasi": "bedroom",
    "cocuk odasi": "bedroom",
    "bedroom": "bedroom",
    "banyo": "bathroom",
    "bathroom": "bathroom",
    "wc": "wc",
    "antre": "entrance",
    "giris": "entrance",
    "entrance": "entrance",
    "hol": "corridor",
    "koridor": "corridor",
    "corridor": "corridor",
    "balkon": "balcony",
    "balcony": "balcony",
    "kiler": "storage",
    "depo": "storage",
    "storage": "storage",
    "calisma odasi": "office",
    "ofis": "office",
    "office": "office",
    "study": "office",
    "toplanti odasi": "meeting",
    "meeting room": "meeting",
    "camasirhane": "utility",
    "utility": "utility",
}

AREA_RE = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(m²|m2|m\^2|sqm|sq\.?\s*m)?$", re.I)
UNIT_RE = re.compile(r"^(m²|m2|m\^2|sqm)$", re.I)


def room_type(name: str) -> RoomType:
    key = fold(name).replace("ı", "i")
    key = re.sub(r"\s+\d+$", "", key)
    return ROOM_TYPES.get(key, "other")


def _labels(poly: Polygon, prims: Prims) -> tuple[str | None, str | None, float | None, list[str]]:
    """Name, number and area label of a room from the texts inside it.

    Tag-like words are set aside first: door/window tags (K1, P3 …) are ignored, other codes (Z01,
    101) are room numbers. PDF/OCR words are joined into lines only when they are adjacent on the
    same baseline; DXF texts are whole strings already."""
    inside = [t for t in prims.texts if poly.contains(Point(t.x, t.y))]
    notes: list[str] = []
    number = None
    words = []
    for t in inside:
        if UNIT_RE.match(t.text.strip()):
            words.append(t)  # an area unit ("m²"), joined to its number below
            continue
        tag = normalise_tag(t.text) if len(t.text) <= 6 else None
        if tag is not None:
            prefix = re.match(r"[^\d]*", tag)
            if prefix is not None and prefix.group(0) not in TAG_KINDS:
                number = number or tag
            continue
        words.append(t)
    lines: list[list[TextPrim]] = []
    for t in sorted(words, key=lambda q: (q.source != "dxf_text", -q.y, q.x)):
        if t.source == "dxf_text":
            lines.append([t])
            continue
        for line in lines:
            last = line[-1]
            if last.source == "dxf_text" or abs(last.angle - t.angle) > 5:
                continue
            h = max(t.height, last.height, 1e-3)
            gap = t.x - last.x
            if abs(last.y - t.y) < 0.4 * h and 0 < gap < h * (len(last.text) * 0.75 + 1.5):
                line.append(t)
                break
        else:
            lines.append([t])
    names: list[tuple[float, str]] = []
    area = None
    for line in lines:
        text = " ".join(q.text for q in line).strip()
        compact = text.replace(" ", "")
        m = AREA_RE.match(text) or AREA_RE.match(compact)
        if m and m.group(2):
            v = parse_number(m.group(1), locale="tr" if "," in m.group(1) else "en")
            if v is not None and 0.5 <= v <= 2000:
                area = v
            continue
        if UNIT_RE.match(text):
            continue
        if any(ch.isalpha() for ch in text) and not text.startswith("+"):
            names.append((max(q.height for q in line), text))
    # names from the room vocabulary first (a stamp or a note may be written larger), then size
    name = max(names, key=lambda n: (room_type(n[1]) != "other", n[0]))[1] if names else None
    if len(names) > 1:
        notes.append(f"several name candidates: {[n for _, n in names]}")
    return name, number, area, notes


# ---------------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------------
def _p(v: NDArray[np.float64] | tuple[float, float]) -> Point2:
    return Point2(x=round(float(v[0]), 6), y=round(float(v[1]), 6))


def build_plan(
    prims: Prims,
    *,
    project: str,
    version: str,
    level_name: str = "Zemin Kat",
    source_doc: str | None = None,
    source: Literal["dxf", "pdf_vector", "raster", "ifc"] = "dxf",
) -> BuildResult:
    method: Method = prims.method
    notes = list(prims.notes)
    segs, rings, mode = _evidence(prims)
    notes.append(f"wall evidence: {mode} ({len(segs)} segments)")
    pieces = _pairs(segs, rings)
    arc_walls, rest = _arc_walls(pieces)
    walls = _merge(rest) + arc_walls
    # keep the connected network(s) of walls: isolated strips (scale bars, frames) are not walls
    walls = _main_network(walls)
    _close_junctions(walls)
    walls = [w for w in walls if w.arc is not None or w.s1 - w.s0 > 0.08]
    openings, unexplained = _openings(walls, prims)

    def f(value: float, conf: float, note: str | None = None) -> Fact[float]:
        return fact(value, method, conf, source_doc=source_doc, note=note)

    assumptions: list[Assumption] = []
    for key, value in DEFAULTS.items():
        assumptions.append(
            Assumption(
                key=f"default/{key}",
                value=value,
                reason="not found in the drawing; architectural default (ARCHITECTURE §S2 step 8)",
                stage="S2",
            )
        )

    def default(key: str) -> Fact[float]:
        return fact(DEFAULTS[key], "default", 0.5, note=f"default/{key}")

    plan_walls: list[Wall] = []
    for i, w in enumerate(walls):
        conf = min(0.95, 0.6 + 0.1 * w.support)
        if w.arc is None:
            cl: Segment | Arc = Segment(a=_p(w.a), b=_p(w.b))
        else:
            cx, cy, r, a0, a1 = w.arc
            cl = Arc(
                center=_p((cx, cy)),
                radius=round(r, 6),
                start_deg=round(a0 % 360, 6),
                end_deg=round(a1 % 360, 6),
            )
        plan_walls.append(
            Wall(
                id=f"W{i + 1}",
                level="L0",
                centerline=cl,
                thickness_m=f(round(w.t, 4), conf),
                height_m=default("wall_height_m"),
                kind="unknown",
            )
        )
    plan_openings: list[Opening] = []
    for k, o in enumerate(openings):
        w = walls[o.wall]
        offset = o.s - w.s0
        door_like = o.kind != "window"
        plan_openings.append(
            Opening(
                id=f"O{k + 1}",
                host_wall=f"W{o.wall + 1}",
                offset_m=f(round(offset, 4), o.confidence),
                width_m=f(round(o.width, 4), o.confidence),
                height_m=default("door_height_m" if door_like else "window_height_m"),
                sill_m=fact(0.0, "default", 0.5, note="door sill")
                if door_like
                else default("window_sill_m"),
                type=o.kind,
                hinge=o.hinge,
                swing=None,
                swing_side=o.swing_side,
            )
        )
    # rooms: holes of the closed wall body
    body = unary_union([_solid(w) for w in walls]) if walls else Polygon()
    polys = list(body.geoms) if hasattr(body, "geoms") else [body]
    holes = [Polygon(h) for p in polys if not p.is_empty for h in p.interiors]
    holes = [h for h in holes if h.area >= 0.5]
    plan_rooms: list[Room] = []
    for i, h in enumerate(sorted(holes, key=lambda q: -q.area)):
        name, number, area, label_notes = _labels(h, prims)
        notes += [f"room {i + 1}: {n}" for n in label_notes]
        ring = list(h.exterior.coords)[:-1]
        if not Polygon(ring).exterior.is_ccw:
            ring.reverse()
        text_conf = 0.9 if name else 0.3
        plan_rooms.append(
            Room(
                id=f"R{i + 1}",
                level="L0",
                polygon=[_p(q) for q in ring],
                name=fact(
                    name or f"Mahal {i + 1}",
                    "pdf_text" if method == "pdf_vector" else method if name else "default",
                    text_conf,
                ),
                number=fact(number, method, 0.9) if number else None,
                type=fact(room_type(name) if name else "other", "derived", text_conf),
                area_label_m2=fact(area, method, 0.9, unit=Unit.M2) if area is not None else None,
                ceiling_height_m=default("ceiling_height_m"),
            )
        )
    _mark_exterior(plan_walls, walls, body)
    plan = PlanGraph(
        version=version,
        project=project,
        levels=[Level(id="L0", name=level_name, elevation_m=0.0)],
        north_angle_deg=fact(0.0, "default", 0.3, note="north not determined yet"),
        walls=plan_walls,
        openings=plan_openings,
        rooms=plan_rooms,
        assumptions=assumptions,
        source=source,
    )
    return BuildResult(plan, notes, len(pieces), unexplained)


def _solid(w: BWall) -> Polygon:
    line = w.line()
    return line.buffer(w.t / 2, cap_style="flat", join_style="mitre")


def _main_network(walls: list[BWall]) -> list[BWall]:
    if len(walls) <= 1:
        return walls
    geoms = [_solid(w).buffer(0.05) for w in walls]
    parent = list(range(len(walls)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    prepared = [prep(g) for g in geoms]
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            if prepared[i].intersects(geoms[j]):
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(len(walls)):
        comps.setdefault(find(i), []).append(i)
    size = {k: sum(walls[i].support for i in v) for k, v in comps.items()}
    top = max(size.values())
    keep = {k for k, v in size.items() if v >= 0.25 * top and len(comps[k]) >= 3}
    return [w for i, w in enumerate(walls) if find(i) in keep]


def _mark_exterior(plan_walls: list[Wall], walls: list[BWall], body: BaseGeometry) -> None:
    polys = list(getattr(body, "geoms", [body]))
    outer = unary_union(
        [Polygon(p.exterior) for p in polys if isinstance(p, Polygon) and not p.is_empty]
    )
    for pw, w in zip(plan_walls, walls, strict=True):
        line = w.line()
        mid = line.interpolate(0.5, normalized=True)
        d = line.project(mid)
        q = line.interpolate(min(line.length, d + 0.01))
        r = line.interpolate(max(0.0, d - 0.01))
        ux, uy = q.x - r.x, q.y - r.y
        n = math.hypot(ux, uy) or 1.0
        nx, ny = -uy / n, ux / n
        off = w.t / 2 + 0.05
        outside = [
            not outer.contains(Point(mid.x + s * nx * off, mid.y + s * ny * off)) for s in (1, -1)
        ]
        pw.kind = "exterior" if any(outside) else "interior"
