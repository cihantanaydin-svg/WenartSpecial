"""On-demand coordinate assist: trigger → hint → snap to measured evidence (ADR-S19).

The assist runs only when a named trigger fires on an extracted raster plan:

- ``validator_failed``: an open room, an unreachable room, an area label that disagrees;
- ``no_candidates_with_ink``: wall-like ink (the vectoriser's wall body) that no extracted wall
  explains;
- ``low_confidence``: a wall whose evidence is weak.

For each trigger (at most :data:`BUDGET` per sheet) a hint source looks at a full-resolution tile
around the trigger and answers with points in page pixels: the two ends of a wall's centre line, or
the two jambs of an opening. A hint is never used as geometry. It defines a search window, and the
element is measured from the page's own evidence inside it:

- a wall from cross-sections of the wall body (or, failing that, two parallel face strokes) along
  the hint: centre line by least squares, thickness from the measured widths;
- an opening from the gap in its host wall's body, its type from the symbol found in the gap
  (a swing arc for a door, glazing lines for a window).

The element is accepted only if the evidence covers ≥ 80 % of it and the fit residual is within
1.5 × the extractor's tolerance. Accepted walls are added to the drawing primitives and the plan is
rebuilt (so junctions, rooms and openings are derived as for any wall); accepted openings are
hosted directly. Both carry ``method="vlm_assisted"`` with the hint and the evidence score, and
block Gate A until a person confirms them (validator ``PLAN_ASSIST_UNCONFIRMED``). Hints without
evidence become suggestions, shown dashed at Gate A and never inserted. Scale never comes from
hints: everything is measured in the page's pixels with the scale the estimators measured.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString, Polygon

from archrender.core.errors import ArchRenderError
from archrender.core.schemas.common import BBox, ModelRef, Point2
from archrender.core.schemas.plan import Opening, PlanGraph, Segment, Wall
from archrender.core.schemas.provenance import AssistTrigger, Fact, Provenance, VlmAssist
from archrender.plan.prims import Polyline, Prims
from archrender.plan.validate import validate_plan

BUDGET = 20  # hint requests per sheet
MIN_COVERAGE = 0.8
RESIDUAL_FACTOR = 1.5
WINDOW_PX_AT_300DPI = 40.0
MAX_TILE_PX = 1536  # the VLM sees tiles at full resolution, never a downscaled sheet
ASSISTED_CONFIDENCE = 0.6  # capped: a person confirms at Gate A
TRIGGER_CODES = {"ROOM_OPEN", "ROOM_UNREACHABLE", "ROOM_AREA_MISMATCH"}
LOW_CONFIDENCE = 0.5
MAX_OPENING_M = 1.6  # widest gap read as an opening inside a snapped wall

HintKind = Literal["wall", "door", "window", "opening"]
Want = Literal["wall", "opening", "any"]


@dataclass(frozen=True)
class Trigger:
    kind: AssistTrigger
    detail: str
    window_px: tuple[float, float, float, float]  # page pixels (x0, y0, x1, y1)
    want: Want = "any"


@dataclass(frozen=True)
class Hint:
    """Two points in page pixels: a wall's centre-line ends, or an opening's jambs."""

    kind: HintKind
    a: tuple[float, float]
    b: tuple[float, float]
    confidence: float = 0.5


class HintSource(Protocol):
    name: str

    def model(self) -> ModelRef | None: ...

    def hints(
        self, tile: NDArray[np.uint8], origin: tuple[int, int], trigger: Trigger
    ) -> list[Hint]:
        """Hints for ``trigger`` from ``tile`` (the page crop starting at ``origin``), in page px."""
        ...


@dataclass
class PageEvidence:
    rgb: NDArray[np.uint8]
    body: NDArray[np.bool_]  # wall body (page px)
    ink: NDArray[np.bool_]  # drawing ink without text
    stroke_px: float
    m_per_px: float
    page_to_plan: NDArray[np.float64]  # 2×3
    dpi: float | None = None

    @property
    def window_px(self) -> float:
        return WINDOW_PX_AT_300DPI * (self.dpi / 300.0 if self.dpi else 1.0)

    @property
    def tolerance_px(self) -> float:
        """The extractor's normal positional tolerance: a stroke width (≥ 1 px)."""
        return max(1.0, self.stroke_px)

    def to_plan(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        m = self.page_to_plan
        return np.asarray(p, float) @ m[:, :2].T + m[:, 2]

    def to_page(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        m = self.page_to_plan
        return (np.asarray(p, float) - m[:, 2]) @ np.linalg.inv(m[:, :2]).T


@dataclass
class Snap:
    hint: Hint
    trigger: Trigger
    accepted: bool
    coverage: float
    residual_px: float
    reason: str
    geometry: dict[str, Any] = field(default_factory=dict)

    def record(self, ev: PageEvidence) -> dict[str, Any]:
        return {
            "kind": self.hint.kind,
            "hint_px": [list(self.hint.a), list(self.hint.b)],
            "hint_plan": ev.to_plan(np.array([self.hint.a, self.hint.b])).round(4).tolist(),
            "trigger": self.trigger.kind,
            "trigger_detail": self.trigger.detail,
            "accepted": self.accepted,
            "coverage": round(self.coverage, 3),
            "residual_px": round(self.residual_px, 2),
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------------------------
# triggers
# ---------------------------------------------------------------------------------------------
def _odd(v: float) -> int:
    return max(3, round(v) // 2 * 2 + 1)


def _box(pts: NDArray[np.float64], pad: float, shape: tuple[int, ...]) -> tuple[float, ...]:
    x0, y0 = pts.min(axis=0) - pad
    x1, y1 = pts.max(axis=0) + pad
    return (
        max(0.0, float(x0)),
        max(0.0, float(y0)),
        min(float(shape[1]), float(x1)),
        min(float(shape[0]), float(y1)),
    )


def walls_mask(plan: PlanGraph, ev: PageEvidence, grow_px: float) -> NDArray[np.bool_]:
    """The plan's wall solids rasterised in page pixels (grown by ``grow_px``)."""
    mask = np.zeros(ev.body.shape, np.uint8)
    k = ev.m_per_px
    for w in plan.walls:
        line = LineString(_centreline(w))
        poly = line.buffer(w.thickness_m.value / 2 + grow_px * k, cap_style="square")
        if poly.is_empty or not isinstance(poly, Polygon):
            continue
        px = ev.to_page(np.array(poly.exterior.coords))
        cv2.fillPoly(mask, [np.round(px * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
    return mask > 0


def _centreline(w: Wall) -> list[tuple[float, float]]:
    from archrender.scene.geometry import centerline_coords

    return centerline_coords(w)


def find_triggers(plan: PlanGraph, ev: PageEvidence) -> list[Trigger]:
    out: list[Trigger] = []
    pad = 2 * ev.window_px
    rooms = {r.id: r for r in plan.rooms}
    for issue in validate_plan(plan):
        if issue.code not in TRIGGER_CODES:
            continue
        pts = [(p.x, p.y) for rid in issue.element_ids if rid in rooms for p in rooms[rid].polygon]
        if not pts and issue.location is not None:
            pts = [(issue.location.x, issue.location.y)]
        if not pts:
            continue
        px = ev.to_page(np.array(pts, float))
        out.append(
            Trigger(
                "validator_failed",
                f"{issue.code} {','.join(issue.element_ids)}".strip(),
                _box(px, pad, ev.body.shape),  # type: ignore[arg-type]
                "wall" if issue.code in ("ROOM_OPEN", "ROOM_AREA_MISMATCH") else "opening",
            )
        )
    # wall body that no extracted wall explains, joined to the walls (isolated ink such as a scale
    # bar or a filled north arrow is not a wall the builder missed)
    explained = walls_mask(plan, ev, 2 * ev.stroke_px)
    unexplained = ev.body & ~explained
    near = cv2.dilate(
        explained.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_odd(4 * ev.stroke_px), _odd(4 * ev.stroke_px))
        ),
    ).astype(bool)
    u8 = cv2.morphologyEx(
        unexplained.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    n, lab, st, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    touching = np.zeros(n, bool)
    touching[np.unique(lab[near & (lab > 0)])] = True
    min_len = 0.4 / ev.m_per_px
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        if max(bw, bh) < min_len or area < 0.05 * min_len * min_len / 4 or not touching[i]:
            continue
        corners = np.array([[x, y], [x + bw, y + bh]], float)
        out.append(
            Trigger(
                "no_candidates_with_ink",
                f"wall body {max(bw, bh) * ev.m_per_px:.2f} m long without a wall",
                _box(corners, pad, ev.body.shape),  # type: ignore[arg-type]
                "wall",
            )
        )
    for w in plan.walls:
        if w.thickness_m.confidence < LOW_CONFIDENCE:
            px = ev.to_page(np.array(_centreline(w)))
            out.append(
                Trigger(
                    "low_confidence",
                    f"wall {w.id}",
                    _box(px, pad, ev.body.shape),  # type: ignore[arg-type]
                    "wall",
                )
            )
    return _dedupe(out)


def _dedupe(triggers: list[Trigger]) -> list[Trigger]:
    out: list[Trigger] = []
    for t in triggers:
        a = t.window_px
        dup = False
        for u in out:
            b = u.window_px
            ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if small > 0 and ix * iy >= 0.8 * small and u.want == t.want:
                dup = True
                break
        if not dup:
            out.append(t)
    return out


# ---------------------------------------------------------------------------------------------
# snapping
# ---------------------------------------------------------------------------------------------
def _sample(mask: NDArray[np.bool_], pts: NDArray[np.float64]) -> NDArray[np.bool_]:
    h, w = mask.shape
    xi = np.round(pts[..., 0]).astype(int)
    yi = np.round(pts[..., 1]).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(xi.shape, bool)
    out[ok] = mask[yi[ok], xi[ok]]
    return out


def _runs(v: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """[start, end) index runs of True."""
    d = np.diff(np.concatenate([[0], v.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist(), strict=True))


def _cross_section(
    body_line: NDArray[np.bool_],
    ink_line: NDArray[np.bool_],
    offs: NDArray[np.float64],
    ev: PageEvidence,
) -> tuple[float, float] | None:
    """(centre offset, outer width) in px of the wall crossing this normal line nearest to 0."""
    k = ev.m_per_px
    best: tuple[float, float] | None = None
    for s, e in _runs(body_line):
        width = offs[e - 1] - offs[s] + 1
        if not 0.05 / k <= width - ev.stroke_px <= 0.6 / k:
            continue
        c = (offs[s] + offs[e - 1]) / 2
        if best is None or abs(c) < abs(best[0]):
            best = (c, width)
    if best is not None:
        return best
    # an outline wall missing from the body: two thin face strokes at a wall's distance
    thin = [
        ((offs[s] + offs[e - 1]) / 2)
        for s, e in _runs(ink_line)
        if offs[e - 1] - offs[s] + 1 <= 2.5 * ev.stroke_px
    ]
    for i, ci in enumerate(thin):
        for cj in thin[i + 1 :]:
            gap = cj - ci
            if 0.05 / k <= gap <= 0.6 / k:
                c = (ci + cj) / 2
                if best is None or abs(c) < abs(best[0]):
                    best = (c, gap + ev.stroke_px)
    return best


def _close_gaps(v: NDArray[np.bool_], max_gap: float) -> NDArray[np.bool_]:
    """Fill False runs shorter than ``max_gap`` between True runs (doors, windows, crossings)."""
    out = v.copy()
    runs = _runs(v)
    for (_, e), (s2, _) in itertools.pairwise(runs):
        if s2 - e <= max_gap:
            out[e:s2] = True
    return out


def snap_wall(hint: Hint, trigger: Trigger, ev: PageEvidence, existing: NDArray[np.bool_]) -> Snap:
    """A wall measured along a hint (see the module doc).

    1. Cross-sections of the wall body (or two face strokes) along the hint, ±window either side;
       a robust line fit through their centres (junctions and crossings are outliers).
    2. Evidence at each station of the fitted line: a consistent cross-section, or wall body on
       the line itself (two of three samples across its middle).
    3. The wall spans the evidence overlapping the hint, with gaps of an opening's width between
       evidence on both sides read as openings, as the builder reads them.
    4. A new wall needs evidence over ≥ 80 % of the hinted span. A wall that is mostly in the
       plan already counts only where it extends that wall: the extension must be drawn wall
       body, or an opening (door/window symbol) between walls.
    """
    a, b = np.array(hint.a, float), np.array(hint.b, float)
    length = float(np.hypot(*(b - a)))
    if length * ev.m_per_px < 0.3:
        return Snap(hint, trigger, False, 0.0, 0.0, "hint shorter than 0.3 m")
    d = (b - a) / length
    n = np.array([-d[1], d[0]])
    win = ev.window_px
    ts = np.arange(-win, length + win + 1e-9, 1.0)  # the hinted ends may be off by the window
    offs = np.arange(-win, win + 0.5, 0.5)
    grid = a + ts[:, None, None] * d + offs[None, :, None] * n  # (T, O, 2)
    body, ink = _sample(ev.body, grid), _sample(ev.ink, grid)
    cs = [_cross_section(body[i], ink[i], offs, ev) for i in range(len(ts))]
    have = np.array([c is not None for c in cs])
    inner = (ts >= 0) & (ts <= length)
    if not have[inner].any():
        return Snap(hint, trigger, False, 0.0, 0.0, "no wall evidence along the hint")
    # 1. robust fit
    tt = ts[have]
    cc = np.array([c[0] for c in cs if c is not None])
    ww = np.array([c[1] for c in cs if c is not None])
    keep = np.abs(ww - np.median(ww)) <= max(2.0, 0.3 * float(np.median(ww)))
    for _ in range(3):
        if int(keep.sum()) < 4:
            return Snap(hint, trigger, False, 0.0, 0.0, "evidence too short")
        beta, alpha = np.polyfit(tt[keep], cc[keep], 1)
        dev = cc - (alpha + beta * tt)
        mad = float(np.median(np.abs(dev[keep])))
        keep &= np.abs(dev) <= max(1.5, 3.0 * 1.4826 * mad)
    if int(keep.sum()) < 4:
        return Snap(hint, trigger, False, 0.0, 0.0, "evidence too short")
    beta, alpha = np.polyfit(tt[keep], cc[keep], 1)
    residual = float(np.sqrt(np.mean((cc[keep] - (alpha + beta * tt[keep])) ** 2)))
    thickness_px = float(np.median(ww[keep])) - ev.stroke_px
    t_m = thickness_px * ev.m_per_px
    line = a + ts[:, None] * d + (alpha + beta * ts)[:, None] * n
    # 2. evidence per station
    ok = np.zeros(len(ts), bool)
    ok[np.nonzero(have)[0][keep]] = True
    q = 0.25 * thickness_px
    votes = (
        _sample(ev.body, line).astype(int)
        + _sample(ev.body, line + q * n)
        + _sample(ev.body, line - q * n)
    )
    drawn = ok | (votes >= 2)
    # 3. the span
    bridged = _close_gaps(drawn, MAX_OPENING_M / ev.m_per_px)
    runs = [(i, j) for i, j in _runs(bridged) if inner[i:j].any()]
    if not runs:
        return Snap(hint, trigger, False, 0.0, residual, "no wall evidence along the hint")
    s0, e0 = max(runs, key=lambda r: int(inner[r[0] : r[1]].sum()))
    span_inner = np.zeros(len(ts), bool)
    span_inner[s0:e0] = inner[s0:e0]
    coverage = float(np.sum(bridged & span_inner)) / max(1, int(np.sum(inner)))
    if float(np.sum(drawn & inner)) < 0.5 * float(np.sum(inner)):
        coverage = min(coverage, float(np.sum(drawn & inner)) / max(1, int(np.sum(inner))))
    known = _sample(existing, line)
    # the drawn pieces become wall primitives (the gaps between them stay openings); the end
    # pieces run on to the middle of a wall they meet (the junction is drawn as that wall)
    pieces = [(s0 + i, s0 + j) for i, j in _runs(drawn[s0:e0]) if (j - i) * ev.m_per_px >= 0.05]
    reach = int(win)
    if pieces:
        before = [r for r in _runs(known[: pieces[0][0] + 1]) if r[1] >= pieces[0][0] - reach]
        if before:
            pieces[0] = (min(pieces[0][0], (before[-1][0] + before[-1][1]) // 2), pieces[0][1])
        last = pieces[-1][1] - 1
        after = [(last + i, last + j) for i, j in _runs(known[last:]) if i <= reach]
        if after:
            pieces[-1] = (pieces[-1][0], max(pieces[-1][1], (after[0][0] + after[0][1]) // 2))
    geom = {
        "a_px": line[s0].tolist(),
        "b_px": line[e0 - 1].tolist(),
        "thickness_m": round(t_m, 4),
        "pieces_px": [[line[i].tolist(), line[j - 1].tolist()] for i, j in pieces],
    }
    if residual > RESIDUAL_FACTOR * ev.tolerance_px:
        return Snap(hint, trigger, False, coverage, residual, "fit residual too large", geom)
    # 4. new wall, or the extension of one
    if float(np.mean(known[s0:e0])) <= 0.5:
        if coverage < MIN_COVERAGE:
            return Snap(hint, trigger, False, coverage, residual, "evidence covers < 80 %", geom)
        return Snap(hint, trigger, True, coverage, residual, "snapped to the wall body", geom)
    for i, j in _runs(~known[s0:e0]):
        i, j = s0 + i, s0 + j
        if (j - i) * ev.m_per_px < 0.15:
            continue
        seg = drawn[i:j]
        left = bool(known[max(0, i - reach) : i].any())
        right = bool(known[j : j + reach].any())
        ext = seg.copy()
        kinds: list[str] = []
        plain = False
        for g0, g1 in _runs(~seg):
            width = (g1 - g0) * ev.m_per_px
            walled = (g0 > 0 or left) and (g1 < len(seg) or right)
            if not walled or not 0.5 <= width <= MAX_OPENING_M:
                continue
            q0, q1 = ev.to_plan(line[i + g0]), ev.to_plan(line[i + g1 - 1])
            gw = float(np.hypot(*(q1 - q0)))
            u = (q1 - q0) / max(1e-9, gw)
            kind, cov, _ = _opening_symbol(ev, q0, u, np.array([-u[1], u[0]]), 0.0, gw, t_m)
            if cov >= MIN_COVERAGE:
                ext[g0:g1] = True
                kinds.append(f"{kind} {width:.2f} m")
                plain |= kind == "opening"
        cont, body_frac = float(ext.mean()), float(seg.mean())
        # a gap with a door or window symbol is evidence in itself; a bare gap is not, so the
        # extension must then be mostly drawn wall
        if cont >= MIN_COVERAGE and (body_frac >= 0.5 or not plain):
            across = f" (across {', '.join(kinds)})" if kinds else ""
            return Snap(
                hint,
                trigger,
                True,
                cont,
                residual,
                f"extends a wall of the plan by {(j - i) * ev.m_per_px:.2f} m{across}",
                geom,
            )
    return Snap(hint, trigger, False, coverage, residual, "already a wall of the plan", geom)


def _host(
    plan: PlanGraph, mid: NDArray[np.float64], dirn: NDArray[np.float64] | None
) -> Wall | None:
    best: tuple[float, Wall] | None = None
    for w in plan.walls:
        cl = w.centerline
        if not isinstance(cl, Segment):
            continue
        a = np.array([cl.a.x, cl.a.y])
        b = np.array([cl.b.x, cl.b.y])
        u = (b - a) / max(1e-9, float(np.hypot(*(b - a))))
        if dirn is not None and abs(float(np.dot(u, dirn))) < 0.9:
            continue
        t = float(np.clip(np.dot(mid - a, u), 0, np.hypot(*(b - a))))
        dist = float(np.hypot(*(a + t * u - mid)))
        if best is None or dist < best[0]:
            best = (dist, w)
    if best is None:
        return None
    return best[1]


def snap_opening(hint: Hint, trigger: Trigger, plan: PlanGraph, ev: PageEvidence) -> Snap:
    k = ev.m_per_px
    pa, pb = ev.to_plan(np.array([hint.a, hint.b]))
    span = float(np.hypot(*(pb - pa)))
    mid = (pa + pb) / 2
    host = _host(plan, mid, (pb - pa) / span if span > 0.2 else None)
    if host is None:
        return Snap(hint, trigger, False, 0.0, 0.0, "no straight wall to host it")
    cl = host.centerline
    assert isinstance(cl, Segment)
    a = np.array([cl.a.x, cl.a.y])
    b = np.array([cl.b.x, cl.b.y])
    wl = float(np.hypot(*(b - a)))
    u = (b - a) / wl
    nrm = np.array([-u[1], u[0]])
    t = host.thickness_m.value
    if abs(float(np.dot(mid - a, nrm))) > t / 2 + ev.window_px * k:
        return Snap(hint, trigger, False, 0.0, 0.0, "the hint is not on a wall")
    s_mid = float(np.dot(mid - a, u))
    half = max(span, 0.6) / 2 + ev.window_px * k
    ss = np.arange(max(0.0, s_mid - half), min(wl, s_mid + half), k)
    if len(ss) < 4:
        return Snap(hint, trigger, False, 0.0, 0.0, "the hint is outside its host wall")
    # wall material across the thickness at each station; face lines for the residual
    across = np.linspace(-0.35 * t, 0.35 * t, 7)
    pts = a + ss[:, None, None] * u + across[None, :, None] * nrm
    present = _sample(ev.body, ev.to_page(pts.reshape(-1, 2)).reshape(pts.shape)).mean(axis=1) > 0.5
    gaps = [(s, e) for s, e in _runs(~present) if s > 0 and e < len(ss)]  # jambs on both sides
    if not gaps:
        return Snap(hint, trigger, False, 0.0, 0.0, "no gap with jambs in the wall body")
    i_mid = int(np.argmin(np.abs(ss - s_mid)))
    s0, e0 = min(
        gaps, key=lambda g: 0 if g[0] <= i_mid < g[1] else min(abs(g[0] - i_mid), abs(g[1] - i_mid))
    )
    g0, g1 = float(ss[s0]), float(ss[e0 - 1]) + k
    width = g1 - g0
    if not 0.5 <= width <= 3.0:
        return Snap(hint, trigger, False, 0.0, 0.0, f"gap {width:.2f} m is not an opening width")
    # residual: the jambs measured on each face line separately
    jambs = []
    for side in (-0.35 * t, 0.35 * t):
        line = a + ss[:, None] * u + side * nrm
        pres = _sample(ev.body, ev.to_page(line))
        runs = [(s, e) for s, e in _runs(~pres) if s <= e0 and e >= s0]
        if runs:
            s1, e1 = min(runs, key=lambda g: abs(g[0] - s0) + abs(g[1] - e0))
            jambs.append((float(ss[s1]), float(ss[e1 - 1]) + k))
    residual = (
        max(abs(jambs[0][0] - jambs[1][0]), abs(jambs[0][1] - jambs[1][1])) / k
        if len(jambs) == 2
        else math.inf
    )
    for o in plan.openings:
        if o.host_wall == host.id:
            c, w = o.offset_m.value, o.width_m.value
            if min(g1, c + w / 2) - max(g0, c - w / 2) > 0.5 * min(width, w):
                return Snap(hint, trigger, False, 1.0, residual, "already an opening of the plan")
    # the symbol in the gap decides the type
    kind, coverage, extra = _opening_symbol(ev, a, u, nrm, g0, g1, t)
    geom = {
        "host_wall": host.id,
        "offset_m": round((g0 + g1) / 2, 4),
        "width_m": round(width, 4),
        "type": kind,
        **extra,
    }
    if coverage < MIN_COVERAGE:
        return Snap(
            hint, trigger, False, coverage, residual, f"{kind} evidence covers < 80 %", geom
        )
    if residual > RESIDUAL_FACTOR * ev.tolerance_px:
        return Snap(
            hint, trigger, False, coverage, residual, "jambs disagree between the faces", geom
        )
    return Snap(hint, trigger, True, coverage, residual, f"snapped to a {kind} in the wall", geom)


def _opening_symbol(
    ev: PageEvidence,
    a: NDArray[np.float64],
    u: NDArray[np.float64],
    nrm: NDArray[np.float64],
    g0: float,
    g1: float,
    t: float,
) -> tuple[str, float, dict[str, Any]]:
    k = ev.m_per_px
    ink = cv2.dilate(
        ev.ink.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    ss = np.arange(g0 + 2 * k, g1 - 2 * k, k)
    # window: glazing lines run inside the wall band along the whole gap
    offsets = np.linspace(-0.4 * t, 0.4 * t, 9)
    band = a + ss[:, None, None] * u + offsets[None, :, None] * nrm
    glazing = _sample(ink, ev.to_page(band.reshape(-1, 2)).reshape(band.shape))
    win_cov = float(glazing.any(axis=1).mean()) if len(ss) else 0.0
    # door: a quarter swing about one jamb, on either side; its centre on the centre line or on
    # the face (drafting conventions differ) and its radius the leaf width (≈ the gap measured)
    width = g1 - g0
    best = (0.0, "start", "pos")
    ang = np.radians(np.linspace(8, 82, 38))
    for hinge, origin, along in (("start", g0, 1.0), ("end", g1, -1.0)):
        for side, sgn in (("pos", 1.0), ("neg", -1.0)):
            c = a + origin * u
            for off in (0.0, t / 2):
                for rad in width * np.array([0.9, 1.0, 1.1, 1.2, 1.3]):
                    pts = (
                        c
                        + (rad * np.cos(ang))[:, None] * (along * u)
                        + (sgn * (off + rad * np.sin(ang)))[:, None] * nrm
                    )
                    cov = float(_sample(ink, ev.to_page(pts)).mean())
                    if cov > best[0]:
                        best = (cov, hinge, side)
    if best[0] >= win_cov and best[0] >= MIN_COVERAGE:
        return "door", best[0], {"hinge": best[1], "swing_side": best[2]}
    if win_cov >= MIN_COVERAGE:
        return "window", win_cov, {}
    # a plain passage: the gap itself is the evidence (clean of wall body along its length)
    clean = a + ss[:, None] * u
    gap_cov = float(1.0 - _sample(ev.body, ev.to_page(clean)).mean()) if len(ss) else 0.0
    return "opening", gap_cov, {}


# ---------------------------------------------------------------------------------------------
# hint sources
# ---------------------------------------------------------------------------------------------
def tile_for(ev: PageEvidence, trigger: Trigger) -> tuple[NDArray[np.uint8], tuple[int, int]]:
    """The full-resolution page crop for a trigger (centred, at most MAX_TILE_PX a side)."""
    h, w = ev.rgb.shape[:2]
    x0, y0, x1, y1 = trigger.window_px
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sw = min(MAX_TILE_PX, max(64.0, x1 - x0))
    sh = min(MAX_TILE_PX, max(64.0, y1 - y0))
    ox = int(max(0, min(w - sw, cx - sw / 2)))
    oy = int(max(0, min(h - sh, cy - sh / 2)))
    return ev.rgb[oy : oy + int(sh), ox : ox + int(sw)], (ox, oy)


class VlmHintSource:
    """The VLM role answering schema-constrained point questions on a tile (UNVERIFIED-ON-GPU)."""

    name = "vlm"

    def __init__(self, vlm: Any) -> None:
        self.vlm = vlm

    def model(self) -> ModelRef | None:
        return self.vlm.ref()  # type: ignore[no-any-return]

    def hints(
        self, tile: NDArray[np.uint8], origin: tuple[int, int], trigger: Trigger
    ) -> list[Hint]:
        what = {
            "wall": "the walls (two end points of each wall's centre line)",
            "opening": "the doors and windows (the two jambs of each opening)",
            "any": "the walls (centre-line ends) and openings (two jambs)",
        }[trigger.want]
        question = (
            f"Extraction problem here: {trigger.detail}. Locate {what} that a careful drafter "
            "would see in this tile. Only elements that are actually drawn; none is a valid answer."
        )
        answer = self.vlm.locate_elements(tile, question)
        out = []
        ox, oy = origin
        for el in answer.get("elements", []):
            pts = el.get("points") or []
            if len(pts) != 2 or el.get("kind") not in ("wall", "door", "window", "opening"):
                continue
            (x0, y0), (x1, y1) = pts
            out.append(
                Hint(
                    el["kind"],
                    (float(x0) + ox, float(y0) + oy),
                    (float(x1) + ox, float(y1) + oy),
                    float(el.get("confidence", 0.5)),
                )
            )
        return out


# ---------------------------------------------------------------------------------------------
# the assist pass
# ---------------------------------------------------------------------------------------------
@dataclass
class AssistResult:
    plan: PlanGraph
    prims: Prims
    triggers: list[Trigger]
    snaps: list[Snap]
    calls: int
    source: str | None
    log: list[dict[str, Any]]

    def report(self, ev: PageEvidence, page_id: str = "") -> dict[str, Any]:
        """The extraction record (suggestion ids are ``<page id>/sg<n>``)."""
        accepted = [s for s in self.snaps if s.accepted]
        return {
            "source": self.source,
            "triggers": [
                {
                    "kind": t.kind,
                    "detail": t.detail,
                    "window_px": [round(v, 1) for v in t.window_px],
                }
                for t in self.triggers
            ],
            "calls": self.calls,
            "budget": BUDGET,
            "hints": len(self.snaps),
            "accepted": len(accepted),
            "suggestions": [
                {"id": f"{page_id}/sg{i + 1}", "page": page_id, **s.record(ev)}
                for i, s in enumerate(x for x in self.snaps if not x.accepted)
            ],
            "accepted_elements": [s.record(ev) | {"geometry": s.geometry} for s in accepted],
            "log": self.log,
        }


def _assist_prov(snap: Snap, model: ModelRef | None, source_doc: str | None) -> Provenance:
    return Provenance(
        method="vlm_assisted",
        confidence=ASSISTED_CONFIDENCE,
        source_doc=source_doc,
        model=model,
        bbox_px=BBox(
            x0=min(snap.hint.a[0], snap.hint.b[0]),
            y0=min(snap.hint.a[1], snap.hint.b[1]),
            x1=max(snap.hint.a[0], snap.hint.b[0]),
            y1=max(snap.hint.a[1], snap.hint.b[1]),
        ),
        assist=VlmAssist(
            trigger=snap.trigger.kind,
            hint_points_px=[
                Point2(x=snap.hint.a[0], y=snap.hint.a[1]),
                Point2(x=snap.hint.b[0], y=snap.hint.b[1]),
            ],
            evidence_coverage=round(min(1.0, snap.coverage), 4),
            snap_residual_px=round(snap.residual_px, 3),
            accepted=True,
        ),
        note=snap.reason,
    )


def _assisted_fact(value: float, prov: Provenance) -> Fact[float]:
    return Fact(value=value, provenance=[prov], status="extracted")


def _same_line(p0: NDArray[np.float64], p1: NDArray[np.float64], w: Wall, tol: float) -> float:
    """Fraction of wall ``w``'s centre line lying within ``tol`` of the segment p0–p1 (and
    parallel to it)."""
    cl = w.centerline
    if not isinstance(cl, Segment):
        return 0.0
    a = np.array([cl.a.x, cl.a.y])
    b = np.array([cl.b.x, cl.b.y])
    la = float(np.hypot(*(b - a)))
    lp = float(np.hypot(*(p1 - p0)))
    if la < 1e-6 or lp < 1e-6:
        return 0.0
    u = (p1 - p0) / lp
    if abs(float(np.dot((b - a) / la, u))) < math.cos(math.radians(5)):
        return 0.0
    nrm = np.array([-u[1], u[0]])
    if max(abs(float(np.dot(a - p0, nrm))), abs(float(np.dot(b - p0, nrm)))) > tol:
        return 0.0
    ta, tb = sorted((float(np.dot(a - p0, u)), float(np.dot(b - p0, u))))
    return max(0.0, min(tb, lp + tol) - max(ta, -tol)) / la


def _unchanged(w: Wall, old: Wall, tol: float = 0.03) -> bool:
    a, b = w.centerline, old.centerline
    if not (isinstance(a, Segment) and isinstance(b, Segment)):
        return a == b
    ends = np.array([[a.a.x, a.a.y], [a.b.x, a.b.y]])
    olds = np.array([[b.a.x, b.a.y], [b.b.x, b.b.y]])
    return (
        bool(np.abs(ends - olds).max() <= tol or np.abs(ends - olds[::-1]).max() <= tol)
        and abs(w.thickness_m.value - old.thickness_m.value) <= tol
    )


def _mark_walls(
    before: PlanGraph,
    rebuilt: PlanGraph,
    snaps: list[Snap],
    ev: PageEvidence,
    model: ModelRef | None,
    source_doc: str | None,
) -> PlanGraph:
    """Walls of the rebuilt plan that come from a snapped hint (and were not there before) get
    ``vlm_assisted`` provenance."""
    plan = rebuilt.model_copy(deep=True)
    for s in snaps:
        p0 = ev.to_plan(np.array(s.geometry["a_px"]))
        p1 = ev.to_plan(np.array(s.geometry["b_px"]))
        tol = max(0.05, s.geometry["thickness_m"])
        prov = _assist_prov(s, model, source_doc)
        for w in plan.walls:
            if _same_line(p0, p1, w, tol) < 0.5:
                continue
            if any(_unchanged(w, old) for old in before.walls):
                continue  # the rebuild left it as it was: still the extractor's wall
            w.thickness_m = _assisted_fact(w.thickness_m.value, prov)
    return PlanGraph.model_validate(plan.model_dump())


def _add_openings(
    plan: PlanGraph, snaps: list[Snap], model: ModelRef | None, source_doc: str | None
) -> PlanGraph:
    from archrender.core.schemas.provenance import fact
    from archrender.plan.builder import DEFAULTS

    if not snaps:
        return plan
    plan = plan.model_copy(deep=True)
    for i, s in enumerate(snaps):
        g = s.geometry
        prov = _assist_prov(s, model, source_doc)
        door_like = g["type"] != "window"

        def default(key: str) -> Fact[float]:
            return fact(DEFAULTS[key], "default", 0.5, note=f"default/{key}")

        plan.openings.append(
            Opening(
                id=f"OA{i + 1}",
                host_wall=g["host_wall"],
                offset_m=_assisted_fact(g["offset_m"], prov),
                width_m=_assisted_fact(g["width_m"], prov),
                height_m=default("door_height_m" if door_like else "window_height_m"),
                sill_m=fact(0.0, "default", 0.5, note="door sill")
                if door_like
                else default("window_sill_m"),
                type=g["type"],
                hinge=g.get("hinge"),
                swing=None,
                swing_side=g.get("swing_side"),
            )
        )
    return PlanGraph.model_validate(plan.model_dump())


def run_assist(
    plan: PlanGraph,
    prims: Prims,
    ev: PageEvidence,
    source: HintSource | None,
    rebuild: Callable[[Prims], PlanGraph],
    *,
    source_doc: str | None = None,
    budget: int = BUDGET,
) -> AssistResult:
    """Triggers → hints → snaps → the plan with the accepted elements (see the module doc)."""
    triggers = find_triggers(plan, ev)
    if source is None or not triggers:
        return AssistResult(plan, prims, triggers, [], 0, source.name if source else None, [])
    model = source.model()
    log: list[dict[str, Any]] = []
    asked: list[tuple[Trigger, Hint]] = []
    calls = 0
    for trig in triggers:
        if calls >= budget:
            log.append({"trigger": trig.kind, "detail": trig.detail, "skipped": "budget"})
            continue
        tile, origin = tile_for(ev, trig)
        calls += 1
        try:
            hints = source.hints(tile, origin, trig)
        except ArchRenderError as e:  # the assist is optional: a failed call is logged, not fatal
            log.append({"trigger": trig.kind, "detail": trig.detail, "error": e.message})
            continue
        log.append(
            {
                "trigger": trig.kind,
                "detail": trig.detail,
                "tile_px": [
                    origin[0],
                    origin[1],
                    origin[0] + tile.shape[1],
                    origin[1] + tile.shape[0],
                ],
                "hints": [{"kind": h.kind, "a": list(h.a), "b": list(h.b)} for h in hints],
            }
        )
        asked += [(trig, h) for h in hints]
    # walls first, rebuilt from the measured rectangles as primitives, so junctions, rooms and
    # the openings in them are derived as for any other wall
    existing = walls_mask(plan, ev, 0.0)
    wall_snaps = [snap_wall(h, t, ev, existing) for t, h in asked if h.kind == "wall"]
    new_walls = [s for s in wall_snaps if s.accepted]
    out_prims = prims
    if new_walls:
        out_prims = Prims(
            method=prims.method,
            polylines=list(prims.polylines),
            arcs=prims.arcs,
            texts=prims.texts,
            dims=prims.dims,
            notes=[*prims.notes, f"{len(new_walls)} wall(s) added from snapped hints"],
            resolution=prims.resolution,
        )
        for i, s in enumerate(new_walls):
            for k, (q0, q1) in enumerate(s.geometry["pieces_px"]):
                p0, p1 = ev.to_plan(np.array(q0)), ev.to_plan(np.array(q1))
                rect = LineString([p0, p1]).buffer(
                    s.geometry["thickness_m"] / 2, cap_style="flat", join_style="mitre"
                )
                out_prims.polylines.append(
                    Polyline(
                        np.array(rect.exterior.coords)[:-1],
                        closed=True,
                        fill=True,
                        group=f"assist{i}.{k}",
                    )
                )
        plan = _mark_walls(plan, rebuild(out_prims), new_walls, ev, model, source_doc)
    # then openings, hosted on the (rebuilt) walls; overlapping accepted openings keep the first
    opening_snaps: list[Snap] = []
    for t, h in asked:
        if h.kind == "wall":
            continue
        snap = snap_opening(h, t, plan, ev)
        if snap.accepted:
            g = snap.geometry
            for o in opening_snaps:
                og = o.geometry
                if (
                    o.accepted
                    and og["host_wall"] == g["host_wall"]
                    and abs(og["offset_m"] - g["offset_m"]) < (og["width_m"] + g["width_m"]) / 2
                ):
                    snap.accepted, snap.reason = False, "duplicate of another snapped hint"
                    break
        opening_snaps.append(snap)
    plan = _add_openings(plan, [s for s in opening_snaps if s.accepted], model, source_doc)
    return AssistResult(
        plan, out_prims, triggers, wall_snaps + opening_snaps, calls, source.name, log
    )
