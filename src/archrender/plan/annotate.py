"""Plan annotations from the other sheets of the set (ARCHITECTURE §S2 steps 6–7).

- **Ceiling heights from the reflected ceiling plan.** The RCP is extracted like a floor plan, then
  registered onto the floor plan of its level (similarity transform, ICP on the wall centre lines;
  RMS residual > 20 mm is reported). Each height label ("+2,70", "h=2.80", "TH 3.00") lands in a
  floor-plan room; that room's ceiling height becomes the measured value (a double-height space is
  flagged), and the walls around it reach the highest ceiling they bound.
- **Opening tags and schedules.** S1 found the door/window tags (K1, P3 …) on the plan page; each
  tag goes to the nearest compatible opening. A door/window schedule row for that tag gives the
  opening's height and sill (replacing defaults) and its width: when the schedule and the drawing
  disagree by more than 3 cm, the schedule value is used (it is the specification) and the
  disagreement is a Gate A conflict.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon

from archrender.core.schemas.common import Severity
from archrender.core.schemas.plan import PlanGraph, Segment, ValidationIssue
from archrender.core.schemas.provenance import Conflict, Fact, Method, Provenance, fact
from archrender.plan.prims import TextPrim
from archrender.scene.geometry import centerline_coords

REGISTRATION_MAX_RMS_M = 0.02
WIDTH_TOLERANCE_M = 0.03
TAG_REACH_M = 1.5
# "+2,70", "h=2.80", "TH 3.00", "tavan +2,60" (a bare number is not a ceiling label)
HEIGHT_RE = re.compile(
    r"^(?:(?:h|th|tav|tavan|ceiling|clg|c\.?h)\s*[=:.]?\s*\+?|\+)\s*(\d{1,2}[.,]\d{1,3})\s*(?:m)?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------------------------
def wall_samples(
    plan: PlanGraph, level: str | None = None, step: float = 0.1
) -> NDArray[np.float64]:
    pts: list[NDArray[np.float64]] = []
    for w in plan.walls:
        if level is not None and w.level != level:
            continue
        line = LineString(centerline_coords(w))
        n = max(2, int(line.length / step) + 1)
        pts.append(
            np.array([line.interpolate(i / (n - 1), normalized=True).coords[0] for i in range(n)])
        )
    return np.vstack(pts) if pts else np.zeros((0, 2))


def _dominant(pts: NDArray[np.float64]) -> float:
    """The plan's main wall direction modulo 90° (radians), from consecutive samples along the
    walls (circular mean of 4θ)."""
    d = np.diff(pts, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    step = float(np.median(n)) if len(n) else 0.0
    d = d[(n > 0) & (n < 2 * step)]
    if not len(d):
        return 0.0
    a = 4 * np.arctan2(d[:, 1], d[:, 0])
    return float(math.atan2(np.sin(a).sum(), np.cos(a).sum()) / 4)


def _umeyama(src: NDArray[np.float64], dst: NDArray[np.float64]) -> NDArray[np.float64]:
    """Least-squares similarity src → dst as a 2×3 matrix."""
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    var = float((a**2).sum() / len(src))
    k = float(np.trace(np.diag(s) @ d) / var) if var > 0 else 1.0
    t = md - k * r @ ms
    return np.hstack([k * r, t[:, None]])


def _apply(m: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
    return p @ m[:, :2].T + m[:, 2]


@dataclass
class Registration:
    matrix: NDArray[np.float64]  # source plan → target plan
    rms_m: float
    inliers: float  # fraction of source samples matched


def register(src: NDArray[np.float64], dst: NDArray[np.float64]) -> Registration | None:
    """Similarity transform taking wall samples ``src`` onto ``dst`` (ICP from four orientations;
    the best final fit wins). Sample ``dst`` densely (≈ 1 cm): the residual is then a distance to
    the target's centre lines, not to its nearest sample."""
    if len(src) < 10 or len(dst) < 10:
        return None
    tree = cKDTree(dst)
    best: Registration | None = None
    scale0 = math.sqrt(dst.var(axis=0).sum() / max(1e-12, src.var(axis=0).sum()))
    offset = _dominant(dst) - _dominant(src)
    starts = [math.radians(r) for r in (0, 90, 180, 270)]
    if abs(offset) > math.radians(1):  # the main wall directions differ: also start aligned
        starts += [a + offset for a in starts]
    for th in starts:
        r = scale0 * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        m = np.hstack([r, (dst.mean(axis=0) - r @ src.mean(axis=0))[:, None]])
        keep = np.ones(len(src), bool)
        for _ in range(40):
            moved = _apply(m, src)
            d, idx = tree.query(moved)
            cut = max(0.05, 3.0 * float(np.median(d)))
            keep = d <= cut
            if keep.sum() < 10:
                break
            new = _umeyama(src[keep], dst[idx[keep]])
            if np.abs(new - m).max() < 1e-7:
                m = new
                break
            m = new
        d, _ = tree.query(_apply(m, src))
        inl = d <= 0.05
        rms = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else math.inf
        reg = Registration(m, rms, float(inl.mean()))
        if best is None or (reg.inliers, -reg.rms_m) > (best.inliers, -best.rms_m):
            best = reg
    return best


# ---------------------------------------------------------------------------------------------
# ceiling heights from the RCP
# ---------------------------------------------------------------------------------------------
def height_labels(texts: list[TextPrim]) -> list[tuple[float, float, float]]:
    """(x, y, metres) of the ceiling height labels among an RCP's texts (plan frame)."""
    from archrender.understand.text import parse_number

    out = []
    for t in texts:
        m = HEIGHT_RE.match(t.text.strip())
        if not m:
            continue
        v = parse_number(m.group(1))
        if v is not None and 2.0 <= v <= 15.0:
            out.append((t.x, t.y, v))
    return out


def rcp_heights(
    plan: PlanGraph,
    level: str,
    rcp: PlanGraph,
    texts: list[TextPrim],
    *,
    source_doc: str | None,
    method: Method,
    page_id: str,
) -> tuple[PlanGraph, list[str], list[ValidationIssue]]:
    notes: list[str] = []
    labels = height_labels(texts)
    if not labels:
        return plan, [f"RCP {page_id}: no ceiling height labels found"], []
    reg = register(wall_samples(rcp), wall_samples(plan, level, step=0.01))
    if reg is None:
        return plan, [f"RCP {page_id}: too few walls to register onto the floor plan"], []
    scale = float(np.sqrt(abs(np.linalg.det(reg.matrix[:, :2]))))
    notes.append(
        f"RCP {page_id} registered onto level {level}: RMS {reg.rms_m * 1000:.1f} mm, "
        f"{reg.inliers:.0%} of its walls matched, scale ×{scale:.4f}"
    )
    issues: list[ValidationIssue] = []
    if reg.rms_m > REGISTRATION_MAX_RMS_M or reg.inliers < 0.7:
        issues.append(
            ValidationIssue(
                code="PLAN_RCP_REGISTRATION",
                severity=Severity.WARNING,
                message=f"The ceiling plan {page_id} does not line up with the floor plan "
                f"(RMS {reg.rms_m * 1000:.0f} mm, {reg.inliers:.0%} of walls matched); its "
                "ceiling heights were not used.",
                fix_hint="Check that the RCP shows the same level, or enter ceiling heights per "
                "room at Gate A.",
            )
        )
        return plan, notes, issues
    out = plan.model_copy(deep=True)
    rooms = [(r, Polygon([(p.x, p.y) for p in r.polygon])) for r in out.rooms if r.level == level]
    storey = next((lv.floor_to_floor_m for lv in out.levels if lv.id == level), None)
    base = storey.value if storey is not None else 3.0
    set_rooms: dict[str, float] = {}
    for x, y, v in labels:
        q = _apply(reg.matrix, np.array([[x, y]]))[0]
        hit = next((r for r, poly in rooms if poly.contains(Point(q[0], q[1]))), None)
        if hit is None:
            notes.append(f"RCP label {v:.2f} m at ({q[0]:.2f}, {q[1]:.2f}) is in no room")
            continue
        if hit.id in set_rooms and abs(set_rooms[hit.id] - v) > 0.01:
            notes.append(f"room {hit.id}: several ceiling heights on the RCP; the highest is used")
            v = max(v, set_rooms[hit.id])
        set_rooms[hit.id] = v
        hit.ceiling_height_m = Fact(
            value=v,
            unit=None,
            provenance=[
                Provenance(
                    source_doc=source_doc,
                    method=method,
                    confidence=0.9,
                    note=f"ceiling height label on the RCP ({page_id})",
                )
            ],
            status="extracted",
        )
        hit.double_height = v >= 1.6 * min(base, 3.0)
    # walls reach the highest ceiling of the rooms they bound
    for w in out.walls:
        if w.level != level:
            continue
        line = LineString(centerline_coords(w))
        near = [
            set_rooms[r.id]
            for r, poly in rooms
            if r.id in set_rooms and poly.distance(line) <= w.thickness_m.value / 2 + 0.05
        ]
        if near:
            w.height_m = fact(
                max(near),
                "derived",
                0.85,
                note=f"highest RCP ceiling of the rooms it bounds ({page_id})",
            )
    notes.append(f"ceiling heights from the RCP for {len(set_rooms)} room(s)")
    return PlanGraph.model_validate(out.model_dump()), notes, issues


# ---------------------------------------------------------------------------------------------
# opening tags and schedules
# ---------------------------------------------------------------------------------------------
TAG_KINDS = {
    "door": ("door", "double_door", "sliding_door", "french_door", "opening", "pass"),
    "window": ("window", "french_door"),
}


def _opening_point(plan: PlanGraph, oi: int) -> NDArray[np.float64] | None:
    o = plan.openings[oi]
    w = next((x for x in plan.walls if x.id == o.host_wall), None)
    if w is None or not isinstance(w.centerline, Segment):
        return None
    a = np.array([w.centerline.a.x, w.centerline.a.y], float)
    b = np.array([w.centerline.b.x, w.centerline.b.y], float)
    out: NDArray[np.float64] = a + (b - a) / float(np.hypot(*(b - a))) * o.offset_m.value
    return out


def assign_tags(plan: PlanGraph, texts: list[TextPrim], level: str) -> tuple[PlanGraph, int]:
    """Door/window tags among the extraction's texts (plan frame) → the nearest compatible
    opening of ``level`` (each tag and opening used once, nearest pairs first)."""
    from archrender.understand.text import TAG_KINDS as KIND_OF
    from archrender.understand.text import normalise_tag

    def _prefix(tag: str) -> str:
        return re.match(r"[^\d]*", tag).group(0)  # type: ignore[union-attr]

    out = plan.model_copy(deep=True)
    pts = {i: _opening_point(out, i) for i in range(len(out.openings))}
    walls = {w.id: w.level for w in out.walls}
    pairs = []
    for t in texts:
        tag = normalise_tag(t.text) if len(t.text) <= 6 else None
        kind = KIND_OF.get(_prefix(tag)) if tag else None
        if tag is None or kind is None:
            continue
        q = np.array([t.x, t.y])
        for i, p in pts.items():
            o = out.openings[i]
            if p is None or walls.get(o.host_wall) != level or o.type not in TAG_KINDS[kind]:
                continue
            d = float(np.hypot(*(q - p)))
            if d <= TAG_REACH_M:
                pairs.append((d, i, tag))
    used_o: set[int] = set()
    used_t: set[str] = set()
    for _, i, tag in sorted(pairs):
        if i in used_o or tag in used_t:
            continue
        out.openings[i].tag = tag
        used_o.add(i)
        used_t.add(tag)
    return PlanGraph.model_validate(out.model_dump()), len(used_o)


def apply_schedules(
    plan: PlanGraph, schedules: list[dict[str, Any]]
) -> tuple[PlanGraph, list[str]]:
    """Heights, sills and widths from door/window schedule rows of tagged openings."""
    from archrender.understand.text import normalise_tag

    rows: dict[str, tuple[dict[str, Any], str]] = {}
    for s in schedules:
        if s.get("kind") != "door_window":
            continue
        for r in s["rows"]:
            if r.get("tag"):
                rows[normalise_tag(r["tag"]) or r["tag"]] = (r, s["source_page"])
    out = plan.model_copy(deep=True)
    notes: list[str] = []
    applied = 0
    for o in out.openings:
        if not o.tag:
            continue
        hit = rows.get(normalise_tag(o.tag) or o.tag)
        if hit is None:
            continue
        row, page = hit
        f = row["fields"]
        prov = Provenance(
            method="schedule", confidence=0.95, note=f"schedule row {row['id']} ({page})"
        )
        applied += 1
        for key, attr in (("height", "height_m"), ("sill", "sill_m")):
            v = f.get(key)
            if isinstance(v, (int, float)) and 0.0 <= v <= 6.0:
                setattr(o, attr, Fact(value=float(v), provenance=[prov], status="extracted"))
        w = f.get("width")
        if not isinstance(w, (int, float)) or not 0.3 <= w <= 6.0:
            continue
        measured = o.width_m
        if abs(measured.value - w) <= WIDTH_TOLERANCE_M:
            o.width_m = Fact(
                value=measured.value,
                provenance=[*measured.provenance, prov],
                status="corroborated",
            )
            continue
        out.conflicts.append(
            Conflict(
                key=f"opening/{o.id}/width",
                candidates=[
                    {"method": measured.provenance[0].method, "value": round(measured.value, 4)},
                    {"method": "schedule", "value": float(w), "detail": row["id"]},
                ],
                proposed=1,
                rule="the schedule specifies the opening; the drawing disagrees by more than 3 cm",
                severity=Severity.WARNING,
            )
        )
        o.width_m = Fact(
            value=float(w), provenance=[prov, *measured.provenance], status="conflicted"
        )
        notes.append(
            f"opening {o.id} ({o.tag}): schedule width {w:.2f} m vs drawn {measured.value:.2f} m"
        )
    if applied:
        notes.insert(0, f"schedule rows applied to {applied} tagged opening(s)")
    return PlanGraph.model_validate(out.model_dump()), notes
