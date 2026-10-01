"""Plan extraction metrics against ground truth (``make eval`` plan table, Phase-3 acceptance).

Both plans must be in the same frame (see :func:`transform_plan`). Metrics:

- **Wall F1 (length based).** Centerlines are sampled every 2 cm. A predicted sample is correct if a
  ground-truth centerline lies within ``tol`` (5 cm) with a direction difference ≤ 5° and a
  thickness difference ≤ max(3 cm, 20 %); recall is the same test the other way round. Wall
  segmentation (where one wall ends and the next begins) does not matter.
- **Opening F1.** One-to-one greedy matching by centre distance: a match needs ≤ 15 cm between
  centres, a width difference ≤ 10 cm and the same kind (door-like or window).
- **Room F1.** One-to-one matching with IoU ≥ 0.8; mean IoU and name accuracy of the matches.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree

from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import Arc, Opening, PlanGraph, Segment, Wall
from archrender.understand.text import fold

STEP = 0.02
DOORS = {"door", "double_door", "sliding_door", "french_door", "pass", "opening"}


def wall_line(w: Wall) -> LineString:
    c = w.centerline
    if isinstance(c, Segment):
        return LineString([(c.a.x, c.a.y), (c.b.x, c.b.y)])
    sweep = (c.end_deg - c.start_deg) % 360 or 360
    n = max(8, int(sweep / 3))
    return LineString(
        [
            (
                c.center.x + c.radius * math.cos(math.radians(c.start_deg + sweep * k / n)),
                c.center.y + c.radius * math.sin(math.radians(c.start_deg + sweep * k / n)),
            )
            for k in range(n + 1)
        ]
    )


def opening_center(plan: PlanGraph, o: Opening) -> tuple[float, float]:
    line = wall_line(plan.wall(o.host_wall))
    p = line.interpolate(min(max(o.offset_m.value, 0.0), line.length))
    return float(p.x), float(p.y)


def _samples(plan: PlanGraph) -> list[tuple[float, float, float, float]]:
    """(x, y, direction angle mod π, thickness) every STEP metres along every wall."""
    out = []
    for w in plan.walls:
        line = wall_line(w)
        n = max(1, int(line.length / STEP))
        for k in range(n):
            d = (k + 0.5) * line.length / n
            p = line.interpolate(d)
            q = line.interpolate(min(line.length, d + 0.01))
            r = line.interpolate(max(0.0, d - 0.01))
            ang = math.atan2(q.y - r.y, q.x - r.x) % math.pi
            out.append((float(p.x), float(p.y), ang, w.thickness_m.value))
    return out


def _coverage(
    samples: list[tuple[float, float, float, float]], other: PlanGraph, tol: float
) -> float:
    if not samples:
        return 0.0
    lines = [wall_line(w) for w in other.walls]
    if not lines:
        return 0.0
    tree = STRtree(lines)
    hit = 0
    for x, y, ang, t in samples:
        p = Point(x, y)
        for i in tree.query(p.buffer(tol)):
            w = other.walls[int(i)]
            line = lines[int(i)]
            if line.distance(p) > tol:
                continue
            d = line.project(p)
            q = line.interpolate(min(line.length, d + 0.01))
            r = line.interpolate(max(0.0, d - 0.01))
            ang2 = math.atan2(q.y - r.y, q.x - r.x) % math.pi
            dang = abs(ang - ang2)
            dang = min(dang, math.pi - dang)
            t2 = w.thickness_m.value
            if dang <= math.radians(5) and abs(t - t2) <= max(0.03, 0.2 * t2):
                hit += 1
                break
    return hit / len(samples)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def wall_scores(pred: PlanGraph, gt: PlanGraph, tol: float = 0.05) -> dict[str, float]:
    p = _coverage(_samples(pred), gt, tol)
    r = _coverage(_samples(gt), pred, tol)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(_f1(p, r), 4)}


def opening_scores(pred: PlanGraph, gt: PlanGraph) -> dict[str, Any]:
    cands = []
    for i, o in enumerate(pred.openings):
        po = opening_center(pred, o)
        for j, g in enumerate(gt.openings):
            pg = opening_center(gt, g)
            d = math.hypot(po[0] - pg[0], po[1] - pg[1])
            same_kind = (o.type in DOORS) == (g.type in DOORS)
            if d <= 0.15 and abs(o.width_m.value - g.width_m.value) <= 0.10 and same_kind:
                cands.append((d, i, j))
    cands.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    for _, i, j in cands:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
    tp = len(used_p)
    precision = tp / len(pred.openings) if pred.openings else (1.0 if not gt.openings else 0.0)
    recall = tp / len(gt.openings) if gt.openings else 1.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "matched": tp,
        "predicted": len(pred.openings),
        "truth": len(gt.openings),
    }


def _poly(points: list[Point2]) -> Polygon:
    # snap to a 1 µm grid: overlays of nearly coincident edges (1e-15 apart) are not robust in GEOS
    # (two identical squares had an empty intersection)
    return shapely.set_precision(Polygon([(p.x, p.y) for p in points]).buffer(0), 1e-6)


def room_scores(pred: PlanGraph, gt: PlanGraph, iou_min: float = 0.8) -> dict[str, Any]:
    pp = [_poly(r.polygon) for r in pred.rooms]
    gp = [_poly(r.polygon) for r in gt.rooms]
    cands = []
    for i, a in enumerate(pp):
        for j, b in enumerate(gp):
            if not a.intersects(b):
                continue
            iou = a.intersection(b).area / max(a.union(b).area, 1e-12)
            if iou >= iou_min:
                cands.append((-iou, i, j))
    cands.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    ious, names = [], []
    for neg, i, j in cands:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        ious.append(-neg)
        names.append(fold(pred.rooms[i].name.value) == fold(gt.rooms[j].name.value))
    tp = len(ious)
    precision = tp / len(pp) if pp else 0.0
    recall = tp / len(gp) if gp else 1.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(_f1(precision, recall), 4),
        "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        "name_accuracy": round(float(np.mean(names)), 4) if names else None,
    }


def plan_scores(pred: PlanGraph, gt: PlanGraph) -> dict[str, Any]:
    return {
        "walls": wall_scores(pred, gt),
        "openings": opening_scores(pred, gt),
        "rooms": room_scores(pred, gt),
    }


# ---------------------------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------------------------
def compose(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """2×3 affines: apply ``a`` then ``b``."""
    ma = np.vstack([np.asarray(a, np.float64), [0, 0, 1]])
    mb = np.vstack([np.asarray(b, np.float64), [0, 0, 1]])
    return [[float(v) for v in row] for row in (mb @ ma)[:2]]


def invert(a: list[list[float]]) -> list[list[float]]:
    m = np.vstack([np.asarray(a, np.float64), [0, 0, 1]])
    return [[float(v) for v in row] for row in np.linalg.inv(m)[:2]]


def transform_plan(plan: PlanGraph, affine: list[list[float]]) -> PlanGraph:
    """Apply a similarity transform (rotation, uniform scale, translation; no reflection) to all
    geometry. Offsets and widths scale with it; thicknesses too."""
    m = np.asarray(affine, np.float64)
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if det <= 0:
        raise ValueError("transform_plan needs an orientation-preserving similarity")
    s = math.sqrt(det)
    rot = math.degrees(math.atan2(m[1, 0], m[0, 0]))

    def pt(p: Point2) -> Point2:
        return Point2(
            x=m[0, 0] * p.x + m[0, 1] * p.y + m[0, 2], y=m[1, 0] * p.x + m[1, 1] * p.y + m[1, 2]
        )

    def scaled(f: Any) -> Any:
        return f.model_copy(update={"value": f.value * s})

    walls = []
    for w in plan.walls:
        c = w.centerline
        if isinstance(c, Segment):
            cl: Segment | Arc = Segment(a=pt(c.a), b=pt(c.b))
        else:
            cl = Arc(
                center=pt(c.center),
                radius=c.radius * s,
                start_deg=(c.start_deg + rot) % 360,
                end_deg=(c.end_deg + rot) % 360,
            )
        walls.append(w.model_copy(update={"centerline": cl, "thickness_m": scaled(w.thickness_m)}))
    openings = [
        o.model_copy(update={"offset_m": scaled(o.offset_m), "width_m": scaled(o.width_m)})
        for o in plan.openings
    ]
    rooms = [r.model_copy(update={"polygon": [pt(p) for p in r.polygon]}) for r in plan.rooms]
    return plan.model_copy(update={"walls": walls, "openings": openings, "rooms": rooms})
