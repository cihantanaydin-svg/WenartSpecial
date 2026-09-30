"""S6 camera proposal: eye-level, zero-pitch (two-point perspective) cameras scored on the plan."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.prepared import prep

from archrender.core.assumptions import AssumptionRegister
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import Vec3
from archrender.core.schemas.plan import PlanGraph, Segment
from archrender.core.schemas.scene import CameraSpec
from archrender.scene.geometry import room_polygon

CLEARANCE_M = 0.25
CORNER_INSET_M = 0.5
ENTRY_INSET_M = 0.4
GRID_M = 0.1
NMS_DIST_M = 1.0
NMS_YAW_DEG = 30.0


@dataclass(frozen=True)
class ScoredCamera:
    camera: CameraSpec
    score: float
    visible_floor_fraction: float
    room_id: str
    origin: str  # "corner" | "entry"


def _yaw_to(src: tuple[float, float], dst: tuple[float, float]) -> float:
    return math.degrees(math.atan2(dst[1] - src[1], dst[0] - src[0]))


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _floor_samples(poly: Polygon) -> np.ndarray:
    minx, miny, maxx, maxy = poly.bounds
    xs = np.arange(minx + GRID_M / 2, maxx, GRID_M)
    ys = np.arange(miny + GRID_M / 2, maxy, GRID_M)
    grid = np.array([(x, y) for x in xs for y in ys])
    pp = prep(poly)
    inside = np.array([pp.contains(Point(p)) for p in grid], dtype=bool)
    return np.asarray(grid[inside])


def visible_floor_fraction(
    poly: Polygon,
    samples: np.ndarray,
    pos: tuple[float, float],
    yaw_deg: float,
    hfov_deg: float,
    vfov_deg: float,
    eye_h: float,
) -> float:
    if len(samples) == 0:
        return 0.0
    d = samples - np.array(pos)
    dist = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    in_h = np.abs((ang - yaw_deg + 180.0) % 360.0 - 180.0) <= hfov_deg / 2
    below = np.degrees(np.arctan2(eye_h, np.maximum(dist, 1e-6)))
    in_v = below <= vfov_deg / 2
    cand = np.nonzero(in_h & in_v)[0]
    if poly.convex_hull.area - poly.area < 1e-6:
        return float(len(cand) / len(samples))
    pp = prep(poly.buffer(1e-6))
    vis = sum(1 for i in cand if pp.contains(LineString([pos, tuple(samples[i])])))
    return float(vis / len(samples))


def _candidates(
    plan: PlanGraph, room_id: str, poly: Polygon
) -> list[tuple[tuple[float, float], str]]:
    out: list[tuple[tuple[float, float], str]] = []
    ring = list(poly.exterior.coords)[:-1]
    n = len(ring)
    for i in range(n):
        p = np.array(ring[i])
        a = np.array(ring[i - 1]) - p
        b = np.array(ring[(i + 1) % n]) - p
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            continue
        bis = a / np.linalg.norm(a) + b / np.linalg.norm(b)
        if np.linalg.norm(bis) < 1e-6:
            continue
        c = p + bis / np.linalg.norm(bis) * CORNER_INSET_M
        if poly.contains(Point(c)):
            out.append(((float(c[0]), float(c[1])), "corner"))
    for o in plan.openings:
        if o.type == "window":
            continue
        w = plan.wall(o.host_wall)
        if not isinstance(w.centerline, Segment):
            continue
        a2 = np.array([w.centerline.a.x, w.centerline.a.y])
        b2 = np.array([w.centerline.b.x, w.centerline.b.y])
        d = (b2 - a2) / np.linalg.norm(b2 - a2)
        mid = a2 + d * o.offset_m.value
        normal = np.array([-d[1], d[0]])
        for sgn in (1.0, -1.0):
            p = mid + normal * sgn * (w.thickness_m.value / 2 + ENTRY_INSET_M)
            if poly.contains(Point(p)):
                out.append(((float(p[0]), float(p[1])), "entry"))
    return out


def propose_cameras(
    plan: PlanGraph,
    room_ids: list[str],
    n_views: int,
    width: int,
    height: int,
    register: AssumptionRegister,
) -> list[ScoredCamera]:
    eye_h = float(register.default("camera_height_m"))
    focal = float(register.default("camera_focal_mm"))
    sensor_w = 36.0
    hfov = math.degrees(2 * math.atan(sensor_w / (2 * focal)))
    vfov = math.degrees(2 * math.atan(sensor_w * height / width / (2 * focal)))
    scored: list[ScoredCamera] = []
    for rid in room_ids:
        poly = room_polygon(plan, rid)
        inner = poly.buffer(-CLEARANCE_M)
        samples = _floor_samples(poly)
        centroid = (poly.centroid.x, poly.centroid.y)
        ring = list(poly.exterior.coords)[:-1]
        for pos, origin in _candidates(plan, rid, poly):
            if inner.is_empty or not inner.contains(Point(pos)):
                continue
            far = max(ring, key=lambda q: math.hypot(q[0] - pos[0], q[1] - pos[1]))
            for yaw in {round(_yaw_to(pos, centroid), 3), round(_yaw_to(pos, far), 3)}:
                frac = visible_floor_fraction(poly, samples, pos, yaw, hfov, vfov, eye_h)
                depth_bonus = min(1.0, math.hypot(far[0] - pos[0], far[1] - pos[1]) / 6.0)
                score = 0.8 * frac + 0.2 * depth_bonus
                cam = CameraSpec(
                    id="cam",
                    position=Vec3(x=pos[0], y=pos[1], z=eye_h),
                    yaw_deg=yaw,
                    pitch_deg=0.0,
                    focal_mm=focal,
                    clip_start=0.05,
                )
                scored.append(ScoredCamera(cam, score, frac, rid, origin))
    if not scored:
        raise ArchRenderError(
            ErrorCode.SCENE_INVALID,
            "No valid camera position found (rooms too small for the clearance rule).",
            "Place cameras manually at Gate C or check the room boundaries.",
        )
    scored.sort(
        key=lambda s: (-s.score, s.camera.position.x, s.camera.position.y, s.camera.yaw_deg)
    )
    chosen: list[ScoredCamera] = []
    for s in scored:
        if len(chosen) >= n_views:
            break
        if any(
            math.hypot(
                s.camera.position.x - c.camera.position.x, s.camera.position.y - c.camera.position.y
            )
            < NMS_DIST_M
            and _angle_diff(s.camera.yaw_deg, c.camera.yaw_deg) < NMS_YAW_DEG
            for c in chosen
        ):
            continue
        chosen.append(s)
    return [
        ScoredCamera(
            s.camera.model_copy(update={"id": f"cam_{i + 1}"}),
            s.score,
            s.visible_floor_fraction,
            s.room_id,
            s.origin,
        )
        for i, s in enumerate(chosen)
    ]
