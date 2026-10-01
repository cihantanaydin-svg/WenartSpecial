"""Synthetic plan specs: ground-truth geometry checked against independent constructions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from archrender.synth.layout import random_layout
from archrender.synth.plan import EXT_THICKNESS, INT_THICKNESS, from_layout, gt_plan, random_spec


def _half_thickness(layout, x: float, y: float, horizontal: bool) -> float:  # type: ignore[no-untyped-def]
    """Thickness/2 of the wall whose centerline passes through (x, y) in the given direction."""
    for w in layout.walls:
        if w.horizontal != horizontal:
            continue
        if (
            horizontal
            and abs(w.ay - y) < 1e-9
            and min(w.ax, w.bx) - 1e-9 <= x <= max(w.ax, w.bx) + 1e-9
        ):
            return w.thickness / 2
        if (
            not horizontal
            and abs(w.ax - x) < 1e-9
            and min(w.ay, w.by) - 1e-9 <= y <= max(w.ay, w.by) + 1e-9
        ):
            return w.thickness / 2
    raise AssertionError("no wall on the room edge")


@pytest.mark.parametrize("seed", range(12))
def test_manhattan_room_faces_are_the_cells_shrunk_by_half_wall_thickness(seed: int) -> None:
    layout = random_layout(np.random.default_rng(seed))
    spec = from_layout(layout)
    for r in layout.rooms:
        cx, cy = r.center
        left = _half_thickness(layout, r.x0, cy, horizontal=False)
        right = _half_thickness(layout, r.x1, cy, horizontal=False)
        bottom = _half_thickness(layout, cx, r.y0, horizontal=True)
        top = _half_thickness(layout, cx, r.y1, horizontal=True)
        expected = box(r.x0 + left, r.y0 + bottom, r.x1 - right, r.y1 - top)
        got = spec.faces[r.id]
        assert got.symmetric_difference(expected).area < 1e-6, (r.id, got.wkt, expected.wkt)


@pytest.mark.parametrize("seed", range(6))
def test_rotation_is_rigid_and_skew_scales_areas_by_the_sine_of_the_axis_angle(seed: int) -> None:
    layout = random_layout(np.random.default_rng(seed))
    flat = from_layout(layout)
    rot = from_layout(layout, variant="rotated", rng=np.random.default_rng(seed))
    for r in layout.rooms:
        assert rot.faces[r.id].area == pytest.approx(flat.faces[r.id].area, abs=1e-6)
    skew = from_layout(layout, variant="skewed", rng=np.random.default_rng(seed))
    sin_phi = abs(skew.e1[0] * skew.e2[1] - skew.e1[1] * skew.e2[0])
    # the cell (between centerlines) scales exactly by sin φ
    for r in skew.rooms:
        lattice = next(x for x in layout.rooms if x.id == r.id)
        assert Polygon(r.cell).area == pytest.approx(lattice.area * sin_phi, rel=1e-9)
    # wall lengths and opening offsets are preserved along the lattice axes
    for a, b in zip(layout.walls, skew.walls, strict=True):
        assert b.length == pytest.approx(a.length, abs=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_arc_wall_geometry_and_ground_truth_plan(seed: int) -> None:
    spec = random_spec(np.random.default_rng(seed), variant="arc")
    arcs = [w for w in spec.walls if w.is_arc]
    assert len(arcs) == 1
    w = arcs[0]
    (cx, cy), r, _, sweep = w.arc_params()
    assert 0 < sweep < math.pi  # a shallow outward bay
    for t in np.linspace(0, w.length, 7):
        x, y = w.point(float(t))
        assert math.hypot(x - cx, y - cy) == pytest.approx(r, abs=1e-9)
    assert w.point(w.length)[0] == pytest.approx(w.b[0], abs=1e-9)
    assert not any(o.wall == spec.walls.index(w) for o in spec.openings)
    plan = gt_plan(spec)
    assert len(plan.rooms) == len(spec.rooms)
    for room in plan.rooms:  # CCW rings, net area = label
        poly = Polygon([(p.x, p.y) for p in room.polygon])
        assert poly.exterior.is_ccw
        assert room.area_label_m2 is not None
        assert poly.area == pytest.approx(room.area_label_m2.value, abs=0.006)
    ext = [x for x in plan.walls if x.kind == "exterior"]
    t_ext = {x.thickness_m.value for x in ext}
    assert len(t_ext) == 1 and t_ext <= set(EXT_THICKNESS)  # one facade thickness
    assert {x.thickness_m.value for x in plan.walls if x.kind == "interior"} <= set(INT_THICKNESS)
