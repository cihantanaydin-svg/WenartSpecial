"""Property-based: random valid plans → watertight meshes, openings inside hosts, validators clean."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from archrender.brief.defaults import default_brief
from archrender.core.assumptions import AssumptionRegister
from archrender.core.config import REPO_ROOT
from archrender.core.schemas.common import Point2
from archrender.core.schemas.plan import Opening, PlanGraph, Segment, Wall
from archrender.core.schemas.provenance import fact
from archrender.core.schemas.scene import RenderSettings
from archrender.core.schemas.section import Section
from archrender.plan.mock import mock_plan
from archrender.plan.validate import blocking, validate_plan
from archrender.scene.compiler import SceneCompiler
from archrender.scene.geometry import is_watertight, wall_footprints
from archrender.scene.materials import MaterialLibrary

LIB = MaterialLibrary.load(REPO_ROOT / "configs")


@st.composite
def room_plans(draw: st.DrawFn) -> PlanGraph:
    width = draw(st.floats(2.5, 9.0))
    depth = draw(st.floats(2.5, 7.0))
    thickness = draw(st.floats(0.1, 0.4))
    base = mock_plan("prj_p", [], width=width, depth=depth)
    walls = [
        w.model_copy(update={"thickness_m": fact(thickness, "mock", 1.0)}) for w in base.walls
    ]
    # rebuild centerlines for the chosen thickness
    t = thickness
    x0, y0, x1, y1 = -t / 2, -t / 2, width + t / 2, depth + t / 2
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = [
        Wall(
            id=w.id, level="L0",
            centerline=Segment(a=Point2(x=corners[i][0], y=corners[i][1]), b=Point2(x=corners[(i + 1) % 4][0], y=corners[(i + 1) % 4][1])),
            thickness_m=fact(t, "mock", 1.0), height_m=w.height_m, kind="exterior",
        )
        for i, w in enumerate(walls)
    ]
    openings: list[Opening] = []
    for i, w in enumerate(walls):
        length = w.centerline.length()
        max_n = max(n for n in (0, 1, 2) if n == 0 or length / (n + 1) - 0.3 >= 0.6)
        n = draw(st.integers(0, max_n))
        slots = n + 1
        for k in range(n):
            width_o = draw(st.floats(0.6, min(1.6, length / slots - 0.3)))
            center = length * (k + 1) / slots
            is_door = draw(st.booleans())
            openings.append(
                Opening(
                    id=f"O{i}{k}", host_wall=w.id, offset_m=fact(center, "mock", 1.0),
                    width_m=fact(width_o, "mock", 1.0),
                    height_m=fact(2.1 if is_door else 1.2, "mock", 1.0),
                    sill_m=fact(0.0 if is_door else 0.9, "mock", 1.0),
                    type="door" if is_door else "window",
                )
            )
    return base.model_copy(update={"walls": walls, "openings": openings})


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(room_plans())
def test_random_rooms_compile_watertight(plan: PlanGraph) -> None:
    issues = blocking(validate_plan(plan))
    assert not issues, [i.message for i in issues]
    reg = AssumptionRegister("test")
    section = Section(id="s", level="L0", kind="rooms", room_ids=["R1"])
    compiled = SceneCompiler(LIB).compile(
        plan, section, default_brief("s", reg), RenderSettings(width=64, height=36), reg, scene_id="scn"
    )
    assert compiled.spec.objects
    for obj in compiled.spec.objects:
        assert obj.mesh.watertight, obj.id
    # wall pieces partition the network: no overlaps
    pieces = list(wall_footprints(plan).values())
    for i, a in enumerate(pieces):
        for b in pieces[i + 1 :]:
            assert a.intersection(b).area < 1e-9


def test_opening_outside_host_is_reported() -> None:
    plan = mock_plan("prj_x", [])
    bad = plan.openings[0].model_copy(update={"offset_m": fact(0.1, "mock", 1.0)})
    plan = plan.model_copy(update={"openings": [bad, *plan.openings[1:]]})
    codes = {i.code for i in validate_plan(plan)}
    assert "OPENING_OUTSIDE_HOST" in codes


def test_area_label_mismatch_reported() -> None:
    plan = mock_plan("prj_x", [])
    room = plan.rooms[0].model_copy(update={"area_label_m2": fact(25.0, "mock", 1.0)})
    plan = plan.model_copy(update={"rooms": [room]})
    assert "ROOM_AREA_MISMATCH" in {i.code for i in validate_plan(plan)}


def test_is_watertight_detects_open_mesh() -> None:
    import numpy as np

    from archrender.scene.geometry import TriMesh

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    closed = TriMesh(v, np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]))
    assert is_watertight(closed)
    assert not is_watertight(TriMesh(v, closed.faces[:3]))
