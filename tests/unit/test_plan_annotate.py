"""Plan annotations: RCP registration and ceiling heights, opening tags, schedule sizes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from archrender.core.schemas.plan import Segment
from archrender.plan.annotate import (
    apply_schedules,
    assign_tags,
    height_labels,
    rcp_heights,
    register,
    wall_samples,
)
from archrender.plan.metrics import transform_plan
from archrender.plan.prims import TextPrim
from archrender.synth.plan import gt_plan, random_spec


def _plan(seed: int = 4):  # type: ignore[no-untyped-def]
    return gt_plan(random_spec(np.random.default_rng(seed), variant="manhattan"))


@settings(max_examples=15, deadline=None)
@given(
    st.floats(-180, 180),
    st.floats(0.97, 1.03),
    st.floats(-20, 20),
    st.floats(-20, 20),
)
def test_registration_recovers_a_similarity(rot: float, k: float, tx: float, ty: float) -> None:
    plan = _plan()
    th = math.radians(rot)
    m = [
        [k * math.cos(th), -k * math.sin(th), tx],
        [k * math.sin(th), k * math.cos(th), ty],
    ]
    moved = transform_plan(plan, m)
    reg = register(wall_samples(moved), wall_samples(plan, step=0.01))
    assert reg is not None and reg.inliers > 0.95 and reg.rms_m < 0.01
    # moved → plan is the inverse of m
    back = np.array(reg.matrix) @ np.vstack([np.array(m), [0, 0, 1]])
    assert np.allclose(back, [[1, 0, 0], [0, 1, 0]], atol=1e-3)


@pytest.mark.parametrize(
    ("text", "value"),
    [("+2,70", 2.7), ("+2.80", 2.8), ("h=3.00", 3.0), ("TH 2,60", 2.6), ("+5,40", 5.4)],
)
def test_ceiling_height_labels(text: str, value: float) -> None:
    assert height_labels([TextPrim(text, 1.0, 2.0)]) == [(1.0, 2.0, value)]


@pytest.mark.parametrize("text", ["+0,00", "2,70 m²", "K3", "Salon", "12,40", "+25,00"])
def test_other_texts_are_not_ceiling_heights(text: str) -> None:
    assert height_labels([TextPrim(text, 0.0, 0.0)]) == []


def test_rcp_heights_land_in_rooms_and_flag_double_height() -> None:
    plan = _plan()
    rcp = transform_plan(plan, [[1, 0, 7.5], [0, 1, -3.0]])  # the RCP sheet in its own frame
    texts = []
    for i, r in enumerate(plan.rooms):
        cx = float(np.mean([p.x for p in r.polygon])) + 7.5
        cy = float(np.mean([p.y for p in r.polygon])) - 3.0
        texts.append(TextPrim("+5,40" if i == 0 else "+2,80", cx, cy))
    out, notes, issues = rcp_heights(
        plan, "L0", rcp, texts, source_doc="d", method="pdf_text", page_id="pg"
    )
    assert not issues and any("registered" in n for n in notes)
    heights = {r.id: r.ceiling_height_m.value for r in out.rooms}
    assert heights[plan.rooms[0].id] == 5.4 and out.rooms[0].double_height
    assert all(v == 2.8 for k, v in heights.items() if k != plan.rooms[0].id)
    assert out.rooms[0].ceiling_height_m.provenance[0].method == "pdf_text"
    # walls around the double-height room reach its ceiling
    assert max(w.height_m.value for w in out.walls) == 5.4


def test_an_rcp_that_does_not_match_is_reported_not_used() -> None:
    plan = _plan(4)
    other = _plan(9)  # a different building
    texts = [
        TextPrim("+3,00", float(r.polygon[0].x) + 0.5, float(r.polygon[0].y) + 0.5)
        for r in other.rooms
    ]
    out, _, issues = rcp_heights(
        plan, "L0", other, texts, source_doc=None, method="ocr", page_id="pg"
    )
    assert [i.code for i in issues] == ["PLAN_RCP_REGISTRATION"]
    assert [r.ceiling_height_m for r in out.rooms] == [r.ceiling_height_m for r in plan.rooms]


def _with_tags(plan, offset: float = 0.4):  # type: ignore[no-untyped-def]
    texts = []
    for o in plan.openings:
        w = next(x for x in plan.walls if x.id == o.host_wall)
        assert isinstance(w.centerline, Segment)
        a = np.array([w.centerline.a.x, w.centerline.a.y])
        b = np.array([w.centerline.b.x, w.centerline.b.y])
        u = (b - a) / np.hypot(*(b - a))
        p = a + u * o.offset_m.value + np.array([-u[1], u[0]]) * offset
        tag = ("P" if o.type == "window" else "K") + o.id[1:]
        texts.append(TextPrim(tag, float(p[0]), float(p[1])))
    return texts


def test_tags_go_to_the_nearest_compatible_opening() -> None:
    plan = _plan()
    out, n = assign_tags(plan, _with_tags(plan), "L0")
    assert n == len(plan.openings)
    for o in out.openings:
        assert o.tag == ("P" if o.type == "window" else "K") + o.id[1:]


def test_schedule_sizes_apply_and_disagreements_become_conflicts() -> None:
    plan = _plan()
    tagged, _ = assign_tags(plan, _with_tags(plan), "L0")
    door = next(o for o in tagged.openings if o.type != "window")
    window = next(o for o in tagged.openings if o.type == "window")
    schedule = {
        "kind": "door_window",
        "source_page": "pg_s",
        "rows": [
            {
                "id": "r1",
                "tag": door.tag,
                "fields": {"width": door.width_m.value + 0.01, "height": 2.2},
            },
            {
                "id": "r2",
                "tag": window.tag,
                "fields": {"width": window.width_m.value + 0.2, "height": 1.4, "sill": 0.8},
            },
        ],
    }
    out, notes = apply_schedules(tagged, [schedule])
    d = next(o for o in out.openings if o.id == door.id)
    w = next(o for o in out.openings if o.id == window.id)
    assert d.width_m.value == door.width_m.value and d.width_m.status == "corroborated"
    assert d.height_m.value == 2.2 and d.height_m.provenance[0].method == "schedule"
    assert w.width_m.value == pytest.approx(window.width_m.value + 0.2)
    assert w.width_m.status == "conflicted" and (w.sill_m.value, w.height_m.value) == (0.8, 1.4)
    [c] = out.conflicts
    assert c.key == f"opening/{window.id}/width" and c.proposed == 1 and c.severity == "warning"
    assert any("schedule width" in n for n in notes)
