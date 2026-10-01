"""Gate A edit primitives: RFC 6902 JSON Patch, user provenance of edited facts, rescaling."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from archrender.core.errors import ArchRenderError
from archrender.plan.mock import mock_plan
from archrender.plan.versions import _mark_user, apply_patch, content_sha, rescale

DOC: dict[str, Any] = {"a": 1, "b": {"c": [1, 2, 3]}, "d/e": "slash", "f~g": "tilde"}


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        ([{"op": "add", "path": "/x", "value": 5}], {**DOC, "x": 5}),
        (
            [{"op": "add", "path": "/b/c/1", "value": 9}],
            {**DOC, "b": {"c": [1, 9, 2, 3]}},
        ),
        ([{"op": "add", "path": "/b/c/-", "value": 4}], {**DOC, "b": {"c": [1, 2, 3, 4]}}),
        ([{"op": "remove", "path": "/b/c/0"}], {**DOC, "b": {"c": [2, 3]}}),
        ([{"op": "replace", "path": "/a", "value": 2}], {**DOC, "a": 2}),
        ([{"op": "replace", "path": "/d~1e", "value": "x"}], {**DOC, "d/e": "x"}),
        ([{"op": "replace", "path": "/f~0g", "value": "y"}], {**DOC, "f~g": "y"}),
        (
            [{"op": "move", "from": "/a", "path": "/b/a"}],
            {"b": {"c": [1, 2, 3], "a": 1}, "d/e": "slash", "f~g": "tilde"},
        ),
        ([{"op": "copy", "from": "/b/c", "path": "/z"}], {**DOC, "z": [1, 2, 3]}),
        ([{"op": "test", "path": "/b/c/2", "value": 3}], DOC),
    ],
)
def test_rfc6902_operations(ops: list[dict[str, Any]], expected: dict[str, Any]) -> None:
    before = copy.deepcopy(DOC)
    assert apply_patch(DOC, ops) == expected
    assert before == DOC  # the source document is never modified


@pytest.mark.parametrize(
    "op",
    [
        {"op": "replace", "path": "/missing", "value": 1},
        {"op": "remove", "path": "/b/c/7"},
        {"op": "test", "path": "/a", "value": 2},
        {"op": "add", "path": "a", "value": 1},
        {"op": "add", "path": "", "value": 1},
        {"op": "frobnicate", "path": "/a"},
        {"op": "add", "path": "/b/c/x", "value": 1},
    ],
)
def test_a_failing_operation_applies_nothing(op: dict[str, Any]) -> None:
    ops = [{"op": "replace", "path": "/a", "value": 99}, op]
    with pytest.raises(ArchRenderError) as e:
        apply_patch(DOC, ops)
    assert "operation 1" in e.value.message
    assert DOC["a"] == 1


@given(st.lists(st.integers(), min_size=1, max_size=8), st.data())
def test_remove_then_add_round_trips(items: list[int], data: st.DataObject) -> None:
    i = data.draw(st.integers(0, len(items) - 1))
    doc = {"xs": items}
    out = apply_patch(
        doc,
        [
            {"op": "remove", "path": f"/xs/{i}"},
            {"op": "add", "path": f"/xs/{i}", "value": items[i]},
        ],
    )
    assert out == doc


def _plan_json() -> dict[str, Any]:
    return mock_plan("prj_u", []).model_dump(mode="json")


def test_only_changed_facts_become_user_facts() -> None:
    old = _plan_json()
    new = copy.deepcopy(old)
    new["openings"][0]["width_m"]["value"] += 0.1
    new["openings"].reverse()  # reordering is not an edit
    out = _mark_user(old, new, "site measure")
    by_id = {o["id"]: o for o in out["openings"]}
    edited = by_id[old["openings"][0]["id"]]["width_m"]
    assert edited["status"] == "user_confirmed"
    assert edited["provenance"][0] == {
        **edited["provenance"][0],
        "method": "user",
        "note": "site measure",
    }
    for o in old["openings"][1:]:
        assert by_id[o["id"]]["width_m"] == o["width_m"]
    assert out["walls"] == old["walls"]


def test_rescale_is_invertible_and_scales_areas() -> None:
    plan = mock_plan("prj_u", [])
    k = 1.0237
    back = rescale(rescale(plan, k), 1 / k)
    for a, b in zip(plan.walls, back.walls, strict=True):
        assert b.centerline.length() == pytest.approx(a.centerline.length())  # type: ignore[union-attr]
    big = rescale(plan, k)
    assert big.walls[0].thickness_m.value == pytest.approx(plan.walls[0].thickness_m.value * k)
    assert big.walls[0].height_m.value == plan.walls[0].height_m.value
    room = plan.rooms[0]
    scaled = big.rooms[0]
    if room.area_label_m2 is not None:  # a stated value is not a plan measurement
        assert scaled.area_label_m2 == room.area_label_m2
    assert scaled.polygon[1].x == pytest.approx(room.polygon[1].x * k)


def test_content_hash_ignores_version_and_timestamps() -> None:
    a = mock_plan("prj_u", [])
    b = a.model_copy(update={"version": "other"})
    assert content_sha(a) == content_sha(b)
    c = a.model_copy(deep=True)
    c.walls[0].thickness_m.value += 0.01
    assert content_sha(a) != content_sha(c)


@pytest.mark.parametrize(
    ("name", "rank"),
    [
        ("Zemin Kat", 0.0),
        ("ZEMİN KAT PLANI", 0.0),
        ("Giriş Katı", 0.0),
        ("Ground Floor", 0.0),
        ("1. Kat", 1.0),
        ("Birinci Kat", 1.0),
        ("2. KAT PLANI", 2.0),
        ("First Floor", 1.0),
        ("Level 3", 3.0),
        ("2nd", 2.0),
        ("Bodrum Kat", -1.0),
        ("2. Bodrum Kat", -2.0),
        ("Basement", -1.0),
        ("Asma Kat", 0.5),
        ("Çatı Katı", 1000.0),
        ("Roof", 1000.0),
        ("Kesit A-A", None),
        ("Daire 3", None),
    ],
)
def test_level_names_give_the_storey_order(name: str, rank: float | None) -> None:
    from archrender.plan.stage import level_rank

    assert level_rank(name) == rank


def test_levels_stack_in_storey_order() -> None:
    from archrender.plan.stage import _stack

    names, elev, guessed = _stack(["1. Kat", "Çatı Katı", "Bodrum Kat", "Zemin Kat"])
    assert names == ["Bodrum Kat", "Zemin Kat", "1. Kat", "Çatı Katı"]
    assert elev == [-3.0, 0.0, 3.0, 6.0] and not guessed
    names, elev, guessed = _stack(["Blok B", "Zemin Kat"])
    assert names == ["Zemin Kat", "Blok B"] and elev == [0.0, 3.0] and guessed
