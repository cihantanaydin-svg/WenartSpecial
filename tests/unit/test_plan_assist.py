"""VLM coordinate assist (ADR-S19): triggers, hint → snap → evidence, adversarial hints."""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image

from archrender.core.schemas.document import Word
from archrender.core.schemas.plan import PlanGraph
from archrender.plan.assist import (
    MAX_TILE_PX,
    Hint,
    PageEvidence,
    Trigger,
    VlmHintSource,
    find_triggers,
    run_assist,
    snap_opening,
    snap_wall,
    walls_mask,
)
from archrender.plan.builder import build_plan
from archrender.plan.metrics import compose, plan_scores, transform_plan
from archrender.plan.raster import raster_extract, vectorise
from archrender.plan.validate import validate_plan
from archrender.synth.hints import AdversarialHints, OracleHints, gt_elements_px
from archrender.synth.plan import gt_plan, random_spec
from archrender.synth.raster import scan
from archrender.synth.sheets import floor_plan_page

VARIANTS = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]
STYLES = ["solid", "grey", "hatch", "outline"]


@dataclass
class Case:
    ev: PageEvidence
    plan: PlanGraph
    prims: Any
    truth: PlanGraph
    gt_hints: list[Hint]
    words: list[Word]

    def rebuild(self, prims: Any) -> PlanGraph:
        return build_plan(prims, project="p", version="v", source="raster").plan


@cache
def case(seed: int, quality: str = "clean", dpi: float = 300.0) -> Case:
    """A scanned synthetic sheet extracted with ground-truth words (perfect OCR)."""
    rng = np.random.default_rng(700 + seed)
    spec = random_spec(rng, variant=VARIANTS[seed % 5])  # type: ignore[arg-type]
    sheet = floor_plan_page(rng, spec=spec, wall_style=STYLES[seed % 4])
    data, _, gt = scan(sheet.pdf, sheet.gt, rng, dpi=dpi, quality=quality)
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    words = [
        Word(
            text=w["text"],
            x0=w["bbox"][0],
            y0=w["bbox"][1],
            x1=w["bbox"][2],
            y1=w["bbox"][3],
            angle_deg=w.get("angle_deg", 0.0),
            source="ocr:ground-truth",
        )
        for w in gt["words"]
    ]
    m = gt["scale"] * 0.0254 / dpi
    rv = vectorise(gray, words, m)
    ex = raster_extract(gray, words, m, vectors=rv)
    plan = build_plan(ex.prims, project="p", version="v", source="raster").plan
    assert rv.body is not None and rv.ink is not None
    ev = PageEvidence(rgb, rv.body, rv.ink, rv.stroke_px, m, np.array(ex.doc_to_plan), dpi)
    truth = transform_plan(gt_plan(spec), compose(gt["plan_to_page_px"], ex.doc_to_plan))
    return Case(
        ev, plan, ex.prims, truth, gt_elements_px(gt_plan(spec), gt["plan_to_page_px"]), words
    )


def test_a_clean_extraction_fires_no_trigger_and_makes_no_call() -> None:
    c = case(0)
    assert find_triggers(c.plan, c.ev) == []
    spy = OracleHints(c.gt_hints, np.random.default_rng(0))
    out = run_assist(c.plan, c.prims, c.ev, spy, c.rebuild)
    assert out.calls == 0 and spy.calls == [] and out.plan is c.plan


def test_the_wall_lost_under_a_stamp_is_measured_back_and_awaits_confirmation() -> None:
    # hatch sheet with an approval stamp across a wall: the extractor loses the junction there and
    # two rooms merge (a known limitation of the CV baseline)
    c = case(2)
    before = plan_scores(c.plan, c.truth)
    assert before["rooms"]["f1"] < 0.8
    triggers = find_triggers(c.plan, c.ev)
    assert [t.kind for t in triggers] == ["validator_failed"]
    assert "ROOM_AREA_MISMATCH" in triggers[0].detail
    oracle = OracleHints(c.gt_hints, np.random.default_rng(1), noise_px=6.0)
    out = run_assist(c.plan, c.prims, c.ev, oracle, c.rebuild, source_doc="doc")
    assert out.calls == 1 and len(oracle.calls) == 1
    accepted = [s for s in out.snaps if s.accepted]
    assert len(accepted) == 1 and "extends a wall" in accepted[0].reason
    after = plan_scores(out.plan, c.truth)
    assert after["rooms"]["f1"] == 1.0 and after["walls"]["f1"] >= before["walls"]["f1"]
    assisted = [w for w in out.plan.walls if w.thickness_m.provenance[0].method == "vlm_assisted"]
    assert len(assisted) == 1
    prov = assisted[0].thickness_m.provenance[0]
    assert prov.assist is not None and prov.assist.trigger == "validator_failed"
    assert prov.assist.evidence_coverage >= 0.8 and not prov.assist.user_confirmed
    assert prov.confidence <= 0.6
    codes = {i.code for i in validate_plan(out.plan)}
    assert "PLAN_ASSIST_UNCONFIRMED" in codes and "ROOM_AREA_MISMATCH" not in codes
    # the other hints of the answer were already walls of the plan: suggestions, never inserted
    report = out.report(c.ev, "pg")
    assert report["accepted"] == 1 and len(report["suggestions"]) == len(out.snaps) - 1
    assert all(s["id"].startswith("pg/sg") for s in report["suggestions"])


def test_the_tile_is_full_resolution_and_answers_map_back_to_the_page() -> None:
    c = case(2)
    trig = find_triggers(c.plan, c.ev)[0]
    seen: dict[str, Any] = {}

    class FakeVlm:
        def ref(self) -> Any:
            return None

        def locate_elements(self, tile: Any, question: str) -> dict[str, Any]:
            seen["shape"], seen["question"] = tile.shape, question
            return {
                "elements": [
                    {"kind": "wall", "points": [[10, 20], [110, 20]], "confidence": 0.7},
                    {"kind": "chair", "points": [[1, 1], [2, 2]], "confidence": 0.9},
                    {"kind": "door", "points": [[5, 5]], "confidence": 0.9},
                ]
            }

    from archrender.plan.assist import tile_for

    tile, origin = tile_for(c.ev, trig)
    hints = VlmHintSource(FakeVlm()).hints(tile, origin, trig)
    assert max(seen["shape"][:2]) <= MAX_TILE_PX
    assert (
        tile.shape == seen["shape"]
        and (
            tile
            == c.ev.rgb[
                origin[1] : origin[1] + tile.shape[0], origin[0] : origin[0] + tile.shape[1]
            ]
        ).all()
    )
    assert "ROOM_AREA_MISMATCH" in seen["question"]
    assert hints == [
        Hint(
            "wall", (origin[0] + 10.0, origin[1] + 20.0), (origin[0] + 110.0, origin[1] + 20.0), 0.7
        )
    ]


def _matches_truth(plan: PlanGraph, truth: PlanGraph, tol: float = 0.1) -> bool:
    """Every element placed with VLM help lies on a ground-truth element."""
    from archrender.plan.assist import _same_line

    for w in plan.walls:
        if w.thickness_m.provenance[0].method != "vlm_assisted":
            continue
        cl = w.centerline
        p0, p1 = np.array([cl.a.x, cl.a.y]), np.array([cl.b.x, cl.b.y])  # type: ignore[union-attr]
        if not any(_same_line(p0, p1, g, tol) >= 0.9 for g in truth.walls):
            return False
    return True


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["wall", "door", "window", "opening"]),
            st.floats(0.02, 0.98),
            st.floats(0.02, 0.98),
            st.floats(0, 360),
            st.floats(0.3, 6.0),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_adversarial_hints_never_enter_the_plan(hints: list[tuple[Any, ...]]) -> None:
    """Confident hints anywhere on the sheet (paper, text, dimensions, furniture, beside walls):
    whatever is accepted lies on a drawn element of the ground truth, and the plan only gets
    better or stays the same."""
    c = case(2)
    h, w = c.ev.body.shape
    planted = []
    for kind, fx, fy, ang, length_m in hints:
        a = np.array([fx * w, fy * h])
        d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        b = a + d * length_m / c.ev.m_per_px
        planted.append(Hint(kind, (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), 0.99))
    out = run_assist(c.plan, c.prims, c.ev, AdversarialHints(planted), c.rebuild)
    assert _matches_truth(out.plan, c.truth)
    before, after = plan_scores(c.plan, c.truth), plan_scores(out.plan, c.truth)
    assert after["walls"]["precision"] >= before["walls"]["precision"] - 1e-9
    for o in out.plan.openings:
        if o.width_m.provenance[0].method == "vlm_assisted":
            assert after["openings"]["precision"] >= before["openings"]["precision"] - 1e-9


@pytest.mark.parametrize("seed", [0, 2])
def test_hints_on_blank_paper_text_and_beside_walls_are_rejected(seed: int) -> None:
    c = case(seed)
    trig = Trigger("validator_failed", "test", (0, 0, 10, 10), "any")
    existing = walls_mask(c.plan, c.ev, 0.0)
    k = c.ev.m_per_px
    rejected = []
    # blank paper in the margin, and every room label (text, not walls)
    rejected.append(Hint("wall", (60.0, 60.0), (60.0 + 2 / k, 60.0), 0.99))
    for wd in c.words[:8]:
        rejected.append(
            Hint("wall", (wd.x0, (wd.y0 + wd.y1) / 2), (wd.x0 + 1 / k, (wd.y0 + wd.y1) / 2), 0.99)
        )
    # parallel to a real wall, one metre into the room
    g = next(x for x in c.gt_hints if x.kind == "wall")
    a, b = np.array(g.a), np.array(g.b)
    n = np.array([-(b - a)[1], (b - a)[0]]) / np.hypot(*(b - a))
    off = n * 1.0 / k
    rejected.append(Hint("wall", tuple(a + off), tuple(b + off), 0.99))  # type: ignore[arg-type]
    for hint in rejected:
        s = snap_wall(hint, trig, c.ev, existing)
        assert not s.accepted, (hint, s.reason)
    # a door in a solid stretch of wall, and one in the middle of a room
    for hint in [
        Hint(
            "door",
            tuple(a + (b - a) * 0.02),
            tuple(a + (b - a) * 0.02 + (b - a) / np.hypot(*(b - a)) * 0.9 / k),
            0.99,
        ),  # type: ignore[arg-type]
        Hint("door", tuple(a + off * 2), tuple(a + off * 2 + np.array([0.9 / k, 0])), 0.99),  # type: ignore[arg-type]
    ]:
        s = snap_opening(hint, trig, c.plan, c.ev)
        assert not s.accepted or s.reason.startswith("already"), (hint, s.reason)
