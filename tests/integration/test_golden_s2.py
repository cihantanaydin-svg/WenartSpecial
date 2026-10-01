"""Phase-3 acceptance on the golden projects: S0 → S1 → S2 end to end (worker jobs), scored
against the ground-truth plan in the frame of the source S2 chose."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from archrender.core.config import Settings
from archrender.core.schemas.plan import PlanGraph
from archrender.pipeline.engine import StageContext
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from archrender.plan import versions
from archrender.plan.metrics import compose, plan_scores, transform_plan
from archrender.plan.stage import Extractor, s2_input
from archrender.synth.golden import GoldenProject, g1_daire, g2_loft, g3_office
from archrender.synth.plan import gt_plan
from tests.helpers import upload_bytes


@pytest.fixture
def svc(settings: Settings, project_id: str) -> Services:
    return Services.create(settings)


def _run(svc: Services, project_id: str, g: GoldenProject) -> tuple[PlanGraph, dict[str, Any]]:
    for d in g.docs:
        upload_bytes(svc, project_id, d.filename, d.data)
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    bad = svc.db.query(
        "SELECT kind, status, error_json FROM jobs WHERE project_id = ? AND status != 'succeeded'",
        (project_id,),
    )
    assert not bad, [dict(r) for r in bad]
    [v] = versions.list_versions(svc, project_id)
    return versions.load(svc, project_id, v["id"]), v["extraction"]


def _truth(svc: Services, g: GoldenProject, plan: PlanGraph, level: str = "L0") -> PlanGraph:
    """The ground truth in the plan's frame, through the chosen document's own transform."""
    assert g.spec is not None
    t = plan.doc_transforms[0]
    name = svc.db.one("SELECT filename FROM documents WHERE id = ?", (t.doc_id,))["filename"]
    doc = next(d for d in g.docs if d.filename == name)
    assert doc.gt is not None
    to_doc = doc.gt.get("plan_to_page_px") or doc.gt["plan_to_doc"]
    truth = transform_plan(gt_plan(g.spec), compose(to_doc, [list(r) for r in t.matrix]))
    if level != "L0":
        return truth
    return truth


def _level(plan: PlanGraph, level: str) -> PlanGraph:
    d = plan.model_dump(mode="json")
    walls = {w["id"] for w in d["walls"] if w["level"] == level}
    d["walls"] = [w for w in d["walls"] if w["id"] in walls]
    d["openings"] = [o for o in d["openings"] if o["host_wall"] in walls]
    d["rooms"] = [r for r in d["rooms"] if r["level"] == level]
    d["levels"] = [lv for lv in d["levels"] if lv["id"] == level]
    return PlanGraph.model_validate(d)


def test_g1_daire_from_its_dxf_with_tags_and_schedules(svc: Services, project_id: str) -> None:
    g = g1_daire()
    plan, ex = _run(svc, project_id, g)
    # both vector sources are read; the DXF wins the level
    assert plan.source == "dxf" and sorted(s["source"] for s in ex["sources"]) == [
        "dxf",
        "pdf_vector",
    ]
    assert len(plan.doc_transforms) == 1
    s = plan_scores(plan, _truth(svc, g, plan))
    assert s["walls"]["f1"] >= 0.98 and s["openings"]["f1"] >= 0.95, s
    assert s["rooms"]["f1"] >= 0.95, s
    # tags read on the drawing, schedule rows applied (drawn and specified widths agree)
    tagged = [o for o in plan.openings if o.tag]
    assert len(tagged) >= 0.9 * len(plan.openings), [o.tag for o in plan.openings]
    assert all(o.height_m.provenance[0].method == "schedule" for o in tagged)
    assert not [c for c in plan.conflicts if c.key.startswith("opening/")]
    assert {i.code for i in plan.issues if i.severity in ("error", "blocker")} == set()


def test_g1_vector_pdf_alone_meets_the_vector_targets(svc: Services, project_id: str) -> None:
    g = g1_daire()
    g.docs = [d for d in g.docs if d.filename.endswith(".pdf") and "Plan" in d.filename]
    plan, ex = _run(svc, project_id, g)
    assert plan.source == "pdf_vector"
    s = plan_scores(plan, _truth(svc, g, plan))
    assert s["walls"]["f1"] >= 0.98 and s["openings"]["f1"] >= 0.95, s
    scale = ex["sources"][0]["scale"]
    gt = g.docs[0].gt
    assert gt is not None
    true_m = gt["scale"] * 0.0254 / 300.0
    assert abs(scale["m_per_unit"] - true_m) / true_m <= 0.01, scale


def test_g2_loft_from_a_noisy_scan_and_a_photo_with_rcp_heights(
    svc: Services, project_id: str
) -> None:
    g = g2_loft()
    plan, ex = _run(svc, project_id, g)
    assert plan.source == "raster"
    assert sorted(s["source"] for s in ex["sources"]) == ["raster", "raster"]
    assert ex["ceiling_plans"], ex
    s = (
        plan_scores(plan, _truth(svc, g, plan))
        if "plan_to_page_px" in _gt_of(svc, g, plan)
        else None
    )
    if s is not None:  # the scan won (the photo's frame is projective; scored in make eval)
        assert s["walls"]["f1"] >= 0.85, s
    # ceiling heights from the RCP: the double-height living room
    assert any("registered onto level" in n for n in ex["annotations"]), ex["annotations"]
    salon = [r for r in plan.rooms if r.double_height]
    assert len(salon) == 1, [(r.name.value, r.ceiling_height_m.value) for r in plan.rooms]
    assert salon[0].ceiling_height_m.provenance[0].method in ("pdf_text", "ocr")
    expected = max(r.ceiling for r in g.spec.rooms)  # type: ignore[union-attr]
    assert salon[0].ceiling_height_m.value == pytest.approx(expected, abs=0.01)
    reg = next(n for n in ex["annotations"] if "registered onto level" in n)
    assert "RMS" in reg and "100% of its walls" in reg, reg
    # tags read on the scan; where the drawn width disagrees with the schedule the schedule is
    # used and the disagreement waits at Gate A as a (non-blocking) conflict
    tagged = [o for o in plan.openings if o.tag]
    assert tagged and all(o.height_m.provenance[0].method == "schedule" for o in tagged)
    for c in plan.conflicts:
        assert c.key.startswith("opening/") and c.severity == "warning"
        o = next(x for x in plan.openings if x.id == c.key.split("/")[1])
        assert o.width_m.value == c.candidates[c.proposed]["value"]


def _gt_of(svc: Services, g: GoldenProject, plan: PlanGraph) -> dict[str, Any]:
    t = plan.doc_transforms[0]
    name = svc.db.one("SELECT filename FROM documents WHERE id = ?", (t.doc_id,))["filename"]
    gt = next(d for d in g.docs if d.filename == name).gt
    assert gt is not None
    return gt


def test_g3_office_from_ifc_and_its_imperial_sheet(svc: Services, project_id: str) -> None:
    g = g3_office()
    plan, _ = _run(svc, project_id, g)
    assert plan.source == "ifc" and len(plan.levels) == 2
    ground = _level(plan, plan.levels[0].id)
    s = plan_scores(ground, transform_plan(gt_plan(g.spec), [[1, 0, 0], [0, 1, 0]]))  # type: ignore[arg-type]
    assert s["walls"]["f1"] >= 0.98 and s["rooms"]["f1"] >= 0.95, s
    # the imperial sheet is not needed (the model wins) but S2 reads it to scale: feet and inches
    gt = g.docs[1].gt
    assert gt is not None and gt["scale"] in (48, 96)  # 1/4" or 1/8" = 1'-0"
    inp = s2_input(svc, project_id)
    page = next(p for p in inp.pages if p.page.kind == "pdf_page")
    assert page.label == "floor_plan" and page.analysis is not None
    assert page.analysis.scale is not None and page.analysis.scale.value == gt["scale"]
    store = svc.store(project_id)
    res = Extractor(svc, store, project_id).page(page, "pdf_vector")
    true_m = gt["scale"] * 0.0254 / 300.0
    assert res.scale is not None
    assert abs(res.scale["m_per_unit"] - true_m) / true_m <= 0.01, res.scale
    methods = {e["method"] for e in res.scale["estimates"]}
    assert "dimensions" in methods, res.scale  # 12'-6" strings read as lengths
    sp = plan_scores(
        res.plan,
        transform_plan(
            gt_plan(g.spec), compose(gt["plan_to_page_px"], [list(r) for r in res.transform.matrix])
        ),
    )  # type: ignore[arg-type]
    assert sp["walls"]["f1"] >= 0.98, sp
    _ = (json, np, StageContext)
