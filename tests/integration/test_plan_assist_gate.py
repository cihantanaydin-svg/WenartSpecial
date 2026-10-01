"""The assist path in S2 and at Gate A: a raster plan through S1 OCR and S2 with a hint source
standing in for the VLM (ground-truth hints), confirmation, suggestions and training examples."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from archrender.core.config import Settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.plan import PlanGraph
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from archrender.plan import stage as s2
from archrender.plan import versions
from archrender.synth.hints import OracleHints, gt_elements_px
from archrender.synth.plan import gt_plan, random_spec
from archrender.synth.raster import scan
from archrender.synth.sheets import floor_plan_page
from tests.helpers import plan_dxf, upload_bytes


@pytest.fixture
def svc(settings: Settings, project_id: str) -> Services:
    return Services.create(settings)


def _stamp_sheet() -> tuple[bytes, str, list[Any]]:
    """The hatch sheet whose approval stamp makes the extractor lose a wall junction."""
    rng = np.random.default_rng(702)
    spec = random_spec(rng, variant="skewed")
    sheet = floor_plan_page(rng, spec=spec, wall_style="hatch")
    data, media, gt = scan(sheet.pdf, sheet.gt, rng, dpi=300.0, quality="clean")
    return data, media, gt_elements_px(gt_plan(spec), gt["plan_to_page_px"])


def test_s2_measures_a_hinted_wall_and_gate_a_waits_for_confirmation(
    svc: Services, project_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, media, hints = _stamp_sheet()
    oracle = OracleHints(hints, np.random.default_rng(3), noise_px=6.0)
    monkeypatch.setattr(s2, "hint_source", lambda _svc: oracle)
    upload_bytes(svc, project_id, f"Tarama.{media.split('/')[1]}", data)
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    [v1] = versions.list_versions(svc, project_id)
    src = v1["extraction"]["sources"][0]
    assert src["source"] == "raster"
    assist = src["assist"]
    assert assist["source"] == "oracle" and assist["calls"] == len(oracle.calls) >= 1
    assert assist["accepted"] >= 1 and assist["triggers"][0]["kind"] == "validator_failed"
    codes = {i["code"] for i in v1["issues"]}
    assert "PLAN_ASSIST_UNCONFIRMED" in codes and "ROOM_AREA_MISMATCH" not in codes
    with pytest.raises(ArchRenderError) as e:
        versions.approve(svc, project_id, v1["id"], user_id="usr_test")
    assert "PLAN_ASSIST_UNCONFIRMED" in e.value.message

    # unsnapped hints are suggestions; rejecting one is recorded once
    sugg = versions.suggestions(svc, project_id, v1["id"])
    assert sugg and all(s["decision"] is None and not s["accepted"] for s in sugg)
    assert (
        versions.decide_suggestion(
            svc, project_id, v1["id"], sugg[0]["id"], accept=False, user_id="usr_test"
        )
        is None
    )
    with pytest.raises(ArchRenderError) as e:
        versions.decide_suggestion(
            svc, project_id, v1["id"], sugg[0]["id"], accept=False, user_id="usr_test"
        )
    assert e.value.code == ErrorCode.CONFLICT

    plan = versions.load(svc, project_id, v1["id"])
    ids = [w.id for w in plan.walls if w.thickness_m.provenance[0].method == "vlm_assisted"]
    with pytest.raises(ArchRenderError):
        versions.confirm_assists(
            svc, project_id, v1["id"], [*["W1", *ids][:1], "nope"], user_id="u"
        )
    v2 = versions.confirm_assists(svc, project_id, v1["id"], ids, user_id="usr_test")
    p2 = versions.load(svc, project_id, v2)
    confirmed = [w for w in p2.walls if w.id in ids]
    assert all(w.thickness_m.provenance[0].assist.user_confirmed for w in confirmed)  # type: ignore[union-attr]
    assert "PLAN_ASSIST_UNCONFIRMED" not in {i.code for i in p2.issues}
    examples = versions.training_examples(svc, project_id)
    kinds = sorted(e["payload"]["action"] for e in examples if e["kind"] == "assist_decision")
    assert kinds == ["confirmed", "rejected"]
    assert all(e["training_use_allowed"] is False for e in examples)
    # the decision follows the extraction into later versions
    assert versions.suggestions(svc, project_id, v2)[0]["decision"] == "rejected"


def test_accepting_a_suggestion_inserts_the_users_wall_and_rederives_rooms(
    svc: Services, project_id: str
) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    [v] = versions.list_versions(svc, project_id)
    full = versions.load(svc, project_id, v["id"])
    t = full.doc_transforms[0]
    page = f"{t.doc_id}_p{t.page}"
    # an extraction that missed an interior wall, with the VLM's hint for it as a suggestion
    inner = next(w for w in full.walls if w.kind != "exterior")
    hosted = {o.id for o in full.openings if o.host_wall == inner.id}
    d = full.model_dump(mode="json")
    d["walls"] = [w for w in d["walls"] if w["id"] != inner.id]
    d["openings"] = [o for o in d["openings"] if o["id"] not in hosted]
    from archrender.plan.rooms import rederive_rooms

    missed = rederive_rooms(PlanGraph.model_validate(d))
    assert len(missed.rooms) < len(full.rooms)
    m = np.array(t.matrix, float)
    to_px = np.linalg.inv(m[:, :2])
    ends = [
        (np.array([p.x, p.y]) - m[:, 2]) @ to_px.T for p in (inner.centerline.a, inner.centerline.b)
    ]  # type: ignore[union-attr]
    extraction = json.loads(versions.get_row(svc, project_id, v["id"])["extraction_json"])
    extraction["sources"][0]["assist"] = {
        "suggestions": [
            {
                "id": f"{page}/sg1",
                "page": page,
                "kind": "wall",
                "hint_px": [e.tolist() for e in ends],
                "reason": "evidence covers < 80 %",
                "accepted": False,
            }
        ]
    }
    vm = versions.record_extraction(svc, project_id, missed, extraction)
    vid = versions.decide_suggestion(
        svc,
        project_id,
        vm,
        f"{page}/sg1",
        accept=True,
        user_id="usr_test",
        thickness_m=inner.thickness_m.value,
    )
    assert vid is not None
    plan = versions.load(svc, project_id, vid)
    added = [w for w in plan.walls if w.id.startswith("WU")]
    assert len(added) == 1 and added[0].thickness_m.provenance[0].method == "user"
    assert len(plan.rooms) == len(full.rooms)

    # the rooms come back as drawn (their names were lost with the merge: new rooms are named
    # "Mahal n", an assumption the user sees)
    def areas(p: PlanGraph) -> list[float]:
        from archrender.plan.rooms import _poly

        return sorted(round(_poly(r).area, 2) for r in p.rooms)

    assert areas(plan) == pytest.approx(areas(full), abs=0.02)
    kept = {r.name.value for r in missed.rooms}
    assert kept <= {r.name.value for r in plan.rooms}
    ex = versions.training_examples(svc, project_id)
    assert [e["payload"]["action"] for e in ex if e["kind"] == "assist_decision"] == ["accepted"]


def test_without_a_hint_source_triggers_are_reported_for_gate_a(
    svc: Services, project_id: str, tmp_path: Path
) -> None:
    data, media, _ = _stamp_sheet()
    upload_bytes(svc, project_id, f"Tarama.{media.split('/')[1]}", data)
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    [v1] = versions.list_versions(svc, project_id)
    src = v1["extraction"]["sources"][0]
    assert src["assist"]["source"] is None and src["assist"]["calls"] == 0
    assert src["assist"]["triggers"]
    assert any("no VLM is serving" in n for n in src["notes"])
    Image.open(io.BytesIO(data))
