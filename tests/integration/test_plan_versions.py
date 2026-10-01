"""S2 in the pipeline and Gate A: extraction → draft version → edits → approval → training example."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from shapely.geometry import Polygon

from archrender.core.config import Settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import Severity
from archrender.core.schemas.plan import PlanGraph
from archrender.core.schemas.provenance import Conflict
from archrender.pipeline.engine import StageContext
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from archrender.plan import versions
from archrender.plan.stage import current_plan, has_plan_source
from tests.helpers import MINIMAL_DXF, plan_dxf, upload_bytes


@pytest.fixture
def svc(settings: Settings, project_id: str) -> Services:
    return Services.create(settings)


def _ctx(svc: Services, project_id: str) -> StageContext:
    return StageContext(project_id, svc.store(project_id), lambda *_: None, lambda: False)


def _jobs(svc: Services, project_id: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in svc.db.query(
        "SELECT kind, status FROM jobs WHERE project_id = ? ORDER BY rowid", (project_id,)
    ):
        out.setdefault(r["kind"], []).append(r["status"])
    return out


def _extracted(svc: Services, project_id: str) -> str:
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    rows = versions.list_versions(svc, project_id)
    assert len(rows) == 1, rows
    return str(rows[0]["id"])


def test_upload_queues_s1_then_s2_and_stores_a_draft(svc: Services, project_id: str) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    vid = _extracted(svc, project_id)
    assert _jobs(svc, project_id) == {
        "intake": ["succeeded"],
        "understand": ["succeeded"],
        "plan": ["succeeded"],
    }
    job = svc.db.one("SELECT id FROM jobs WHERE kind = 'plan' AND project_id = ?", (project_id,))
    res = svc.queue.get(job["id"]).result
    assert res is not None
    assert res["plan_version"] == vid and res["status"] == "draft" and res["source"] == "dxf"
    assert res["blocking"] == 0 and res["walls"] >= 4 and res["rooms"] >= 2
    v = versions.summary(versions.get_row(svc, project_id, vid))
    assert v["origin"] == "extraction" and v["root_id"] == vid and v["number"] == 1
    assert v["extraction"]["sources"][0]["scale"] == {"m_per_unit": 0.01, "method": "dxf_entity"}
    plan = versions.load(svc, project_id, vid)
    assert plan.version == vid and plan.project == project_id

    # the same pages again: S2 is a cache hit and the draft is reused, not duplicated
    cur = current_plan(svc, _ctx(svc, project_id))
    assert cur.version_id == vid and cur.status == "draft"
    assert len(versions.list_versions(svc, project_id)) == 1


def test_no_plan_job_without_a_plan_source(svc: Services, project_id: str) -> None:
    # one line is no floor plan: S1 runs, S2 is not queued (nothing to extract)
    upload_bytes(svc, project_id, "çizgi.dxf", MINIMAL_DXF)
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    assert not has_plan_source(svc, project_id)
    assert "plan" not in _jobs(svc, project_id)


def _fact(plan: dict[str, Any], path: str) -> Any:
    cur: Any = plan
    for t in path.strip("/").split("/"):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


def test_edit_approve_round_trip_keeps_versions_immutable(svc: Services, project_id: str) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    v1 = _extracted(svc, project_id)
    p1 = versions.load(svc, project_id, v1)
    old_width = p1.openings[0].width_m.value
    ops = [
        {"op": "test", "path": "/openings/0/id", "value": p1.openings[0].id},
        {"op": "replace", "path": "/openings/0/width_m/value", "value": old_width + 0.1},
        {"op": "replace", "path": "/rooms/0/name/value", "value": "Çalışma Odası"},
    ]
    v2 = versions.edit(svc, project_id, v1, ops, user_id="usr_test", note="door width per site")
    p2 = versions.load(svc, project_id, v2)
    # the edited facts carry user provenance; the others keep their measured provenance
    w = p2.openings[0].width_m
    assert w.value == pytest.approx(old_width + 0.1)
    assert w.status == "user_confirmed" and w.provenance[0].method == "user"
    assert w.provenance[0].note == "door width per site"
    assert p2.rooms[0].name.value == "Çalışma Odası"
    assert p2.openings[1].width_m.provenance == p1.openings[1].width_m.provenance
    assert p2.source == "dxf" and p2.version == v2
    # the source version is unchanged
    assert versions.load(svc, project_id, v1).openings[0].width_m.value == old_width
    row2 = versions.get_row(svc, project_id, v2)
    assert row2["parent_id"] == v1 and row2["root_id"] == v1 and row2["origin"] == "edit"
    assert row2["status"] == "draft"

    # the run uses the latest edit while nothing is approved, the approved one afterwards
    ctx = _ctx(svc, project_id)
    assert current_plan(svc, ctx).version_id == v2
    out = versions.approve(svc, project_id, v2, user_id="usr_test")
    assert out["status"] == "approved" and out["approved_by"] == "usr_test"
    v3 = versions.edit(
        svc,
        project_id,
        v2,
        [{"op": "replace", "path": "/rooms/0/name/value", "value": "Ofis"}],
        user_id="usr_test",
    )
    assert current_plan(svc, ctx).version_id == v2  # a later draft needs its own approval
    assert versions.head(svc, project_id, v1)["id"] == v3
    versions.approve(svc, project_id, v3, user_id="usr_test")
    assert versions.get_row(svc, project_id, v2)["status"] == "superseded"
    assert current_plan(svc, ctx).version_id == v3

    # the corrections are kept as training examples, never allowed for training by default
    ex = versions.training_examples(svc, project_id)
    assert [e["plan_version"] for e in ex] == [v2, v3]
    assert all(e["training_use_allowed"] is False for e in ex)
    raw = svc.db.query("SELECT training_use_allowed FROM training_examples")
    assert {r["training_use_allowed"] for r in raw} == {0}
    last = ex[1]["payload"]
    assert last["extraction_version"] == v1 and last["approved_version"] == v3
    assert [e["to"] for e in last["edits"]] == [v2, v3]
    assert last["edits"][0]["patch"] == ops and last["edits"][0]["by"] == "usr_test"
    assert last["pages"] and last["pages"][0].endswith("_p0")


def test_approval_is_refused_while_blocking_issues_remain(svc: Services, project_id: str) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    v1 = _extracted(svc, project_id)
    plan = versions.load(svc, project_id, v1)
    # an opening wider than its host wall
    i = 0
    host = next(w for w in plan.walls if w.id == plan.openings[i].host_wall)
    too_wide = host.centerline.length() + 1.0  # type: ignore[union-attr]
    v2 = versions.edit(
        svc,
        project_id,
        v1,
        [{"op": "replace", "path": f"/openings/{i}/width_m/value", "value": too_wide}],
        user_id="usr_test",
    )
    summary = versions.summary(versions.get_row(svc, project_id, v2))
    assert summary["blocking"] >= 1
    with pytest.raises(ArchRenderError) as e:
        versions.approve(svc, project_id, v2, user_id="usr_test")
    assert e.value.code == ErrorCode.PLAN_INVALID
    assert versions.get_row(svc, project_id, v2)["status"] == "draft"


@pytest.mark.parametrize(
    ("ops", "message"),
    [
        ([{"op": "remove", "path": "/walls/0"}], "unknown wall"),  # an opening loses its host
        ([{"op": "replace", "path": "/rooms/0/polygon", "value": []}], "at least 3"),
        ([{"op": "replace", "path": "/nope/0", "value": 1}], "cannot be applied"),
        ([{"op": "test", "path": "/source", "value": "ifc"}], "test failed"),
        ([{"op": "jump", "path": "/source"}], "unknown operation"),
    ],
)
def test_invalid_edits_are_refused_and_create_nothing(
    svc: Services, project_id: str, ops: list[dict[str, Any]], message: str
) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    v1 = _extracted(svc, project_id)
    plan = versions.load(svc, project_id, v1)
    if message == "unknown wall":  # remove a wall that hosts an opening
        hosts = {o.host_wall for o in plan.openings}
        idx = next(i for i, w in enumerate(plan.walls) if w.id in hosts)
        ops = [{"op": "remove", "path": f"/walls/{idx}"}]
    with pytest.raises(ArchRenderError) as e:
        versions.edit(svc, project_id, v1, ops, user_id="usr_test")
    assert e.value.code == ErrorCode.VALIDATION
    assert message in e.value.message, e.value.message
    assert len(versions.list_versions(svc, project_id)) == 1


def _with_scale_conflict(svc: Services, project_id: str, v1: str, k: float) -> str:
    """The extraction as if a second estimator had measured a scale ``k`` × the one used."""
    plan = versions.load(svc, project_id, v1)
    row = versions.get_row(svc, project_id, v1)
    page = json.loads(row["extraction_json"])["sources"][0]["page"]
    m = 0.01
    plan.conflicts = [
        Conflict(
            key=f"scale/{page}",
            candidates=[
                {"method": "dxf_insunits", "m_per_unit": m},
                {"method": "dimensions", "m_per_unit": m * k},
            ],
            proposed=0,
            rule="test",
            severity=Severity.ERROR,
        )
    ]
    return versions.record_extraction(
        svc,
        project_id,
        PlanGraph.model_validate(plan.model_dump()),
        json.loads(row["extraction_json"]),
    )


def test_resolving_a_scale_conflict_rescales_and_unblocks(svc: Services, project_id: str) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    v1 = _extracted(svc, project_id)
    vc = _with_scale_conflict(svc, project_id, v1, 1.02)
    assert vc != v1  # a different extraction (it carries the conflict)
    s = versions.summary(versions.get_row(svc, project_id, vc))
    assert "PLAN_SCALE_CONFLICT" in {i["code"] for i in s["issues"]} and s["blocking"] >= 1
    with pytest.raises(ArchRenderError):
        versions.approve(svc, project_id, vc, user_id="usr_test")
    base = versions.load(svc, project_id, vc)
    key = base.conflicts[0].key
    with pytest.raises(ArchRenderError):
        versions.resolve_conflict(svc, project_id, vc, key, 5, user_id="usr_test")

    vr = versions.resolve_conflict(svc, project_id, vc, key, 1, user_id="usr_test")
    res = versions.load(svc, project_id, vr)
    c = res.conflicts[0]
    assert c.resolution == 1 and c.resolved_by == "usr_test"
    for a, b in zip(base.walls, res.walls, strict=True):
        assert b.centerline.length() == pytest.approx(a.centerline.length() * 1.02)  # type: ignore[union-attr]
        assert b.thickness_m.value == pytest.approx(a.thickness_m.value * 1.02)
    for a, b in zip(base.rooms, res.rooms, strict=True):
        pa = Polygon([(p.x, p.y) for p in a.polygon])
        pb = Polygon([(p.x, p.y) for p in b.polygon])
        assert pb.area == pytest.approx(pa.area * 1.02**2)
    for a, b in zip(base.openings, res.openings, strict=True):
        assert b.width_m.value == pytest.approx(a.width_m.value * 1.02)
        assert b.height_m.value == a.height_m.value  # heights are not plan measurements
    m0 = base.doc_transforms[0].matrix
    assert res.doc_transforms[0].matrix[0][0] == pytest.approx(m0[0][0] * 1.02)
    assert "PLAN_SCALE_CONFLICT" not in {i.code for i in res.issues}
    edits = svc.db.query("SELECT patch_json, summary FROM plan_edits WHERE to_version = ?", (vr,))
    assert json.loads(edits[0]["patch_json"]) == [{"op": "resolve", "key": key, "choice": 1}]
    # the wrong scale is caught by the room area labels (4 % off): approval stays refused
    assert "ROOM_AREA_MISMATCH" in {i.code for i in res.issues}
    with pytest.raises(ArchRenderError) as e:
        versions.approve(svc, project_id, vr, user_id="usr_test")
    assert "ROOM_AREA_MISMATCH" in e.value.message

    # keeping the proposed candidate changes no geometry, and the plan can be approved
    vk = versions.resolve_conflict(svc, project_id, vc, key, 0, user_id="usr_test")
    kept = versions.load(svc, project_id, vk)
    assert [w.centerline for w in kept.walls] == [w.centerline for w in base.walls]
    assert versions.approve(svc, project_id, vk, user_id="usr_test")["status"] == "approved"


def test_levels_from_separate_sheets_are_stacked_with_an_assumed_storey_height() -> None:
    import numpy as np

    from archrender.core.schemas.plan import DocTransform
    from archrender.plan.stage import STOREY_HEIGHT_M, PageResult, assemble
    from archrender.synth.plan import gt_plan, random_spec

    def page(seed: int, level: str) -> PageResult:
        plan = gt_plan(random_spec(np.random.default_rng(seed), variant="manhattan"))
        t = DocTransform(doc_id=f"doc{seed}", page=0, matrix=((1, 0, 0), (0, 1, 0)), method="dxf")
        return PageResult(f"doc{seed}_p0", "dxf", level, plan, t)

    ground, first = page(1, "Zemin Kat"), page(2, "1. Kat")
    raster_dup = page(3, "Zemin Kat")
    raster_dup.source = "raster"
    plan = assemble("prj", [raster_dup, ground, first], (12.5, "north arrow"))
    assert [lv.name for lv in plan.levels] == ["Zemin Kat", "1. Kat"]
    assert [lv.elevation_m for lv in plan.levels] == [0.0, STOREY_HEIGHT_M]
    assert plan.levels[1].floor_to_floor_m is not None
    assert plan.levels[1].floor_to_floor_m.provenance[0].method == "default"
    # the DXF wins its level over the raster of the same level; ids stay unique per level
    assert len(plan.walls) == len(ground.plan.walls) + len(first.plan.walls)
    assert {w.level for w in plan.walls} == {"L0", "L1"}
    assert all(w.id.startswith(("L0-", "L1-")) for w in plan.walls)
    assert {o.host_wall for o in plan.openings} <= {w.id for w in plan.walls}
    assert [t.doc_id for t in plan.doc_transforms] == ["doc1", "doc2"]
    assert "default/storey_height_m" in {a.key for a in plan.assumptions}
    assert plan.north_angle_deg.value == 12.5


def test_a_scan_of_a_sheet_that_is_also_a_dxf_is_not_extracted(
    svc: Services, project_id: str
) -> None:
    import io

    import numpy as np
    from PIL import Image

    from archrender.synth.raster import scan
    from archrender.synth.sheets import floor_plan_page

    rng = np.random.default_rng(5)
    sheet = floor_plan_page(rng, wall_style="solid")
    data, media, _ = scan(sheet.pdf, sheet.gt, rng, dpi=150.0, quality="clean")
    Image.open(io.BytesIO(data))  # a readable image
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    upload_bytes(svc, project_id, f"Tarama.{media.split('/')[1]}", data)
    Worker(svc, ["cpu", "gpu"], name="w").run_until_idle()
    rows = svc.db.query("SELECT label FROM page_analysis WHERE project_id = ?", (project_id,))
    assert [r["label"] for r in rows] == ["floor_plan", "floor_plan"]
    cur = current_plan(svc, _ctx(svc, project_id))
    ex = cur.extraction
    assert [s["source"] for s in ex["sources"]] == ["dxf"]
    assert len(ex["skipped_rasters"]) == 1 and ex["failed"] == []
    assert cur.plan.source == "dxf"


def test_two_point_calibration_rescales_and_records_the_users_scale(
    svc: Services, project_id: str
) -> None:
    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    v1 = _extracted(svc, project_id)
    plan = versions.load(svc, project_id, v1)
    w = plan.walls[0]
    a = (w.centerline.a.x, w.centerline.a.y)  # type: ignore[union-attr]
    b = (w.centerline.b.x, w.centerline.b.y)  # type: ignore[union-attr]
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    v2 = versions.calibrate(svc, project_id, v1, a, b, length * 1.05, user_id="usr_test")
    p2 = versions.load(svc, project_id, v2)
    assert p2.walls[0].centerline.length() == pytest.approx(length * 1.05)  # type: ignore[union-attr]
    assert "user/scale_calibration" in {x.key for x in p2.assumptions}
    assert all(t.method == "user_calibration" for t in p2.doc_transforms)
    # a calibration also settles an open scale question of the page
    vq = _with_scale_conflict(svc, project_id, v1, 1.0)
    vc2 = versions.calibrate(svc, project_id, vq, a, b, length, user_id="usr_test")
    pc = versions.load(svc, project_id, vc2)
    assert pc.conflicts[0].resolved_by == "usr_test"
    assert "PLAN_SCALE_CONFLICT" not in {i.code for i in pc.issues}
    for bad in ((a, a, 3.0), (a, b, length * 40)):
        with pytest.raises(ArchRenderError):
            versions.calibrate(svc, project_id, v1, bad[0], bad[1], bad[2], user_id="usr_test")


def test_choosing_the_drawn_width_over_the_schedule(svc: Services, project_id: str) -> None:
    from archrender.plan.annotate import apply_schedules

    upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    v1 = _extracted(svc, project_id)
    plan = versions.load(svc, project_id, v1)
    o = plan.openings[0]
    plan.openings[0].tag = "K9"
    sched = {
        "kind": "door_window",
        "source_page": "pg",
        "rows": [{"id": "r", "tag": "K9", "fields": {"width": o.width_m.value + 0.25}}],
    }
    with_sched, _ = apply_schedules(plan, [sched])
    row = versions.get_row(svc, project_id, v1)
    vc = versions.record_extraction(svc, project_id, with_sched, json.loads(row["extraction_json"]))
    pc = versions.load(svc, project_id, vc)
    assert pc.openings[0].width_m.value == pytest.approx(o.width_m.value + 0.25)
    key = f"opening/{o.id}/width"
    vr = versions.resolve_conflict(svc, project_id, vc, key, 0, user_id="usr_test")
    pr = versions.load(svc, project_id, vr)
    assert pr.openings[0].width_m.value == pytest.approx(o.width_m.value, abs=1e-4)
    assert pr.openings[0].width_m.provenance[0].method == "user"
    assert pr.conflicts[0].resolution == 0
