from __future__ import annotations

import io
import json
import zipfile

import pytest

from archrender.core.config import Settings
from archrender.core.schemas.jobs import JobStatus
from archrender.pipeline.gates import decide
from archrender.pipeline.run import RunConfig, create_run, decide_gate
from archrender.pipeline.services import Services
from archrender.pipeline.worker import Worker
from tests.helpers import plan_dxf, upload_bytes

pytestmark = pytest.mark.blender


@pytest.fixture
def svc(settings: Settings, project_id: str) -> Services:
    s = Services.create(settings)
    return s


def _run(svc: Services, project_id: str, **cfg: object) -> tuple[str, str]:
    config = RunConfig(views=2, width=96, height=54, samples=8, **cfg)  # type: ignore[arg-type]
    return create_run(svc, project_id, config, "usr_test")


def test_skeleton_end_to_end_with_gates_and_cache(svc: Services, project_id: str) -> None:
    worker = Worker(svc, ["cpu", "gpu"], name="test-worker")
    intake_job = upload_bytes(svc, project_id, "Kat Planı.dxf", plan_dxf())
    worker.run_until_idle()
    intake = svc.queue.get(intake_job)
    assert intake.status == JobStatus.SUCCEEDED, intake.error
    assert intake.result is not None and intake.result["kind"] == "dxf"

    # on_low_confidence: A/B/C auto-pass (validators pass), D waits because QA used mocks
    run_id, job_id = _run(svc, project_id)
    worker.run_until_idle()
    job = svc.queue.get(job_id)
    assert job.status == JobStatus.WAITING_GATE, job.error
    gates = {
        r["gate"]: r["status"]
        for r in svc.db.query("SELECT gate, status FROM gates WHERE run_id = ?", (run_id,))
    }
    assert gates == {
        "A_plan": "auto_passed",
        "B_brief": "auto_passed",
        "C_cameras": "auto_passed",
        "D_final": "pending",
    }
    evidence = json.loads(
        svc.db.one(
            "SELECT evidence_json FROM gates WHERE run_id = ? AND gate = 'D_final'", (run_id,)
        )["evidence_json"]
    )
    assert "brief" in evidence["families_not_covered_by_real_models"]

    decide(svc.db, run_id, "D_final", approve=True, user_id="usr_test", notes="plumbing test")
    svc.queue.resume(job_id)
    worker.run_until_idle()
    job = svc.queue.get(job_id)
    assert job.status == JobStatus.SUCCEEDED, job.error
    result = job.result
    assert result is not None and result["uses_mocks"] is True
    assert len(result["views"]) == 2
    for v in result["views"]:
        assert v["status"] in ("refined", "hard_composite", "fallback_base")
        if v["status"] != "fallback_base":
            assert all(c["passed"] for c in v["checks"])
    # resumed run: everything before Gate D came from the cache
    timings = {t["stage"]: t["cached"] for t in result["timings"]}
    assert timings["S7_render"] is True and timings["S5_scene"] is True

    store = svc.store(project_id)
    zdata = store.read_bytes(store_ref(result["bundle"]))
    with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
        names = set(zf.namelist())
        assert {
            "model/scene.glb",
            "plan/plan.json",
            "qa/qa_report.html",
            "manifests/run_manifest.json",
        } <= names
        assert {"renders/view_1.png", "renders/view_1.jpg", "base/view_1_cycles.png"} <= names
        report = zf.read("qa/qa_report.html").decode()
        assert "Mock models were used" in report and "Mock plan" not in report
        bundled = json.loads(zf.read("plan/plan.json"))
        assert bundled["source"] == "dxf" and bundled["version"] == result["plan_version"]
        manifest = json.loads(zf.read("manifests/run_manifest.json"))
        assert manifest["uses_mocks"] is True and manifest["seeds"]


def test_rerun_hits_cache_and_policy_never(svc: Services, project_id: str) -> None:
    worker = Worker(svc, ["cpu", "gpu"], name="w")
    upload_bytes(svc, project_id, "plan.dxf", plan_dxf())
    worker.run_until_idle()
    _, j1 = _run(svc, project_id, gate_policy="never")
    worker.run_until_idle()
    assert svc.queue.get(j1).status == JobStatus.SUCCEEDED
    _, j2 = _run(svc, project_id, gate_policy="never")
    worker.run_until_idle()
    r2 = svc.queue.get(j2).result
    assert r2 is not None
    assert all(t["cached"] for t in r2["timings"]), r2["timings"]


def test_changing_floor_material_reruns_only_scene_onwards(svc: Services, project_id: str) -> None:
    worker = Worker(svc, ["cpu", "gpu"], name="w")
    upload_bytes(svc, project_id, "plan.dxf", plan_dxf())
    worker.run_until_idle()
    _, j1 = _run(svc, project_id, gate_policy="never")
    worker.run_until_idle()
    assert svc.queue.get(j1).status == JobStatus.SUCCEEDED
    run2, j2 = _run(
        svc, project_id, gate_policy="never", materials={"floor": "stone_porcelain_grey"}
    )
    worker.run_until_idle()
    job = svc.queue.get(j2)
    assert job.status == JobStatus.SUCCEEDED, job.error
    assert job.result is not None
    cached = {t["stage"]: t["cached"] for t in job.result["timings"]}
    # material choices do not change the plan, brief extraction or camera placement …
    assert cached["S2_plan"] and cached["S4_brief"] and cached["S6_cameras"], cached
    # … but everything that depends on the scene re-runs
    assert not cached["S5_scene"] and not cached["S7_render"], cached
    assert not any(v for k, v in cached.items() if k.startswith("S8")), cached
    evidence = json.loads(
        svc.db.one(
            "SELECT evidence_json FROM gates WHERE run_id = ? AND gate = 'B_brief'", (run2,)
        )["evidence_json"]
    )
    assert evidence["material_overrides"] == {"floor": "stone_porcelain_grey"}
    assert "floor_material" not in {a["key"] for a in evidence["assumptions"]}


@pytest.mark.parametrize(
    ("materials", "message"),
    [
        ({"floor": "paint_warm_white"}, "is a wall material"),
        ({"floor": "no_such_material"}, "not in the library"),
        ({"roof": "oak_floor_natural"}, "Unknown surface"),
    ],
)
def test_invalid_material_overrides_are_rejected_up_front(
    svc: Services, project_id: str, materials: dict[str, str], message: str
) -> None:
    from archrender.core.errors import ArchRenderError, ErrorCode

    with pytest.raises(ArchRenderError) as e:
        _run(svc, project_id, materials=materials)
    assert e.value.code == ErrorCode.VALIDATION and message in e.value.message


def test_unknown_plan_element_in_override_fails_the_run(svc: Services, project_id: str) -> None:
    worker = Worker(svc, ["cpu", "gpu"], name="w")
    upload_bytes(svc, project_id, "plan.dxf", plan_dxf())
    worker.run_until_idle()
    _, job_id = _run(
        svc, project_id, gate_policy="never", materials={"wall:W99": "paint_warm_white"}
    )
    worker.run_until_idle()
    job = svc.queue.get(job_id)
    assert job.status == JobStatus.FAILED and job.error is not None
    assert job.error.code == "VALIDATION" and "W99" in job.error.message


def test_gate_a_approves_the_edited_plan_version_the_run_then_renders(
    svc: Services, project_id: str
) -> None:
    from archrender.plan import versions

    worker = Worker(svc, ["cpu", "gpu"], name="w")
    upload_bytes(svc, project_id, "plan.dxf", plan_dxf())
    worker.run_until_idle()
    run_id, job_id = _run(svc, project_id, gate_policy="always")
    worker.run_until_idle()
    assert svc.queue.get(job_id).status == JobStatus.WAITING_GATE
    gate = svc.db.one(
        "SELECT status, evidence_json FROM gates WHERE run_id = ? AND gate = 'A_plan'", (run_id,)
    )
    evidence = json.loads(gate["evidence_json"])
    v1 = evidence["plan_version"]
    assert gate["status"] == "pending" and evidence["plan_status"] == "draft"
    assert evidence["plan_source"] == "dxf" and evidence["sources"][0]["source"] == "dxf"

    # the reviewer renames a room in the editor, then approves Gate A
    v2 = versions.edit(
        svc,
        project_id,
        v1,
        [{"op": "replace", "path": "/rooms/0/name/value", "value": "Atölye"}],
        user_id="usr_test",
    )
    decide_gate(svc, run_id, "A_plan", approve=True, user_id="usr_test", notes="checked")
    assert versions.get_row(svc, project_id, v2)["status"] == "approved"
    assert versions.get_row(svc, project_id, v1)["status"] == "draft"
    for g in ("B_brief", "C_cameras", "D_final"):
        svc.queue.resume(job_id)
        worker.run_until_idle()
        assert svc.queue.get(job_id).status == JobStatus.WAITING_GATE, g
        decide_gate(svc, run_id, g, approve=True, user_id="usr_test", notes=None)
    svc.queue.resume(job_id)
    worker.run_until_idle()
    job = svc.queue.get(job_id)
    assert job.status == JobStatus.SUCCEEDED, job.error
    assert job.result is not None and job.result["plan_version"] == v2
    notes = svc.db.one("SELECT notes FROM gates WHERE run_id = ? AND gate = 'A_plan'", (run_id,))
    assert v2 in notes["notes"]

    # the next run starts from the approved version: Gate A is recorded as approved, not asked
    run2, _ = _run(svc, project_id, gate_policy="always")
    worker.run_until_idle()
    rows = {
        r["gate"]: (r["status"], r["decided_by"])
        for r in svc.db.query(
            "SELECT gate, status, decided_by FROM gates WHERE run_id = ?", (run2,)
        )
    }
    assert rows["A_plan"] == ("approved", "usr_test") and rows["B_brief"][0] == "pending"
    assert svc.db.one("SELECT plan_version FROM runs WHERE id = ?", (run2,))[0] == v2


def test_rejected_gate_fails_run(svc: Services, project_id: str) -> None:
    worker = Worker(svc, ["cpu", "gpu"], name="w")
    upload_bytes(svc, project_id, "plan.dxf", plan_dxf())
    worker.run_until_idle()
    run_id, job_id = _run(svc, project_id, gate_policy="always")
    worker.run_until_idle()
    assert svc.queue.get(job_id).status == JobStatus.WAITING_GATE
    decide(svc.db, run_id, "A_plan", approve=False, user_id="usr_test", notes="wrong scale")
    svc.queue.resume(job_id)
    worker.run_until_idle()
    job = svc.queue.get(job_id)
    assert (
        job.status == JobStatus.FAILED
        and job.error is not None
        and job.error.code == "GATE_REJECTED"
    )


def store_ref(d: dict[str, object]):  # type: ignore[no-untyped-def]
    from archrender.core.cas import CasRef

    return CasRef.model_validate(d)
