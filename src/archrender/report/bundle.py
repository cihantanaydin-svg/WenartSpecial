"""S10 bundle: renders, model, manifests and the QA report, behind the DeliverableGuard."""

from __future__ import annotations

import io
import json
import zipfile
from importlib import resources
from typing import Any

from jinja2 import Environment, StrictUndefined, select_autoescape

from archrender.core.cas import CasRef, ProjectStore
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.ids import now_iso
from archrender.core.schemas.common import Strict
from archrender.core.schemas.manifest import RunManifest
from archrender.core.schemas.plan import PlanGraph
from archrender.core.schemas.provenance import Assumption
from archrender.core.schemas.qa import ViewOutcome
from archrender.qa.policy import delivered_checks

_FIXED_DATE = (1980, 1, 1, 0, 0, 0)


class BundleView(Strict):
    outcome: ViewOutcome
    delivered_png: CasRef
    delivered_jpg: CasRef
    base_png: CasRef
    camera_json: CasRef
    log: list[str]


class BundleInput(Strict):
    run_id: str
    project_name: str
    mode: str
    plan_json: CasRef
    scene_json: CasRef
    glb: CasRef
    blend_file: CasRef | None
    views: list[BundleView]
    assumptions: list[Assumption]
    gates: list[dict[str, Any]]
    manifest: RunManifest


def deliverable_guard(views: list[BundleView]) -> None:
    """Defense in depth: every delivered image must be the Cycles base or a QA-passing candidate."""
    for v in views:
        o = v.outcome
        if v.delivered_png.sha256 != o.delivered_sha256:
            raise _guard(f"view {o.view_id}: delivered file hash does not match the QA record")
        if o.status == "fallback_base":
            if o.delivered_sha256 != o.base_sha256:
                raise _guard(f"view {o.view_id}: fallback must deliver the Cycles render")
            continue
        checks = delivered_checks(o)
        if not checks:
            raise _guard(f"view {o.view_id}: no QA record for the delivered image")
        failed = [c.name for c in checks if not c.passed]
        if failed:
            raise _guard(f"view {o.view_id}: delivered image failed {failed}")


def _guard(msg: str) -> ArchRenderError:
    return ArchRenderError(
        ErrorCode.QA_DELIVERABLE_GUARD,
        f"Deliverable guard blocked the bundle: {msg}.",
        "This indicates a pipeline bug; the run's QA records are inconsistent. Re-run the section.",
    )


def render_report(inp: BundleInput, plan: PlanGraph) -> str:
    env = Environment(
        undefined=StrictUndefined,
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    text = (
        resources.files("archrender.report")
        .joinpath("templates", "qa_report.html.j2")
        .read_text("utf-8")
    )
    views = [v.outcome for v in inp.views]
    return env.from_string(text).render(
        project_name=inp.project_name,
        run_id=inp.run_id,
        generated_at=now_iso(),
        mode=inp.mode,
        manifest=inp.manifest,
        views=views,
        delivered_checks={v.view_id: delivered_checks(v) for v in views},
        logs={v.outcome.view_id: v.log for v in inp.views},
        gates=inp.gates,
        assumptions=inp.assumptions,
        plan_issues=plan.issues,
        plan_source=plan.source,
    )


def build_bundle(inp: BundleInput, store: ProjectStore) -> tuple[CasRef, CasRef]:
    deliverable_guard(inp.views)
    plan = PlanGraph.model_validate_json(store.read_bytes(inp.plan_json))
    report_html = render_report(inp, plan)
    report_ref = store.put_bytes(report_html.encode("utf-8"), "text/html", "qa_report.html")

    files: list[tuple[str, bytes]] = []
    for v in inp.views:
        vid = v.outcome.view_id
        files.append((f"renders/{vid}.png", store.read_bytes(v.delivered_png)))
        files.append((f"renders/{vid}.jpg", store.read_bytes(v.delivered_jpg)))
        files.append((f"base/{vid}_cycles.png", store.read_bytes(v.base_png)))
        files.append((f"cameras/{vid}.json", store.read_bytes(v.camera_json)))
    files.append(("model/scene.glb", store.read_bytes(inp.glb)))
    if inp.blend_file is not None:
        files.append(("model/scene.blend", store.read_bytes(inp.blend_file)))
    files.append(("plan/plan.json", store.read_bytes(inp.plan_json)))
    scene = json.loads(store.read_bytes(inp.scene_json))
    files.append(
        (
            "manifests/scene_manifest.json",
            json.dumps({"assets": scene.get("assets", [])}, indent=1).encode(),
        )
    )
    files.append(("manifests/run_manifest.json", inp.manifest.model_dump_json(indent=1).encode()))
    files.append(
        (
            "qa/view_outcomes.json",
            json.dumps([v.outcome.model_dump(mode="json") for v in inp.views], indent=1).encode(),
        )
    )
    files.append(("qa/qa_report.html", report_html.encode("utf-8")))
    readme = (
        f"ArchRender bundle for run {inp.run_id}\n"
        "renders/   delivered images (16-bit PNG + JPEG)\n"
        "base/      unrefined Cycles renders (ground truth)\n"
        "cameras/   camera intrinsics/extrinsics (OpenCV + Blender conventions)\n"
        "model/     GLB (and optional .blend)\n"
        "plan/      the approved PlanGraph\n"
        "manifests/ asset licences and the run manifest (models, seeds, environment)\n"
        "qa/        QA report and per-view QA records\n"
    )
    if inp.manifest.uses_mocks:
        readme += "\nWARNING: mock models were used; this bundle is a pipeline test, not a client deliverable.\n"
    files.append(("README.txt", readme.encode()))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in sorted(files):
            info = zipfile.ZipInfo(name, date_time=_FIXED_DATE)
            info.compress_type = (
                zipfile.ZIP_STORED
                if name.endswith((".png", ".jpg", ".glb"))
                else zipfile.ZIP_DEFLATED
            )
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)
    zip_ref = store.put_bytes(buf.getvalue(), "application/zip", f"archrender_{inp.run_id}.zip")
    return zip_ref, report_ref
