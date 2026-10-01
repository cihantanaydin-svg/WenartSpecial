"""Run orchestration: the section pipeline S2 → Gate A → S4 → Gate B → S5/S6 → Gate C → S7–S9 → Gate D → S10.

A run job is re-entrant. When a gate needs a human, the job is parked. On resume it starts again from
the top, completed stages hit the cache, and decided gates are read from the database.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any, Literal

from pydantic import Field

from archrender.brief.overrides import (
    apply_material_overrides,
    check_elements_exist,
    validate_material_overrides,
)
from archrender.core.config import REPO_ROOT
from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.hashing import sha256_json
from archrender.core.ids import new_id, now_iso
from archrender.core.schemas.common import Strict
from archrender.core.schemas.jobs import GateName, JobKind
from archrender.core.schemas.manifest import RunManifest
from archrender.core.schemas.provenance import Assumption
from archrender.core.schemas.scene import RenderSettings
from archrender.pipeline.engine import StageContext, StageEngine
from archrender.pipeline.gates import GatePolicy, Gates, decide
from archrender.pipeline.services import Services
from archrender.pipeline.stages import (
    BriefIn,
    CamerasIn,
    RefineQAIn,
    RenderIn,
    SceneIn,
    build_stages,
    resolve_section,
)
from archrender.plan import versions
from archrender.plan.stage import current_plan, version_plan
from archrender.qa.policy import delivered_checks
from archrender.qa.runner import families_covered
from archrender.refine.prompt import compile_prompt
from archrender.report.bundle import BundleInput, BundleView, build_bundle


class RunConfig(Strict):
    room_ids: list[str] | None = None  # None = all rooms of the approved plan
    mode: Literal["faithful", "concept"] = "faithful"
    views: int = Field(default=3, ge=1, le=12)
    width: int = Field(default=3840, ge=16, le=8192)
    height: int = Field(default=2160, ge=16, le=8192)
    samples: int | None = Field(default=None, ge=1, le=8192)
    gate_policy: GatePolicy = "on_low_confidence"
    seed: int = Field(default=0, ge=0)
    blend_file: bool = False
    # Gate B edit: surface → library material id (see archrender.brief.overrides)
    materials: dict[str, str] = Field(default_factory=dict)


def create_run(svc: Services, project_id: str, config: RunConfig, user_id: str) -> tuple[str, str]:
    prof = svc.profile.render
    if config.width > prof.max_width or config.height > prof.max_height:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"{config.width}×{config.height} exceeds the {svc.profile.name} profile limit "
            f"{prof.max_width}×{prof.max_height}.",
            "Lower the resolution or deploy a larger GPU profile.",
        )
    validate_material_overrides(config.materials, svc.library)
    run_id = new_id("run")
    job_id = new_id("job")
    with svc.db.tx(immediate=True) as c:
        svc.queue.enqueue(
            project_id,
            JobKind.RUN,
            "gpu",
            {"run_id": run_id},
            created_by=user_id,
            job_id=job_id,
            conn=c,
        )
        c.execute(
            "INSERT INTO runs(id, project_id, job_id, config_json, status, created_by, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (run_id, project_id, job_id, config.model_dump_json(), "queued", user_id, now_iso()),
        )
    return run_id, job_id


def decide_gate(
    svc: Services,
    run_id: str,
    gate: str,
    *,
    approve: bool,
    user_id: str,
    notes: str | None,
    plan_version: str | None = None,
) -> None:
    """A reviewer's gate decision. Approving Gate A approves a plan version: the one named, or
    the latest edit of the version the run was parked with; the run then renders that version.
    The plan is approved first, so a plan with blocking issues leaves the gate pending."""
    run = svc.db.one("SELECT project_id, plan_version FROM runs WHERE id = ?", (run_id,))
    if run is None:
        raise not_found("Run", run_id)
    if plan_version is not None and gate != GateName.A.value:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"A plan version can only be chosen at Gate {GateName.A.value}.",
            "Drop plan_version from the decision.",
        )
    if gate == GateName.A.value and approve:
        g = svc.db.one("SELECT status FROM gates WHERE run_id = ? AND gate = ?", (run_id, gate))
        if g is not None and g["status"] == "pending":
            pid = run["project_id"]
            if plan_version is None:
                if not run["plan_version"]:
                    raise ArchRenderError(
                        ErrorCode.CONFLICT,
                        "The run has no plan version yet.",
                        "Wait until the run reaches Gate A.",
                    )
                parked = versions.get_row(svc, pid, run["plan_version"])
                plan_version = str(versions.head(svc, pid, parked["root_id"])["id"])
            versions.approve(svc, pid, plan_version, user_id=user_id)
            svc.db.execute("UPDATE runs SET plan_version = ? WHERE id = ?", (plan_version, run_id))
            notes = (
                f"{notes} (plan version {plan_version})"
                if notes
                else f"plan version {plan_version}"
            )
    decide(svc.db, run_id, gate, approve=approve, user_id=user_id, notes=notes)


def _git_commit() -> str | None:
    env = os.environ.get("ARCHRENDER_GIT_COMMIT")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or None
    except OSError:
        return None


class RunOrchestrator:
    def __init__(self, svc: Services, job_id: str, run_id: str, ctx: StageContext) -> None:
        self.svc = svc
        self.job_id = job_id
        self.run_id = run_id
        self.ctx = ctx
        self.engine = StageEngine(svc.db)
        self.stages = build_stages(svc)

    def _project_name(self) -> str:
        row = self.svc.db.one("SELECT name FROM projects WHERE id = ?", (self.ctx.project_id,))
        if row is None:
            raise not_found("Project", self.ctx.project_id)
        return str(row["name"])

    def execute(self) -> dict[str, Any]:
        svc, ctx, eng, st = self.svc, self.ctx, self.engine, self.stages
        row = svc.db.one("SELECT config_json FROM runs WHERE id = ?", (self.run_id,))
        if row is None:
            raise not_found("Run", self.run_id)
        cfg = RunConfig.model_validate_json(row["config_json"])
        svc.db.execute("UPDATE runs SET status = 'running' WHERE id = ?", (self.run_id,))
        gates = Gates(svc.db, self.run_id, cfg.gate_policy)
        if not svc.db.one("SELECT 1 FROM documents WHERE project_id = ?", (ctx.project_id,)):
            raise ArchRenderError(
                ErrorCode.PLAN_NO_PLAN_FOUND,
                "The project has no ingested documents.",
                "Upload at least one plan document (PDF, DXF, IFC or image) before starting a run.",
            )

        ctx.progress(0.02, "S2 plan")
        # a run keeps the plan version it started with (a resumed run must not pick up a plan
        # approved in the meantime); a Gate A approval of an edited version re-pins it
        pinned = svc.db.one("SELECT plan_version FROM runs WHERE id = ?", (self.run_id,))
        if pinned is not None and pinned["plan_version"]:
            cur = version_plan(svc, ctx.project_id, pinned["plan_version"])
        else:
            cur = current_plan(svc, ctx, eng)
            svc.db.execute(
                "UPDATE runs SET plan_version = ? WHERE id = ?", (cur.version_id, self.run_id)
            )
        plan = cur.plan
        plan_evidence = {
            "plan_version": cur.version_id,
            "plan_number": cur.number,
            "plan_status": cur.status,
            "extraction_version": cur.extraction_version,
            "plan_source": plan.source,
            "issues": [i.model_dump(mode="json") for i in cur.issues],
            "open_conflicts": [c.key for c in plan.conflicts if c.resolution is None],
            "sources": cur.extraction["sources"],
        }
        if cur.status == "approved":
            # the reviewer approved this plan version at Gate A (possibly for an earlier run)
            gates.record_approval(GateName.A, cur.approved_by, plan_evidence)
        else:
            gates.check(
                GateName.A,
                auto_ok=not cur.blocking,
                mandatory=plan.source == "raster",
                evidence=plan_evidence,
            )

        section = resolve_section(plan, cfg.room_ids)
        check_elements_exist(cfg.materials, plan)
        ctx.progress(0.08, "S4 brief")
        brief_out = eng.run(st["brief"], BriefIn(plan_version=plan.version, section=section), ctx)
        brief, brief_assumptions = apply_material_overrides(
            brief_out.brief, brief_out.assumptions, cfg.materials
        )
        gates.check(
            GateName.B,
            auto_ok=not brief.contradictions,
            evidence={
                "assumptions": [a.model_dump(mode="json") for a in brief_assumptions],
                "material_overrides": cfg.materials,
            },
        )

        render = RenderSettings(
            device=svc.profile.render.device,
            width=cfg.width,
            height=cfg.height,
            samples=cfg.samples or svc.profile.render.samples,
            seed=cfg.seed,
        )
        ctx.progress(0.12, "S5 scene")
        scene_out = eng.run(
            st["scene"],
            SceneIn(
                plan_json=cur.plan_json,
                section=section,
                brief=brief,
                render=render,
                blend=cfg.blend_file,
            ),
            ctx,
        )
        ctx.progress(0.2, "S6 cameras")
        cams_out = eng.run(
            st["cameras"],
            CamerasIn(
                plan_json=cur.plan_json,
                room_ids=section.room_ids,
                views=cfg.views,
                width=cfg.width,
                height=cfg.height,
            ),
            ctx,
        )
        gates.check(
            GateName.C,
            auto_ok=all(s > 0.2 for s in cams_out.scores),
            evidence={
                "cameras": [c.model_dump(mode="json") for c in cams_out.cameras],
                "scores": cams_out.scores,
            },
        )

        materials = {
            s.surface: svc.library.get(s.material_id).id.replace("_", " ") for s in brief.surfaces
        }
        prompt, template_hash = compile_prompt(brief, materials)
        views: list[BundleView] = []
        seeds: dict[str, int] = {}
        render_info: dict[str, Any] = {}
        n = len(cams_out.cameras)
        for i, cam in enumerate(cams_out.cameras):
            base_p = 0.25 + 0.65 * i / n
            ctx.progress(base_p, f"S7 render {cam.id}")
            r = eng.run(
                st["render"],
                RenderIn(scene_json=scene_out.scene_json, meshes=scene_out.meshes, camera=cam),
                ctx,
            )
            render_info = r.info or render_info
            ctx.progress(base_p + 0.65 / n * 0.4, f"S8/S9 refine+QA {cam.id}")
            view_id = f"view_{i + 1}"
            seed = cfg.seed * 10_000 + (i + 1) * 101
            seeds[view_id] = seed
            q = eng.run(
                st["refine_qa"],
                RefineQAIn(
                    view_id=view_id,
                    camera_id=cam.id,
                    render=r,
                    scene_json=scene_out.scene_json,
                    prompt=prompt,
                    prompt_template=template_hash,
                    seed=seed,
                ),
                ctx,
            )
            views.append(
                BundleView(
                    outcome=q.outcome,
                    delivered_png=q.delivered_png,
                    delivered_jpg=q.delivered_jpg,
                    base_png=r.beauty_png,
                    camera_json=r.camera_json,
                    log=q.log,
                )
            )

        all_checks = [c for v in views for c in delivered_checks(v.outcome)]
        covered = families_covered(all_checks)
        missing = sorted(set(svc.qa_config.required_families) - covered)
        gate_d_auto = (
            all(v.outcome.status == "refined" for v in views)
            and all(c.passed for c in all_checks)
            and not missing
        )
        gates.check(
            GateName.D,
            auto_ok=gate_d_auto,
            evidence={
                "views": [
                    {
                        "view_id": v.outcome.view_id,
                        "status": v.outcome.status,
                        "reason": v.outcome.reason,
                    }
                    for v in views
                ],
                "families_not_covered_by_real_models": missing,
            },
        )

        ctx.progress(0.95, "S10 bundle")
        assumptions: list[Assumption] = [
            *brief_assumptions,
            *scene_out.assumptions,
            *cams_out.assumptions,
        ]
        manifest = RunManifest(
            run_id=self.run_id,
            project_id=ctx.project_id,
            git_commit=_git_commit(),
            image_digest=os.environ.get("ARCHRENDER_IMAGE_DIGEST"),
            config_hash=sha256_json({"profile": svc.profile, "qa": svc.qa_config, "run": cfg})[:16],
            profile=svc.profile.name,
            models=[svc.models.ref(r) for r in ("refiner", "depth", "segmenter")],
            seeds=seeds,
            environment={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "blender_mode": svc.settings.blender_mode,
                "blender": render_info.get("blender"),
                "render_device": json.dumps(render_info.get("device", {}), sort_keys=True),
            },
            degradations=svc.models.degradations,
            timings=eng.timings,
            uses_mocks=plan.source == "mock"
            or any(v.outcome.uses_mocks for v in views)
            or svc.models.any_mock_used(),
        )
        gate_rows = [
            dict(r)
            for r in svc.db.query(
                "SELECT gate, status, policy, decided_by, notes FROM gates WHERE run_id = ?",
                (self.run_id,),
            )
        ]
        bundle_input = BundleInput(
            run_id=self.run_id,
            project_name=self._project_name(),
            mode=cfg.mode,
            plan_json=cur.plan_json,
            scene_json=scene_out.scene_json,
            glb=scene_out.glb,
            blend_file=scene_out.blend_file,
            views=views,
            assumptions=assumptions,
            gates=gate_rows,
            manifest=manifest,
        )
        zip_ref, report_ref = build_bundle(bundle_input, ctx.store)
        bundle_id = new_id("bdl")
        with svc.db.tx(immediate=True) as c:
            c.execute(
                "INSERT INTO bundles(id, project_id, run_id, job_id, sha256, size, status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    bundle_id,
                    ctx.project_id,
                    self.run_id,
                    self.job_id,
                    zip_ref.sha256,
                    zip_ref.size,
                    "ready",
                    now_iso(),
                ),
            )
        result = {
            "run_id": self.run_id,
            "bundle_id": bundle_id,
            "bundle": zip_ref.model_dump(),
            "report": report_ref.model_dump(),
            "glb": scene_out.glb.model_dump(),
            "uses_mocks": manifest.uses_mocks,
            "views": [
                {
                    "view_id": v.outcome.view_id,
                    "camera_id": v.outcome.camera_id,
                    "status": v.outcome.status,
                    "reason": v.outcome.reason,
                    "delivered": v.delivered_png.model_dump(),
                    "delivered_jpg": v.delivered_jpg.model_dump(),
                    "base": v.base_png.model_dump(),
                    "checks": [c.model_dump(mode="json") for c in delivered_checks(v.outcome)],
                    "attempts": len(v.outcome.attempts),
                    "uses_mocks": v.outcome.uses_mocks,
                }
                for v in views
            ],
            "assumptions": [a.model_dump(mode="json") for a in assumptions],
            "plan_version": cur.version_id,
            "plan_issues": [i.model_dump(mode="json") for i in cur.issues],
            "timings": [t.model_dump() for t in eng.timings],
        }
        svc.db.execute(
            "UPDATE runs SET status = 'succeeded', result_json = ? WHERE id = ?",
            (json.dumps(result), self.run_id),
        )
        return result
