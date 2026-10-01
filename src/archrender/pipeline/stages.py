"""Stage definitions (typed I/O, cached by the engine): S4…S9. S1 and S2 live with their
packages (``understand.stage``, ``plan.stage``)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pydantic import Field

from archrender.brief.defaults import default_brief
from archrender.camera.intrinsics import camera_json
from archrender.camera.propose import propose_cameras
from archrender.core.assumptions import AssumptionRegister
from archrender.core.cas import CasRef
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.hashing import sha256_json
from archrender.core.schemas.brief import DesignBrief
from archrender.core.schemas.common import Strict
from archrender.core.schemas.plan import PlanGraph
from archrender.core.schemas.provenance import Assumption
from archrender.core.schemas.qa import ViewOutcome
from archrender.core.schemas.scene import CameraSpec, RenderSettings, SceneSpec
from archrender.core.schemas.section import Section
from archrender.pipeline.engine import StageContext, StageDef
from archrender.pipeline.services import Services
from archrender.qa.images import decode, encode_jpeg, encode_png16
from archrender.qa.policy import run_view_policy
from archrender.qa.runner import QAContext
from archrender.render.passes import RenderPasses, line_art, read_passes
from archrender.scene.compiler import SceneCompiler, section_rooms

BLENDER_VERSION = "5.2.2"


# ---- I/O models ------------------------------------------------------------------------------
class BriefIn(Strict):
    plan_version: str
    section: Section


class BriefOut(Strict):
    brief: DesignBrief
    assumptions: list[Assumption]


class SceneIn(Strict):
    plan_json: CasRef
    section: Section
    brief: DesignBrief
    render: RenderSettings
    blend: bool = False


class SceneOut(Strict):
    spec: SceneSpec
    scene_json: CasRef
    meshes: dict[str, CasRef]
    glb: CasRef
    blend_file: CasRef | None = None
    assumptions: list[Assumption]


class CamerasIn(Strict):
    plan_json: CasRef
    room_ids: list[str]
    views: int
    width: int
    height: int


class CamerasOut(Strict):
    cameras: list[CameraSpec]
    scores: list[float]
    visible_floor: list[float]
    assumptions: list[Assumption]


class RenderIn(Strict):
    scene_json: CasRef
    meshes: dict[str, CasRef]
    camera: CameraSpec


class RenderOut(Strict):
    beauty_png: CasRef
    passes_npz: CasRef
    passes_exr: CasRef
    lineart_png: CasRef
    camera_json: CasRef
    info: dict[str, Any] = Field(default_factory=dict)


class RefineQAIn(Strict):
    view_id: str
    camera_id: str
    render: RenderOut
    scene_json: CasRef
    prompt: str
    prompt_template: str
    seed: int


class RefineQAOut(Strict):
    outcome: ViewOutcome
    delivered_png: CasRef
    delivered_jpg: CasRef
    candidates: list[CasRef]
    log: list[str]


# ---- helpers ---------------------------------------------------------------------------------
def load_model[M: Strict](ctx: StageContext, ref: CasRef, model: type[M]) -> M:
    return model.model_validate_json(ctx.store.read_bytes(ref))


def materialize_package(
    ctx: StageContext,
    scene_json: CasRef,
    meshes: dict[str, CasRef],
    spec: SceneSpec | None,
    prefix: str,
) -> Path:
    pkg = ctx.store.scratch_dir(prefix)
    (pkg / "meshes").mkdir()
    for rel, ref in meshes.items():
        dest = pkg / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ctx.store.path(ref), dest)
    data = (
        spec.model_dump_json(indent=1)
        if spec is not None
        else ctx.store.read_bytes(scene_json).decode()
    )
    (pkg / "scene.json").write_text(data, encoding="utf-8")
    return pkg


def build_stages(svc: Services) -> dict[str, StageDef[Any, Any]]:
    """Stage definitions bound to the service container (config subsets + model identities)."""
    prof = svc.profile

    def s_brief(inp: BriefIn, ctx: StageContext) -> BriefOut:
        reg = AssumptionRegister("S4")
        brief = default_brief(inp.section.id, reg)
        return BriefOut(brief=brief, assumptions=reg.items())

    def s_scene(inp: SceneIn, ctx: StageContext) -> SceneOut:
        plan = load_model(ctx, inp.plan_json, PlanGraph)
        reg = AssumptionRegister("S5")
        scene_id = (
            "scn_"
            + sha256_json(
                {"plan": inp.plan_json.sha256, "section": inp.section, "brief": inp.brief}
            )[:16]
        )
        compiled = SceneCompiler(svc.library).compile(
            plan, inp.section, inp.brief, inp.render, reg, scene_id=scene_id
        )
        spec = compiled.spec.model_copy(
            update={"exports": compiled.spec.exports.model_copy(update={"blend": inp.blend})}
        )
        meshes = {
            p: ctx.store.put_bytes(b, "application/x-npz", p.rsplit("/", 1)[-1])
            for p, b in compiled.mesh_blobs.items()
        }
        scene_ref = ctx.store.put_bytes(
            spec.model_dump_json(indent=1).encode(), "application/json", "scene.json"
        )
        pkg = materialize_package(ctx, scene_ref, meshes, None, "scene")
        out_dir = pkg / "out"
        try:
            svc.blender.run(["export", str(pkg), str(out_dir)], cwd=pkg, cancelled=ctx.cancelled)
            glb = ctx.store.put_file(out_dir / "scene.glb", "model/gltf-binary", "scene.glb")
            blend = (
                ctx.store.put_file(out_dir / "scene.blend", "application/x-blender", "scene.blend")
                if inp.blend
                else None
            )
        finally:
            shutil.rmtree(pkg, ignore_errors=True)
        return SceneOut(
            spec=spec,
            scene_json=scene_ref,
            meshes=meshes,
            glb=glb,
            blend_file=blend,
            assumptions=reg.items(),
        )

    def s_cameras(inp: CamerasIn, ctx: StageContext) -> CamerasOut:
        plan = load_model(ctx, inp.plan_json, PlanGraph)
        reg = AssumptionRegister("S6")
        scored = propose_cameras(plan, inp.room_ids, inp.views, inp.width, inp.height, reg)
        return CamerasOut(
            cameras=[s.camera for s in scored],
            scores=[round(s.score, 4) for s in scored],
            visible_floor=[round(s.visible_floor_fraction, 4) for s in scored],
            assumptions=reg.items(),
        )

    def s_render(inp: RenderIn, ctx: StageContext) -> RenderOut:
        spec = load_model(ctx, inp.scene_json, SceneSpec)
        spec = spec.model_copy(update={"cameras": [inp.camera]})
        pkg = materialize_package(ctx, inp.scene_json, inp.meshes, spec, f"render-{inp.camera.id}")
        out_dir = pkg / "out"
        try:
            svc.blender.run(
                ["render", str(pkg), str(out_dir), inp.camera.id], cwd=pkg, cancelled=ctx.cancelled
            )
            passes = read_passes(out_dir / "passes.exr")
            info = json.loads((out_dir / "render.json").read_text())
            lineart = line_art(passes, spec)
            ok, buf = cv2.imencode(".png", lineart)
            if not ok:
                raise ArchRenderError(
                    ErrorCode.INTERNAL, "Line-art encode failed.", "Retry the render."
                )
            cam = camera_json(inp.camera, spec.render.width, spec.render.height)
            return RenderOut(
                beauty_png=ctx.store.put_file(
                    out_dir / "beauty.png", "image/png", f"{inp.camera.id}_cycles.png"
                ),
                passes_exr=ctx.store.put_file(
                    out_dir / "passes.exr", "image/x-exr", f"{inp.camera.id}_passes.exr"
                ),
                passes_npz=ctx.store.put_bytes(
                    passes.to_npz(), "application/x-npz", f"{inp.camera.id}_passes.npz"
                ),
                lineart_png=ctx.store.put_bytes(
                    bytes(buf), "image/png", f"{inp.camera.id}_lineart.png"
                ),
                camera_json=ctx.store.put_bytes(
                    json.dumps(cam, indent=1).encode(),
                    "application/json",
                    f"{inp.camera.id}_camera.json",
                ),
                info=info,
            )
        finally:
            shutil.rmtree(pkg, ignore_errors=True)

    def s_refine_qa(inp: RefineQAIn, ctx: StageContext) -> RefineQAOut:
        spec = load_model(ctx, inp.scene_json, SceneSpec)
        base = decode(ctx.store.read_bytes(inp.render.beauty_png))
        passes = RenderPasses.from_npz(ctx.store.read_bytes(inp.render.passes_npz))
        qa = QAContext(
            passes=passes,
            spec=spec,
            qa_width=prof.qa_width,
            config=svc.qa_config,
            depth=svc.models.get_with_fallback("depth", "S9")[0],
            segmenter=svc.models.get_with_fallback("segmenter", "S9")[0],
        )
        stored: list[CasRef] = []

        def store(img: np.ndarray, label: str) -> str:
            ref = ctx.store.put_bytes(encode_png16(img), "image/png", f"{label}.png")
            stored.append(ref)
            return ref.sha256

        result = run_view_policy(
            view_id=inp.view_id,
            camera_id=inp.camera_id,
            base=base,
            base_sha=inp.render.beauty_png.sha256,
            passes=passes,
            spec=spec,
            refiner=svc.models.get_with_fallback("refiner", "S8")[0],
            qa=qa,
            profile=prof.refine,
            prompt=inp.prompt,
            seed=inp.seed,
            store=store,
            cancelled=ctx.cancelled,
        )
        if result.outcome.status == "fallback_base":
            delivered_png = inp.render.beauty_png
        else:
            delivered_png = next(r for r in stored if r.sha256 == result.outcome.delivered_sha256)
        jpg = ctx.store.put_bytes(encode_jpeg(result.delivered), "image/jpeg", f"{inp.view_id}.jpg")
        return RefineQAOut(
            outcome=result.outcome,
            delivered_png=delivered_png,
            delivered_jpg=jpg,
            candidates=stored,
            log=result.log,
        )

    def models_refine() -> list[Any]:
        return [svc.models.ref(r) for r in ("refiner", "depth", "segmenter")]

    return {
        "brief": StageDef("S4_brief", "1", BriefOut, s_brief),
        "scene": StageDef(
            "S5_scene",
            "1",
            SceneOut,
            s_scene,
            config=lambda: {
                "materials": sha256_json(
                    [svc.library.get(i).model_dump() for i in svc.library.ids()]
                ),
                "blender": BLENDER_VERSION,
            },
        ),
        "cameras": StageDef("S6_cameras", "1", CamerasOut, s_cameras),
        "render": StageDef(
            "S7_render", "1", RenderOut, s_render, config=lambda: {"blender": BLENDER_VERSION}
        ),
        "refine_qa": StageDef(
            "S8S9_refine_qa",
            "1",
            RefineQAOut,
            s_refine_qa,
            config=lambda: {
                "refine": prof.refine.model_dump(),
                "qa": svc.qa_config.model_dump(),
                "qa_width": prof.qa_width,
            },
            models=models_refine,
        ),
    }


def resolve_section(plan: PlanGraph, room_ids: list[str] | None) -> Section:
    """Section with a content-derived id, so identical selections share cached stages across runs."""
    ids = sorted(room_ids or [r.id for r in plan.rooms])
    level = plan.levels[0].id
    sid = "sec_" + sha256_json({"plan": plan.version, "level": level, "rooms": ids})[:16]
    sec = Section(id=sid, level=level, kind="rooms", room_ids=ids)
    section_rooms(plan, sec)
    return sec
