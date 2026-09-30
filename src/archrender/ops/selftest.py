"""Boot self-test (entrypoint): DB, licence gate, Blender device probe and a tiny render on the
profile's device (which also warms the Cycles kernel cache). Writes the marker that /readyz needs."""

from __future__ import annotations

import json
import shutil
import sys
import time

from archrender.brief.defaults import default_brief
from archrender.camera.propose import propose_cameras
from archrender.core.assumptions import AssumptionRegister
from archrender.core.config import get_settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.scene import RenderSettings
from archrender.core.schemas.section import Section
from archrender.ops.readiness import write_selftest_marker
from archrender.pipeline.services import Services
from archrender.plan.mock import mock_plan
from archrender.render.passes import read_passes
from archrender.scene.compiler import SceneCompiler


def run(svc: Services) -> dict[str, object]:
    t0 = time.time()
    report: dict[str, object] = {"profile": svc.profile.name}
    svc.db.migrate()
    for name in svc.profile.roles.values():
        svc.gate.check(svc.registry.get(name))
    report["license_gate"] = "ok"
    work = svc.settings.data_dir / "cache" / "selftest"
    shutil.rmtree(work, ignore_errors=True)
    probe = svc.blender.probe(work / "probe")
    report["blender_probe"] = probe
    if svc.profile.render.device == "GPU" and probe.get("device") != "GPU":
        raise ArchRenderError(
            ErrorCode.RENDER_DEVICE_UNAVAILABLE,
            f"Profile {svc.profile.name} needs a GPU render device but Blender found {probe}.",
            "Check NVIDIA_DRIVER_CAPABILITIES=all in the image, the host driver (≥ 580) and nvidia-smi.",
        )
    plan = mock_plan("selftest", [])
    reg = AssumptionRegister("selftest")
    render = RenderSettings(
        width=64, height=36, samples=4, device=svc.profile.render.device, denoise=False
    )
    compiled = SceneCompiler(svc.library).compile(
        plan,
        Section(id="s", level="L0", kind="rooms", room_ids=["R1"]),
        default_brief("s", reg),
        render,
        reg,
        scene_id="selftest",
    )
    cam = propose_cameras(plan, ["R1"], 1, 64, 36, reg)[0].camera
    pkg = work / "pkg"
    (pkg / "meshes").mkdir(parents=True)
    for rel, blob in compiled.mesh_blobs.items():
        (pkg / rel).write_bytes(blob)
    (pkg / "scene.json").write_text(
        compiled.spec.model_copy(update={"cameras": [cam]}).model_dump_json()
    )
    t_render = time.time()
    svc.blender.run(["render", str(pkg), str(work / "out"), cam.id], cwd=pkg, timeout_s=1800)
    passes = read_passes(work / "out" / "passes.exr")
    if not (passes.object_index > 0).any():
        raise ArchRenderError(
            ErrorCode.BLENDER_FAILED, "Self-test render is empty.", "Inspect the Blender log."
        )
    info = json.loads((work / "out" / "render.json").read_text())
    report.update(
        render_seconds=round(time.time() - t_render, 2),
        render_device=info.get("device"),
        blender=info.get("blender"),
        total_seconds=round(time.time() - t0, 2),
    )
    write_selftest_marker(svc.settings.data_dir)
    (svc.settings.data_dir / "cache" / "selftest.json").write_text(
        json.dumps(report, indent=1, default=str)
    )
    return report


def main() -> int:
    svc = Services.create(get_settings())
    try:
        report = run(svc)
    except ArchRenderError as e:
        print(f"SELF-TEST FAILED [{e.code}]: {e.message}\n  → {e.fix_hint}", file=sys.stderr)
        tail = e.context.get("log_tail")
        if tail:
            print(tail, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
