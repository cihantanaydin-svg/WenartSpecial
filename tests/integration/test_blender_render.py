"""Blender CPU tests at tiny resolution: camera model and depth pass match the analytic model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from archrender.brief.defaults import default_brief
from archrender.camera.intrinsics import extrinsics_cv, intrinsics, project
from archrender.camera.propose import propose_cameras
from archrender.core.assumptions import AssumptionRegister
from archrender.core.config import REPO_ROOT, Settings
from archrender.core.schemas.scene import RenderSettings
from archrender.core.schemas.section import Section
from archrender.plan.mock import mock_plan
from archrender.render.blender import BlenderRunner
from archrender.render.passes import line_art, read_passes, structural_mask
from archrender.scene.compiler import SceneCompiler
from archrender.scene.materials import MaterialLibrary

pytestmark = pytest.mark.blender
W, H = 160, 90


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    tmp = tmp_path_factory.mktemp("blender")
    mode = "module" if (REPO_ROOT / ".venv-blender" / "bin" / "python").exists() else "binary"
    settings = Settings(data_dir=tmp / "ws", blender_mode=mode)  # type: ignore[arg-type]
    plan = mock_plan("prj_b", [])
    reg = AssumptionRegister("t")
    compiled = SceneCompiler(MaterialLibrary.load(REPO_ROOT / "configs")).compile(
        plan,
        Section(id="s", level="L0", kind="rooms", room_ids=["R1"]),
        default_brief("s", reg),
        RenderSettings(width=W, height=H, samples=8, device="CPU"),
        reg,
        scene_id="scn",
    )
    cam = propose_cameras(plan, ["R1"], 1, W, H, reg)[0].camera
    spec = compiled.spec.model_copy(update={"cameras": [cam]})
    pkg = tmp / "pkg"
    (pkg / "meshes").mkdir(parents=True)
    for rel, blob in compiled.mesh_blobs.items():
        (pkg / rel).write_bytes(blob)
    (pkg / "scene.json").write_text(spec.model_dump_json())
    BlenderRunner(settings).run(
        ["render", str(pkg), str(tmp / "out"), cam.id], cwd=pkg, timeout_s=600
    )
    return tmp / "out", spec


def test_depth_is_planar_and_matches_camera_model(rendered: tuple[Path, object]) -> None:
    out, spec = rendered
    passes = read_passes(out / "passes.exr")
    cam = spec.cameras[0]  # type: ignore[attr-defined]
    k = intrinsics(cam, W, H)
    r, _ = extrinsics_cv(cam)
    c = np.array(cam.position.as_tuple())
    floor_ids = {o.pass_index for o in spec.objects if o.category == "floor"}  # type: ignore[attr-defined]
    checked = 0
    for v in range(H - 12, H):
        for u in range(20, W - 20, 10):
            if int(passes.object_index[v, u]) not in floor_ids:
                continue
            d_w = r.T @ (np.linalg.inv(k) @ np.array([u + 0.5, v + 0.5, 1.0]))
            t = -c[2] / d_w[2]  # ray/floor intersection → planar depth
            assert abs(float(passes.depth[v, u]) - t) < 2e-3 * t
            checked += 1
    assert checked >= 5


def test_reprojected_room_corner_lies_on_line_art(rendered: tuple[Path, object]) -> None:
    out, spec = rendered
    passes = read_passes(out / "passes.exr")
    edges = line_art(passes, spec)  # type: ignore[arg-type]
    assert structural_mask(passes, spec).mean() > 0.8  # type: ignore[arg-type]
    cam = spec.cameras[0]  # type: ignore[attr-defined]
    # far vertical room corner (5, 4) at mid-height is in view for the corner camera
    uv, z = project(cam, W, H, np.array([[5.0, 4.0, 1.3]]))
    u, v = round(uv[0, 0]), round(uv[0, 1])
    assert z[0] > 0 and 0 <= u < W and 0 <= v < H
    window = edges[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3]
    assert window.max() == 255, "corner not within 2 px of the line art"


def test_render_info_reports_device(rendered: tuple[Path, object]) -> None:
    import json

    out, _ = rendered
    info = json.loads((out / "render.json").read_text())
    assert info["blender"].startswith("5.2")
    assert info["device"]["device"] == "CPU"
    assert (out / "beauty.png").stat().st_size > 1000


def test_deterministic_qa_flags_shift_and_removed_window_on_real_render(
    rendered: tuple[Path, object], settings: Settings
) -> None:
    """Geometry faults on the Cycles render are caught by the deterministic checks (edge F-score
    against the line art, verticals); the unmodified render passes (no false alarm)."""
    from archrender.pipeline.services import Services
    from archrender.qa.images import decode
    from archrender.qa.runner import QAContext
    from archrender.render.passes import category_mask

    out, spec = rendered
    passes = read_passes(out / "passes.exr")
    svc = Services.create(settings)
    qa = QAContext(
        passes=passes,
        spec=spec,  # type: ignore[arg-type]
        qa_width=W,
        config=svc.qa_config,
        depth=svc.models.get("depth"),
        segmenter=svc.models.get("segmenter"),
    )
    base = decode((out / "beauty.png").read_bytes())
    base_m = qa.measure_base(base)

    def failed(img: np.ndarray) -> set[str]:
        return {c.name for c in qa.evaluate(img, base_m) if not c.passed and not c.mock}

    assert failed(base) == set()
    assert "structural_edge_f" in failed(np.roll(base, 4, axis=1))  # 2.5% horizontal shift

    opening = category_mask(passes, spec, {"glass", "opening_frame"})  # type: ignore[arg-type]
    walls = category_mask(passes, spec, {"wall"})  # type: ignore[arg-type]
    assert opening.sum() > 50, "the test camera must see a window"
    no_window = base.copy()
    no_window[opening] = np.median(base[walls], axis=0)  # window painted over with wall colour
    assert "structural_edge_f" in failed(no_window)
