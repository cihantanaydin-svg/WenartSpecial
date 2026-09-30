#!/usr/bin/env python3
"""Evaluation table (docs/PLAN.md, "Evaluation table").

    python scripts/eval.py                   # CPU subset: cpu_test profile, real Blender, mocks
    python scripts/eval.py --json out.json   # also write the raw numbers

Runs the golden input through the real pipeline (worker, stages, Blender, QA, bundle) in a
throw-away workspace and reports, per profile:

- plan metrics per source type,
- render QA (pass rate, attempts, fallbacks, hard composites),
- cache behaviour (identical re-run, material change),
- fault injection (detection / false-alarm rates),
- stage timings and VRAM peaks.

Columns whose measuring component does not exist yet are printed as "not measured" with the phase
that adds it; they are never filled with placeholder numbers. Results computed with mock
estimators are labelled as plumbing-only. Exits non-zero if a run fails or a pipeline invariant
(cache hits, material-change scope) is violated.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.helpers import MINIMAL_DXF, upload_bytes  # noqa: E402

from archrender.core.config import Settings  # noqa: E402
from archrender.core.ids import now_iso  # noqa: E402
from archrender.core.schemas.jobs import JobStatus  # noqa: E402
from archrender.pipeline.run import RunConfig, create_run  # noqa: E402
from archrender.pipeline.services import Services  # noqa: E402
from archrender.pipeline.worker import Worker  # noqa: E402

NOT_MEASURED = {
    "plan": "not measured: the Phase-1 plan is a mock; extractors + metrics arrive in Phase 3",
    "fault_injection": "not measured: the fault-injection harness arrives in Phase 6",
    "vram": "not recorded: no GPU model stages in this build (cpu_test profile)",
}


def _stage_no(stage: str) -> int:
    m = re.match(r"S(\d+)", stage)
    if m is None:
        raise ValueError(f"unexpected stage id {stage!r}")
    return int(m.group(1))


def _run(svc: Services, worker: Worker, project_id: str, cfg: RunConfig) -> dict[str, Any]:
    t0 = time.perf_counter()
    _, job_id = create_run(svc, project_id, cfg, "usr_eval")
    worker.run_until_idle()
    job = svc.queue.get(job_id)
    if job.status != JobStatus.SUCCEEDED or job.result is None:
        raise SystemExit(f"eval run failed: {job.status} {job.error}")
    return {"wall_s": time.perf_counter() - t0, **job.result}


def evaluate(args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    blender_venv = ROOT / ".venv-blender" / "bin" / "python"
    settings = Settings(
        data_dir=workdir / "workspace",
        db_path=workdir / "db" / "archrender.sqlite",
        profile="cpu_test",
        blender_mode="module" if blender_venv.exists() else "binary",
        cookie_secure=False,
        job_lease_s=60.0,
        worker_poll_s=0.05,
    )
    settings.ensure_dirs()
    svc = Services.create(settings)
    svc.db.migrate()
    svc.db.execute(
        "INSERT INTO users(id, name, role, created_at) VALUES ('usr_eval', 'eval', 'admin', ?)",
        (now_iso(),),
    )
    svc.db.execute(
        "INSERT INTO projects(id, name, created_by, created_at) VALUES ('prj_eval', 'Eval', 'usr_eval', ?)",
        (now_iso(),),
    )
    worker = Worker(svc, ["cpu", "gpu"], name="eval")
    upload_bytes(svc, "prj_eval", "Kat Planı.dxf", MINIMAL_DXF)
    worker.run_until_idle()

    base = {
        "views": args.views,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "gate_policy": "never",
    }
    first = _run(svc, worker, "prj_eval", RunConfig.model_validate(base))
    again = _run(svc, worker, "prj_eval", RunConfig.model_validate(base))
    floor = _run(
        svc,
        worker,
        "prj_eval",
        RunConfig.model_validate({**base, "materials": {"floor": "stone_porcelain_grey"}}),
    )

    views = first["views"]
    statuses = [v["status"] for v in views]
    checks = [c for v in views for c in v["checks"]]
    rerun_stages = [t["stage"] for t in floor["timings"] if not t["cached"]]
    invariants = {
        "identical_rerun_all_cached": all(t["cached"] for t in again["timings"]),
        "material_change_reruns_only_S5_onwards": bool(rerun_stages)
        and all(_stage_no(s) >= 5 for s in rerun_stages)
        and "S6_cameras" not in rerun_stages,
    }
    per_stage: dict[str, list[float]] = {}
    for t in first["timings"]:
        per_stage.setdefault(t["stage"], []).append(t["seconds"])
    return {
        "profile": svc.profile.name,
        "blender_mode": settings.blender_mode,
        "config": base,
        "uses_mocks": first["uses_mocks"],
        "plan": NOT_MEASURED["plan"],
        "render_qa": {
            "views": len(views),
            "refined": statuses.count("refined"),
            "hard_composite": statuses.count("hard_composite"),
            "fallback_base": statuses.count("fallback_base"),
            "needs_review": statuses.count("needs_review"),
            "pass_rate": sum(s == "refined" for s in statuses) / len(statuses),
            "mean_attempts": statistics.fmean(v["attempts"] for v in views),
            "checks_passed": f"{sum(c['passed'] for c in checks)}/{len(checks)}",
            "mock_checks": sum(c["mock"] for c in checks),
            "checks": {
                c["name"]: {"passed": c["passed"], "value": c["value"], "delta": c["delta"]}
                for c in views[0]["checks"]
            },
        },
        "cache": {
            "identical_rerun_hits": f"{sum(t['cached'] for t in again['timings'])}/{len(again['timings'])}",
            "floor_material_change_reran": sorted(set(rerun_stages)),
        },
        "invariants": invariants,
        "fault_injection": NOT_MEASURED["fault_injection"],
        "timings_s": {k: round(sum(v), 3) for k, v in per_stage.items()},
        "wall_s": {
            "first_run": round(first["wall_s"], 2),
            "identical_rerun": round(again["wall_s"], 2),
            "material_change": round(floor["wall_s"], 2),
        },
        "vram_peaks_mb": NOT_MEASURED["vram"],
    }


def render_table(r: dict[str, Any]) -> str:
    qa = r["render_qa"]
    label = " (mock estimators: plumbing only)" if r["uses_mocks"] else ""
    rows = [
        (
            "profile",
            f"{r['profile']} · Blender {r['blender_mode']} · "
            f"{r['config']['width']}×{r['config']['height']} @ {r['config']['samples']} spp",
        ),
        ("plan (wall F1, opening F1, scale err)", r["plan"]),
        (
            f"render QA{label}",
            f"pass {qa['pass_rate']:.0%} ({qa['refined']}/{qa['views']} refined) · "
            f"attempts {qa['mean_attempts']:.1f} · hard composite {qa['hard_composite']} · "
            f"fallback {qa['fallback_base']} · checks {qa['checks_passed']} ({qa['mock_checks']} mock)",
        ),
        ("cache: identical re-run", f"{r['cache']['identical_rerun_hits']} stages cached"),
        (
            "cache: floor material change",
            "re-ran " + ", ".join(r["cache"]["floor_material_change_reran"]),
        ),
        ("fault injection", r["fault_injection"]),
        ("timings (s, first run)", " · ".join(f"{k} {v:.2f}" for k, v in r["timings_s"].items())),
        (
            "wall clock (s)",
            f"first {r['wall_s']['first_run']} · re-run {r['wall_s']['identical_rerun']} · "
            f"material change {r['wall_s']['material_change']}",
        ),
        ("VRAM peaks", r["vram_peaks_mb"]),
        (
            "invariants",
            " · ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r["invariants"].items()),
        ),
    ]
    width = max(len(k) for k, _ in rows)
    out = ["| " + "metric".ljust(width) + " | value |", "|" + "-" * (width + 2) + "|---|"]
    out += [f"| {k.ljust(width)} | {v} |" for k, v in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--json", type=Path, help="also write the raw results here")
    ap.add_argument("--keep", action="store_true", help="keep the throw-away workspace")
    args = ap.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="archrender-eval-"))
    try:
        result = evaluate(args, workdir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    print(render_table(result))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not all(result["invariants"].values()):
        print("\nEVAL FAILED: pipeline invariant violated (see table)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
