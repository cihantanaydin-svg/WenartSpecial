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

from tests.helpers import plan_dxf, upload_bytes  # noqa: E402

from archrender.core.config import Settings  # noqa: E402
from archrender.core.ids import now_iso  # noqa: E402
from archrender.core.schemas.jobs import JobStatus  # noqa: E402
from archrender.pipeline.run import RunConfig, create_run  # noqa: E402
from archrender.pipeline.services import Services  # noqa: E402
from archrender.pipeline.worker import Worker  # noqa: E402

NOT_MEASURED = {
    "plan": "see the S2 table below (skipped with --plan-per-source 0)",
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
    upload_bytes(svc, "prj_eval", "Kat Planı.dxf", plan_dxf())
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


def golden_s1(workdir: Path) -> list[dict[str, Any]]:
    """Golden projects G1/G2 through intake + S1 in the real pipeline (Phase-2 acceptance)."""
    from archrender.core.schemas.understanding import Schedule
    from archrender.synth.golden import g1_daire, g2_loft

    settings = Settings(
        data_dir=workdir / "golden",
        db_path=workdir / "golden-db" / "archrender.sqlite",
        profile="cpu_test",
        cookie_secure=False,
        job_lease_s=600.0,
        worker_poll_s=0.05,
    )
    settings.ensure_dirs()
    svc = Services.create(settings)
    svc.db.migrate()
    svc.db.execute(
        "INSERT INTO users(id, name, role, created_at) VALUES ('usr_eval', 'eval', 'admin', ?)",
        (now_iso(),),
    )
    worker = Worker(svc, ["cpu", "gpu"], name="eval-golden")
    out = []
    for i, make in enumerate((g1_daire, g2_loft)):
        g = make()
        pid = f"prj_golden{i + 1}"
        svc.db.execute(
            "INSERT INTO projects(id, name, created_by, created_at) VALUES (?, ?, 'usr_eval', ?)",
            (pid, g.name, now_iso()),
        )
        t0 = time.perf_counter()
        for d in g.docs:
            upload_bytes(svc, pid, d.filename, d.data, user_id="usr_eval")
        worker.run_until_idle()
        failed = svc.db.query(
            "SELECT kind, error_json FROM jobs WHERE project_id = ? AND status != ?",
            (pid, JobStatus.SUCCEEDED.value),
        )
        labels = {
            (r["filename"], r["label"])
            for r in svc.db.query(
                "SELECT d.filename, a.label FROM page_analysis a JOIN documents d"
                " ON d.id = a.document_id WHERE a.project_id = ?",
                (pid,),
            )
        }
        expected = [d for d in g.docs if d.expected_class]
        correct = [d for d in expected if (d.filename, d.expected_class) in labels]
        rows = [
            row
            for r in svc.db.query(
                "SELECT schedule_json FROM schedules WHERE project_id = ?", (pid,)
            )
            for row in Schedule.model_validate_json(r["schedule_json"]).rows
            if row.tag
        ]
        out.append(
            {
                "project": g.name,
                "documents": len(g.docs),
                "failed_jobs": len(failed),
                "classes": f"{len(correct)}/{len(expected)}",
                "misclassified": sorted(
                    f"{d.filename}→{','.join(sorted(lb for f, lb in labels if f == d.filename))}"
                    for d in expected
                    if d not in correct
                ),
                "schedule_rows_linked": f"{sum(1 for r in rows if r.links)}/{len(g.schedule_tags)}",
                "unlinked": sorted(r.tag for r in rows if not r.links and r.tag),
                "wall_s": round(time.perf_counter() - t0, 1),
            }
        )
    return out


def render_s1(r: dict[str, Any]) -> str:
    c = r["classification"]
    src = " · ".join(f"{k} F1 {v['macro_f1']:.2f} (n={v['n']})" for k, v in c["by_source"].items())
    ocr = r["ocr_on_scans"]
    na = r["north_arrow"]
    sl = r["schedule_linking"]
    rows = [
        (
            "S1 corpus (synthetic, held out)",
            f"seed {r['corpus']['seed']}, {r['corpus']['n']} pages {r['corpus']['sources']}",
        ),
        (
            "page classification (heuristic model)",
            f"macro-F1 {c['macro_f1']:.3f} · acc {c['accuracy']:.3f} · review rate {c['review_rate']:.0%}"
            f" ({c['outside_training_range']} outside training range) · errors not sent to review "
            f"{c['errors_not_sent_to_review']}",
        ),
        ("  by source", src),
        ("  top confusions", ", ".join(c["top_confusions"]) or "none"),
        ("title-block fields (vector + scans)", r["title_block_fields"]),
        ("scale from title block", r["scale_from_title_block"]),
        (
            "north arrow (vector)",
            f"{na['measured']} measured, {na['missing']} missing, "
            f"mean |err| {na['mean_abs_err_deg']}°, max {na['max_abs_err_deg']}°",
        ),
        (
            f"OCR on scans ({ocr['engine']})",
            f"dimensions {ocr['dimension_strings']} · title text {ocr['title_block_text']} · "
            f"room names {ocr['room_names_exact']}",
        ),
        ("  Turkish char accuracy (room names)", str(ocr["turkish_char_accuracy_room_names"])),
        ("  bubble tags", ocr["bubble_tags"]),
        (
            "schedule ↔ plan tags",
            f"{sl['rows_linked']} rows linked, "
            f"{sl['projects_fully_linked']}/{sl['projects']} projects fully linked",
        ),
        *(
            (
                f"golden {g['project']} (S0+S1 pipeline)",
                f"classes {g['classes']} · schedule rows linked {g['schedule_rows_linked']} · "
                f"failed jobs {g['failed_jobs']} · {g['wall_s']} s"
                + (f" · misclassified {g['misclassified']}" if g["misclassified"] else "")
                + (f" · unlinked {g['unlinked']}" if g["unlinked"] else ""),
            )
            for g in r.get("golden", [])
        ),
        ("VLM classification / PaddleOCR-VL", "not measured here: UNVERIFIED-ON-GPU (pod run)"),
    ]
    width = max(len(k) for k, _ in rows)
    out = ["| " + "S1 metric".ljust(width) + " | value |", "|" + "-" * (width + 2) + "|---|"]
    out += [f"| {k.ljust(width)} | {v} |" for k, v in rows]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=180)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--json", type=Path, help="also write the raw results here")
    ap.add_argument(
        "--understanding-per-class",
        type=int,
        default=3,
        help="S1 eval corpus size per class (0 = skip)",
    )
    ap.add_argument(
        "--plan-per-source",
        type=int,
        default=3,
        help="S2 eval sheets per source type (0 = skip)",
    )
    ap.add_argument("--keep", action="store_true", help="keep the throw-away workspace")
    args = ap.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="archrender-eval-"))
    try:
        result = evaluate(args, workdir)
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
    print(render_table(result))
    if args.understanding_per_class:
        from archrender.understand.evaluate import evaluate as evaluate_s1

        s1 = evaluate_s1(seed=3, per_class=args.understanding_per_class)
        golden_dir = Path(tempfile.mkdtemp(prefix="archrender-eval-golden-"))
        try:
            s1["golden"] = golden_s1(golden_dir)
        finally:
            shutil.rmtree(golden_dir, ignore_errors=True)
        result["understanding"] = s1
        result["invariants"]["golden S0+S1 jobs succeed"] = all(
            g["failed_jobs"] == 0 for g in s1["golden"]
        )
        print()
        print(render_s1(s1))
    if args.plan_per_source:
        from archrender.plan.evaluate import evaluate as evaluate_s2
        from archrender.plan.evaluate import render as render_s2

        plan_dir = Path(tempfile.mkdtemp(prefix="archrender-eval-plan-"))
        try:
            s2 = evaluate_s2(plan_dir, per_source=args.plan_per_source)
        finally:
            shutil.rmtree(plan_dir, ignore_errors=True)
        result["plan_eval"] = s2
        result["invariants"]["S2 sheets extracted"] = all(
            row["failed"] == 0 for row in s2["rows"] if row["source"] != "photo"
        )
        result["invariants"]["validators detect every injected defect"] = (
            s2["validators"]["detected"] == s2["validators"]["injected"]
        )
        print()
        print(render_s2(s2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not all(result["invariants"].values()):
        print("\nEVAL FAILED: pipeline invariant violated (see table)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
