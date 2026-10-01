"""S2 evaluation on synthetic sheets with exact ground truth (the ``make eval`` plan table).

Every sheet goes through the real path: S0 intake (sandboxed parsers) → S1 (text layer or real
OCR, sheet rectification for photos) → the S2 extractor of its source type. Scores are F1 of walls,
openings and rooms against the ground truth mapped into the extractor's frame, and the relative
scale error. The assist path is measured with ground-truth hints (``synth.hints.OracleHints``: what
a good VLM would answer) and adversarial ones are covered by the property tests; the VLM itself is
UNVERIFIED-ON-GPU. Targets are the Phase-3 acceptance numbers (docs/PLAN.md); misses are reported,
never hidden.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from archrender.core.config import Settings
from archrender.core.ids import now_iso
from archrender.core.schemas.document import PageRef
from archrender.core.schemas.plan import PlanGraph
from archrender.ingest.intake import ingest_file, page_refs
from archrender.pipeline.engine import StageContext, StageEngine
from archrender.pipeline.services import Services
from archrender.plan.defects import DEFECTS, detected
from archrender.plan.metrics import compose, plan_scores, transform_plan
from archrender.plan.stage import Extractor, PageResult, S2Page
from archrender.synth.dxf import random_plan_dxf
from archrender.synth.hints import OracleHints, gt_elements_px
from archrender.synth.plan import PlanSpec, Variant, gt_plan, random_spec
from archrender.synth.raster import phone_photo, scan
from archrender.synth.sheets import floor_plan_page
from archrender.understand.stage import S1In, S1Out, build_s1

VARIANTS: list[Variant] = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]
STYLES = ["solid", "grey", "hatch", "outline"]
TARGETS: dict[str, dict[str, float]] = {
    "dxf": {"walls": 0.98, "openings": 0.95, "scale_err": 0.01},
    "pdf_vector": {"walls": 0.98, "openings": 0.95, "scale_err": 0.01},
    "raster_clean": {"walls": 0.92, "openings": 0.88},
}
ASSIST_PRECISION_TARGET = 0.95


class _Bench:
    def __init__(self, workdir: Path) -> None:
        settings = Settings(
            data_dir=workdir / "plan-eval",
            db_path=workdir / "plan-eval-db" / "archrender.sqlite",
            profile="cpu_test",
            cookie_secure=False,
            job_lease_s=600.0,
            worker_poll_s=0.05,
        )
        settings.ensure_dirs()
        self.svc = Services.create(settings)
        self.svc.db.migrate()
        self.svc.db.execute(
            "INSERT INTO users(id, name, role, created_at) VALUES ('usr_eval', 'eval', 'admin', ?)",
            (now_iso(),),
        )
        self.pid = "prj_planeval"
        self.svc.db.execute(
            "INSERT INTO projects(id, name, created_by, created_at) VALUES (?, 'plan eval', 'usr_eval', ?)",
            (self.pid, now_iso()),
        )
        self.store = self.svc.store(self.pid)
        self.tmp = workdir / "plan-eval-files"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.engine = StageEngine(self.svc.db)
        self.ctx = StageContext(self.pid, self.store, lambda *_: None, lambda: False)

    def ingest(self, name: str, data: bytes) -> PageRef:
        path = self.tmp / name
        path.write_bytes(data)
        res = ingest_file(self.svc, self.pid, path, name)
        return next(p for p in page_refs(self.svc, self.pid) if p.document_id == res.document_id)

    def s1(self, page: PageRef) -> S1Out:
        return self.engine.run(
            build_s1(self.svc), S1In(page=page, doc_meta={}, langs=["tr", "en"]), self.ctx
        )

    def extract(
        self, page: PageRef, kind: Any, analysis: S1Out | None, hints: Any = None
    ) -> PageResult:
        ex = Extractor(self.svc, self.store, self.pid, hints=hints)
        return ex.page(S2Page(page=page, label="floor_plan", analysis=analysis), kind)


def _f1(s: dict[str, Any]) -> dict[str, float]:
    return {k: round(float(s[k]["f1"]), 4) for k in ("walls", "openings", "rooms")}


def _truth(spec: PlanSpec, to_doc: list[list[float]], res: PageResult) -> PlanGraph:
    return transform_plan(gt_plan(spec), compose(to_doc, [list(r) for r in res.transform.matrix]))


def _photo_truth(
    spec: PlanSpec, gt: dict[str, Any], analysis: S1Out, res: PageResult
) -> tuple[PlanGraph, float]:
    """GT plan → photo px → rectified px (projective) → plan: the best similarity, and its scale
    error (the scale of a photo comes from dimension strings or door swings)."""
    import cv2

    full = (
        np.vstack([np.array(res.transform.matrix, float), [0, 0, 1]])
        @ np.array((analysis.rectification or {})["h"])
        @ np.array(gt["plan_to_photo_h"])
    )
    src = np.array(
        [
            (p.x, p.y)
            for w in gt_plan(spec).walls
            for p in (w.centerline.a, w.centerline.b)  # type: ignore[union-attr]
        ],
        np.float32,
    )
    q = np.column_stack([src, np.ones(len(src))]) @ full.T
    dst = (q[:, :2] / q[:, 2:3]).astype(np.float32)
    sim, _ = cv2.estimateAffinePartial2D(src, dst)
    k = float(np.hypot(sim[0, 0], sim[1, 0]))
    return transform_plan(gt_plan(spec), sim.tolist()), abs(1 / k - 1)


def _row(
    source: str, scores: list[dict[str, float]], errs: list[float], n_fail: int, secs: float
) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        vals = [s[key] for s in scores]
        return round(statistics.fmean(vals), 4) if vals else None

    row: dict[str, Any] = {
        "source": source,
        "sheets": len(scores) + n_fail,
        "failed": n_fail,
        "walls_f1": mean("walls"),
        "openings_f1": mean("openings"),
        "rooms_f1": mean("rooms"),
        "scale_err_max": round(max(errs), 4) if errs else None,
        "scale_err_mean": round(statistics.fmean(errs), 4) if errs else None,
        "seconds": round(secs, 1),
    }
    target = TARGETS.get(source)
    if target:
        misses = []
        if row["walls_f1"] is None or row["walls_f1"] < target["walls"]:
            misses.append(f"walls {row['walls_f1']} < {target['walls']}")
        if row["openings_f1"] is None or row["openings_f1"] < target["openings"]:
            misses.append(f"openings {row['openings_f1']} < {target['openings']}")
        if "scale_err" in target and (
            row["scale_err_max"] is None or row["scale_err_max"] > target["scale_err"]
        ):
            misses.append(f"scale error {row['scale_err_max']} > {target['scale_err']}")
        row["target"] = target
        row["misses"] = misses
    return row


def evaluate(workdir: Path, *, per_source: int = 4, seed: int = 11) -> dict[str, Any]:
    b = _Bench(workdir)
    rng = np.random.default_rng(seed)
    rows = []
    assist: dict[str, Any] = {
        "sheets": 0,
        "with_triggers": 0,
        "calls": 0,
        "hints": 0,
        "accepted": 0,
        "accepted_on_truth": 0,
        "rooms_f1_before": [],
        "rooms_f1_after": [],
    }

    def spec_of(i: int) -> PlanSpec:
        return random_spec(rng, variant=VARIANTS[i % len(VARIANTS)])

    # vector DXF
    t0, scores, errs, fails = time.perf_counter(), [], [], 0
    for i in range(per_source):
        spec = spec_of(i)
        data, gt = random_plan_dxf(spec, rng)
        page = b.ingest(f"dxf{i}.dxf", data)
        try:
            res = b.extract(page, "dxf", None)
        except Exception:
            fails += 1
            continue
        scores.append(_f1(plan_scores(res.plan, _truth(spec, gt["plan_to_doc"], res))))
        true_m = 1.0 / float(gt["plan_to_doc"][0][0])
        errs.append(abs(float(res.scale["m_per_unit"]) - true_m) / true_m)  # type: ignore[index]
    rows.append(_row("dxf", scores, errs, fails, time.perf_counter() - t0))

    # vector PDF
    t0, scores, errs, fails = time.perf_counter(), [], [], 0
    for i in range(per_source):
        spec = spec_of(i)
        sheet = floor_plan_page(rng, spec=spec, wall_style=STYLES[i % len(STYLES)])
        page = b.ingest(f"pdf{i}.pdf", sheet.pdf)
        try:
            res = b.extract(page, "pdf_vector", b.s1(page))
        except Exception:
            fails += 1
            continue
        scores.append(_f1(plan_scores(res.plan, _truth(spec, sheet.gt["plan_to_page_px"], res))))
        true_m = sheet.gt["scale"] * 0.0254 / (page.dpi or 300.0)
        errs.append(abs(float(res.scale["m_per_unit"]) - true_m) / true_m)  # type: ignore[index]
    rows.append(_row("pdf_vector", scores, errs, fails, time.perf_counter() - t0))

    # rasters: clean 300 DPI and noisy 200 DPI scans (real OCR), and phone photos
    for source, dpi, quality in (
        ("raster_clean", 300.0, "clean"),
        ("raster_noisy", 200.0, "noisy"),
    ):
        t0, scores, errs, fails = time.perf_counter(), [], [], 0
        detail: list[dict[str, Any]] = []
        for i in range(per_source):
            spec = spec_of(i)
            sheet = floor_plan_page(rng, spec=spec, wall_style=STYLES[i % len(STYLES)])
            data, media, gt = scan(sheet.pdf, sheet.gt, rng, dpi=dpi, quality=quality)
            page = b.ingest(f"{source}{i}.{media.split('/')[1]}", data)
            analysis = b.s1(page)
            try:
                res = b.extract(page, "raster", analysis)
            except Exception:
                fails += 1
                continue
            truth = _truth(spec, gt["plan_to_page_px"], res)
            s = _f1(plan_scores(res.plan, truth))
            scores.append(s)
            true_m = gt["scale"] * 0.0254 / dpi
            errs.append(abs(float(res.scale["m_per_unit"]) - true_m) / true_m)  # type: ignore[index]
            detail.append(
                {
                    "sheet": i,
                    "variant": spec.variant,
                    "style": STYLES[i % len(STYLES)],
                    "f1": s,
                    "scale_err": round(errs[-1], 4),
                    "stated": analysis.scale.value if analysis.scale else None,
                    "true_scale": gt["scale"],
                    "scale": res.scale,
                }
            )
            # the assist with ground-truth hints on the same sheet
            oracle = OracleHints(
                gt_elements_px(gt_plan(spec), gt["plan_to_page_px"]), np.random.default_rng(i)
            )
            ra = b.extract(page, "raster", analysis, hints=oracle)
            rep = ra.assist or {}
            assist["sheets"] += 1
            assist["with_triggers"] += int(bool(rep.get("triggers")))
            assist["calls"] += int(rep.get("calls", 0))
            assist["hints"] += int(rep.get("hints", 0))
            assist["accepted"] += int(rep.get("accepted", 0))
            assist["accepted_on_truth"] += _accepted_on_truth(ra.plan, truth)
            assist["rooms_f1_before"].append(s["rooms"])
            assist["rooms_f1_after"].append(_f1(plan_scores(ra.plan, truth))["rooms"])
        rows.append(_row(source, scores, errs, fails, time.perf_counter() - t0))
        rows[-1]["detail"] = detail

    t0, scores, errs, fails = time.perf_counter(), [], [], 0
    for i in range(max(1, per_source // 2)):
        spec = random_spec(rng, variant="manhattan")
        sheet = floor_plan_page(rng, spec=spec, wall_style="solid")
        data, gt = phone_photo(sheet.pdf, sheet.gt, rng)
        page = b.ingest(f"photo{i}.jpg", data)
        analysis = b.s1(page)
        try:
            res = b.extract(page, "raster", analysis)
        except Exception:
            fails += 1
            continue
        if analysis.rectification is None:
            fails += 1
            continue
        truth, err = _photo_truth(spec, gt, analysis, res)
        scores.append(_f1(plan_scores(res.plan, truth)))
        errs.append(err)
    rows.append(_row("photo", scores, errs, fails, time.perf_counter() - t0))

    # validators: injected defects on correct plans
    total = hit = 0
    for i in range(6):
        for v in VARIANTS:
            plan = gt_plan(random_spec(np.random.default_rng(i), variant=v))
            for inject in DEFECTS:
                total += 1
                hit += int(detected(inject(plan)))
    acc = assist["accepted"]
    return {
        "per_source": per_source,
        "seed": seed,
        "rows": rows,
        "validators": {"injected": total, "detected": hit},
        "assist": {
            **{k: v for k, v in assist.items() if not k.startswith("rooms_f1")},
            "trigger_rate": round(assist["with_triggers"] / max(1, assist["sheets"]), 3),
            "acceptance_rate": round(acc / max(1, assist["hints"]), 3),
            "accepted_precision": round(assist["accepted_on_truth"] / acc, 3) if acc else None,
            "rooms_f1_before": round(statistics.fmean(assist["rooms_f1_before"]), 4)
            if assist["rooms_f1_before"]
            else None,
            "rooms_f1_after": round(statistics.fmean(assist["rooms_f1_after"]), 4)
            if assist["rooms_f1_after"]
            else None,
        },
        "gate_a": "every raster plan (scan or photo) requires a person at Gate A (mandatory); "
        "elements placed with VLM help block approval until confirmed",
    }


def _accepted_on_truth(plan: PlanGraph, truth: PlanGraph, tol: float = 0.1) -> int:
    """Elements placed with hints that lie on a ground-truth element (walls: centre line within
    ``tol``; openings: centre within 0.15 m and width within 0.1 m)."""
    from archrender.plan.annotate import _opening_point
    from archrender.plan.assist import _same_line

    n = 0
    for w in plan.walls:
        if w.thickness_m.provenance[0].method != "vlm_assisted":
            continue
        cl = w.centerline
        p0, p1 = np.array([cl.a.x, cl.a.y]), np.array([cl.b.x, cl.b.y])  # type: ignore[union-attr]
        n += int(any(_same_line(p0, p1, g, tol) >= 0.9 for g in truth.walls))
    for i, o in enumerate(plan.openings):
        if o.width_m.provenance[0].method != "vlm_assisted":
            continue
        p = _opening_point(plan, i)
        ok = False
        for j, g in enumerate(truth.openings):
            q = _opening_point(truth, j)
            if p is not None and q is not None and float(np.hypot(*(p - q))) <= 0.15:
                ok = abs(g.width_m.value - o.width_m.value) <= 0.1
                break
        n += int(ok)
    return n


def render(r: dict[str, Any]) -> str:
    rows = []
    for row in r["rows"]:
        target = row.get("target")
        verdict = ""
        if target is not None:
            verdict = (
                " · MISS: " + "; ".join(row["misses"]) if row["misses"] else " · meets targets"
            )
        scale = (
            f"scale err max {row['scale_err_max']:.2%}"
            if row["scale_err_max"] is not None
            else "scale n/a"
        )
        rows.append(
            (
                f"plan: {row['source']} ({row['sheets']} sheets)",
                f"walls {row['walls_f1']} · openings {row['openings_f1']} · rooms {row['rooms_f1']} · "
                f"{scale} · failed {row['failed']} · {row['seconds']} s{verdict}",
            )
        )
    v = r["validators"]
    rows.append(("validators: injected defects", f"{v['detected']}/{v['injected']} detected"))
    a = r["assist"]
    prec = a["accepted_precision"]
    rows.append(
        (
            "assist (ground-truth hints, scans)",
            f"triggered on {a['with_triggers']}/{a['sheets']} sheets · {a['calls']} calls · "
            f"{a['hints']} hints · accepted {a['accepted']} ({a['acceptance_rate']:.0%}) · "
            f"accepted on truth {prec if prec is not None else 'n/a'}"
            f"{'' if prec is None or prec >= ASSIST_PRECISION_TARGET else ' · MISS (< 0.95)'} · "
            f"rooms F1 {a['rooms_f1_before']} → {a['rooms_f1_after']}",
        )
    )
    rows.append(("assist with the VLM", "not measured here: UNVERIFIED-ON-GPU (pod run)"))
    rows.append(("Gate A routing", r["gate_a"]))
    width = max(len(k) for k, _ in rows)
    out = ["| " + "S2 metric".ljust(width) + " | value |", "|" + "-" * (width + 2) + "|---|"]
    out += [f"| {k.ljust(width)} | {val} |" for k, val in rows]
    return "\n".join(out)
