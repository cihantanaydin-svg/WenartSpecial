"""Raster plans (scans, phone photos) through S0 intake → vectoriser → PlanBuilder, against exact
synthetic ground truth (the GT plan mapped into the extractor's frame, so geometry and scale are
scored separately).

The scan cases use the sheet's ground-truth words (perfect OCR) to test the vectoriser on its own;
one case runs the real OCR role and checks the scale fusion; the phone photo goes through the real
S1 stage (sheet detection, rectification, OCR). The clean-raster acceptance numbers (wall F1 ≥
0.92, opening F1 ≥ 0.88 with real OCR, many seeds) are measured by ``make eval``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from archrender.core.config import Settings
from archrender.core.schemas.document import PageRef, Word
from archrender.ingest.intake import ingest_file, page_refs
from archrender.pipeline.engine import StageContext, StageEngine
from archrender.pipeline.services import Services
from archrender.plan.builder import build_plan
from archrender.plan.metrics import compose, plan_scores, transform_plan
from archrender.plan.raster import raster_extract, raster_scale
from archrender.synth.plan import Variant, gt_plan, random_spec
from archrender.synth.raster import phone_photo, scan
from archrender.synth.sheets import floor_plan_page
from archrender.understand.page import analyse
from archrender.understand.stage import S1In, build_s1

VARIANTS: list[Variant] = ["manhattan", "rotated", "skewed", "arc", "skewed_arc"]
STYLES = ["solid", "grey", "hatch", "outline"]


@pytest.fixture
def svc(settings: Settings, project_id: str) -> Services:
    return Services.create(settings)


def _ingest(svc: Services, tmp: Path, name: str, data: bytes) -> PageRef:
    p = tmp / name
    p.write_bytes(data)
    res = ingest_file(svc, "prj_test", p, name)
    return next(pg for pg in page_refs(svc, "prj_test") if pg.document_id == res.document_id)


def _rgb(svc: Services, ref: Any) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(svc.store("prj_test").read_bytes(ref))).convert("RGB"))


def _gt_words(gt: dict[str, Any]) -> list[Word]:
    return [
        Word(
            text=w["text"],
            x0=w["bbox"][0],
            y0=w["bbox"][1],
            x1=w["bbox"][2],
            y1=w["bbox"][3],
            angle_deg=w.get("angle_deg", 0.0),
            source="ocr:ground-truth",
        )
        for w in gt["words"]
    ]


def _sheet(seed: int) -> tuple[Any, Any]:
    rng = np.random.default_rng(700 + seed)
    spec = random_spec(rng, variant=VARIANTS[seed % 5])
    return rng, (spec, floor_plan_page(rng, spec=spec, wall_style=STYLES[seed % 4]))


def test_clean_scans_of_every_wall_style(svc: Services, tmp_path: Path) -> None:
    scores, rooms = [], []
    for seed in range(4):  # solid, grey, hatch, outline
        rng, (spec, sheet) = _sheet(seed)
        data, media, gt = scan(sheet.pdf, sheet.gt, rng, dpi=300.0, quality="clean")
        page = _ingest(svc, tmp_path, f"scan{seed}.{media.split('/')[1]}", data)
        assert page.dpi == pytest.approx(300.0, abs=0.01)
        rgb = _rgb(svc, page.raster)
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        m = gt["scale"] * 0.0254 / page.dpi
        ex = raster_extract(gray, _gt_words(gt), m, rgb=rgb)
        res = build_plan(ex.prims, project="p", version="v", source="raster")
        truth = transform_plan(gt_plan(spec), compose(gt["plan_to_page_px"], ex.doc_to_plan))
        s = plan_scores(res.plan, truth)
        assert s["walls"]["f1"] >= 0.95, (STYLES[seed], s)
        scores.append(s["openings"]["f1"])
        rooms.append(s["rooms"]["f1"])
    assert float(np.mean(scores)) >= 0.85, scores
    # the hatch sheet carries an approval stamp across a wall (grey scan): its two rooms merge
    # there, which the validators flag (area label mismatch) — see PROGRESS, known limitations
    assert float(np.mean(rooms)) >= 0.85, rooms


def test_a_scan_with_real_ocr_has_a_measured_scale(svc: Services, tmp_path: Path) -> None:
    rng, (spec, sheet) = _sheet(0)
    data, media, gt = scan(sheet.pdf, sheet.gt, rng, dpi=300.0, quality="clean")
    page = _ingest(svc, tmp_path, f"ocr.{media.split('/')[1]}", data)
    rgb = _rgb(svc, page.raster)
    ocr, entry = svc.models.get_with_fallback("ocr", "S2")
    a = analyse(
        page.id, rgb, dpi=page.dpi or 300.0, text_layer=None, tiles=page.tiles, ocr=ocr,
        ocr_name=entry.name, langs=["tr", "en"], page_meta=page.meta, doc_meta={}, vector=None,
    )  # fmt: skip
    assert a.scale is not None and a.scale.value == gt["scale"]
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    sc, rv = raster_scale(gray, a.words, stated_n=a.scale.value, dpi=page.dpi, rgb=rgb, reader=ocr)
    true_m = gt["scale"] * 0.0254 / (page.dpi or 300.0)
    assert sc.result.estimate is not None and not sc.result.conflicted
    assert abs(sc.m_per_px - true_m) / true_m <= 0.01
    ex = raster_extract(gray, a.words, sc.m_per_px, rgb=rgb, vectors=rv)
    res = build_plan(ex.prims, project="p", version="v", source="raster")
    truth = transform_plan(gt_plan(spec), compose(gt["plan_to_page_px"], ex.doc_to_plan))
    s = plan_scores(res.plan, truth)
    assert s["walls"]["f1"] >= 0.95 and s["openings"]["f1"] >= 0.85, s


def test_a_phone_photo_is_rectified_in_s1_and_extracted(
    svc: Services, settings: Settings, tmp_path: Path
) -> None:
    rng = np.random.default_rng(900)
    spec = random_spec(rng, variant="manhattan")
    sheet = floor_plan_page(rng, spec=spec, wall_style="solid")
    data, gt = phone_photo(sheet.pdf, sheet.gt, rng)
    page = _ingest(svc, tmp_path, "photo.jpg", data)
    store = svc.store("prj_test")
    ctx = StageContext("prj_test", store, lambda *_: None, lambda: False)
    out = StageEngine(svc.db).run(
        build_s1(svc), S1In(page=page, doc_meta={}, langs=["tr", "en"]), ctx
    )
    assert out.rectified is not None and out.rectification is not None
    corners = np.array(out.rectification["corners_px"])
    assert np.hypot(*(corners - np.array(gt["sheet_corners_px"])).T).max() <= 2.0
    assert out.rectification["aspect_source"] == "iso216"
    assert out.classification.label == "floor_plan"
    rgb = _rgb(svc, out.rectified)
    words = [Word.model_validate(w) for w in json.loads(store.read_bytes(out.words))]  # type: ignore[arg-type]
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    ocr, _ = svc.models.get_with_fallback("ocr", "S2")
    sc, rv = raster_scale(gray, words, stated_n=None, dpi=None, rgb=rgb, reader=ocr)
    assert sc.result.estimate is not None  # dimensions (no DPI for a photo)
    ex = raster_extract(gray, words, sc.m_per_px, rgb=rgb, vectors=rv)
    res = build_plan(ex.prims, project="p", version="v", source="raster")
    # GT plan metres → photo px → rectified px (projective) → plan frame: fit a similarity
    import cv2

    full = (
        np.vstack([np.array(ex.doc_to_plan), [0, 0, 1]])
        @ np.array(out.rectification["h"])
        @ np.array(gt["plan_to_photo_h"])
    )
    src = np.array(
        [(p.x, p.y) for w in gt_plan(spec).walls for p in (w.centerline.a, w.centerline.b)]  # type: ignore[union-attr]
    )
    q = np.column_stack([src, np.ones(len(src))]) @ full.T
    dst = q[:, :2] / q[:, 2:3]
    sim, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32))
    k = float(np.hypot(sim[0, 0], sim[1, 0]))
    assert abs(1 / k - 1) <= 0.05  # the scale from dimension strings, within 5 % on a photo
    s = plan_scores(res.plan, transform_plan(gt_plan(spec), sim.tolist()))
    assert s["walls"]["f1"] >= 0.9, s
