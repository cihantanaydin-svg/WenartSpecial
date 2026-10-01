"""S1 in the pipeline: a cached per-page stage, then project-level schedules and linking.

The UNDERSTAND job (queued after every intake) analyses every page of the project; unchanged
pages are cache hits. Results land in ``page_analysis``; low-confidence classifications and
schedule rows whose tag appears on no plan go to the review queue.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
from PIL import Image

from archrender.core.cas import CasRef
from archrender.core.hashing import sha256_file
from archrender.core.ids import new_id, now_iso
from archrender.core.schemas.common import ModelRef, Strict
from archrender.core.schemas.document import PageRef, Word
from archrender.core.schemas.provenance import Fact, Method
from archrender.core.schemas.understanding import Classification, NorthArrow, Schedule, TitleBlock
from archrender.ingest.intake import page_refs, tiles
from archrender.ingest.tasks import PREVIEW_PX
from archrender.pipeline.engine import StageContext, StageDef, StageEngine
from archrender.pipeline.services import Services
from archrender.understand.classify import REVIEW_THRESHOLD, PageClassifier, combine_with_vlm
from archrender.understand.features import vector_of
from archrender.understand.page import analyse
from archrender.understand.rectify import find_sheet, rectify
from archrender.understand.schedules import link_rows, rows_from_words, schedule_from_rows
from archrender.understand.tags import TagHit

CLASSIFIER_FILE = "classifier/page_classifier_v1.json"
PLAN_CLASSES = {"floor_plan", "ceiling_plan"}


class S1In(Strict):
    page: PageRef
    doc_meta: dict[str, Any]
    langs: list[str]


class S1Out(Strict):
    page_id: str
    words: CasRef | None  # JSON list[Word] when produced by OCR / DXF (text layer words stay in S0)
    words_method: Method
    word_count: int
    classification: Classification
    title_block: TitleBlock | None
    scale: Fact[float] | None
    north: NorthArrow | None
    tags: list[dict[str, Any]]
    notes: list[str]
    # phone photo of a sheet: the rectified sheet (words, tags and title block are in its pixels)
    rectified: CasRef | None = None
    rectification: dict[str, Any] | None = None


def _rgb(store: Any, ref: CasRef) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(store.read_bytes(ref))).convert("RGB"))


def _json(store: Any, ref: CasRef | None) -> Any:
    return json.loads(store.read_bytes(ref)) if ref is not None else None


def _rule_class(page: PageRef, content: Any) -> tuple[str, float, str] | None:
    """Pages without a raster are classified by their kind (derived, not guessed)."""
    if page.kind in ("docx", "text"):
        return "text_document", 1.0, f"{page.kind} page"
    if page.kind == "sheet":
        rows = (content or {}).get("rows", [])
        return (
            ("schedule", 1.0, "worksheet with a schedule header")
            if schedule_from_rows("x", page.id, rows)
            else ("other", 1.0, "worksheet")
        )
    if page.kind == "slide":
        return (
            ("moodboard", 0.85, "slide with pictures")
            if page.meta.get("pictures")
            else ("text_document", 0.85, "slide with text")
        )
    if page.kind in ("ifc", "model_3dm", "svg"):
        return "other", 1.0, f"{page.kind}: model/vector file, handled in S2"
    return None


def build_s1(svc: Services) -> StageDef[S1In, S1Out]:
    classifier_path = svc.settings.configs_dir / CLASSIFIER_FILE

    def ocr_ref() -> list[ModelRef]:
        entry = svc.models.entry_for("ocr")
        while entry.impl is None and entry.fallback:
            entry = svc.registry.get(entry.fallback)
        return [entry.ref()]

    def vlm_state() -> str | None:
        """Model name if the profile's VLM is serving now, "unavailable", or None (no VLM role)."""
        if "vlm" not in svc.profile.roles:
            return None
        entry = svc.models.entry_for("vlm")
        if entry.impl is None or not entry.impl.endswith(":VllmVlm"):
            return None
        from archrender.models.impls.vllm_vlm import VllmVlm

        return entry.name if VllmVlm(entry).probe() else "unavailable"

    def run(inp: S1In, ctx: StageContext) -> S1Out:
        page, store = inp.page, ctx.store
        content = _json(store, page.content)
        rule = _rule_class(page, content)
        if rule is not None or page.raster is None:
            label, conf, why = rule or ("other", 1.0, f"{page.kind} page without a raster")
            cls = Classification(
                page_id=page.id,
                label=label,
                confidence=conf,
                probabilities={label: conf},
                sources={"rule": why},
                needs_review=conf < REVIEW_THRESHOLD,
            )
            return S1Out(
                page_id=page.id,
                words=None,
                words_method="derived",
                word_count=0,
                classification=cls,
                title_block=None,
                scale=None,
                north=None,
                tags=[],
                notes=[why],
            )
        rgb = _rgb(store, page.raster)
        preview_ref = page.meta.get("preview")
        preview = _rgb(store, CasRef.model_validate(preview_ref)) if preview_ref else None
        text_layer = [Word.model_validate(w) for w in (_json(store, page.words) or [])]
        ocr_impl, ocr_entry = svc.models.get_with_fallback("ocr", "S1")
        page_tiles, dpi = page.tiles, page.dpi or 300.0
        rect_ref, rect_info, rect_notes = None, None, []
        if page.kind == "image":
            quad = find_sheet(rgb)
            if quad is not None:
                r = rectify(rgb, quad)
                rgb = r.image
                hh, ww = rgb.shape[:2]
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, "PNG")
                rect_ref = store.put_bytes(buf.getvalue(), "image/png", f"{page.id}_rectified.png")
                small = Image.fromarray(rgb)
                small.thumbnail((PREVIEW_PX, PREVIEW_PX), Image.Resampling.LANCZOS)
                preview = np.asarray(small)
                page_tiles = tiles(ww, hh, svc.settings.tile_px, svc.settings.tile_overlap)
                # the sheet's physical size is unknown; symbol sizes assume an A3 sheet
                dpi = max(ww, hh) / (420.0 / 25.4)
                rect_info = {
                    "h": r.h.tolist(),
                    "corners_px": quad.corners.round(2).tolist(),
                    "corners_in_frame": quad.corners_in_frame,
                    "coverage": round(quad.coverage, 4),
                    "fit_rms_px": round(quad.fit_rms_px, 3),
                    "aspect": round(r.aspect, 5),
                    "aspect_source": r.aspect_source,
                    "width_px": ww,
                    "height_px": hh,
                    "symbol_dpi_assumed": round(dpi, 1),
                }
                rect_notes.append(
                    "photo of a sheet: rectified ("
                    + (
                        "ISO 216 aspect assumed"
                        if r.aspect_source == "iso216"
                        else "measured aspect"
                    )
                    + ("; a corner is outside the photo" if not quad.corners_in_frame else "")
                    + "); its physical size is unknown, so the scale comes from dimensions"
                )
        a = analyse(
            page.id,
            rgb,
            dpi=dpi,
            text_layer=text_layer or None,
            tiles=page_tiles,
            ocr=ocr_impl,
            ocr_name=ocr_entry.name,
            langs=inp.langs,
            page_meta=page.meta,
            doc_meta=inp.doc_meta,
            vector=content if page.kind == "pdf_page" else None,
            preview=preview,
            dxf_summary=content if page.kind == "dxf" else None,
        )
        model = PageClassifier.load(classifier_path)
        cls = model.classify(
            page.id,
            vector_of(a.features),
            sources={
                "heuristic_model": {
                    "file": CLASSIFIER_FILE,
                    "trained_on": model.info.get("trained_on"),
                },
                "words": a.words_method,
            },
        )
        state = vlm_state()
        if state is None:
            cls.sources["vlm"] = "no VLM role in this profile"
        elif state == "unavailable":
            cls.sources["vlm"] = (
                "VLM not serving (weights missing or still loading): heuristics only"
            )
            svc.models.record_degradation(
                "S1", "vlm: unavailable → heuristics only", "vLLM server not serving"
            )
        else:
            vlm, _ = svc.models.get_with_fallback("vlm", "S1")
            excerpt = " ".join(w.text for w in a.words[:300])
            answer = vlm.classify_page(preview if preview is not None else rgb, excerpt)
            cls = combine_with_vlm(cls, answer, model.classes)
        words_ref = None
        if a.words_method != "pdf_text" and a.words:
            words_ref = store.put_bytes(
                json.dumps([w.model_dump() for w in a.words], ensure_ascii=False).encode(),
                "application/json",
                f"{page.id}_words.json",
            )
        return S1Out(
            page_id=page.id,
            words=words_ref,
            words_method=a.words_method,
            word_count=len(a.words),
            classification=cls,
            title_block=a.title_block,
            scale=a.scale,
            north=a.north,
            tags=[
                {"tag": t.tag, "kind": t.kind, "bbox": list(t.bbox), "source": t.source}
                for t in a.tags
            ],
            notes=rect_notes + a.notes,
            rectified=rect_ref,
            rectification=rect_info,
        )

    return StageDef(
        "S1_understand",
        "2",
        S1Out,
        run,
        config=lambda: {
            "classifier_sha256": sha256_file(classifier_path) if classifier_path.exists() else None,
            "vlm": vlm_state(),  # a page analysed without the VLM is recomputed once it serves
        },
        models=ocr_ref,
    )


def _review(
    svc: Services, project_id: str, kind: str, subject: str, payload: dict[str, Any]
) -> None:
    svc.db.execute(
        "INSERT INTO review_items(id, project_id, kind, subject_id, status, payload_json, created_at)"
        " VALUES (?,?,?,?, 'open', ?, ?)"
        " ON CONFLICT(kind, subject_id) DO UPDATE SET payload_json = excluded.payload_json"
        " WHERE review_items.status = 'open'",
        (
            new_id("rev"),
            project_id,
            kind,
            subject,
            json.dumps(payload, ensure_ascii=False),
            now_iso(),
        ),
    )


def run_understanding(svc: Services, ctx: StageContext) -> dict[str, Any]:
    pages = page_refs(svc, ctx.project_id)
    if not pages:
        return {"pages": 0, "note": "no pages to analyse (every uploaded entry was skipped)"}
    engine = StageEngine(svc.db)
    stage = build_s1(svc)
    docs = {
        r["id"]: json.loads(r["meta_json"])
        for r in svc.db.query(
            "SELECT id, meta_json FROM documents WHERE project_id = ?", (ctx.project_id,)
        )
    }
    results: dict[str, S1Out] = {}
    for i, page in enumerate(pages):
        ctx.progress(0.05 + 0.8 * i / len(pages), f"S1 page {i + 1}/{len(pages)}")
        out = engine.run(
            stage, S1In(page=page, doc_meta=docs.get(page.document_id, {}), langs=["tr", "en"]), ctx
        )
        results[page.id] = out
        c = out.classification
        with svc.db.tx(immediate=True) as tx:
            tx.execute(
                "INSERT INTO page_analysis(page_id, project_id, document_id, label, confidence,"
                " needs_review, analysis_json, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(page_id) DO UPDATE SET label = excluded.label,"
                " confidence = excluded.confidence, needs_review = excluded.needs_review,"
                " analysis_json = excluded.analysis_json, updated_at = excluded.updated_at",
                (
                    page.id,
                    ctx.project_id,
                    page.document_id,
                    c.label,
                    c.confidence,
                    int(c.needs_review),
                    out.model_dump_json(),
                    now_iso(),
                ),
            )
        overridden = svc.db.one(
            "SELECT override_label FROM page_analysis WHERE page_id = ?", (page.id,)
        )
        if c.needs_review and not (overridden and overridden["override_label"]):
            _review(
                svc,
                ctx.project_id,
                "page_class",
                page.id,
                {"label": c.label, "confidence": c.confidence, "probabilities": c.probabilities},
            )
        ocr_failures = [n for n in out.notes if n.startswith("OCR incomplete")]
        if ocr_failures:
            _review(svc, ctx.project_id, "page_ocr", page.id, {"notes": ocr_failures})
    ctx.progress(0.9, "S1 schedules")
    schedules = _schedules(svc, ctx, pages, results)
    labels = {pid: _effective_label(svc, pid, out) for pid, out in results.items()}
    hits = {
        pid: [TagHit(t["tag"], t["kind"], tuple(t["bbox"]), t["source"], 1.0) for t in out.tags]
        for pid, out in results.items()
        if labels[pid] in PLAN_CLASSES
    }
    linked_total = unlinked_total = 0
    for sched in schedules:
        linked, unlinked = link_rows(sched, hits)
        linked_total += sum(1 for r in linked.rows if r.links)
        unlinked_total += len(unlinked)
        svc.db.execute(
            "INSERT INTO schedules(id, project_id, source_page, kind, schedule_json, updated_at)"
            " VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET schedule_json = excluded.schedule_json,"
            " kind = excluded.kind, updated_at = excluded.updated_at",
            (
                linked.id,
                ctx.project_id,
                linked.source_page,
                linked.kind,
                linked.model_dump_json(),
                now_iso(),
            ),
        )
        for tag in unlinked:
            if hits:  # without any plan page there is nothing to link to yet
                _review(
                    svc,
                    ctx.project_id,
                    "schedule_link",
                    f"{linked.id}:{tag}",
                    {"schedule": linked.id, "tag": tag},
                )
    return {
        "pages": len(pages),
        "labels": {pid: labels[pid] for pid in results},
        "needs_review": [pid for pid, o in results.items() if o.classification.needs_review],
        "schedules": len(schedules),
        "schedule_rows_linked": linked_total,
        "schedule_rows_unlinked": unlinked_total,
        "timings": [t.model_dump() for t in engine.timings],
    }


def _effective_label(svc: Services, page_id: str, out: S1Out) -> str:
    row = svc.db.one("SELECT override_label FROM page_analysis WHERE page_id = ?", (page_id,))
    return str(row["override_label"]) if row and row["override_label"] else out.classification.label


def _schedules(
    svc: Services, ctx: StageContext, pages: list[PageRef], results: dict[str, S1Out]
) -> list[Schedule]:
    found: list[Schedule] = []
    for page in pages:
        out = results[page.id]
        label = _effective_label(svc, page.id, out)
        content = _json(ctx.store, page.content)
        sid = f"sch_{page.id}"
        if page.kind == "sheet":
            s = schedule_from_rows(sid, page.id, content.get("rows", []))
            if s:
                found.append(s)
        elif page.kind == "docx":
            for ti, table in enumerate(content.get("tables", [])):
                s = schedule_from_rows(f"{sid}_t{ti}", page.id, table)
                if s:
                    found.append(s)
        elif label == "schedule":
            words = [
                Word.model_validate(w)
                for w in (_json(ctx.store, out.words) or _json(ctx.store, page.words) or [])
            ]
            s = schedule_from_rows(sid, page.id, rows_from_words(words))
            if s:
                found.append(s)
    return found
