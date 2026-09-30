"""Phase-2 evaluation on a held-out synthetic corpus (seed differs from training).

Reports, per source (vector PDF / scan / photo): classification macro-F1 and review rate;
title-block field accuracy and scale accuracy; north-arrow angle error; OCR recall of dimension
strings, title-block text, room names and tags on scans, with Turkish character accuracy; and
schedule-to-plan tag linking on synthetic projects. Everything is measured on synthetic data only:
the real-document numbers come from the pod run with the VLM (UNVERIFIED-ON-GPU).
"""

from __future__ import annotations

import io
from collections import Counter
from multiprocessing import Pool
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from PIL import Image

from archrender.core.config import REPO_ROOT
from archrender.core.schemas.document import Word
from archrender.ingest.intake import tiles
from archrender.ingest.tasks import _exif_meta, _page_to_px, _pdf_vector_stats, _pdf_words
from archrender.synth.corpus import Sample, samples
from archrender.synth.layout import random_layout
from archrender.synth.raster import render_pdf
from archrender.synth.sheets import GT_DPI, floor_plan_page, schedule_page
from archrender.understand.classify import REVIEW_THRESHOLD, PageClassifier
from archrender.understand.features import meta_features, text_features, vector_of, visual_features
from archrender.understand.north import measure_north
from archrender.understand.ocr import ocr_tiles
from archrender.understand.schedules import link_rows, rows_from_words, schedule_from_rows
from archrender.understand.tags import read_bubble_tags, tags_from_words
from archrender.understand.titleblock import extract_title_block
from archrender.understand.train import MODEL_PATH, _engine, macro_f1


def _iou(a: list[float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _char_accuracy(truth: str, got: str) -> float:
    """1 − normalised Levenshtein distance."""
    if not truth:
        return 1.0 if not got else 0.0
    prev = list(range(len(got) + 1))
    for i, ct in enumerate(truth, 1):
        cur = [i]
        for j, cg in enumerate(got, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ct != cg)))
        prev = cur
    return max(0.0, 1.0 - prev[-1] / len(truth))


def _analyse(sample: Sample) -> dict[str, Any]:
    gt = sample.gt
    out: dict[str, Any] = {"id": sample.id, "class": sample.klass, "source": sample.source}
    if sample.source == "vector_pdf":
        doc = pdfium.PdfDocument(sample.data)
        page = doc[0]
        w_pt, h_pt = page.get_size()
        w_px, h_px = round(w_pt * GT_DPI / 72), round(h_pt * GT_DPI / 72)
        words = [
            Word.model_validate(x)
            for x in _pdf_words(page, page.get_textpage(), _page_to_px(page, w_px, h_px))
        ]
        preview = np.asarray(page.render(scale=1024 / max(w_pt, h_pt)).to_pil().convert("RGB"))
        vector = _pdf_vector_stats(page)
        doc.close()
        method = "pdf_text"
        rgb_full = None
        exif: dict[str, Any] = {}
    else:
        img = Image.open(io.BytesIO(sample.data))
        exif = _exif_meta(img)
        rgb_full = np.asarray(img.convert("RGB"))
        h_px, w_px = rgb_full.shape[:2]
        words = ocr_tiles(
            rgb_full,
            tiles(w_px, h_px, 1536, 0.2),
            _engine(),
            langs=["tr", "en"],
            source="ocr:tesseract-5",
        )
        preview = rgb_full
        vector = None
        method = "ocr"
    tb = extract_title_block(sample.id, words, w_px, h_px, method)  # type: ignore[arg-type]
    feats = {
        **visual_features(preview),
        **text_features(words, w_px, h_px, has_title_block=tb is not None),
        **meta_features({}, {"exif": exif} if exif else {}, vector),
    }
    out["x"] = vector_of(feats).tolist()
    if "title_block" in gt:
        fields = gt["title_block"]["fields"]
        got = {k: v.value for k, v in (tb.fields.items() if tb else [])}
        out["tb_fields"] = (sum(got.get(k) == v for k, v in fields.items()), len(fields))
        if "scale" in gt:
            out["scale_ok"] = bool(tb and tb.scale and tb.scale.value == gt["scale"])
    if "north" in gt and sample.source == "vector_pdf":
        gray = np.asarray(render_pdf(sample.data, GT_DPI).convert("L"))
        na = measure_north(sample.id, gray, words, GT_DPI)
        out["north_err"] = (
            None
            if na is None
            else abs((na.angle_deg.value - gt["north"]["angle_deg"] + 180) % 360 - 180)
        )
    if sample.source == "scan" and rgb_full is not None:
        for role in ("dim", "title:", "room_name"):
            g = [w for w in gt["words"] if (w.get("role") or "").startswith(role)]
            hits = 0
            chars: list[float] = []
            for w in g:
                near = [o for o in words if _iou(w["bbox"], (o.x0, o.y0, o.x1, o.y1)) > 0.3]
                hits += any(o.text == w["text"] for o in near)
                if role == "room_name":
                    best = max((_char_accuracy(w["text"], o.text) for o in near), default=0.0)
                    chars.append(best)
            out[f"ocr_{role.rstrip(':')}"] = (hits, len(g))
            if chars:
                out["tr_char_acc"] = (sum(chars), len(chars))
        if gt.get("tags"):
            gray = np.asarray(Image.fromarray(rgb_full).convert("L"))
            found = {
                h.tag for h in read_bubble_tags(gray, float(gt.get("gt_dpi") or 300), _engine())
            }
            truth = {t["tag"] for t in gt["tags"]}
            out["tags"] = (len(found & truth), len(truth), len(found - truth))
    return out


def schedule_linking(seed: int, n: int) -> dict[str, Any]:
    linked = total = unlinked_projects = 0
    for i in range(n):
        rng = np.random.default_rng([seed, 99, i])
        layout = random_layout(rng)
        plan = floor_plan_page(rng, layout)
        sched_page = schedule_page(rng, layout, kind="door_window")

        def words(pdf: bytes) -> list[Word]:
            d = pdfium.PdfDocument(pdf)
            p = d[0]
            w, h = (round(v * GT_DPI / 72) for v in p.get_size())
            return [
                Word.model_validate(x)
                for x in _pdf_words(p, p.get_textpage(), _page_to_px(p, w, h))
            ]

        sched = schedule_from_rows("s", "p", rows_from_words(words(sched_page.pdf)))
        if sched is None:
            unlinked_projects += 1
            continue
        res, unlinked = link_rows(sched, {"plan": tags_from_words(words(plan.pdf))})
        total += len(res.rows)
        linked += sum(1 for r in res.rows if r.links)
        unlinked_projects += bool(unlinked)
    return {
        "projects": n,
        "rows_linked": f"{linked}/{total}",
        "projects_fully_linked": n - unlinked_projects,
    }


def evaluate(seed: int = 3, per_class: int = 4, workers: int = 4) -> dict[str, Any]:
    model = PageClassifier.load(MODEL_PATH)
    items = list(samples(seed, per_class))
    with Pool(workers) as pool:
        rows = pool.map(_analyse, items, chunksize=1)
    x = np.array([r["x"] for r in rows])
    proba = model.predict_proba(x)
    pred = [model.classes[i] for i in proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    unfamiliar = np.array([bool(model.out_of_range(xi)) for xi in x])
    review = (conf < REVIEW_THRESHOLD) | unfamiliar
    truth = [r["class"] for r in rows]
    by_source: dict[str, Any] = {}
    for src in sorted({r["source"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["source"] == src]
        by_source[src] = {
            "n": len(idx),
            "macro_f1": round(
                macro_f1([truth[i] for i in idx], [pred[i] for i in idx], model.classes), 3
            ),
            "accuracy": round(float(np.mean([truth[i] == pred[i] for i in idx])), 3),
        }
    confusions = Counter((t, p) for t, p in zip(truth, pred, strict=True) if t != p).most_common(5)

    def ratio(key: str) -> str:
        vals = [r[key] for r in rows if key in r]
        a, b = sum(v[0] for v in vals), sum(v[1] for v in vals)
        return f"{a}/{b} ({a / b:.0%})" if b else "n/a"

    north = [r["north_err"] for r in rows if r.get("north_err") is not None]
    north_missing = sum(1 for r in rows if "north_err" in r and r["north_err"] is None)
    tag_rows = [r["tags"] for r in rows if "tags" in r]
    tr = [r["tr_char_acc"] for r in rows if "tr_char_acc" in r]
    scaled = [r for r in rows if "scale_ok" in r]
    return {
        "corpus": {
            "seed": seed,
            "per_class": per_class,
            "n": len(rows),
            "sources": dict(Counter(r["source"] for r in rows)),
        },
        "classification": {
            "macro_f1": round(macro_f1(truth, pred, model.classes), 3),
            "accuracy": round(
                float(np.mean([a == b for a, b in zip(truth, pred, strict=True)])), 3
            ),
            "review_rate": round(float(np.mean(review)), 3),
            "outside_training_range": int(unfamiliar.sum()),
            "errors_not_sent_to_review": sum(
                1 for t, p_, r in zip(truth, pred, review, strict=True) if t != p_ and not r
            ),
            "by_source": by_source,
            "top_confusions": [f"{t}→{p} ×{n}" for (t, p), n in confusions],
            "model_trained_on": model.info.get("trained_on"),
        },
        "title_block_fields": ratio("tb_fields"),
        "scale_from_title_block": f"{sum(r['scale_ok'] for r in scaled)}/{len(scaled)}",
        "north_arrow": {
            "measured": len(north),
            "missing": north_missing,
            "mean_abs_err_deg": round(float(np.mean(north)), 2) if north else None,
            "max_abs_err_deg": round(float(np.max(north)), 2) if north else None,
        },
        "ocr_on_scans": {
            "dimension_strings": ratio("ocr_dim"),
            "title_block_text": ratio("ocr_title"),
            "room_names_exact": ratio("ocr_room_name"),
            "turkish_char_accuracy_room_names": round(
                sum(v[0] for v in tr) / max(1, sum(v[1] for v in tr)), 3
            )
            if tr
            else None,
            "bubble_tags": (
                f"{sum(t[0] for t in tag_rows)}/{sum(t[1] for t in tag_rows)} found, "
                f"{sum(t[2] for t in tag_rows)} false"
                if tag_rows
                else "n/a"
            ),
            "engine": _engine().version,
        },
        "schedule_linking": schedule_linking(seed, max(2, per_class)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(), indent=1, ensure_ascii=False))
    print(f"model: {MODEL_PATH.relative_to(REPO_ROOT)}")
