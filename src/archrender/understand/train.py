"""Train the page classifier on synthetic corpora (deterministic, reproducible).

    python -m archrender.understand.train                       # writes configs/classifier/…json
    python -m archrender.understand.train --per-class 20 --workers 4
    python -m archrender.understand.train --cache var/classifier-features   # reuse OCR'd features

Features are computed exactly as S1 computes them in production: vector PDFs use the PDF text
layer and a ≤1024 px preview; scans and photos go through tiled Tesseract OCR at native resolution.
The model records its corpora (seeds, sizes), the OCR version, a hash of the feature-producing code
and calibration metrics. ``--cache`` stores the corpus features keyed by that hash, so refitting
after a change to the model (not the features) skips the OCR; any change to the code that produces
features changes the key.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray
from PIL import Image

from archrender.core.config import REPO_ROOT
from archrender.core.schemas.document import Word
from archrender.ingest.intake import tiles
from archrender.ingest.tasks import _exif_meta, _page_to_px, _pdf_vector_stats, _pdf_words
from archrender.models.impls.tesseract import TesseractOcr
from archrender.models.registry import Registry
from archrender.synth.corpus import Sample, samples
from archrender.understand.classify import PageClassifier, fit
from archrender.understand.features import meta_features, text_features, vector_of, visual_features
from archrender.understand.ocr import ocr_tiles
from archrender.understand.titleblock import extract_title_block

MODEL_PATH = REPO_ROOT / "configs" / "classifier" / "page_classifier_v1.json"
_ocr: TesseractOcr | None = None


def _engine() -> TesseractOcr:
    global _ocr
    if _ocr is None:
        _ocr = TesseractOcr(Registry.load(REPO_ROOT / "configs").get("tesseract-5"))
        _ocr.load()
    return _ocr


def features_for(sample: Sample) -> tuple[NDArray[np.float64], str, str]:
    """(feature vector, class, source) for one synthetic sample, as S1 would compute it."""
    if sample.source == "vector_pdf":
        doc = pdfium.PdfDocument(sample.data)
        page = doc[0]
        w_pt, h_pt = page.get_size()
        preview = np.asarray(page.render(scale=1024 / max(w_pt, h_pt)).to_pil().convert("RGB"))
        w_px, h_px = round(w_pt * 300 / 72), round(h_pt * 300 / 72)
        words = [
            Word.model_validate(x)
            for x in _pdf_words(page, page.get_textpage(), _page_to_px(page, w_px, h_px))
        ]
        vector = _pdf_vector_stats(page)
        tb = extract_title_block("p", words, w_px, h_px, "pdf_text")
        feats = {
            **visual_features(preview),
            **text_features(words, w_px, h_px, has_title_block=tb is not None),
            **meta_features({}, {}, vector),
        }
        doc.close()
        return vector_of(feats), sample.klass, sample.source
    img = Image.open(io.BytesIO(sample.data))
    exif = _exif_meta(img)
    rgb = np.asarray(img.convert("RGB"))
    h_px, w_px = rgb.shape[:2]
    words = ocr_tiles(
        rgb, tiles(w_px, h_px, 1536, 0.2), _engine(), langs=["tr", "en"], source="ocr:tesseract-5"
    )
    tb = extract_title_block("p", words, w_px, h_px, "ocr")
    feats = {
        **visual_features(rgb),
        **text_features(words, w_px, h_px, has_title_block=tb is not None),
        **meta_features({}, {"exif": exif} if exif else {}, None),
    }
    return vector_of(feats), sample.klass, sample.source


FEATURE_CODE = [
    "src/archrender/synth",
    "src/archrender/understand/features.py",
    "src/archrender/understand/ocr.py",
    "src/archrender/understand/titleblock.py",
    "src/archrender/understand/text.py",
    "src/archrender/understand/train.py",
    "src/archrender/ingest/tasks.py",
    "src/archrender/ingest/intake.py",
    "src/archrender/models/impls/tesseract.py",
    "src/archrender/assets",
]


def feature_code_sha() -> str:
    """Hash of everything that shapes the training features (code, fonts, OCR version)."""
    h = hashlib.sha256()
    for rel in FEATURE_CODE:
        root = REPO_ROOT / rel
        for f in sorted(root.rglob("*") if root.is_dir() else [root]):
            if f.is_file() and "__pycache__" not in f.parts:
                h.update(str(f.relative_to(REPO_ROOT)).encode())
                h.update(f.read_bytes())
    h.update(_engine().version.encode())
    return h.hexdigest()[:16]


def corpus_features(
    seed: int, per_class: int, workers: int, cache: Path | None = None
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    path = cache / f"features_{feature_code_sha()}_s{seed}_n{per_class}.npz" if cache else None
    if path is not None and path.exists():
        data = np.load(path)
        return data["x"], [str(v) for v in data["y"]], [str(v) for v in data["src"]]
    items = list(samples(seed, per_class))
    with Pool(workers) as pool:
        rows = pool.map(features_for, items, chunksize=1)
    x = np.stack([r[0] for r in rows])
    y, src = [r[1] for r in rows], [r[2] for r in rows]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, x=x, y=np.array(y), src=np.array(src))
    return x, y, src


def macro_f1(y_true: list[str], y_pred: list[str], classes: list[str]) -> float:
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred, strict=True))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def train(
    seed: int,
    per_class: int,
    cal_seed: int,
    cal_per_class: int,
    workers: int,
    cache: Path | None = None,
) -> PageClassifier:
    from archrender.synth.corpus import CLASSES

    classes = sorted(CLASSES)
    x, y, src = corpus_features(seed, per_class, workers, cache)
    xc, yc, srcc = corpus_features(cal_seed, cal_per_class, workers, cache)
    yi = np.array([classes.index(c) for c in y])
    yci = np.array([classes.index(c) for c in yc])
    git = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    model = fit(
        x,
        yi,
        classes,
        x_cal=xc,
        y_cal=yci,
        info={
            "trained_on": "synthetic corpus only (archrender.synth); no client or third-party documents",
            "train_corpus": {"seed": seed, "per_class": per_class, "sources": dict(Counter(src))},
            "calibration_corpus": {
                "seed": cal_seed,
                "per_class": cal_per_class,
                "sources": dict(Counter(srcc)),
            },
            "ocr": _engine().version,
            "feature_code_sha": feature_code_sha(),
            "git_commit": git,
            "vlm_features": False,
        },
    )
    pred = [classes[i] for i in model.predict_proba(xc).argmax(axis=1)]
    model.info["calibration_macro_f1"] = round(macro_f1(yc, pred, classes), 4)
    model.info["calibration_accuracy"] = round(
        float(np.mean([a == b for a, b in zip(yc, pred, strict=True)])), 4
    )
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="archrender.understand.train")
    ap.add_argument("--out", type=Path, default=MODEL_PATH)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--cal-seed", type=int, default=2)
    ap.add_argument("--cal-per-class", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", type=Path, help="directory for cached corpus features")
    args = ap.parse_args(argv)
    model = train(
        args.seed, args.per_class, args.cal_seed, args.cal_per_class, args.workers, args.cache
    )
    model.save(args.out)
    print(json.dumps({k: v for k, v in model.info.items() if k != "trained_on"}, indent=1))
    print(f"temperature {model.temperature:.3f} → {args.out}")
    return 0


def summary(model: PageClassifier) -> dict[str, Any]:
    return {"classes": model.classes, "temperature": model.temperature, **model.info}


if __name__ == "__main__":
    raise SystemExit(main())
