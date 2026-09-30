"""Deterministic labelled corpora of synthetic pages (vector PDFs, scans, photos).

    python -m archrender.synth.corpus OUT_DIR --per-class 20 --seed 1

Writes one file per sample plus ``labels.jsonl`` (class, source, language, ground truth). The same
seed always produces the same corpus, which makes classification and OCR evaluations repeatable.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from archrender.synth.raster import photo, scan
from archrender.synth.sheets import GENERATORS

CLASSES = [*GENERATORS.keys(), "photo"]


@dataclass
class Sample:
    id: str
    klass: str
    source: str  # vector_pdf | scan | photo
    lang: str
    filename: str
    media_type: str
    data: bytes
    gt: dict[str, Any]


def samples(
    seed: int,
    per_class: int,
    *,
    scan_fraction: float = 0.4,
    langs: tuple[str, ...] = ("tr", "tr", "en"),
) -> Iterator[Sample]:
    for ci, klass in enumerate(CLASSES):
        for i in range(per_class):
            rng = np.random.default_rng([seed, ci, i])
            lang = str(rng.choice(list(langs)))
            sid = f"{klass}_{i:03d}"
            if klass == "photo":
                data, gt = photo(rng)
                yield Sample(sid, klass, "photo", lang, f"{sid}.jpg", "image/jpeg", data, gt)
                continue
            page = GENERATORS[klass](rng, lang)
            if rng.random() < scan_fraction:
                data, media, gt = scan(
                    page.pdf, page.gt, rng, quality=str(rng.choice(["clean", "medium", "noisy"]))
                )
                ext = "jpg" if media == "image/jpeg" else "png"
                yield Sample(sid, klass, "scan", lang, f"{sid}.{ext}", media, data, gt)
            else:
                yield Sample(
                    sid,
                    klass,
                    "vector_pdf",
                    lang,
                    f"{sid}.pdf",
                    "application/pdf",
                    page.pdf,
                    page.gt,
                )


def write_corpus(out: Path, seed: int, per_class: int, **kw: Any) -> list[Sample]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    with (out / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for s in samples(seed, per_class, **kw):
            (out / s.filename).write_bytes(s.data)
            fh.write(
                json.dumps(
                    {
                        "id": s.id,
                        "class": s.klass,
                        "source": s.source,
                        "lang": s.lang,
                        "file": s.filename,
                        "gt": s.gt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written.append(s)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="archrender.synth.corpus")
    ap.add_argument("out", type=Path)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scan-fraction", type=float, default=0.4)
    args = ap.parse_args(argv)
    n = len(write_corpus(args.out, args.seed, args.per_class, scan_fraction=args.scan_fraction))
    print(f"wrote {n} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
