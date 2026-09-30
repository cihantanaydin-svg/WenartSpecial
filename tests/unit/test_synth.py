"""Synthetic sheet generator: determinism, ground truth vs the PDF text layer, scans, layouts."""

from __future__ import annotations

from collections import deque

import numpy as np
import pypdfium2 as pdfium
import pytest

from archrender.ingest.tasks import _page_to_px, _pdf_words
from archrender.synth.corpus import CLASSES, samples
from archrender.synth.layout import opening_point, random_layout
from archrender.synth.raster import render_pdf, scan
from archrender.synth.sheets import GENERATORS, GT_DPI, floor_plan_page, schedule_page


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_generation_is_deterministic() -> None:
    a = [s.data for s in samples(3, 1)]
    b = [s.data for s in samples(3, 1)]
    assert a == b and len(a) == len(CLASSES)


@pytest.mark.parametrize("klass", sorted(GENERATORS))
def test_ground_truth_words_match_the_pdf_text_layer(klass: str) -> None:
    page = GENERATORS[klass](np.random.default_rng(5), "tr")
    doc = pdfium.PdfDocument(page.pdf)
    pg = doc[0]
    w, h = (round(v * GT_DPI / 72) for v in pg.get_size())
    extracted = _pdf_words(pg, pg.get_textpage(), _page_to_px(pg, w, h))
    gt = page.gt["words"]
    assert gt, klass
    matched = 0
    for g in gt:
        if any(e["text"] == g["text"] and _iou(e_box(e), g["bbox"]) > 0.5 for e in extracted):
            matched += 1
    assert matched / len(gt) >= 0.98, (klass, matched, len(gt))


def e_box(e: dict[str, float]) -> list[float]:
    return [e["x0"], e["y0"], e["x1"], e["y1"]]


def test_scan_ground_truth_lands_on_ink() -> None:
    page = floor_plan_page(np.random.default_rng(9))
    data, media, gt = scan(
        page.pdf, page.gt, np.random.default_rng(1), dpi=200, skew_deg=1.2, quality="clean"
    )
    from io import BytesIO

    from PIL import Image

    img = np.asarray(Image.open(BytesIO(data)).convert("L"))
    inked = 0
    for wd in gt["words"]:
        x0, y0, x1, y1 = (round(v) for v in wd["bbox"])
        if (img[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] < 128).any():
            inked += 1
    assert inked / len(gt["words"]) >= 0.98
    assert abs(gt["skew_deg"] - 1.2) < 1e-6 and gt["gt_dpi"] == 200
    assert media in ("image/png", "image/jpeg")


def test_render_matches_page_size() -> None:
    page = floor_plan_page(np.random.default_rng(2))
    img = render_pdf(page.pdf, 72)
    w_mm, h_mm = page.gt["page_mm"]
    assert abs(img.width - w_mm / 25.4 * 72) <= 1 and abs(img.height - h_mm / 25.4 * 72) <= 1


@pytest.mark.parametrize("seed", range(40))
def test_layouts_are_connected_and_openings_do_not_overlap(seed: int) -> None:
    lay = random_layout(np.random.default_rng(seed))
    assert 5 <= len(lay.rooms) <= 8
    # union of rooms tiles the footprint
    assert abs(sum(r.area for r in lay.rooms) - lay.width * lay.depth) < 1e-6
    doors = [o for o in lay.openings if o.kind == "door"]
    adj: dict[str, set[str]] = {r.id: set() for r in lay.rooms}
    for o in doors:
        w = lay.walls[o.wall]
        if w.exterior:
            continue
        x, y = opening_point(lay, o)
        eps = 0.05
        probes = [(x, y - eps), (x, y + eps)] if w.horizontal else [(x - eps, y), (x + eps, y)]
        found = [
            r.id for p in probes for r in lay.rooms if r.x0 <= p[0] <= r.x1 and r.y0 <= p[1] <= r.y1
        ]
        assert len(set(found)) == 2, (seed, o.tag)
        a, b = sorted(set(found))
        adj[a].add(b)
        adj[b].add(a)
    seen = {lay.rooms[0].id}
    q = deque(seen)
    while q:
        for n in adj[q.popleft()] - seen:
            seen.add(n)
            q.append(n)
    assert seen == set(adj), seed
    for i, a in enumerate(lay.openings):
        for b in lay.openings[i + 1 :]:
            if a.wall == b.wall:
                assert abs(a.t - b.t) >= (a.width + b.width) / 2, (seed, a.tag, b.tag)
    assert sum(o.kind == "door" and lay.walls[o.wall].exterior for o in lay.openings) == 1


def test_schedule_ground_truth_lists_every_opening() -> None:
    rng = np.random.default_rng(4)
    lay = random_layout(rng)
    page = schedule_page(rng, lay, kind="door_window")
    tags = [row[0] for row in page.gt["schedule"]["rows"]]
    assert sorted(tags) == sorted(o.tag for o in lay.openings)
