"""S1 building blocks on synthetic sheets: tiled OCR (Tesseract), bubble tags, title blocks,
north arrows and schedules. Vector pages use their text layer; scans go through OCR."""

from __future__ import annotations

import io

import numpy as np
import pypdfium2 as pdfium
import pytest
from PIL import Image

from archrender.core.config import REPO_ROOT
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.document import Tile, Word
from archrender.ingest.intake import tiles
from archrender.ingest.tasks import _page_to_px, _pdf_words
from archrender.models.registry import Registry
from archrender.models.roles import OcrWord
from archrender.synth.layout import random_layout
from archrender.synth.raster import render_pdf, scan
from archrender.synth.sheets import GT_DPI, floor_plan_page, schedule_page, site_plan_page
from archrender.understand.north import measure_north
from archrender.understand.ocr import ocr_tiles
from archrender.understand.schedules import link_rows, rows_from_words, schedule_from_rows
from archrender.understand.tags import (
    _vote,
    bubble_reading_to_tag,
    bubble_text,
    find_bubbles,
    ink_mask,
    read_bubble_tags,
    tags_from_words,
)
from archrender.understand.titleblock import extract_title_block
from tests.conftest import require_tool


def _text_layer(pdf: bytes) -> list[Word]:
    doc = pdfium.PdfDocument(pdf)
    pg = doc[0]
    w, h = (round(v * GT_DPI / 72) for v in pg.get_size())
    return [
        Word.model_validate(x) for x in _pdf_words(pg, pg.get_textpage(), _page_to_px(pg, w, h))
    ]


@pytest.fixture(scope="module")
def tesseract():  # type: ignore[no-untyped-def]
    require_tool("tesseract")
    from archrender.models.impls.tesseract import TesseractOcr

    eng = TesseractOcr(Registry.load(REPO_ROOT / "configs").get("tesseract-5"))
    eng.load()
    return eng


# ---------------------------------------------------------------------------------------------
# tiling logic with a scripted engine (no OCR needed)
# ---------------------------------------------------------------------------------------------
class _BlobReader:
    """Fake OCR: each word is a filled grey rectangle whose grey level encodes the text. A word cut
    by the crop edge is 'misread' as its first half, like a real engine seeing a partial word."""

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts

    def read(self, img: np.ndarray, langs: list[str]) -> list[OcrWord]:
        import cv2

        g = img[..., 0]
        out = []
        for level, text in self.texts.items():
            mask = (g == level).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
            for i in range(1, n):
                x, y, w, h, _ = stats[i]
                cut = x == 0 or y == 0 or x + w == g.shape[1] or y + h == g.shape[0]
                out.append(
                    OcrWord(text[: len(text) // 2] if cut else text, (x, y, x + w, y + h), 0.9)
                )
        return out


def test_tiling_reads_each_word_once_and_drops_words_cut_by_inner_edges() -> None:
    page = np.full((800, 1600, 3), 255, np.uint8)
    words = {10: "DUVAR", 20: "3,50", 30: "SALON"}
    page[100:140, 100:300] = 10  # inside tile 1 only
    page[400:440, 700:900] = 20  # inside both tiles (overlap)
    page[600:640, 950:1150] = 30  # crosses tile 1's right edge (x = 1000): whole only in tile 2
    ts = [Tile(x=0, y=0, w=1000, h=800), Tile(x=600, y=0, w=1000, h=800)]
    got = ocr_tiles(page, ts, _BlobReader(words), langs=["tr"], source="ocr:test", rotations=(0,))
    assert sorted(w.text for w in got) == ["3,50", "DUVAR", "SALON"]
    salon = next(w for w in got if w.text == "SALON")
    assert (salon.x0, salon.x1) == (950, 1150) and salon.source == "ocr:test"


class _FailingOnSecondTile(_BlobReader):
    def read(self, img: np.ndarray, langs: list[str]) -> list[OcrWord]:
        if img.shape[1] == 999:
            raise ArchRenderError(ErrorCode.STAGE_FAILED, "Tesseract did not finish.", "")
        return super().read(img, langs)


def test_an_unreadable_tile_is_reported_and_the_rest_is_still_read() -> None:
    page = np.full((800, 1600, 3), 255, np.uint8)
    page[100:140, 100:300] = 10
    ts = [Tile(x=0, y=0, w=1000, h=800), Tile(x=601, y=0, w=999, h=800)]
    engine = _FailingOnSecondTile({10: "DUVAR"})
    with pytest.raises(ArchRenderError):  # without a failure list the page fails loudly
        ocr_tiles(page, ts, engine, langs=["tr"], source="ocr:test", rotations=(0,))
    failures: list[str] = []
    got = ocr_tiles(
        page, ts, engine, langs=["tr"], source="ocr:test", rotations=(0,), failures=failures
    )
    assert [w.text for w in got] == ["DUVAR"]
    assert failures == ["tile x=601 y=0 999×800 px, 0°: Tesseract did not finish."]


def test_only_noisy_scans_are_despeckled_before_ocr() -> None:
    from archrender.models.impls.tesseract import DESPECKLE_SIGMA, noise_sigma

    page = floor_plan_page(np.random.default_rng(23))
    for quality, noisy in (("clean", False), ("noisy", True)):
        data, _, _ = scan(page.pdf, page.gt, np.random.default_rng(3), dpi=200, quality=quality)
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"))[:1536, :1536]
        assert (noise_sigma(np.ascontiguousarray(gray)) > DESPECKLE_SIGMA) is noisy, quality


def test_tesseract_timeout_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    from archrender.models.impls.tesseract import TesseractOcr

    eng = TesseractOcr(Registry.load(REPO_ROOT / "configs").get("tesseract-5"))
    eng.binary = "/usr/bin/tesseract"

    def hang(cmd: list[str], data: bytes) -> None:
        raise subprocess.TimeoutExpired(cmd, 120)

    monkeypatch.setattr(TesseractOcr, "_run", staticmethod(hang))
    with pytest.raises(ArchRenderError) as e:
        eng.read(np.full((64, 64, 3), 255, np.uint8), ["tr"])
    assert "did not finish" in e.value.message and not e.value.retryable
    assert eng.read_line(np.full((64, 64, 3), 255, np.uint8), "K0123") == ""


# ---------------------------------------------------------------------------------------------
# real OCR on synthetic scans
# ---------------------------------------------------------------------------------------------
def _iou(a: list[float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def test_tesseract_reads_dimensions_and_title_block_on_a_clean_scan(tesseract) -> None:  # type: ignore[no-untyped-def]
    page = floor_plan_page(np.random.default_rng(21))
    data, _, gt = scan(
        page.pdf, page.gt, np.random.default_rng(2), dpi=300, skew_deg=0.0, quality="clean"
    )
    rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    words = ocr_tiles(
        rgb,
        tiles(rgb.shape[1], rgb.shape[0], 1536, 0.2),
        tesseract,
        langs=["tr", "en"],
        source="ocr:tesseract-5",
    )

    def recall(role: str) -> float:
        g = [w for w in gt["words"] if (w["role"] or "").startswith(role)]
        ok = sum(
            any(
                o.text == w["text"] and _iou(w["bbox"], (o.x0, o.y0, o.x1, o.y1)) > 0.3
                for o in words
            )
            for w in g
        )
        return ok / len(g)

    # measured baseline (not a Phase-2 acceptance number; that is the GPU OCR run)
    assert recall("dim") >= 0.95  # incl. vertical dimension strings (rotated pass)
    assert recall("title:") >= 0.95


@pytest.mark.parametrize("quality", ["clean", "noisy"])
def test_bubble_tags_read_without_false_positives(tesseract, quality: str) -> None:  # type: ignore[no-untyped-def]
    found = total = 0
    for seed in (21, 22):
        page = floor_plan_page(np.random.default_rng(seed))
        data, _, gt = scan(
            page.pdf, page.gt, np.random.default_rng(2), dpi=300, skew_deg=0.0, quality=quality
        )
        gray = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        truth = {t["tag"] for t in gt["tags"]}
        got = {h.tag for h in read_bubble_tags(gray, 300, tesseract)}
        assert not got - truth  # nothing invented
        found += len(got & truth)
        total += len(truth)
    # measured baseline at 300 DPI (Phase 2: 95–97 % on 6 plans per quality; swings crossing
    # the lettering are the misses)
    assert found / total >= 0.9


def test_bubbles_touching_a_door_swing_are_found_and_read_apart_from_it() -> None:
    import cv2

    gray = np.full((300, 300), 240, np.uint8)
    cv2.circle(gray, (150, 150), 38, 30, 2)  # 6.4 mm bubble at 300 DPI
    cv2.putText(gray, "K7", (122, 165), cv2.FONT_HERSHEY_DUPLEX, 1.2, 20, 4)
    cv2.ellipse(gray, (60, 60), (150, 150), 0, 0, 90, 30, 2)  # swing arc across the bubble
    bubbles = find_bubbles(gray, 300)
    # the outer boundary merged with the arc; the larger piece of the split hole is kept
    assert len(bubbles) == 1
    cx, cy, ew, eh, _ = bubbles[0]
    assert abs(cx - 150) < 8 and abs(cy - 150) < 8 and 50 < min(ew, eh) <= max(ew, eh) < 80
    crop = bubble_text(gray, ink_mask(gray), bubbles[0], 300, 4)
    assert crop is not None
    ink = crop < 128
    # the lettering survives, the arc does not: no ink reaches the crop's border rows/columns
    assert ink.sum() > 300
    assert not (ink[0].any() or ink[-1].any() or ink[:, 0].any() or ink[:, -1].any())


def test_bubble_readings_repair_digit_look_alikes_only_after_a_tag_prefix() -> None:
    assert bubble_reading_to_tag("PS") == "P5"
    assert bubble_reading_to_tag("KIO") == "K10"
    assert bubble_reading_to_tag("K-07") == "K7"  # plain tags parse as before
    assert bubble_reading_to_tag("SO") is None  # no door/window prefix
    assert bubble_reading_to_tag("PX") is None  # X is no digit look-alike
    assert bubble_reading_to_tag("") is None


def test_bubble_vote_needs_two_agreeing_readings_and_no_conflict() -> None:
    assert _vote(["K7", "K7", None]) == "K7"
    assert _vote(["K7", None, None]) is None
    assert _vote(["K7", "K7", "K1"]) is None
    assert _vote([None, None, None]) is None


# ---------------------------------------------------------------------------------------------
# text-layer based extraction (vector sheets)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_title_block_and_scale_from_the_text_layer(seed: int) -> None:
    page = floor_plan_page(np.random.default_rng(seed), lang="tr" if seed % 2 else "en")
    words = _text_layer(page.pdf)
    w_px, h_px = (v / 25.4 * GT_DPI for v in page.gt["page_mm"])
    tb = extract_title_block("p", words, w_px, h_px, "pdf_text")
    assert tb is not None
    gt = page.gt["title_block"]["fields"]
    for key in ("project", "sheet_title", "sheet_no", "date"):
        assert tb.fields[key].value == gt[key], (key, tb.fields[key].value, gt[key])
    assert tb.scale is not None and tb.scale.value == page.gt["scale"]
    assert tb.scale.provenance[0].method == "pdf_text"


@pytest.mark.parametrize("seed", range(8))
def test_north_arrow_angle_within_two_degrees(seed: int) -> None:
    page = site_plan_page(np.random.default_rng(seed))
    words = _text_layer(page.pdf)
    gray = np.asarray(render_pdf(page.pdf, GT_DPI).convert("L"))
    na = measure_north("p", gray, words, GT_DPI)
    assert na is not None
    err = (na.angle_deg.value - page.gt["north"]["angle_deg"] + 180) % 360 - 180
    assert abs(err) <= 2.0, (na.angle_deg.value, page.gt["north"]["angle_deg"])


def test_door_window_schedule_rebuilt_from_pdf_and_linked_to_plan_tags() -> None:
    rng = np.random.default_rng(3)
    lay = random_layout(rng)
    plan = floor_plan_page(rng, lay)
    sched_page = schedule_page(rng, lay, kind="door_window")
    table = rows_from_words(_text_layer(sched_page.pdf))
    sched = schedule_from_rows("s1", "sched_p0", table)
    assert sched is not None and sched.kind == "door_window"
    gt_rows = sched_page.gt["schedule"]["rows"]
    assert [r.tag for r in sched.rows] == [r[0] for r in gt_rows]
    first = sched.rows[0]
    o = next(o for o in lay.openings if o.tag == first.tag)
    assert first.fields["width"] == pytest.approx(o.width) and first.fields[
        "height"
    ] == pytest.approx(o.height)
    hits = tags_from_words(_text_layer(plan.pdf))
    linked, unlinked = link_rows(sched, {"plan_p0": hits})
    assert not unlinked and all(r.links for r in linked.rows)


def test_finish_schedule_from_sheet_rows() -> None:
    rows = [
        ["MAHAL LİSTESİ", None, None],
        ["MAHAL NO", "MAHAL ADI", "ZEMİN", "DUVAR", "TAVAN"],
        ["Z01", "Salon", "Meşe parke", "Saten boya", "Alçıpan"],
        ["Z02", "Banyo", "Seramik", "Seramik", "Nem dayanımlı alçıpan"],
    ]
    sched = schedule_from_rows("s2", "xlsx_p0", rows)
    assert sched is not None and sched.kind == "finish"
    assert (
        sched.rows[1].fields["room_name"] == "Banyo" and sched.rows[1].fields["floor"] == "Seramik"
    )
    assert sched.columns["floor"] == "ZEMİN"


def test_schedule_units_from_headers() -> None:
    rows = [["POZ", "GENİŞLİK (mm)", "YÜKSEKLİK (mm)", "ADET"], ["P1", "1200", "1400", "2"]]
    sched = schedule_from_rows("s3", "p", rows)
    assert sched is not None
    r = sched.rows[0]
    assert r.tag == "P1" and r.fields["width"] == pytest.approx(1.2) and r.fields["qty"] == 2.0
