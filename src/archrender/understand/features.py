"""Cheap page features for classification (ARCHITECTURE §S1).

Visual features are computed on a ≤1024 px preview; text features on the page's words (text
layer, DXF text or OCR). Every feature is a finite float; the vector layout is fixed by
:data:`FEATURE_NAMES` and versioned with the classifier model.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.document import Word
from archrender.core.schemas.understanding import PAGE_CLASSES
from archrender.understand.text import fold, normalise_tag, parse_dimension

KEYWORDS: dict[str, tuple[str, ...]] = {
    "floor_plan": (
        "kat plani",
        "zemin kat",
        "floor plan",
        "salon",
        "mutfak",
        "yatak",
        "banyo",
        "antre",
        "living",
        "kitchen",
        "bedroom",
        "bathroom",
        "plan",
    ),
    "ceiling_plan": ("tavan", "ceiling", "reflected", "armatur", "spot", "asma tavan"),
    "section": ("kesit", "kesiti", "section"),
    "elevation": ("gorunus", "gorunusu", "elevation", "cephe", "facade"),
    "detail": ("detay", "detail", "yalitim", "insulation", "siva", "plaster", "tugla", "betonarme"),
    "site_plan": (
        "vaziyet",
        "site plan",
        "ada",
        "parsel",
        "sokak",
        "cadde",
        "street",
        "avenue",
        "lot",
    ),
    "schedule": (
        "listesi",
        "schedule",
        "poz",
        "adet",
        "qty",
        "mahal",
        "genislik",
        "width",
        "yukseklik",
    ),
    "text_document": (
        "ozeti",
        "brief",
        "sartname",
        "specification",
        "notlari",
        "notes",
        "sayfa",
        "page",
    ),
    "moodboard": ("konsept", "mood", "palet", "palette", "ral", "ncs", "board"),
    "other": (
        "pafta listesi",
        "drawing list",
        "uygulama projesi",
        "construction documents",
        "kapak",
        "cover",
    ),
    "photo": (),
}
VISUAL = [
    "ink_frac",
    "dark_frac",
    "white_frac",
    "colorfulness",
    "saturation",
    "edge_density",
    "line_count",
    "hv_line_frac",
    "diag_line_frac",
    "long_line_frac",
    "grid_score",
    "small_circles",
    "uniform_colour_frac",
    "aspect",
    "portrait",
]
TEXTUAL = [
    "log_words",
    "numeric_frac",
    "dimension_count",
    "tag_count",
    "avg_word_len",
    "long_line_words",
    "has_title_block",
    *[f"kw_{c}" for c in PAGE_CLASSES],
]
META = ["exif_camera", "is_vector", "log_paths", "image_area_frac"]
FEATURE_NAMES = [*VISUAL, *TEXTUAL, *META]


def visual_features(rgb: NDArray[np.uint8]) -> dict[str, float]:
    h, w = rgb.shape[:2]
    scale = 1024 / max(h, w)
    if scale < 1:
        rgb = np.asarray(
            cv2.resize(
                rgb,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            ),
            dtype=np.uint8,
        )
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    f: dict[str, float] = {}
    f["ink_frac"] = float((gray < 128).mean())
    f["dark_frac"] = float((gray < 60).mean())
    f["white_frac"] = float((gray > 235).mean())
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    f["colorfulness"] = (
        float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())) / 100.0
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    f["saturation"] = float(hsv[..., 1].mean() / 255.0)
    edges = cv2.Canny(gray, 60, 160)
    f["edge_density"] = float((edges > 0).mean())
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=40,
        minLineLength=max(15, int(0.03 * max(h, w))),
        maxLineGap=3,
    )
    total = hv = diag = long_ = 0.0
    long_h = long_v = 0
    diag_len = math.hypot(h, w)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # (N,1,4) in OpenCV 4, (N,4) in 5
            length = math.hypot(x2 - x1, y2 - y1)
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
            total += length
            if min(ang, 180 - ang) < 3 or abs(ang - 90) < 3:
                hv += length
                if length > 0.25 * w and min(ang, 180 - ang) < 3:
                    long_h += 1
                if length > 0.25 * h and abs(ang - 90) < 3:
                    long_v += 1
            elif abs(ang - 45) < 8 or abs(ang - 135) < 8:
                diag += length
            if length > 0.2 * diag_len:
                long_ += length
    f["line_count"] = math.log1p(0 if lines is None else len(lines))
    f["hv_line_frac"] = hv / total if total else 0.0
    f["diag_line_frac"] = diag / total if total else 0.0
    f["long_line_frac"] = long_ / total if total else 0.0
    f["grid_score"] = math.log1p(min(long_h, 60)) * math.log1p(min(long_v, 30)) / 10.0
    ink = (gray < 128).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    circles = 0
    for c in contours:
        a = cv2.contourArea(c)
        if 6 < a < 400:
            p = cv2.arcLength(c, True)
            if p > 0 and 4 * math.pi * a / (p * p) > 0.75:
                circles += 1
    f["small_circles"] = math.log1p(circles)
    small = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    local_std = cv2.blur(small**2, (5, 5)) - cv2.blur(small, (5, 5)) ** 2
    uniform = (np.sqrt(np.maximum(local_std, 0)).mean(axis=2) < 6) & (small.mean(axis=2) < 225)
    f["uniform_colour_frac"] = float(uniform.mean())
    f["aspect"] = float(max(w, h) / max(1, min(w, h)))
    f["portrait"] = float(h > w)
    return f


def text_features(
    words: list[Word], width_px: float, height_px: float, *, has_title_block: bool
) -> dict[str, float]:
    f: dict[str, float] = {}
    texts = [w.text for w in words]
    f["log_words"] = math.log1p(len(texts))
    f["numeric_frac"] = (
        sum(any(ch.isdigit() for ch in t) for t in texts) / len(texts) if texts else 0.0
    )
    f["dimension_count"] = math.log1p(
        sum(1 for t in texts if any(ch.isdigit() for ch in t) and parse_dimension(t) is not None)
    )
    f["tag_count"] = math.log1p(
        sum(1 for t in texts if normalise_tag(t) and t.strip()[:1].upper() in "KPDW")
    )
    f["avg_word_len"] = sum(len(t) for t in texts) / len(texts) / 10.0 if texts else 0.0
    # words that sit on long text lines (running text) vs short labels
    rows: dict[int, int] = {}
    for w in words:
        key = round((w.y0 + w.y1) / 2 / max(1.0, (w.y1 - w.y0)))
        rows[key] = rows.get(key, 0) + 1
    f["long_line_words"] = sum(n for n in rows.values() if n >= 8) / len(words) if words else 0.0
    f["has_title_block"] = float(has_title_block)
    folded = [fold(t) for t in texts]
    joined = " " + " ".join(folded) + " "
    for c in PAGE_CLASSES:
        hits = 0
        for kw in KEYWORDS.get(c, ()):
            hits += (
                joined.count(f" {kw} ")
                if " " in kw
                else sum(1 for t in folded if t == kw or t.startswith(kw))
            )
        f[f"kw_{c}"] = math.log1p(hits)
    return f


def meta_features(
    page_meta: dict[str, Any], doc_meta: dict[str, Any], vector: dict[str, Any] | None
) -> dict[str, float]:
    exif = doc_meta.get("exif") or {}
    return {
        "exif_camera": float(bool(exif.get("make") or exif.get("model"))),
        "is_vector": float(vector is not None and vector.get("objects", {}).get("path", 0) > 0),
        "log_paths": math.log1p(vector.get("objects", {}).get("path", 0)) if vector else 0.0,
        "image_area_frac": float(vector.get("image_area_fraction", 0.0))
        if vector
        else (0.0 if exif else 1.0),
    }


def vector_of(features: dict[str, float]) -> NDArray[np.float64]:
    return np.array([float(features.get(n, 0.0)) for n in FEATURE_NAMES], np.float64)
