"""Raster variants of synthetic sheets (scans) and procedural photos, with ground truth carried over.

Scans: the vector page is rendered at the target DPI, then rotated slightly, blurred, noised,
optionally converted to greyscale and JPEG-compressed. Word boxes are moved with the same affine
transform, so the ground truth stays exact. Photos are procedural interior perspectives (not
drawings), with camera EXIF like a phone picture.
"""

from __future__ import annotations

import io
import math
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium
from numpy.typing import NDArray
from PIL import Image

from archrender.synth.sheets import GT_DPI


def render_pdf(pdf: bytes, dpi: float) -> Image.Image:
    doc = pdfium.PdfDocument(pdf)
    try:
        img: Image.Image = doc[0].render(scale=dpi / 72.0).to_pil().convert("RGB")
        return img
    finally:
        doc.close()


def _transform_box(box: list[float], m: np.ndarray) -> list[float]:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], np.float64) @ m.T
    return [
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    ]


def transform_gt(gt: dict[str, Any], m: np.ndarray, dpi: float) -> dict[str, Any]:
    """Map every ``bbox`` of the ground truth from GT_DPI pixels through scale + affine ``m``."""
    s = dpi / GT_DPI
    full = m @ np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], np.float64)

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            out = {k: walk(x) for k, x in v.items()}
            if "bbox" in v and isinstance(v["bbox"], list) and len(v["bbox"]) == 4:
                out["bbox"] = _transform_box(v["bbox"], full)
            return out
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    out: dict[str, Any] = walk(gt)
    angle = math.degrees(math.atan2(-full[1, 0], full[0, 0]))
    for w in out.get("words", []):
        w["angle_deg"] = round((w.get("angle_deg", 0.0) + angle) % 360.0, 1)
    if "north" in out:
        out["north"]["angle_deg"] = round((out["north"]["angle_deg"] + angle) % 360.0, 2)
    out["gt_dpi"] = dpi
    out["skew_deg"] = round(angle, 3)
    if "scale_bar" in out:
        out["scale_bar"]["segment_px"] *= s
    return out


def scan(
    pdf: bytes,
    gt: dict[str, Any],
    rng: np.random.Generator,
    *,
    dpi: float | None = None,
    skew_deg: float | None = None,
    quality: str = "medium",
) -> tuple[bytes, str, dict[str, Any]]:
    """Scan-like raster of page 0. Returns (file bytes, media type, ground truth in its pixels)."""
    dpi = dpi or float(rng.choice([200.0, 300.0]))
    img: NDArray[Any] = np.asarray(render_pdf(pdf, dpi), dtype=np.float32)
    h, w = img.shape[:2]
    skew = float(rng.uniform(-1.5, 1.5)) if skew_deg is None else skew_deg
    m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    noise = {"clean": 2.0, "medium": 6.0, "noisy": 14.0}[quality]
    blur = {"clean": 0.0, "medium": 0.6, "noisy": 1.1}[quality]
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    paper = np.array(
        [rng.uniform(235, 250), rng.uniform(232, 248), rng.uniform(220, 245)], np.float32
    )
    img = img * (paper / 255.0)
    img = img + rng.normal(0.0, noise, img.shape[:2]).astype(np.float32)[..., None]
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img8, "RGB")
    if rng.random() < 0.5:
        pil = pil.convert("L")
    buf = io.BytesIO()
    if rng.random() < 0.5:
        pil.save(buf, "JPEG", quality=int(rng.integers(55, 92)), dpi=(dpi, dpi))
        media = "image/jpeg"
    else:
        pil.save(buf, "PNG", dpi=(round(dpi), round(dpi)))
        media = "image/png"
    m3 = np.vstack([m, [0, 0, 1]])
    return buf.getvalue(), media, transform_gt(gt, m3, dpi)


def _shade(img: np.ndarray, poly: np.ndarray, color: np.ndarray, rng: np.random.Generator) -> None:
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    gx = (xs - xs.min()) / max(1, int(xs.max() - xs.min()))
    gy = (ys - ys.min()) / max(1, int(ys.max() - ys.min()))
    k = 1.0 + (gx * rng.uniform(-0.25, 0.25) + gy * rng.uniform(-0.3, 0.1))
    img[ys, xs] = np.clip(color[None, :] * k[:, None], 0, 255)


def photo(
    rng: np.random.Generator, size: tuple[int, int] = (1600, 1200)
) -> tuple[bytes, dict[str, Any]]:
    """Procedural interior photo (one-point perspective room, window, furniture blocks) as JPEG."""
    w, h = size
    img: NDArray[Any] = np.zeros((h, w, 3), np.float32)
    vx, vy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.4, 0.55)
    bw, bh = w * rng.uniform(0.3, 0.5), h * rng.uniform(0.3, 0.45)
    back = np.array(
        [
            [vx - bw / 2, vy - bh / 2],
            [vx + bw / 2, vy - bh / 2],
            [vx + bw / 2, vy + bh / 2],
            [vx - bw / 2, vy + bh / 2],
        ]
    )
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    wall = rng.uniform(150, 235, 3)
    floor = rng.uniform(60, 170, 3) * np.array([1.0, 0.8, 0.6])
    ceil = np.full(3, rng.uniform(215, 250))
    _shade(img, np.array([corners[0], corners[1], back[1], back[0]]), ceil, rng)
    _shade(img, np.array([back[3], back[2], corners[2], corners[3]]), floor, rng)
    _shade(img, np.array([corners[0], back[0], back[3], corners[3]]), wall * 0.85, rng)
    _shade(img, np.array([back[1], corners[1], corners[2], back[2]]), wall * 0.9, rng)
    _shade(img, back, wall, rng)
    # window on the back wall (bright)
    wx0, wy0 = vx - bw * rng.uniform(0.1, 0.35), vy - bh * 0.3
    wx1, wy1 = wx0 + bw * rng.uniform(0.25, 0.4), vy + bh * 0.15
    img[int(wy0) : int(wy1), int(wx0) : int(wx1)] = np.array([235, 242, 250]) * rng.uniform(
        0.95, 1.05
    )
    for _ in range(int(rng.integers(2, 5))):  # furniture blocks on the floor
        fx = rng.uniform(0.1, 0.8) * w
        fy = rng.uniform(vy + bh / 2, h * 0.9)
        fw, fh = rng.uniform(0.1, 0.3) * w, rng.uniform(0.05, 0.2) * h
        col = rng.uniform(30, 200, 3)
        img[int(fy - fh) : int(fy), int(fx) : int(min(w, fx + fw))] = col
        img[int(fy - fh - fh * 0.2) : int(fy - fh), int(fx) : int(min(w, fx + fw))] = col * 1.2
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")
    exif = Image.Exif()
    exif[0x010F] = str(rng.choice(["Apple", "samsung", "Xiaomi", "Canon"]))
    exif[0x0110] = str(rng.choice(["iPhone 15", "SM-S918B", "23078PND5G", "EOS R6"]))
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=int(rng.integers(75, 95)), exif=exif)
    return buf.getvalue(), {"class": "photo", "words": [], "gt_dpi": None}
