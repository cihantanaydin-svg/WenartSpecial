"""Raster variants of synthetic sheets (scans) and procedural photos, with ground truth carried over.

Scans: the vector page is rendered at the target DPI, then rotated slightly, blurred, noised,
optionally converted to greyscale and JPEG-compressed. Word boxes and the plan→page transform are
moved with the same affine transform, so the ground truth stays exact. Phone photos of a printed
sheet: the page is warped by a random perspective onto a table background with uneven light,
blur and JPEG, and the ground truth carries the homography. Photos are procedural interior
perspectives (not drawings), with camera EXIF like a phone picture.
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
    if "plan_to_page_px" in gt:  # plan metres → pixels of this raster
        a = np.vstack([np.array(gt["plan_to_page_px"], np.float64), [0, 0, 1]])
        out["plan_to_page_px"] = (full @ a)[:2].tolist()
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


PHONES = [("Apple", "iPhone 15"), ("samsung", "SM-S918B"), ("Xiaomi", "23078PND5G")]


def phone_photo(
    pdf: bytes,
    gt: dict[str, Any],
    rng: np.random.Generator,
    *,
    size: tuple[int, int] = (3264, 2448),
    crop_corner: bool | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """A phone picture of the printed sheet. Ground truth: ``sheet_corners_px`` (TL, TR, BR, BL of
    the paper in the photo), ``page_to_photo_h`` (page px at GT_DPI → photo px) and, for plans,
    ``plan_to_photo_h`` (plan metres → photo px). Word boxes are not carried over (OCR on photos
    is measured after rectification)."""
    w, h = size
    page = np.asarray(render_pdf(pdf, 200.0), np.float32)
    ph, pw = page.shape[:2]
    # the sheet fills 70–90 % of the frame, seen from a slightly tilted, rotated camera
    fill = float(rng.uniform(0.70, 0.90))
    aspect = pw / ph
    sw = min(w * fill, h * fill * aspect)
    sh_ = sw / aspect
    cx, cy = w / 2 + rng.uniform(-0.05, 0.05) * w, h / 2 + rng.uniform(-0.05, 0.05) * h
    rot = math.radians(float(rng.uniform(-8, 8)))
    base = np.array(
        [[-sw / 2, -sh_ / 2], [sw / 2, -sh_ / 2], [sw / 2, sh_ / 2], [-sw / 2, sh_ / 2]]
    )
    keystone = rng.uniform(0.0, 0.10)  # far edge shorter: camera tilted towards the top
    base[0, 0] *= 1 - keystone
    base[1, 0] *= 1 - keystone
    base += rng.normal(0, 0.012, base.shape) * np.array([sw, sh_])
    rm = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    corners = base @ rm.T + np.array([cx, cy])
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    hmat = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    # table: a warm, slightly textured surface
    table = np.ones((h, w, 3), np.float32) * rng.uniform(90, 160, 3)
    grain = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 25.0, sigmaY=2.0)
    table += grain[..., None]
    warped = cv2.warpPerspective(page, hmat, (w, h), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones((ph, pw), np.float32), hmat, (w, h), flags=cv2.INTER_LINEAR)
    img: NDArray[np.float32] = np.asarray(
        table * (1 - mask[..., None]) + warped * mask[..., None], np.float32
    )
    # uneven light: a broad gradient and a soft shadow
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    grad = 1.0 + 0.18 * ((xx / w - 0.5) * math.cos(ang) + (yy / h - 0.5) * math.sin(ang))
    img *= grad[..., None]
    sx, sy, sr = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.2, 0.5) * w
    shadow = 1.0 - 0.18 * np.exp(-(((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * sr * sr)))
    img *= shadow[..., None]
    img *= np.array([1.0, rng.uniform(0.96, 1.0), rng.uniform(0.88, 0.97)], np.float32)  # warm cast
    img = np.asarray(cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.6, 1.3))), np.float32)
    img += rng.normal(0, 3.5, img.shape).astype(np.float32)
    out = np.clip(img, 0, 255).astype(np.uint8)
    full_h = np.asarray(hmat, np.float64) @ np.diag([200.0 / GT_DPI, 200.0 / GT_DPI, 1.0])
    crop = crop_corner if crop_corner is not None else bool(rng.random() < 0.15)
    if crop:  # one paper corner outside the frame
        k = int(rng.integers(4))
        dx = int(max(0.0, corners[k, 0] + 60)) if corners[k, 0] < w / 2 else 0
        dy = int(max(0.0, corners[k, 1] + 60)) if corners[k, 1] < h / 2 else 0
        x1 = w if corners[k, 0] < w / 2 else int(min(w, corners[k, 0] - 60))
        y1 = h if corners[k, 1] < h / 2 else int(min(h, corners[k, 1] - 60))
        out = np.ascontiguousarray(out[dy:y1, dx:x1])
        shift = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], np.float64)
        full_h = shift @ full_h
        corners = corners - np.array([dx, dy])
    maker, model = PHONES[int(rng.integers(len(PHONES)))]
    exif = Image.Exif()
    exif[0x010F] = maker
    exif[0x0110] = model
    buf = io.BytesIO()
    Image.fromarray(out, "RGB").save(buf, "JPEG", quality=int(rng.integers(80, 93)), exif=exif)
    pt = {
        "class": gt.get("class"),
        "source": "phone_photo",
        "words": [],
        "gt_dpi": None,
        "sheet_corners_px": corners.round(2).tolist(),
        "corners_in_frame": not crop,
        "page_to_photo_h": full_h.tolist(),
        "page_mm": gt.get("page_mm"),
        "scale": gt.get("scale"),
    }
    if "plan_to_page_px" in gt:
        a = np.vstack([np.array(gt["plan_to_page_px"], np.float64), [0, 0, 1]])
        pt["plan_to_photo_h"] = (full_h @ a).tolist()
    return buf.getvalue(), pt
