"""Phone photos of a drawing sheet → the sheet, rectified (ARCHITECTURE §S1, raster inputs).

The paper is the largest bright, unsaturated region that does not fill the whole frame (a scan
fills it; a photo shows the table around it). Its four sides are fitted as lines to the region's
outline (points on the image border are not sides, so a corner cut off by the frame is still found
as the intersection of its two sides); the homography maps the quadrilateral to an upright
rectangle. The sheet's aspect ratio is snapped to ISO 216 (√2) when the measured ratio is within
12 % of it (an assumption, recorded); the rectified page keeps the photo's resolution along the
sheet's longest side. The physical sheet size, hence the DPI, is unknown: S2 takes the scale from
dimension strings, not from the stated scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

ISO_ASPECT = math.sqrt(2.0)
WORK_PX = 1000


@dataclass
class SheetQuad:
    corners: NDArray[np.float64]  # TL, TR, BR, BL in photo pixels (may lie outside the frame)
    coverage: float  # quad area / frame area
    corners_in_frame: bool
    fit_rms_px: float  # residual of the side lines (photo px)


@dataclass
class Rectified:
    image: NDArray[np.uint8]
    h: NDArray[np.float64]  # photo px → rectified px
    quad: SheetQuad
    aspect: float
    aspect_source: str  # "iso216" | "measured"


def _order(c: NDArray[np.float64]) -> NDArray[np.float64]:
    """TL, TR, BR, BL (image coordinates, y down)."""
    centre = c.mean(axis=0)
    ang = np.arctan2(c[:, 1] - centre[1], c[:, 0] - centre[0])
    c = c[np.argsort(ang)]  # clockwise on screen starting at the left-most angle
    k = int(np.argmin(c.sum(axis=1)))  # top-left: smallest x + y
    return np.roll(c, -k, axis=0)


def _line(pts: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    c = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    return c, evecs[:, 1], math.sqrt(max(float(evals[0]), 0.0))


def _intersect(
    a: tuple[NDArray[np.float64], NDArray[np.float64], float],
    b: tuple[NDArray[np.float64], NDArray[np.float64], float],
) -> NDArray[np.float64] | None:
    (ca, ua, _), (cb, ub, _) = a, b
    cross = float(ua[0] * ub[1] - ua[1] * ub[0])
    if abs(cross) < 1e-6:
        return None
    t = float(((cb - ca)[0] * ub[1] - (cb - ca)[1] * ub[0]) / cross)
    return ca + ua * t


def find_sheet(rgb: NDArray[np.uint8]) -> SheetQuad | None:
    """The paper sheet in a photo, or None (no sheet, or the sheet fills the frame: a scan)."""
    h, w = rgb.shape[:2]
    s = WORK_PX / max(h, w)
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[..., 2] / 255.0
    sat = hsv[..., 1] / 255.0
    # brightness relative to a broad local level (uneven light), minus saturation (the table)
    paper = cv2.GaussianBlur(v, (0, 0), 3.0) - 0.8 * cv2.GaussianBlur(sat, (0, 0), 3.0)
    lo, hi = float(paper.min()), float(paper.max())
    score = np.asarray((paper - lo) / max(hi - lo, 1e-6) * 255.0, np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(0.02 * max(sh, sw)) | 1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (lab == best).astype(np.uint8)
    frac = float(stats[best, cv2.CC_STAT_AREA]) / (sh * sw)
    if not 0.15 <= frac <= 0.97:
        return None  # nothing sheet-like, or paper everywhere (a scan)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps in np.linspace(0.01, 0.08, 15):
        a = cv2.approxPolyDP(hull, eps * peri, True).reshape(-1, 2).astype(np.float64)
        if len(a) <= 6:
            approx = a
            break
    if approx is None or len(approx) < 4:
        return None
    # sides: the polygon edges not lying on the frame border, the four longest
    m = 6.0
    edges = []
    for i in range(len(approx)):
        p, q = approx[i], approx[(i + 1) % len(approx)]
        on_border = (
            (p[0] <= m and q[0] <= m)
            or (p[1] <= m and q[1] <= m)
            or (p[0] >= sw - 1 - m and q[0] >= sw - 1 - m)
            or (p[1] >= sh - 1 - m and q[1] >= sh - 1 - m)
        )
        if not on_border:
            edges.append((float(np.hypot(*(q - p))), i, p, q))
    if len(edges) < 4:
        return None
    edges = sorted(sorted(edges, key=lambda e: -e[0])[:4], key=lambda e: e[1])
    pts = cnt.reshape(-1, 2).astype(np.float64)
    inner = (pts[:, 0] > m) & (pts[:, 1] > m) & (pts[:, 0] < sw - 1 - m) & (pts[:, 1] < sh - 1 - m)
    lines = []
    rms = []
    for _, _, p, q in edges:
        d = q - p
        length = float(np.hypot(*d))
        u = d / length
        nrm = np.array([-u[1], u[0]])
        along = (pts - p) @ u
        off = np.abs((pts - p) @ nrm)
        sel = inner & (off < 0.015 * peri) & (along > 0.1 * length) & (along < 0.9 * length)
        if sel.sum() < 10:
            return None
        ln = _line(pts[sel])
        lines.append(ln)
        rms.append(ln[2])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    centre = pts.mean(axis=0) / s
    lines = [_refine(gray, (c / s, u, r / s), centre) or (c / s, u, r / s) for c, u, r in lines]
    rms = [ln[2] for ln in lines]
    corners = []
    for i in range(4):
        c = _intersect(lines[i - 1], lines[i])
        if c is None:
            return None
        corners.append(c)
    quad = _order(np.array(corners))
    if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)):
        return None
    area = float(cv2.contourArea(quad.astype(np.float32)))
    inside = bool(
        (
            (quad[:, 0] >= -2) & (quad[:, 0] < w + 2) & (quad[:, 1] >= -2) & (quad[:, 1] < h + 2)
        ).all()
    )
    return SheetQuad(quad, area / (w * h), inside, float(np.mean(rms)))


def _refine(
    gray: NDArray[np.float32],
    line: tuple[NDArray[np.float64], NDArray[np.float64], float],
    centre: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
    """Full-resolution side line: along normals to the coarse line, the strongest step from the
    paper (inside, bright) to the background; two rounds of outlier rejection."""
    h, w = gray.shape
    c, u, _ = line
    nrm = np.array([-u[1], u[0]])
    if float((centre - c) @ nrm) < 0:
        nrm = -nrm  # points into the sheet
    span = 0.004 * max(h, w) + 6.0
    ts = np.arange(-span, span + 0.5, 0.5)
    # the side's extent: where the coarse line crosses the frame
    lo, hi = -1e9, 1e9
    for k, size in ((0, w), (1, h)):
        if abs(u[k]) > 1e-9:
            a, b = (0 - c[k]) / u[k], (size - 1 - c[k]) / u[k]
            lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    edge = []
    for f in np.linspace(lo, hi, 80)[5:-5]:
        p = c + u * f
        q = p[None, :] + nrm[None, :] * ts[:, None]
        ok = (q[:, 0] >= 0) & (q[:, 0] < w - 1) & (q[:, 1] >= 0) & (q[:, 1] < h - 1)
        if ok.mean() < 0.9:
            continue
        prof = cv2.remap(
            gray,
            q[:, 0].astype(np.float32).reshape(1, -1),
            q[:, 1].astype(np.float32).reshape(1, -1),
            cv2.INTER_LINEAR,
        ).ravel()
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        grad = np.gradient(prof)  # bright inside → positive step along nrm
        k = int(np.argmax(grad[2:-2])) + 2
        if grad[k] <= 5.0:
            continue
        # parabolic sub-sample peak
        y0, y1, y2 = grad[k - 1], grad[k], grad[k + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        edge.append(p + nrm * (ts[k] + off * 0.5))
    if len(edge) < 12:
        return None
    e = np.array(edge)
    for _ in range(2):
        lc, lu, _ = _line(e)
        ln = np.array([-lu[1], lu[0]])
        d = np.abs((e - lc) @ ln)
        keep = d <= max(1.5, 3 * float(np.median(d)))
        if keep.sum() < 10:
            break
        e = e[keep]
    return _line(e)


def rectify(rgb: NDArray[np.uint8], quad: SheetQuad) -> Rectified:
    c = quad.corners
    top, bottom = np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[3]))
    left, right = np.hypot(*(c[3] - c[0])), np.hypot(*(c[2] - c[1]))
    wide, tall = (top + bottom) / 2, (left + right) / 2
    measured = max(wide, tall) / max(min(wide, tall), 1.0)
    if abs(measured - ISO_ASPECT) / ISO_ASPECT <= 0.12:
        aspect, source = ISO_ASPECT, "iso216"
    else:
        aspect, source = measured, "measured"
    long_px = float(max(top, bottom, left, right))
    if wide >= tall:
        out_w, out_h = long_px, long_px / aspect
    else:
        out_w, out_h = long_px / aspect, long_px
    out_wi, out_hi = round(out_w), round(out_h)
    dst = np.array([[0, 0], [out_wi, 0], [out_wi, out_hi], [0, out_hi]], np.float32)
    hm = cv2.getPerspectiveTransform(c.astype(np.float32), dst).astype(np.float64)
    img = cv2.warpPerspective(
        rgb,
        hm,
        (out_wi, out_hi),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return Rectified(np.asarray(img, np.uint8), hm, quad, aspect, source)
