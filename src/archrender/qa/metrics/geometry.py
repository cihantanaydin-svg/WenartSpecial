"""Deterministic geometry metrics (edges, verticals, depth alignment, opening IoU matching)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from archrender.qa.images import Img, luminance

QA_WIDTH = 2048
EDGE_TOLERANCE_PX = 3.0


def canny_edges(img: Img) -> NDArray[np.bool_]:
    """Deterministic multi-scale Canny on bilateral-filtered luminance (8-bit)."""
    lum = np.clip(luminance(img) * 255.0, 0, 255).astype(np.uint8)
    lum = cv2.bilateralFilter(lum, d=5, sigmaColor=25, sigmaSpace=5)
    med = float(np.median(lum))
    lo, hi = max(5.0, 0.66 * med * 0.5), max(15.0, 1.33 * med * 0.5)
    e1 = cv2.Canny(lum, lo, hi, L2gradient=True)
    blur = cv2.GaussianBlur(lum, (0, 0), 1.5)
    e2 = cv2.Canny(blur, lo * 0.8, hi * 0.8, L2gradient=True)
    return (e1 > 0) | (e2 > 0)


def edge_fscore(pred: NDArray[np.bool_], gt: NDArray[np.bool_], tol: float = EDGE_TOLERANCE_PX) -> tuple[float, float, float]:
    """Precision, recall, F with a distance tolerance (distance-transform approximation of BSDS)."""
    if not gt.any() or not pred.any():
        return 0.0, 0.0, 0.0
    dt_gt = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 5)
    dt_pred = cv2.distanceTransform((~pred).astype(np.uint8), cv2.DIST_L2, 5)
    precision = float((dt_gt[pred] <= tol).mean())
    recall = float((dt_pred[gt] <= tol).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f


def vertical_deviation_deg(img: Img, mask: NDArray[np.bool_] | None = None, max_tilt_deg: float = 10.0) -> tuple[float, int]:
    """Length-weighted median |angle from vertical| of near-vertical LSD segments."""
    lum = np.clip(luminance(img) * 255.0, 0, 255).astype(np.uint8)
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(lum)[0]
    if lines is None:
        return 0.0, 0
    devs: list[float] = []
    weights: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        if length < 20:
            continue
        tilt = math.degrees(math.atan2(abs(dx), abs(dy)))
        if tilt > max_tilt_deg:
            continue
        if mask is not None:
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            if not (0 <= my < mask.shape[0] and 0 <= mx < mask.shape[1] and mask[my, mx]):
                continue
        devs.append(tilt)
        weights.append(length)
    if not devs:
        return 0.0, 0
    order = np.argsort(devs)
    cum = np.cumsum(np.array(weights)[order])
    median = float(np.array(devs)[order][np.searchsorted(cum, cum[-1] / 2.0)])
    return median, len(devs)


def align_scale_shift(pred: NDArray[np.float32], gt: NDArray[np.float32], mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Least-squares ``s·pred + b`` fitted to ``gt`` on ``mask``."""
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    a = np.stack([p, np.ones_like(p)], axis=1)
    sol, *_ = np.linalg.lstsq(a, g, rcond=None)
    return (pred * sol[0] + sol[1]).astype(np.float32)


def abs_rel(pred: NDArray[np.float32], gt: NDArray[np.float32], mask: NDArray[np.bool_]) -> float:
    m = mask & np.isfinite(gt) & (gt > 1e-3)
    if not m.any():
        return 0.0
    return float(np.mean(np.abs(pred[m] - gt[m]) / gt[m]))


@dataclass(frozen=True)
class OpeningMatch:
    opening_id: str
    iou: float
    matched: bool


def match_openings(
    gt_masks: dict[str, NDArray[np.bool_]], pred_masks: list[NDArray[np.bool_]], min_iou: float = 0.3
) -> tuple[list[OpeningMatch], int]:
    """Hungarian matching of predicted instances to GT opening masks. Returns matches and #extra."""
    ids = list(gt_masks)
    if not ids:
        return [], len(pred_masks)
    iou = np.zeros((len(ids), len(pred_masks)))
    for i, oid in enumerate(ids):
        g = gt_masks[oid]
        for j, p in enumerate(pred_masks):
            inter = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            iou[i, j] = inter / union if union else 0.0
    matches: list[OpeningMatch] = []
    used: set[int] = set()
    if pred_masks:
        rows, cols = linear_sum_assignment(-iou)
        pairs = dict(zip(rows.tolist(), cols.tolist(), strict=True))
    else:
        pairs = {}
    for i, oid in enumerate(ids):
        j = pairs.get(i)
        if j is not None and iou[i, j] >= min_iou:
            matches.append(OpeningMatch(oid, float(iou[i, j]), True))
            used.add(j)
        else:
            matches.append(OpeningMatch(oid, float(iou[i, j]) if j is not None else 0.0, False))
    extra = len(pred_masks) - len(used)
    return matches, extra
