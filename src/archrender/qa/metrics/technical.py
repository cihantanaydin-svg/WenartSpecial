"""Technical image checks: clipping, noise, sharpness relative to base, tile seams, colour shift."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.qa.images import Img, luminance


def clipping_fraction(img: Img, exclude: NDArray[np.bool_] | None = None) -> float:
    clipped = np.any((img <= 1.0 / 510.0) | (img >= 1.0 - 1.0 / 510.0), axis=-1)
    if exclude is not None:
        clipped &= ~exclude
        denom = max(1, int((~exclude).sum()))
    else:
        denom = clipped.size
    return float(clipped.sum() / denom)


def noise_sigma(img: Img) -> float:
    """Robust noise estimate: MAD of the Laplacian residual (Immerkær-style)."""
    lum = luminance(img).astype(np.float32)
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    res = cv2.filter2D(lum, -1, kernel)
    return float(np.median(np.abs(res)) / 0.6745 / 6.0)


def sharpness(img: Img) -> float:
    lum = luminance(img).astype(np.float32)
    return float(cv2.Laplacian(lum, cv2.CV_32F).var())


def mean_lab(img: Img, mask: NDArray[np.bool_]) -> NDArray[np.float64]:
    lab = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2Lab)
    if not mask.any():
        return np.zeros(3)
    return lab[mask].astype(np.float64).mean(axis=0)


def delta_e76(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(a - b))


def seam_score(img: Img, tile: int, overlap: int) -> float:
    """Mean gradient on tile borders divided by mean gradient elsewhere (1.0 ≈ seamless)."""
    lum = luminance(img)
    gx = np.abs(np.diff(lum, axis=1))
    _h, w = lum.shape
    step = tile - overlap
    if step <= 0 or w <= tile:
        return 1.0
    cols = list(range(step, w - 1, step))
    border = (
        np.concatenate([gx[:, c - 1 : c + 1].ravel() for c in cols]) if cols else np.array([0.0])
    )
    return float((border.mean() + 1e-6) / (gx.mean() + 1e-6))
