"""Refinement orchestration, independent of the refiner model.

The global pass runs at the profile's working resolution. When the deliverable is larger, the
global result is upscaled and a tiled low-strength pass runs at full resolution, with feathered
(cosine) tile blending and one seed per tile derived from the candidate seed.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.models.profiles import RefineProfile
from archrender.models.roles import Refiner, RefineRequest
from archrender.refine.strength import resize_map

Img = NDArray[np.float32]


def working_size(width: int, height: int, megapixels: float) -> tuple[int, int]:
    target = megapixels * 1e6
    if width * height <= target:
        return width, height
    k = math.sqrt(target / (width * height))
    return max(16, int(width * k) // 16 * 16), max(16, int(height * k) // 16 * 16)


def _cosine_window(h: int, w: int, overlap: int) -> NDArray[np.float32]:
    def ramp(n: int) -> NDArray[np.float32]:
        r = np.ones(n, dtype=np.float32)
        o = min(overlap, n // 2)
        if o > 0:
            t = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, o, dtype=np.float32))
            r[:o] = np.maximum(t, 1e-3)
            r[-o:] = np.maximum(t[::-1], 1e-3)
        return r

    return np.outer(ramp(h), ramp(w)).astype(np.float32)


def tile_grid(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, size - tile, step)) + [size - tile]
    return sorted(set(starts))


def refine_image(
    refiner: Refiner,
    base: Img,
    strength: NDArray[np.float32],
    prompt: str,
    seed: int,
    profile: RefineProfile,
    *,
    depth_vis: Img | None = None,
    edges_vis: Img | None = None,
    references: list[Img] | None = None,
    tile_strength_scale: float = 0.6,
) -> Img:
    h, w = base.shape[:2]
    gw, gh = working_size(w, h, profile.global_megapixels)
    small = cv2.resize(base, (gw, gh), interpolation=cv2.INTER_AREA) if (gw, gh) != (w, h) else base
    req = RefineRequest(
        base=small,
        strength_map=resize_map(strength, (gh, gw)),
        prompt=prompt,
        seed=seed,
        depth_vis=None if depth_vis is None else cv2.resize(depth_vis, (gw, gh)),
        edges_vis=None if edges_vis is None else cv2.resize(edges_vis, (gw, gh)),
        references=list(references or []),
    )
    global_out = refiner.refine(req)
    if (gw, gh) == (w, h):
        return global_out
    # Upscaled global result carries the refined look. The Cycles render carries fine geometry, so
    # the tile pass starts from a blend weighted by strength: structure stays Cycles-sharp.
    up = cv2.resize(global_out, (w, h), interpolation=cv2.INTER_CUBIC)
    s3 = strength[..., None]
    start = (s3 * up + (1.0 - s3) * base).astype(np.float32)
    acc = np.zeros_like(base)
    weight = np.zeros(base.shape[:2], dtype=np.float32)
    t, ov = profile.tile_px, profile.tile_overlap_px
    for ti, y in enumerate(tile_grid(h, t, ov)):
        for tj, x in enumerate(tile_grid(w, t, ov)):
            th, tw = min(t, h - y), min(t, w - x)
            tile_req = RefineRequest(
                base=start[y : y + th, x : x + tw],
                strength_map=strength[y : y + th, x : x + tw] * tile_strength_scale,
                prompt=prompt,
                seed=seed * 1000 + ti * 97 + tj,
                references=list(references or []),
            )
            out = refiner.refine(tile_req)
            win = _cosine_window(th, tw, ov)
            acc[y : y + th, x : x + tw] += out * win[..., None]
            weight[y : y + th, x : x + tw] += win
    return (acc / np.maximum(weight, 1e-6)[..., None]).astype(np.float32)
