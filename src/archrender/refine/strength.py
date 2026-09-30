"""Per-pixel strength maps (ADR-S07) and the hard structural composite escalation rung."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.scene import STRUCTURAL_CATEGORIES, SceneSpec
from archrender.models.profiles import RefineProfile
from archrender.render.passes import RenderPasses, category_mask

FloatMap = NDArray[np.float32]


def strength_map(
    passes: RenderPasses,
    spec: SceneSpec,
    profile: RefineProfile,
    *,
    scale: float = 1.0,
    decor_enabled: bool = False,
) -> FloatMap:
    """Structural low, furniture moderate, decor-allowed higher (only if optional decor is on).

    Background (sky/window view) gets the structural strength: the view through a window is
    structural context, not something to reinvent.
    """
    s = np.full(passes.shape, profile.strength_structural, dtype=np.float32)
    furniture = category_mask(passes, spec, {"furniture"})
    s[furniture] = profile.strength_furniture
    if decor_enabled:
        decor = category_mask(passes, spec, {"decor"})
        s[decor] = profile.strength_decor
    # soften transitions by a few pixels so latent blending has no hard steps
    blurred = cv2.GaussianBlur(s, (0, 0), max(1.0, passes.shape[1] / 1024))
    return np.asarray(np.clip(blurred * scale, 0.0, 1.0), dtype=np.float32)


def hard_structural_composite(
    base: NDArray[np.float32], candidate: NDArray[np.float32], passes: RenderPasses, spec: SceneSpec
) -> NDArray[np.float32]:
    """Keep Cycles pixels inside structural masks; feathered blend at the borders."""
    mask = category_mask(passes, spec, set(STRUCTURAL_CATEGORIES)).astype(np.float32)
    radius = max(1.0, passes.shape[1] / 800)
    alpha = cv2.GaussianBlur(mask, (0, 0), radius)[..., None]
    return (alpha * base + (1.0 - alpha) * candidate).astype(np.float32)


def resize_map(m: FloatMap, shape: tuple[int, int]) -> FloatMap:
    if m.shape == shape:
        return m
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
