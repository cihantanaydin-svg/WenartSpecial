"""Deterministic mock implementations for CPU tests (principle 5).

Mocks are real, deterministic computations, so the pipeline and QA plumbing are exercised
end-to-end. Every result they produce is flagged ``mock`` in QA records and reports, so a mock
result can never be presented as a real model output.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.common import ModelRef
from archrender.models.registry import RegistryEntry
from archrender.models.roles import Img, Instance, RefineRequest
from archrender.qa.images import luminance


class _MockBase:
    def __init__(self, entry: RegistryEntry) -> None:
        self.entry = entry
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def ref(self) -> ModelRef:
        return self.entry.ref()


class MockRefiner(_MockBase):
    """Geometry-preserving 'refinement': local contrast, gentle warm grade, seeded fine grain,
    all scaled by the per-pixel strength map. Edges and depth structure are unchanged."""

    def refine(self, req: RefineRequest) -> Img:
        base = req.base.astype(np.float32)
        s = np.clip(req.strength_map, 0.0, 1.0)[..., None].astype(np.float32)
        blur = cv2.GaussianBlur(base, (0, 0), 1.2)
        detail = base + 0.35 * s * (base - blur)
        warm = detail * (1.0 + s * np.array([0.02, 0.0, -0.02], dtype=np.float32))
        rng = np.random.default_rng(req.seed)
        grain = rng.normal(0.0, 0.004, size=base.shape[:2]).astype(np.float32)[..., None] * s
        return np.clip(warm + grain, 0.0, 1.0).astype(np.float32)


class MockDepthEstimator(_MockBase):
    """Pseudo-depth from smoothed inverse luminance. Not a depth model: QA checks computed with it
    are flagged mock and only exercise the relative-metric plumbing."""

    def estimate(self, img: Img) -> NDArray[np.float32]:
        lum = luminance(img)
        smooth = cv2.GaussianBlur(lum, (0, 0), max(1.0, img.shape[1] / 512))
        return (1.0 - smooth).astype(np.float32)


class MockTextSegmenter(_MockBase):
    """Returns no instances. Opening checks run in relative mode (same count on base and refined)
    and are flagged mock."""

    def segment(self, img: Img, prompts: list[str]) -> list[Instance]:
        return []
