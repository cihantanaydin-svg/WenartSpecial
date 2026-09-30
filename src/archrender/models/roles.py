"""Typed interfaces per model role. Implementations: primary, fallback, mock (principle 5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from archrender.core.schemas.common import ModelRef

Img = NDArray[np.float32]  # (H, W, 3) RGB in [0, 1]


@runtime_checkable
class ModelImpl(Protocol):
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def ref(self) -> ModelRef: ...


@dataclass
class RefineRequest:
    base: Img
    strength_map: NDArray[np.float32]  # (H, W) per-pixel denoising strength in [0, 1]
    prompt: str
    seed: int
    depth_vis: Img | None = None  # in-context conditioning image (normalised depth)
    edges_vis: Img | None = None
    references: list[Img] = field(default_factory=list)
    steps: int = 30


class Refiner(ModelImpl, Protocol):
    def refine(self, req: RefineRequest) -> Img: ...


class DepthEstimator(ModelImpl, Protocol):
    def estimate(self, img: Img) -> NDArray[np.float32]:
        """Relative depth (larger = farther), same H×W as the input."""
        ...


@dataclass
class Instance:
    mask: NDArray[np.bool_]
    label: str
    score: float


class TextSegmenter(ModelImpl, Protocol):
    def segment(self, img: Img, prompts: list[str]) -> list[Instance]: ...
