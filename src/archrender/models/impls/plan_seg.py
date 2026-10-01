"""The firm-trained plan segmentation network (``plan_segmenter`` role, ADR-M06).

A TorchScript model written by ``archrender.plan.seg_train`` (RGB tile in [0, 1] → class logits,
classes in ``plan_seg.json``), installed like other weights under
``<data_dir>/models/snapshots/<name>/<revision>/``. Inference is tiled at full resolution with
overlap; the wall class becomes the wall body of the raster extractor (``raster_seg``).

UNVERIFIED-ON-GPU: no checkpoint has been trained yet; the role is mapped in no profile until one
passes the promotion rule (``make eval``).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

from archrender.core.config import get_settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef
from archrender.models.registry import RegistryEntry


class TorchPlanSegmenter:
    def __init__(self, entry: RegistryEntry) -> None:
        self.entry = entry
        self.model: Any = None
        self.tile = 512
        self.classes: list[str] = []
        self.device = "cpu"

    def load(self) -> None:
        import torch

        snap = (
            get_settings().data_dir
            / "models"
            / "snapshots"
            / self.entry.name
            / str(self.entry.revision or "local")
        )
        weights, info = snap / "plan_seg.pt", snap / "plan_seg.json"
        if not weights.exists() or not info.exists():
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                f"Plan segmentation weights are not installed at {snap}.",
                "Train with `python -m archrender.plan.seg_train` on the pod and install the "
                "output there (see PROGRESS.md, plan segmentation).",
            )
        meta = json.loads(info.read_text(encoding="utf-8"))
        self.tile = int(meta["tile"])
        self.classes = list(meta["classes"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(str(weights), map_location=self.device).eval()

    def unload(self) -> None:
        self.model = None

    def ref(self) -> ModelRef:
        return self.entry.ref()

    def wall_mask(self, rgb: NDArray[np.uint8]) -> NDArray[np.bool_]:
        """Wall pixels of a page (tiles with a quarter-tile overlap; centres win)."""
        import torch

        if self.model is None:
            raise ArchRenderError(
                ErrorCode.INTERNAL, "The plan segmenter is not loaded.", "Call load() first."
            )
        t, step = self.tile, self.tile // 2
        h, w = rgb.shape[:2]
        logits = np.zeros((len(self.classes), h, w), np.float32)
        weight = np.zeros((h, w), np.float32)
        ramp = np.minimum(np.arange(t) + 1, np.arange(t)[::-1] + 1).astype(np.float32)
        win = np.minimum.outer(ramp, ramp)
        with torch.no_grad():
            for y in range(0, max(1, h - t + step), step):
                for x in range(0, max(1, w - t + step), step):
                    y0, x0 = min(y, max(0, h - t)), min(x, max(0, w - t))
                    crop = np.zeros((t, t, 3), np.uint8)
                    part = rgb[y0 : y0 + t, x0 : x0 + t]
                    crop[: part.shape[0], : part.shape[1]] = part
                    xb = torch.from_numpy(crop).permute(2, 0, 1)[None].float().div(255)
                    out = self.model(xb.to(self.device))[0].float().cpu().numpy()
                    ph, pw = part.shape[:2]
                    logits[:, y0 : y0 + ph, x0 : x0 + pw] += out[:, :ph, :pw] * win[:ph, :pw]
                    weight[y0 : y0 + ph, x0 : x0 + pw] += win[:ph, :pw]
        cls = (logits / np.maximum(weight, 1e-6)).argmax(axis=0)
        mask: NDArray[np.bool_] = cls == self.classes.index("wall")
        return mask
