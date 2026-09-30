"""S9 QA: evaluate a candidate against the unrefined base render (ADR-S06)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel

from archrender.core.schemas.qa import CheckResult
from archrender.core.schemas.scene import STRUCTURAL_CATEGORIES, SceneSpec
from archrender.models.roles import DepthEstimator, TextSegmenter
from archrender.qa.images import Img, resize_to_width
from archrender.qa.metrics import geometry as g
from archrender.qa.metrics import technical as t
from archrender.render.passes import (
    RenderPasses,
    category_mask,
    line_art,
    object_masks,
    resize_mask_to_width,
)

OPENING_PROMPTS = ["window", "door", "glass door"]


class GeometryThresholds(BaseModel):
    edge_f_drop_max: float
    depth_absrel_rise_max: float
    opening_iou_drop_max: float
    verticals_max_deg: float


class TechnicalThresholds(BaseModel):
    clipping_rise_max: float
    base_clipping_review: float
    sharpness_ratio_min: float
    noise_rise_max: float
    material_delta_e_max: float


class QAConfig(BaseModel):
    geometry: GeometryThresholds
    technical: TechnicalThresholds
    required_families: list[str]

    @classmethod
    def load(cls, configs_dir: Path) -> QAConfig:
        return cls.model_validate(yaml.safe_load((configs_dir / "qa.yaml").read_text(encoding="utf-8")))


@dataclass
class BaseMeasurements:
    """Metrics of the base render, computed once per view and reused for every candidate."""

    edge_f: float
    absrel: float
    opening_iou: dict[str, float]
    opening_extra: int
    noise: float
    sharpness: float
    clipping: float
    material_lab: dict[int, NDArray[np.float64]]


@dataclass
class QAContext:
    passes: RenderPasses
    spec: SceneSpec
    qa_width: int
    config: QAConfig
    depth: DepthEstimator
    segmenter: TextSegmenter

    def __post_init__(self) -> None:
        h, w = self.passes.shape
        self.gt_edges = resize_mask_to_width(line_art(self.passes, self.spec), self.qa_width)
        self.struct = resize_mask_to_width(
            category_mask(self.passes, self.spec, set(STRUCTURAL_CATEGORIES)), self.qa_width
        )
        self.glass = resize_mask_to_width(category_mask(self.passes, self.spec, {"glass"}), self.qa_width)
        opening_ids = {o.element_ref for o in self.spec.objects if o.category in ("glass", "door_leaf") and o.element_ref}
        self.opening_masks = {
            k: resize_mask_to_width(v, self.qa_width)
            for k, v in object_masks(self.passes, self.spec, opening_ids).items()
        }
        gt_depth = self.passes.depth
        self.gt_depth = _resize_depth(gt_depth, self.gt_edges.shape)
        self.depth_mask = self.struct & np.isfinite(self.gt_depth) & ~self.glass
        self.materials = {
            m.pass_index: resize_mask_to_width(self.passes.material_index == m.pass_index, self.qa_width)
            for m in self.spec.materials
        }

    def _depth_absrel(self, img: Img) -> float:
        pred = self.depth.estimate(img)
        pred = _resize_depth(pred, self.gt_depth.shape)
        m = self.depth_mask & np.isfinite(pred)
        if m.sum() < 16:
            return 0.0
        aligned = g.align_scale_shift(pred, self.gt_depth, m)
        return g.abs_rel(aligned, self.gt_depth, m)

    def _openings(self, img: Img) -> tuple[dict[str, float], int]:
        inst = self.segmenter.segment(img, OPENING_PROMPTS)
        preds = [
            resize_mask_to_width(i.mask, self.qa_width) if i.mask.shape[1] != self.qa_width else i.mask
            for i in inst
        ]
        matches, extra = g.match_openings(self.opening_masks, preds)
        return {m.opening_id: (m.iou if m.matched else 0.0) for m in matches}, extra

    def measure_base(self, base_full: Img) -> BaseMeasurements:
        base = resize_to_width(base_full, self.qa_width)
        _, _, f = g.edge_fscore(g.canny_edges(base), self.gt_edges)
        iou, extra = self._openings(base)
        return BaseMeasurements(
            edge_f=f,
            absrel=self._depth_absrel(base),
            opening_iou=iou,
            opening_extra=extra,
            noise=t.noise_sigma(base),
            clipping=t.clipping_fraction(base, exclude=self.glass),
            sharpness=t.sharpness(base),
            material_lab={k: t.mean_lab(base, m) for k, m in self.materials.items() if m.any()},
        )

    def evaluate(self, cand_full: Img, base_m: BaseMeasurements) -> list[CheckResult]:
        c = self.config
        cand = resize_to_width(cand_full, self.qa_width)
        depth_ref = self.depth.ref()
        seg_ref = self.segmenter.ref()
        out: list[CheckResult] = []

        prec, rec, f = g.edge_fscore(g.canny_edges(cand), self.gt_edges)
        drop = base_m.edge_f - f
        out.append(
            CheckResult(
                name="structural_edge_f",
                family="geometry",
                passed=drop <= c.geometry.edge_f_drop_max,
                value=f,
                base_value=base_m.edge_f,
                delta=drop,
                threshold=c.geometry.edge_f_drop_max,
                estimator="deterministic",
                critical=True,
                evidence={"precision": prec, "recall": rec, "tolerance_px": g.EDGE_TOLERANCE_PX},
            )
        )
        absrel = self._depth_absrel(cand)
        rise = absrel - base_m.absrel
        out.append(
            CheckResult(
                name="depth_drift",
                family="geometry",
                passed=rise <= c.geometry.depth_absrel_rise_max,
                value=absrel,
                base_value=base_m.absrel,
                delta=rise,
                threshold=c.geometry.depth_absrel_rise_max,
                estimator=depth_ref.name,
                estimator_model=depth_ref,
                mock=depth_ref.mock,
                critical=True,
            )
        )
        iou, extra = self._openings(cand)
        per: dict[str, dict[str, float]] = {}
        worst = 0.0
        new_missing = 0
        for oid, base_iou in base_m.opening_iou.items():
            ci = iou.get(oid, 0.0)
            per[oid] = {"base": base_iou, "refined": ci}
            worst = max(worst, base_iou - ci)
            if base_iou > 0 and ci == 0.0:
                new_missing += 1
        new_extra = max(0, extra - base_m.opening_extra)
        out.append(
            CheckResult(
                name="openings",
                family="geometry",
                passed=worst <= c.geometry.opening_iou_drop_max and new_missing == 0 and new_extra == 0,
                value=float(new_missing + new_extra),
                delta=worst,
                threshold=c.geometry.opening_iou_drop_max,
                estimator=seg_ref.name,
                estimator_model=seg_ref,
                mock=seg_ref.mock,
                critical=True,
                evidence={"per_opening": per, "new_missing": new_missing, "new_extra": new_extra},
            )
        )
        vdev, nlines = g.vertical_deviation_deg(cand, self.struct)
        out.append(
            CheckResult(
                name="verticals",
                family="geometry",
                passed=vdev <= c.geometry.verticals_max_deg,
                value=vdev,
                threshold=c.geometry.verticals_max_deg,
                estimator="deterministic",
                evidence={"lines": nlines},
            )
        )
        clip = t.clipping_fraction(cand, exclude=self.glass)
        out.append(
            CheckResult(
                name="clipping",
                family="technical",
                passed=clip - base_m.clipping <= c.technical.clipping_rise_max,
                value=clip,
                base_value=base_m.clipping,
                delta=clip - base_m.clipping,
                threshold=c.technical.clipping_rise_max,
                estimator="deterministic",
            )
        )
        sharp = t.sharpness(cand) / max(base_m.sharpness, 1e-9)
        out.append(
            CheckResult(
                name="sharpness_ratio",
                family="technical",
                passed=sharp >= c.technical.sharpness_ratio_min,
                value=sharp,
                threshold=c.technical.sharpness_ratio_min,
                comparator=">=",
                estimator="deterministic",
            )
        )
        noise_rise = t.noise_sigma(cand) - base_m.noise
        out.append(
            CheckResult(
                name="noise",
                family="technical",
                passed=noise_rise <= c.technical.noise_rise_max,
                value=noise_rise,
                threshold=c.technical.noise_rise_max,
                estimator="deterministic",
            )
        )
        worst_de, worst_mat = 0.0, None
        for idx, lab in base_m.material_lab.items():
            de = t.delta_e76(t.mean_lab(cand, self.materials[idx]), lab)
            if de > worst_de:
                worst_de, worst_mat = de, idx
        out.append(
            CheckResult(
                name="material_color_shift",
                family="technical",
                passed=worst_de <= c.technical.material_delta_e_max,
                value=worst_de,
                threshold=c.technical.material_delta_e_max,
                estimator="deterministic",
                evidence={"material_pass_index": worst_mat},
            )
        )
        return out


def _resize_depth(depth: NDArray[np.float32], shape: tuple[int, ...]) -> NDArray[np.float32]:
    import cv2

    h, w = shape[:2]
    if depth.shape[:2] == (h, w):
        return depth.astype(np.float32)
    finite = np.isfinite(depth)
    filled = np.where(finite, depth, 0.0).astype(np.float32)
    out = cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)
    fin = cv2.resize(finite.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    out[~fin] = np.inf
    return out.astype(np.float32)


def families_covered(checks: list[CheckResult]) -> set[str]:
    return {c.family for c in checks if not c.mock}
