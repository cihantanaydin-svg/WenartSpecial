"""Read Blender's multilayer EXR, derive structural masks and deterministic line art (ADR-S09).

Depth is Cycles' Z pass = planar depth along the optical axis (verified against the analytic camera
model in tests). Background pixels (sky, depth ≥ 1e9) get ``inf``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import OpenEXR
from numpy.typing import NDArray

from archrender.core.schemas.scene import STRUCTURAL_CATEGORIES, SceneSpec
from archrender.scene.meshio import npz_bytes

BACKGROUND_DEPTH = 1e9
NORMAL_CREASE_DEG = 30.0
DEPTH_JUMP_REL = 0.03


@dataclass
class RenderPasses:
    depth: NDArray[np.float32]  # (H, W) planar depth, inf = background
    normal: NDArray[np.float32]  # (H, W, 3) world-space normals
    object_index: NDArray[np.uint16]  # (H, W) SceneObject.pass_index, 0 = background
    material_index: NDArray[np.uint16]
    albedo: NDArray[np.float32]  # (H, W, 3) diffuse colour, linear

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.depth.shape[0]), int(self.depth.shape[1]))

    def to_npz(self) -> bytes:
        return npz_bytes(
            {
                "depth": self.depth.astype(np.float32),
                "normal": self.normal.astype(np.float16),
                "object_index": self.object_index.astype(np.uint16),
                "material_index": self.material_index.astype(np.uint16),
                "albedo": self.albedo.astype(np.float16),
            }
        )

    @classmethod
    def from_npz(cls, data: bytes) -> RenderPasses:
        z = np.load(io.BytesIO(data))
        return cls(
            depth=z["depth"].astype(np.float32),
            normal=z["normal"].astype(np.float32),
            object_index=z["object_index"].astype(np.uint16),
            material_index=z["material_index"].astype(np.uint16),
            albedo=z["albedo"].astype(np.float32),
        )


def _channels(path: Path) -> dict[str, NDArray[np.float32]]:
    out: dict[str, NDArray[np.float32]] = {}
    with OpenEXR.File(str(path), separate_channels=True) as f:
        for part in f.parts:
            for name, ch in part.channels.items():
                out[name] = np.asarray(ch.pixels, dtype=np.float32)
    return out


def _find(ch: dict[str, NDArray[np.float32]], layer: str, comp: str) -> NDArray[np.float32]:
    for name, arr in ch.items():
        parts = name.split(".")
        if len(parts) >= 2 and parts[-2] == layer and parts[-1] == comp:
            return arr
    raise KeyError(f"EXR has no channel {layer}.{comp}; channels: {sorted(ch)}")


def read_passes(exr_path: Path) -> RenderPasses:
    ch = _channels(exr_path)
    depth = _find(ch, "Depth", "Z").copy()
    depth[depth >= BACKGROUND_DEPTH] = np.inf
    normal = np.stack([_find(ch, "Normal", c) for c in "XYZ"], axis=-1)
    obj = np.rint(_find(ch, "Object Index", "X")).astype(np.uint16)
    mat = np.rint(_find(ch, "Material Index", "X")).astype(np.uint16)
    albedo = np.stack([_find(ch, "Diffuse Color", c) for c in "RGB"], axis=-1)
    return RenderPasses(
        depth=depth, normal=normal, object_index=obj, material_index=mat, albedo=albedo
    )


def category_lut(spec: SceneSpec) -> dict[int, str]:
    return {o.pass_index: o.category for o in spec.objects}


def structural_mask(passes: RenderPasses, spec: SceneSpec) -> NDArray[np.bool_]:
    ids = [o.pass_index for o in spec.objects if o.category in STRUCTURAL_CATEGORIES]
    return np.isin(passes.object_index, ids)


def category_mask(passes: RenderPasses, spec: SceneSpec, categories: set[str]) -> NDArray[np.bool_]:
    ids = [o.pass_index for o in spec.objects if o.category in categories]
    return np.isin(passes.object_index, ids)


def object_masks(
    passes: RenderPasses, spec: SceneSpec, element_refs: set[str]
) -> dict[str, NDArray[np.bool_]]:
    """Union mask per plan element (e.g. per opening id: frame + glass + leaf)."""
    out: dict[str, NDArray[np.bool_]] = {}
    for o in spec.objects:
        if o.element_ref in element_refs:
            m = passes.object_index == o.pass_index
            out[o.element_ref] = out[o.element_ref] | m if o.element_ref in out else m
    return out


def line_art(passes: RenderPasses, spec: SceneSpec) -> NDArray[np.uint8]:
    """Structural edges from object-id, normal-crease and depth discontinuities (0/255)."""
    obj = passes.object_index.astype(np.int32)
    depth = passes.depth
    normal = passes.normal
    struct = structural_mask(passes, spec)
    h, w = obj.shape
    edges = np.zeros((h, w), dtype=bool)
    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(0, h - dy), slice(0, w - dx))
        b = (slice(dy, h), slice(dx, w))
        involved = struct[a] | struct[b]
        id_edge = obj[a] != obj[b]
        n_dot = np.sum(normal[a] * normal[b], axis=-1)
        crease = n_dot < np.cos(np.radians(NORMAL_CREASE_DEG))
        za, zb = depth[a], depth[b]
        finite = np.isfinite(za) & np.isfinite(zb)
        with np.errstate(invalid="ignore", divide="ignore"):
            jump = np.where(
                finite, np.abs(za - zb) / np.minimum(za, zb), np.isfinite(za) != np.isfinite(zb)
            )
        depth_edge = jump > DEPTH_JUMP_REL
        edges[a] |= involved & (id_edge | crease | depth_edge)
    return (edges * 255).astype(np.uint8)


def resize_mask_to_width(
    mask: NDArray[np.uint8] | NDArray[np.bool_], width: int
) -> NDArray[np.bool_]:
    m = mask.astype(np.uint8)
    h, w = m.shape
    if w == width:
        return m > 0
    height = max(1, round(h * width / w))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_NEAREST
    return cv2.resize(m * 255, (width, height), interpolation=interp) > 0
