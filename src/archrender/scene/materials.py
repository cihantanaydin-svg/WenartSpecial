"""Material library (procedural seed set; PBR assets join in Phase 5 with the same interface)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from archrender.core.errors import ArchRenderError, ErrorCode


class LibraryMaterial(BaseModel):
    id: str
    category: str
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    roughness: float = 0.5
    metallic: float = 0.0
    transmission: float = 0.0
    ior: float = 1.5
    texture_size_m: tuple[float, float] = (1.0, 1.0)
    license: str
    source: str = "procedural"


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))


class MaterialLibrary:
    def __init__(self, materials: list[LibraryMaterial]) -> None:
        self._by_id = {m.id: m for m in materials}

    @classmethod
    def load(cls, configs_dir: Path) -> MaterialLibrary:
        data = yaml.safe_load((configs_dir / "materials.yaml").read_text(encoding="utf-8"))
        return cls([LibraryMaterial.model_validate(m) for m in data["materials"]])

    def get(self, material_id: str) -> LibraryMaterial:
        try:
            return self._by_id[material_id]
        except KeyError:
            raise ArchRenderError(
                ErrorCode.SCENE_INVALID,
                f"Material {material_id!r} is not in the library.",
                "Pick a library material at Gate B or import the material into the asset library.",
            ) from None

    def ids(self) -> list[str]:
        return sorted(self._by_id)
