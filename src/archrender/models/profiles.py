"""Hardware profiles (``configs/profiles/*.yaml``): model per role, budgets, quality knobs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from archrender.core.errors import ArchRenderError, ErrorCode


class RefineProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_megapixels: float = Field(gt=0)
    tile_px: int = Field(gt=0)
    tile_overlap_px: int = Field(ge=0)
    best_of_n: int = Field(ge=1, le=8)
    max_retries: int = Field(ge=0, le=8)
    strength_structural: float = Field(ge=0, le=1)
    strength_furniture: float = Field(ge=0, le=1)
    strength_decor: float = Field(ge=0, le=1)


class RenderProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["AUTO", "CPU", "GPU"]
    samples: int = Field(ge=1)
    max_width: int = Field(ge=16)
    max_height: int = Field(ge=16)


class HardwareProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    vram_budget_gb: float = Field(ge=0)
    min_system_ram_gb: float = Field(ge=0)
    roles: dict[str, str]
    co_resident: list[list[str]] = Field(default_factory=list)
    render: RenderProfile
    refine: RefineProfile
    qa_width: int = 2048

    @classmethod
    def load(cls, configs_dir: Path, name: str) -> HardwareProfile:
        path = configs_dir / "profiles" / f"{name}.yaml"
        if not path.exists():
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Unknown hardware profile {name!r}.",
                "Use one of cpu_test, gpu48, gpu80, gpu96plus (ARCHRENDER_PROFILE).",
            )
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
