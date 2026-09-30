"""Model registry (``configs/models.yaml``) and pins (``configs/models.lock.yaml``)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef

LicenseClass = Literal["permissive", "conditional", "blocked", "proprietary"]
Role = Literal[
    "vlm",
    "judge2",
    "ocr",
    "layout",
    "segmenter",
    "depth",
    "plan_segmenter",
    "embedder",
    "refiner",
    "concept",
    "upscaler",
    "iqa",
    "image_to_3d",
]


class LicenseInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    license_class: LicenseClass = Field(alias="class")
    url: str | None = None
    evidence_url: str | None = None
    checked_at: date | None = None
    terms_id: str | None = None  # for conditional licences: what the owner must accept
    obligations: list[str] = Field(default_factory=list)


class ModelFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size: int | None = None


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: Role
    repo: str | None = None
    revision: str | None = None
    files: list[ModelFile] = Field(default_factory=list)
    allow_patterns: list[str] = Field(default_factory=list)
    license: LicenseInfo
    commercial_ok: bool
    territorial_exclusions: list[str] = Field(default_factory=list)
    caps: dict[str, str] = Field(default_factory=dict)
    gated: bool = False
    runtime: str
    impl: str | None = None  # "module:Class"; None = weights registered, runtime not yet built
    vram_gb: float = 0.0
    disk_gb: float = 0.0
    fallback: str | None = None
    mock: bool = False
    notes: str = ""

    def ref(self) -> ModelRef:
        return ModelRef(
            role=self.role,
            name=self.name,
            repo=self.repo or "",
            revision=self.revision,
            weights_sha256=None,
            mock=self.mock,
        )


class Registry:
    def __init__(self, entries: list[RegistryEntry]) -> None:
        self._by_name: dict[str, RegistryEntry] = {}
        for e in entries:
            if e.name in self._by_name:
                raise ValueError(f"duplicate registry entry {e.name}")
            self._by_name[e.name] = e

    @classmethod
    def load(cls, configs_dir: Path) -> Registry:
        data = yaml.safe_load((configs_dir / "models.yaml").read_text(encoding="utf-8"))
        entries = [RegistryEntry.model_validate(e) for e in data["models"]]
        lock_path = configs_dir / "models.lock.yaml"
        if lock_path.exists():
            lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) or {}
            pins = lock.get("models", {})
            for e in entries:
                pin = pins.get(e.name)
                if pin:
                    e.revision = pin["revision"]
                    e.files = [ModelFile.model_validate(f) for f in pin.get("files", [])]
        return cls(entries)

    def get(self, name: str) -> RegistryEntry:
        try:
            return self._by_name[name]
        except KeyError:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Model {name!r} is not in configs/models.yaml.",
                "Add a registry entry (with licence evidence) or fix the profile.",
            ) from None

    def entries(self) -> list[RegistryEntry]:
        return list(self._by_name.values())
