"""Run manifest: everything needed to reproduce a job (principle 7)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from archrender.core.schemas.common import ModelRef, Strict, utcnow


class Degradation(Strict):
    stage: str
    rung: str
    reason: str


class StageTiming(Strict):
    stage: str
    key: str
    cached: bool
    seconds: float
    vram_peak_mb: float | None = None


class RunManifest(Strict):
    run_id: str
    project_id: str
    created_at: datetime = Field(default_factory=utcnow)
    git_commit: str | None = None
    image_digest: str | None = None
    config_hash: str
    profile: str
    models: list[ModelRef] = Field(default_factory=list)
    seeds: dict[str, int] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)  # GPU, driver, CUDA, torch, Blender
    degradations: list[Degradation] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    uses_mocks: bool = False
