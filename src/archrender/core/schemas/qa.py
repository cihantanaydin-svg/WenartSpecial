"""QA results. Every check says which estimator produced it (real model, deterministic code or mock)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from archrender.core.schemas.common import ModelRef, Strict

CheckFamily = Literal["geometry", "brief", "artifact", "technical", "cross_view"]


class CheckResult(Strict):
    name: str
    family: CheckFamily
    passed: bool
    value: float | None = None
    base_value: float | None = None  # same metric on the unrefined base render
    delta: float | None = None
    threshold: float | None = None
    comparator: Literal["<=", ">=", "==", "bool"] = "<="
    estimator: str  # "deterministic" | model name
    estimator_model: ModelRef | None = None
    mock: bool = False
    critical: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)


class CandidateQA(Strict):
    view_id: str
    candidate_id: str
    attempt: int
    seed: int
    params: dict[str, Any] = Field(default_factory=dict)
    checks: list[CheckResult]
    image_sha256: str

    @property
    def geometry_passed(self) -> bool:
        return all(c.passed for c in self.checks if c.family == "geometry")

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def uses_mocks(self) -> bool:
        return any(c.mock for c in self.checks)


class ViewOutcome(Strict):
    view_id: str
    camera_id: str
    status: Literal["refined", "fallback_base", "hard_composite", "needs_review"]
    delivered_sha256: str
    base_sha256: str
    attempts: list[CandidateQA]
    reason: str
    uses_mocks: bool
