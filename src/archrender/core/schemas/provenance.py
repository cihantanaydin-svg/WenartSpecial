"""Provenance, facts, conflicts and assumptions (principles 2 and 3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from archrender.core.schemas.common import BBox, ModelRef, Point2, Severity, Strict, Unit, utcnow

Method = Literal[
    "ifc",
    "dxf_entity",
    "dxf_dimension",
    "pdf_vector",
    "pdf_text",
    "raster_cv",
    "raster_seg",
    "ocr",
    "vlm",
    "vlm_assisted",
    "schedule",
    "user",
    "default",
    "derived",
    "mock",
]

AssistTrigger = Literal[
    "no_candidates_with_ink",
    "validator_failed",
    "low_confidence",
    "title_block_missing",
    "north_arrow_missing",
    "scale_bar_missing",
    "sheet_corners_missing",
]


class VlmAssist(Strict):
    """Record of an on-demand VLM coordinate hint and its deterministic snap (ADR-S19)."""

    trigger: AssistTrigger
    hint_points_px: list[Point2] = Field(default_factory=list)
    hint_box_px: BBox | None = None
    evidence_coverage: float = Field(ge=0.0, le=1.0)
    snap_residual_px: float = Field(ge=0.0)
    accepted: bool
    user_confirmed: bool = False


class Provenance(Strict):
    source_doc: str | None = None
    page: int | None = None
    bbox_px: BBox | None = None
    bbox_plan: BBox | None = None
    method: Method
    model: ModelRef | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    assist: VlmAssist | None = None
    note: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class Fact[T](BaseModel):
    """A value with provenance. ``status`` tracks corroboration and conflicts."""

    value: T
    unit: Unit | None = None
    provenance: list[Provenance] = Field(min_length=1)
    status: Literal["extracted", "corroborated", "conflicted", "user_confirmed", "assumed"] = (
        "extracted"
    )

    @property
    def confidence(self) -> float:
        return max(p.confidence for p in self.provenance)


def fact[T](
    value: T,
    method: Method,
    confidence: float,
    *,
    unit: Unit | None = None,
    source_doc: str | None = None,
    note: str | None = None,
    model: ModelRef | None = None,
) -> Fact[T]:
    status: Literal["extracted", "assumed"] = "assumed" if method == "default" else "extracted"
    return Fact(
        value=value,
        unit=unit,
        status=status,
        provenance=[
            Provenance(
                method=method,
                confidence=confidence,
                source_doc=source_doc,
                note=note,
                model=model,
            )
        ],
    )


class Conflict(Strict):
    key: str
    candidates: list[dict[str, Any]]
    proposed: int = Field(ge=0)
    rule: str
    severity: Severity = Severity.WARNING
    resolved_by: str | None = None
    resolution: int | None = None


class Assumption(Strict):
    key: str
    value: Any
    reason: str
    stage: str
    overridable: bool = True
    requires_review: bool = False
