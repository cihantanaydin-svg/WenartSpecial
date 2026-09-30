"""Jobs, job events and stage-run records."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from archrender.core.errors import ErrorInfo
from archrender.core.schemas.common import Strict

Queue = Literal["cpu", "gpu"]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_GATE = "waiting_gate"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


class JobKind(StrEnum):
    INTAKE = "intake"
    UNDERSTAND = "understand"
    RUN = "run"
    BUNDLE = "bundle"


class Job(Strict):
    id: str
    project_id: str
    kind: JobKind
    queue: Queue
    status: JobStatus
    priority: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: ErrorInfo | None = None
    attempts: int = 0
    max_attempts: int = 3
    progress: float = 0.0
    stage: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_by: str | None = None


class JobEvent(Strict):
    id: int
    job_id: str
    ts: datetime
    type: Literal["status", "stage", "progress", "log", "gate", "result", "error"]
    data: dict[str, Any] = Field(default_factory=dict)


class GateName(StrEnum):
    A = "A_plan"
    B = "B_brief"
    C = "C_cameras"
    D = "D_final"


class GateStatus(StrEnum):
    PENDING = "pending"
    AUTO_PASSED = "auto_passed"
    APPROVED = "approved"
    REJECTED = "rejected"
