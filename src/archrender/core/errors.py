"""Structured errors with stable codes and fix hints.

Every failure that reaches a user (API, CLI, UI, QA report) is an :class:`ArchRenderError` with a
stable :class:`ErrorCode`, a human message and an actionable ``fix_hint``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    # generic / API
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    VALIDATION = "VALIDATION"
    RATE_LIMITED = "RATE_LIMITED"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    INTERNAL = "INTERNAL"
    # ingest
    INGEST_UNSUPPORTED_TYPE = "INGEST_UNSUPPORTED_TYPE"
    INGEST_UNSUPPORTED_NATIVE = "INGEST_UNSUPPORTED_NATIVE"
    INGEST_TOO_LARGE = "INGEST_TOO_LARGE"
    INGEST_UNSAFE_ARCHIVE = "INGEST_UNSAFE_ARCHIVE"
    INGEST_CHECKSUM_MISMATCH = "INGEST_CHECKSUM_MISMATCH"
    INGEST_INCOMPLETE_UPLOAD = "INGEST_INCOMPLETE_UPLOAD"
    INGEST_PARSER_FAILED = "INGEST_PARSER_FAILED"
    INGEST_DWG_CONVERSION_FAILED = "INGEST_DWG_CONVERSION_FAILED"
    INGEST_CORRUPT = "INGEST_CORRUPT"
    INGEST_ENCRYPTED = "INGEST_ENCRYPTED"
    INGEST_LIMIT_EXCEEDED = "INGEST_LIMIT_EXCEEDED"
    INGEST_TOOL_MISSING = "INGEST_TOOL_MISSING"
    # plan
    PLAN_NO_PLAN_FOUND = "PLAN_NO_PLAN_FOUND"
    PLAN_SCALE_CONFLICT = "PLAN_SCALE_CONFLICT"
    PLAN_INVALID = "PLAN_INVALID"
    # scene / render
    SCENE_INVALID = "SCENE_INVALID"
    SCENE_NOT_WATERTIGHT = "SCENE_NOT_WATERTIGHT"
    BLENDER_NOT_FOUND = "BLENDER_NOT_FOUND"
    BLENDER_FAILED = "BLENDER_FAILED"
    RENDER_DEVICE_UNAVAILABLE = "RENDER_DEVICE_UNAVAILABLE"
    # models
    MODEL_LICENSE_BLOCKED = "MODEL_LICENSE_BLOCKED"
    MODEL_NOT_DOWNLOADED = "MODEL_NOT_DOWNLOADED"
    MODEL_GATED_ACCESS = "MODEL_GATED_ACCESS"
    MODEL_CHECKSUM_MISMATCH = "MODEL_CHECKSUM_MISMATCH"
    MODEL_LICENSE_MISMATCH = "MODEL_LICENSE_MISMATCH"
    CUDA_OOM_EXHAUSTED = "CUDA_OOM_EXHAUSTED"
    VRAM_BUDGET_EXCEEDED = "VRAM_BUDGET_EXCEEDED"
    # pipeline
    STAGE_TIMEOUT = "STAGE_TIMEOUT"
    STAGE_FAILED = "STAGE_FAILED"
    GATE_REJECTED = "GATE_REJECTED"
    JOB_CANCELLED = "JOB_CANCELLED"
    QA_DELIVERABLE_GUARD = "QA_DELIVERABLE_GUARD"


class ErrorInfo(BaseModel):
    """Serializable error payload (API responses, job records, QA report)."""

    code: ErrorCode
    message: str
    fix_hint: str
    stage: str | None = None
    retryable: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.CONFLICT: 409,
    ErrorCode.VALIDATION: 422,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.REQUEST_TIMEOUT: 504,
    ErrorCode.INTERNAL: 500,
    ErrorCode.INGEST_TOO_LARGE: 413,
    ErrorCode.INGEST_LIMIT_EXCEEDED: 413,
    ErrorCode.INGEST_TOOL_MISSING: 503,
    ErrorCode.INGEST_CHECKSUM_MISMATCH: 422,
    ErrorCode.INGEST_INCOMPLETE_UPLOAD: 409,
}


class ArchRenderError(Exception):
    """An error with a stable code and an actionable fix hint."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        fix_hint: str,
        *,
        stage: str | None = None,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.fix_hint = fix_hint
        self.stage = stage
        self.retryable = retryable
        self.context = dict(context or {})

    @property
    def http_status(self) -> int:
        return _HTTP_STATUS.get(self.code, 400)

    def to_info(self) -> ErrorInfo:
        return ErrorInfo(
            code=self.code,
            message=self.message,
            fix_hint=self.fix_hint,
            stage=self.stage,
            retryable=self.retryable,
            context=self.context,
        )


def not_found(what: str, ident: str) -> ArchRenderError:
    return ArchRenderError(
        ErrorCode.NOT_FOUND,
        f"{what} '{ident}' does not exist or you have no access to it.",
        "Check the identifier, or ask an admin to add you to the project.",
    )
