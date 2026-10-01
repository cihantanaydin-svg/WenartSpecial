"""Gate A: the project's plan versions, JSON Patch edits, conflict resolution and approval."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from archrender.api.deps import CurrentUser, Svc, client_ip, project_access
from archrender.api.security import audit
from archrender.core.paths import check_id
from archrender.core.schemas.jobs import JobKind
from archrender.plan import versions

router = APIRouter(tags=["plans"])

MAX_OPS = 2000


class PlanEdit(BaseModel):
    # RFC 6902 operations on the PlanGraph JSON (add / remove / replace / move / copy / test)
    ops: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_OPS)
    note: str = Field(default="", max_length=500)


class ConflictChoice(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    choice: int = Field(ge=0)


def _version(svc: Svc, project_id: str, version_id: str) -> dict[str, Any]:
    check_id(version_id, "plan version id")
    out = versions.summary(versions.get_row(svc, project_id, version_id))
    out["plan"] = json.loads(versions.load(svc, project_id, version_id).model_dump_json())
    return out


@router.get("/projects/{project_id}/plans")
def list_plan_versions(project_id: str, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    """All versions (newest first) and the latest PLAN job (the UI polls it after uploads)."""
    project_access(svc, user, project_id)
    job = svc.db.one(
        "SELECT id FROM jobs WHERE project_id = ? AND kind = ? ORDER BY created_at DESC, rowid DESC"
        " LIMIT 1",
        (project_id, JobKind.PLAN.value),
    )
    return {
        "versions": versions.list_versions(svc, project_id),
        "job": svc.queue.get(job["id"]).model_dump(mode="json") if job else None,
    }


@router.post("/projects/{project_id}/plans/extract", status_code=202)
def extract(project_id: str, request: Request, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    """Queue S2 for the project's current pages (an unchanged extraction keeps its version)."""
    project_access(svc, user, project_id, "editor")
    job_id = svc.queue.enqueue(project_id, JobKind.PLAN, "gpu", {}, created_by=user.id)
    audit(svc.db, user.id, "plan.extract", project_id, ip=client_ip(request))
    return {"job_id": job_id}


@router.get("/projects/{project_id}/plans/{version_id}")
def get_plan_version(
    project_id: str, version_id: str, user: CurrentUser, svc: Svc
) -> dict[str, Any]:
    project_access(svc, user, project_id)
    return _version(svc, project_id, version_id)


@router.post("/projects/{project_id}/plans/{version_id}/edits", status_code=201)
def edit_plan(
    project_id: str,
    version_id: str,
    body: PlanEdit,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    """A new draft version: ``version_id`` with the patch applied (the source never changes)."""
    project_access(svc, user, project_id, "editor")
    check_id(version_id, "plan version id")
    vid = versions.edit(svc, project_id, version_id, body.ops, user_id=user.id, note=body.note)
    audit(
        svc.db,
        user.id,
        "plan.edit",
        vid,
        {"from": version_id, "ops": len(body.ops), "note": body.note},
        client_ip(request),
    )
    return _version(svc, project_id, vid)


@router.post("/projects/{project_id}/plans/{version_id}/resolve", status_code=201)
def resolve(
    project_id: str,
    version_id: str,
    body: ConflictChoice,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    """Pick a candidate of a conflict (a scale conflict rescales the plan) → a new draft."""
    project_access(svc, user, project_id, "editor")
    check_id(version_id, "plan version id")
    vid = versions.resolve_conflict(
        svc, project_id, version_id, body.key, body.choice, user_id=user.id
    )
    audit(
        svc.db,
        user.id,
        "plan.resolve",
        vid,
        {"from": version_id, "key": body.key, "choice": body.choice},
        client_ip(request),
    )
    return _version(svc, project_id, vid)


@router.post("/projects/{project_id}/plans/{version_id}/approve")
def approve(
    project_id: str, version_id: str, request: Request, user: CurrentUser, svc: Svc
) -> dict[str, Any]:
    """Approve the version for runs (refused while it has blocking issues)."""
    project_access(svc, user, project_id, "reviewer")
    check_id(version_id, "plan version id")
    out = versions.approve(svc, project_id, version_id, user_id=user.id)
    audit(svc.db, user.id, "plan.approve", version_id, ip=client_ip(request))
    return out


@router.get("/projects/{project_id}/training-examples")
def list_training_examples(project_id: str, user: CurrentUser, svc: Svc) -> list[dict[str, Any]]:
    """Gate A corrections kept as examples (never used for training unless the owner allows)."""
    project_access(svc, user, project_id, "editor")
    return versions.training_examples(svc, project_id)
