"""Gate A: the project's plan versions, JSON Patch edits, conflict resolution and approval."""

from __future__ import annotations

import json
from typing import Any, Literal

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
    rederive_rooms: bool = False  # recompute rooms from the edited walls


class AssistConfirmation(BaseModel):
    element_ids: list[str] = Field(min_length=1, max_length=500)


class Calibration(BaseModel):
    a: tuple[float, float]  # plan metres, in the version's frame
    b: tuple[float, float]
    length_m: float = Field(gt=0)
    level: str | None = Field(default=None, max_length=64)


class SuggestionDecision(BaseModel):
    action: Literal["accept", "reject"]
    thickness_m: float | None = Field(default=None, gt=0.03, le=1.0)  # for an accepted wall


class ConflictChoice(BaseModel):
    key: str = Field(min_length=1, max_length=200)
    choice: int = Field(ge=0)


def _version(svc: Svc, project_id: str, version_id: str) -> dict[str, Any]:
    check_id(version_id, "plan version id")
    out = versions.summary(versions.get_row(svc, project_id, version_id))
    out["plan"] = json.loads(versions.load(svc, project_id, version_id).model_dump_json())
    out["suggestions"] = versions.suggestions(svc, project_id, version_id)
    plan = versions.load(svc, project_id, version_id)
    out["backgrounds"] = versions.backgrounds(svc, project_id, plan)
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
    vid = versions.edit(
        svc,
        project_id,
        version_id,
        body.ops,
        user_id=user.id,
        note=body.note,
        rederive_rooms=body.rederive_rooms,
    )
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


@router.post("/projects/{project_id}/plans/{version_id}/assists/confirm", status_code=201)
def confirm_assists(
    project_id: str,
    version_id: str,
    body: AssistConfirmation,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    """Confirm elements measured from VLM hints (they block approval until confirmed)."""
    project_access(svc, user, project_id, "editor")
    check_id(version_id, "plan version id")
    for eid in body.element_ids:
        check_id(eid, "element id")
    vid = versions.confirm_assists(svc, project_id, version_id, body.element_ids, user_id=user.id)
    audit(
        svc.db,
        user.id,
        "plan.assist_confirm",
        vid,
        {"from": version_id, "elements": body.element_ids},
        client_ip(request),
    )
    return _version(svc, project_id, vid)


@router.post("/projects/{project_id}/plans/{version_id}/suggestions/{page_id}/{suggestion}")
def decide_suggestion(
    project_id: str,
    version_id: str,
    page_id: str,
    suggestion: str,
    body: SuggestionDecision,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    """Accept a VLM suggestion without drawing evidence as your own element (new version), or
    reject it. Either decision is kept as a training example (never used without the owner)."""
    project_access(svc, user, project_id, "editor")
    check_id(version_id, "plan version id")
    check_id(page_id, "page id")
    check_id(suggestion, "suggestion id")
    sid = f"{page_id}/{suggestion}"
    vid = versions.decide_suggestion(
        svc,
        project_id,
        version_id,
        sid,
        accept=body.action == "accept",
        user_id=user.id,
        thickness_m=body.thickness_m,
    )
    audit(
        svc.db,
        user.id,
        f"plan.suggestion_{body.action}",
        sid,
        {"version": version_id, "new_version": vid},
        client_ip(request),
    )
    return _version(svc, project_id, vid or version_id)


@router.post("/projects/{project_id}/plans/{version_id}/calibrate", status_code=201)
def calibrate(
    project_id: str,
    version_id: str,
    body: Calibration,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    """Two-point scale calibration: a known length between two clicked points → a new draft."""
    project_access(svc, user, project_id, "editor")
    check_id(version_id, "plan version id")
    vid = versions.calibrate(
        svc,
        project_id,
        version_id,
        body.a,
        body.b,
        body.length_m,
        user_id=user.id,
        level=body.level,
    )
    audit(
        svc.db,
        user.id,
        "plan.calibrate",
        vid,
        {"from": version_id, "length_m": body.length_m},
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
