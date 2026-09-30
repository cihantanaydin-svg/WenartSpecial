"""Projects, members, documents, uploads, retention/purge."""

from __future__ import annotations

import json
import shutil
from typing import Any, Literal

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field

from archrender.api.deps import CurrentUser, Svc, client_ip, project_access, require
from archrender.api.security import audit
from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.ids import new_id, now_iso
from archrender.ingest.uploads import complete_upload, create_upload, put_chunk, upload_status

router = APIRouter(tags=["projects"])


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    timezone: str = Field(default="Europe/Istanbul", max_length=64)
    units: Literal["metric", "imperial"] = "metric"


class MemberIn(BaseModel):
    user_id: str
    role: Literal["editor", "reviewer", "viewer"]


class UploadIn(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _project(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "latitude": row["latitude"],
        "longitude": row["longitude"],
        "timezone": row["timezone"],
        "units": row["units"],
    }


@router.post("/projects", status_code=201)
def create_project(
    body: ProjectIn, request: Request, user: CurrentUser, svc: Svc
) -> dict[str, Any]:
    require(user, "editor")
    pid = new_id("prj")
    svc.db.execute(
        "INSERT INTO projects(id, name, created_by, created_at, latitude, longitude, timezone, units)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (
            pid,
            body.name,
            user.id,
            now_iso(),
            body.latitude,
            body.longitude,
            body.timezone,
            body.units,
        ),
    )
    svc.store(pid)
    audit(svc.db, user.id, "project.create", pid, {"name": body.name}, client_ip(request))
    row = svc.db.one("SELECT * FROM projects WHERE id = ?", (pid,))
    return _project(row)


@router.get("/projects")
def list_projects(user: CurrentUser, svc: Svc) -> list[dict[str, Any]]:
    if user.role == "admin":
        rows = svc.db.query(
            "SELECT * FROM projects WHERE purged_at IS NULL ORDER BY created_at DESC"
        )
    else:
        rows = svc.db.query(
            "SELECT p.* FROM projects p LEFT JOIN project_members m ON m.project_id = p.id AND m.user_id = ?"
            " WHERE p.purged_at IS NULL AND (p.created_by = ? OR m.user_id IS NOT NULL) ORDER BY p.created_at DESC",
            (user.id, user.id),
        )
    return [_project(r) for r in rows]


@router.get("/projects/{project_id}")
def get_project(project_id: str, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    project_access(svc, user, project_id)
    return _project(svc.db.one("SELECT * FROM projects WHERE id = ?", (project_id,)))


@router.post("/projects/{project_id}/members", status_code=201)
def add_member(
    project_id: str, body: MemberIn, request: Request, user: CurrentUser, svc: Svc
) -> dict[str, bool]:
    project_access(svc, user, project_id, "editor")
    if svc.db.one("SELECT 1 FROM users WHERE id = ?", (body.user_id,)) is None:
        raise not_found("User", body.user_id)
    svc.db.execute(
        "INSERT OR REPLACE INTO project_members(project_id, user_id, role) VALUES (?,?,?)",
        (project_id, body.user_id, body.role),
    )
    audit(svc.db, user.id, "project.member_add", project_id, body.model_dump(), client_ip(request))
    return {"ok": True}


@router.delete("/projects/{project_id}")
def purge_project(
    project_id: str, request: Request, user: CurrentUser, svc: Svc
) -> dict[str, bool]:
    """Retention/purge: deletes every blob and row of the project (irreversible)."""
    require(user, "admin")
    project_access(svc, user, project_id)
    running = svc.db.one(
        "SELECT 1 FROM jobs WHERE project_id = ? AND status IN ('running')", (project_id,)
    )
    if running:
        raise ArchRenderError(
            ErrorCode.CONFLICT, "Project has a running job.", "Cancel the job first, then purge."
        )
    shutil.rmtree(svc.store(project_id).root, ignore_errors=True)
    with svc.db.tx(immediate=True) as c:
        for table in (
            "bundles",
            "gates",
            "runs",
            "stage_runs",
            "job_events",
            "jobs",
            "upload_chunks",
            "uploads",
            "documents",
            "project_members",
        ):
            if table in ("gates",):
                c.execute(
                    "DELETE FROM gates WHERE run_id IN (SELECT id FROM runs WHERE project_id = ?)",
                    (project_id,),
                )
            elif table == "job_events":
                c.execute(
                    "DELETE FROM job_events WHERE job_id IN (SELECT id FROM jobs WHERE project_id = ?)",
                    (project_id,),
                )
            elif table == "upload_chunks":
                c.execute(
                    "DELETE FROM upload_chunks WHERE upload_id IN (SELECT id FROM uploads WHERE project_id = ?)",
                    (project_id,),
                )
            else:
                c.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))  # noqa: S608
        c.execute(
            "UPDATE projects SET purged_at = ?, name = '[purged]', meta_json = '{}' WHERE id = ?",
            (now_iso(), project_id),
        )
    audit(svc.db, user.id, "project.purge", project_id, ip=client_ip(request))
    return {"ok": True}


@router.get("/projects/{project_id}/documents")
def list_documents(project_id: str, user: CurrentUser, svc: Svc) -> list[dict[str, Any]]:
    project_access(svc, user, project_id)
    rows = svc.db.query(
        "SELECT id, sha256, filename, kind, media_type, size, created_at, meta_json FROM documents"
        " WHERE project_id = ? ORDER BY created_at",
        (project_id,),
    )
    return [{**dict(r), "meta": json.loads(r["meta_json"])} for r in rows]


@router.post("/projects/{project_id}/uploads", status_code=201)
def start_upload(project_id: str, body: UploadIn, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    project_access(svc, user, project_id, "editor")
    return create_upload(svc, project_id, body.filename, body.size, body.sha256, user.id)


def _upload_project(svc: Svc, upload_id: str) -> str:
    row = svc.db.one("SELECT project_id FROM uploads WHERE id = ?", (upload_id,))
    if row is None:
        raise not_found("Upload", upload_id)
    return str(row["project_id"])


@router.put("/uploads/{upload_id}/chunks/{index}")
async def upload_chunk(
    upload_id: str,
    index: int,
    request: Request,
    user: CurrentUser,
    svc: Svc,
    x_chunk_sha256: str | None = Header(default=None),
) -> dict[str, Any]:
    from starlette.concurrency import run_in_threadpool

    pid = await run_in_threadpool(_upload_project, svc, upload_id)
    await run_in_threadpool(project_access, svc, user, pid, "editor")
    limit = svc.settings.upload_chunk_bytes
    declared = int(request.headers.get("content-length") or 0)
    if declared > limit:
        raise ArchRenderError(
            ErrorCode.INGEST_TOO_LARGE,
            f"Chunk larger than {limit} bytes.",
            "Use the chunk_size from the upload.",
        )
    body = bytearray()
    async for part in request.stream():
        body.extend(part)
        if len(body) > limit:
            raise ArchRenderError(
                ErrorCode.INGEST_TOO_LARGE,
                f"Chunk larger than {limit} bytes.",
                "Use the chunk_size from the upload.",
            )
    return await run_in_threadpool(put_chunk, svc, upload_id, index, bytes(body), x_chunk_sha256)


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: str, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    project_access(svc, user, _upload_project(svc, upload_id))
    return upload_status(svc, upload_id)


@router.post("/uploads/{upload_id}/complete", status_code=202)
def finish_upload(upload_id: str, request: Request, user: CurrentUser, svc: Svc) -> dict[str, str]:
    pid = _upload_project(svc, upload_id)
    project_access(svc, user, pid, "editor")
    job_id = complete_upload(svc, upload_id, user.id)
    audit(svc.db, user.id, "upload.complete", upload_id, {"project": pid}, client_ip(request))
    return {"job_id": job_id}
