"""Runs, gates, jobs (+SSE), blobs and bundle downloads."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Header, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from archrender.api.deps import CurrentUser, Svc, client_ip, project_access
from archrender.api.security import audit
from archrender.core.cas import CasRef
from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.paths import check_sha256
from archrender.pipeline.gates import decide
from archrender.pipeline.run import RunConfig, create_run

router = APIRouter(tags=["runs"])

SSE_MAX_SECONDS = 600.0


class GateDecision(BaseModel):
    approve: bool
    notes: str | None = Field(default=None, max_length=2000)


def _run_row(svc: Svc, run_id: str) -> Any:
    row = svc.db.one("SELECT * FROM runs WHERE id = ?", (run_id,))
    if row is None:
        raise not_found("Run", run_id)
    return row


def _run_view(svc: Svc, row: Any) -> dict[str, Any]:
    gates = [
        dict(g)
        for g in svc.db.query(
            "SELECT gate, status, policy, evidence_json, decided_by, decided_at, notes FROM gates"
            " WHERE run_id = ? ORDER BY gate",
            (row["id"],),
        )
    ]
    for g in gates:
        g["evidence"] = json.loads(g.pop("evidence_json"))
    bundle = svc.db.one(
        "SELECT id, sha256, size, status FROM bundles WHERE run_id = ? ORDER BY created_at DESC",
        (row["id"],),
    )
    return {
        "id": row["id"],
        "project_id": row["project_id"],
        "job_id": row["job_id"],
        "status": row["status"],
        "config": json.loads(row["config_json"]),
        "created_at": row["created_at"],
        "gates": gates,
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "bundle": dict(bundle) if bundle else None,
    }


@router.post("/projects/{project_id}/runs", status_code=202)
def start_run(
    project_id: str, body: RunConfig, request: Request, user: CurrentUser, svc: Svc
) -> dict[str, str]:
    project_access(svc, user, project_id, "editor")
    run_id, job_id = create_run(svc, project_id, body, user.id)
    audit(
        svc.db,
        user.id,
        "run.start",
        run_id,
        {"project": project_id, "config": body.model_dump()},
        client_ip(request),
    )
    return {"run_id": run_id, "job_id": job_id}


@router.get("/projects/{project_id}/runs")
def list_runs(project_id: str, user: CurrentUser, svc: Svc) -> list[dict[str, Any]]:
    project_access(svc, user, project_id)
    rows = svc.db.query(
        "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC LIMIT 100", (project_id,)
    )
    return [_run_view(svc, r) for r in rows]


@router.get("/runs/{run_id}")
def get_run(run_id: str, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    row = _run_row(svc, run_id)
    project_access(svc, user, row["project_id"])
    return _run_view(svc, row)


@router.post("/runs/{run_id}/gates/{gate}")
def decide_gate(
    run_id: str, gate: str, body: GateDecision, request: Request, user: CurrentUser, svc: Svc
) -> dict[str, Any]:
    row = _run_row(svc, run_id)
    project_access(svc, user, row["project_id"], "reviewer")
    decide(svc.db, run_id, gate, approve=body.approve, user_id=user.id, notes=body.notes)
    svc.queue.resume(row["job_id"])
    audit(
        svc.db,
        user.id,
        "gate.approve" if body.approve else "gate.reject",
        f"{run_id}/{gate}",
        {"notes": body.notes},
        client_ip(request),
    )
    return _run_view(svc, _run_row(svc, run_id))


@router.get("/jobs/{job_id}")
def get_job(job_id: str, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    job = svc.queue.get(job_id)
    project_access(svc, user, job.project_id)
    return job.model_dump(mode="json")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    job = svc.queue.get(job_id)
    project_access(svc, user, job.project_id, "editor")
    audit(svc.db, user.id, "job.cancel", job_id, ip=client_ip(request))
    return svc.queue.request_cancel(job_id).model_dump(mode="json")


@router.get("/jobs/{job_id}/events")
async def job_events(
    job_id: str,
    user: CurrentUser,
    svc: Svc,
    after: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None),
) -> StreamingResponse:
    """Server-sent events with heartbeats and ``Last-Event-ID`` resume (ADR-S08)."""
    job = await run_in_threadpool(svc.queue.get, job_id)
    await run_in_threadpool(project_access, svc, user, job.project_id)
    start = int(last_event_id) if last_event_id and last_event_id.isdigit() else after
    heartbeat = svc.settings.sse_heartbeat_s

    async def stream() -> AsyncIterator[bytes]:
        cursor = start
        t0 = time.monotonic()
        last_beat = t0
        yield b"retry: 3000\n\n"
        while True:
            events = await run_in_threadpool(svc.queue.events, job_id, cursor)
            for ev in events:
                cursor = ev.id
                payload = json.dumps(
                    {"type": ev.type, "ts": ev.ts.isoformat(), **ev.data}, default=str
                )
                yield f"id: {ev.id}\nevent: {ev.type}\ndata: {payload}\n\n".encode()
            now = time.monotonic()
            if not events:
                current = await run_in_threadpool(svc.queue.get, job_id)
                if current.status.terminal:
                    yield b"event: end\ndata: {}\n\n"
                    return
            if now - last_beat >= heartbeat:
                last_beat = now
                yield b": ping\n\n"
            if now - t0 > SSE_MAX_SECONDS:
                return  # client reconnects with Last-Event-ID
            await asyncio.sleep(0.5)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"glTF", "model/gltf-binary"),
    (b"PK", "application/zip"),
    (b"%PDF", "application/pdf"),
    (b"<!doctype html", "text/html; charset=utf-8"),
    (b"{", "application/json"),
]


def _sniff(path_head: bytes) -> str:
    low = path_head[:16].lower()
    for magic, mt in _MAGIC:
        if low.startswith(magic.lower()):
            return mt
    return "application/octet-stream"


@router.get("/projects/{project_id}/blobs/{sha256}")
def get_blob(
    project_id: str, sha256: str, user: CurrentUser, svc: Svc, download: str | None = None
) -> FileResponse:
    project_access(svc, user, project_id)
    check_sha256(sha256)
    path = svc.store(project_id).path(CasRef(sha256=sha256, size=0))
    with path.open("rb") as fh:
        media = _sniff(fh.read(16))
    headers = {
        "Cache-Control": "private, max-age=31536000, immutable",
        "X-Content-Type-Options": "nosniff",
    }
    if media.startswith("text/html"):
        headers["Content-Security-Policy"] = (
            "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
        )
    return FileResponse(path, media_type=media, headers=headers, filename=download)


@router.get("/bundles/{bundle_id}/download")
def download_bundle(bundle_id: str, request: Request, user: CurrentUser, svc: Svc) -> FileResponse:
    row = svc.db.one("SELECT * FROM bundles WHERE id = ?", (bundle_id,))
    if row is None:
        raise not_found("Bundle", bundle_id)
    project_access(svc, user, row["project_id"])
    if row["status"] != "ready":
        raise ArchRenderError(
            ErrorCode.CONFLICT, "Bundle is not ready yet.", "Wait for the run to finish."
        )
    path = svc.store(row["project_id"]).path(CasRef(sha256=row["sha256"], size=row["size"]))
    audit(svc.db, user.id, "bundle.download", bundle_id, ip=client_ip(request))
    return FileResponse(
        path, media_type="application/zip", filename=f"archrender_{row['run_id']}.zip"
    )
