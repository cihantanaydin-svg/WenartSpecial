"""S1 results: pages with their classification, user overrides, the review queue and schedules."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from archrender.api.deps import CurrentUser, Svc, client_ip, project_access
from archrender.api.security import audit
from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.ids import now_iso
from archrender.core.paths import check_id
from archrender.core.schemas.jobs import JobKind
from archrender.core.schemas.understanding import PageClass

router = APIRouter(tags=["understanding"])


class ClassOverride(BaseModel):
    label: PageClass
    note: str | None = Field(default=None, max_length=500)


class ReviewDecision(BaseModel):
    action: Literal["resolve", "dismiss"]
    note: str | None = Field(default=None, max_length=500)


@router.get("/projects/{project_id}/pages")
def list_pages(project_id: str, user: CurrentUser, svc: Svc) -> list[dict[str, Any]]:
    project_access(svc, user, project_id)
    rows = svc.db.query(
        "SELECT p.id, p.document_id, p.idx, p.kind, p.page_json, d.filename, a.label, a.confidence,"
        " a.needs_review, a.override_label, a.analysis_json FROM pages p"
        " JOIN documents d ON d.id = p.document_id LEFT JOIN page_analysis a ON a.page_id = p.id"
        " WHERE p.project_id = ? ORDER BY d.created_at, p.document_id, p.idx",
        (project_id,),
    )
    out = []
    for r in rows:
        page = json.loads(r["page_json"])
        analysis = json.loads(r["analysis_json"]) if r["analysis_json"] else None
        tb = (analysis or {}).get("title_block") or {}
        out.append(
            {
                "page_id": r["id"],
                "document_id": r["document_id"],
                "filename": r["filename"],
                "index": r["idx"],
                "kind": r["kind"],
                "width_px": page.get("width_px"),
                "height_px": page.get("height_px"),
                "dpi": page.get("dpi"),
                "preview": page.get("meta", {}).get("preview"),
                "label": r["override_label"] or r["label"],
                "model_label": r["label"],
                "confidence": r["confidence"],
                "needs_review": bool(r["needs_review"]) and not r["override_label"],
                "overridden": bool(r["override_label"]),
                "scale": ((analysis or {}).get("scale") or {}).get("value"),
                "north_deg": (((analysis or {}).get("north") or {}).get("angle_deg") or {}).get(
                    "value"
                ),
                "title": {k: v.get("value") for k, v in (tb.get("fields") or {}).items()},
                "tags": len((analysis or {}).get("tags", [])),
                "analysed": analysis is not None,
            }
        )
    return out


@router.put("/projects/{project_id}/pages/{page_id}/class")
def override_class(
    project_id: str,
    page_id: str,
    body: ClassOverride,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    project_access(svc, user, project_id, "editor")
    check_id(page_id, "page id")
    row = svc.db.one(
        "SELECT page_id FROM page_analysis WHERE page_id = ? AND project_id = ?",
        (page_id, project_id),
    )
    if row is None:
        raise not_found("Analysed page", page_id)
    with svc.db.tx(immediate=True) as c:
        c.execute(
            "UPDATE page_analysis SET override_label = ?, override_by = ?, updated_at = ? WHERE page_id = ?",
            (body.label, user.id, now_iso(), page_id),
        )
        c.execute(
            "UPDATE review_items SET status = 'resolved', resolved_by = ?, resolved_at = ?, resolution_json = ?"
            " WHERE kind = 'page_class' AND subject_id = ? AND status = 'open'",
            (user.id, now_iso(), json.dumps({"label": body.label, "note": body.note}), page_id),
        )
    audit(
        svc.db,
        user.id,
        "page.class_override",
        page_id,
        ip=client_ip(request),
        detail={"label": body.label},
    )
    return {"page_id": page_id, "label": body.label}


@router.get("/projects/{project_id}/review")
def list_review(
    project_id: str, user: CurrentUser, svc: Svc, status: str = "open"
) -> list[dict[str, Any]]:
    project_access(svc, user, project_id)
    if status not in ("open", "resolved", "dismissed", "all"):
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Unknown status {status!r}.",
            "Use open, resolved, dismissed or all.",
        )
    sql = "SELECT * FROM review_items WHERE project_id = ?"
    args: tuple[str, ...] = (project_id,)
    if status != "all":
        sql += " AND status = ?"
        args = (project_id, status)
    rows = svc.db.query(sql + " ORDER BY created_at", args)
    return [
        {
            "id": r["id"],
            "kind": r["kind"],
            "subject_id": r["subject_id"],
            "status": r["status"],
            "payload": json.loads(r["payload_json"]),
            "resolution": json.loads(r["resolution_json"]) if r["resolution_json"] else None,
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.post("/projects/{project_id}/review/{item_id}")
def decide_review(
    project_id: str,
    item_id: str,
    body: ReviewDecision,
    request: Request,
    user: CurrentUser,
    svc: Svc,
) -> dict[str, Any]:
    project_access(svc, user, project_id, "reviewer")
    check_id(item_id, "review item id")
    row = svc.db.one(
        "SELECT status FROM review_items WHERE id = ? AND project_id = ?", (item_id, project_id)
    )
    if row is None:
        raise not_found("Review item", item_id)
    if row["status"] != "open":
        raise ArchRenderError(
            ErrorCode.CONFLICT, f"The item is already {row['status']}.", "Refresh the review list."
        )
    status = "resolved" if body.action == "resolve" else "dismissed"
    svc.db.execute(
        "UPDATE review_items SET status = ?, resolved_by = ?, resolved_at = ?, resolution_json = ? WHERE id = ?",
        (status, user.id, now_iso(), json.dumps({"note": body.note}), item_id),
    )
    audit(svc.db, user.id, f"review.{body.action}", item_id, ip=client_ip(request))
    return {"id": item_id, "status": status}


@router.get("/projects/{project_id}/schedules")
def list_schedules(project_id: str, user: CurrentUser, svc: Svc) -> list[dict[str, Any]]:
    project_access(svc, user, project_id)
    rows = svc.db.query(
        "SELECT schedule_json FROM schedules WHERE project_id = ? ORDER BY id", (project_id,)
    )
    return [json.loads(r["schedule_json"]) for r in rows]


@router.get("/projects/{project_id}/understand")
def understanding_status(project_id: str, user: CurrentUser, svc: Svc) -> dict[str, Any] | None:
    """The project's latest S1 job (the UI polls it while pages are being analysed), or null."""
    project_access(svc, user, project_id)
    row = svc.db.one(
        "SELECT id FROM jobs WHERE project_id = ? AND kind = ? ORDER BY created_at DESC, rowid DESC"
        " LIMIT 1",
        (project_id, JobKind.UNDERSTAND.value),
    )
    return svc.queue.get(row["id"]).model_dump(mode="json") if row else None


@router.post("/projects/{project_id}/understand", status_code=202)
def reanalyse(project_id: str, request: Request, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    """Queue S1 for all pages (unchanged pages are cache hits; e.g. after the VLM came up)."""
    project_access(svc, user, project_id, "editor")
    job_id = svc.queue.enqueue(project_id, JobKind.UNDERSTAND, "gpu", {}, created_by=user.id)
    audit(svc.db, user.id, "project.understand", project_id, ip=client_ip(request))
    return {"job_id": job_id}
