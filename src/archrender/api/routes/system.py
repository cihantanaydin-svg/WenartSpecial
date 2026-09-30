"""System info, model/licence status, audit log."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query, Request

from archrender.api.deps import CurrentUser, Svc, require
from archrender.ops.readiness import ReadinessState, evaluate

router = APIRouter(tags=["system"])


@router.get("/system/info")
def system_info(request: Request, user: CurrentUser, svc: Svc) -> dict[str, Any]:
    state: ReadinessState = request.app.state.readiness
    evaluate(svc, state)
    models = []
    for role, name in svc.profile.roles.items():
        e = svc.registry.get(name)
        models.append(
            {
                "role": role,
                "name": name,
                "repo": e.repo,
                "license": e.license.id,
                "license_class": e.license.license_class,
                "allowed": svc.gate.allowed(e),
                "blocked_reasons": svc.gate.reasons_blocked(e),
                "mock": e.mock,
                "implemented": e.impl is not None,
            }
        )
    return {
        "profile": svc.profile.name,
        "profile_description": svc.profile.description,
        "ready": state.ready,
        "checks": state.checks,
        "models": models,
        "jurisdictions": svc.gate.policy.jurisdictions,
        "version": request.app.version,
    }


@router.get("/audit")
def audit_log(
    user: CurrentUser, svc: Svc, limit: int = Query(default=200, ge=1, le=2000)
) -> list[dict[str, Any]]:
    require(user, "admin")
    rows = svc.db.query("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
    return [{**dict(r), "detail": json.loads(r["detail_json"])} for r in rows]
