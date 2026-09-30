"""Authentication, users and API keys."""

from __future__ import annotations

import hmac

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from archrender.api.deps import SESSION_COOKIE, CurrentUser, Svc, client_ip, limiter, require
from archrender.api.security import (
    Role,
    audit,
    create_session,
    create_user,
    delete_session,
    issue_key,
    user_from_key,
)
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.ids import now_iso

router = APIRouter(tags=["auth"])


class BootstrapIn(BaseModel):
    token: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")


class SessionIn(BaseModel):
    api_key: str


class UserIn(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[\w.@-]+$")
    role: Role


@router.post("/auth/bootstrap", status_code=201)
def bootstrap(body: BootstrapIn, request: Request, svc: Svc) -> dict[str, object]:
    """One-shot: exchange the RunPod-secret bootstrap token for the first admin API key."""
    lim = limiter(request)
    ip = client_ip(request)
    lim.check(ip)
    expected = svc.settings.admin_token
    if expected is None:
        raise ArchRenderError(
            ErrorCode.FORBIDDEN,
            "Bootstrap is disabled (no admin token configured).",
            "Set the ARCHRENDER_ADMIN_TOKEN secret on the pod, or ask an existing admin for a key.",
        )
    if not hmac.compare_digest(body.token.encode(), expected.get_secret_value().encode()):
        lim.fail(ip)
        raise ArchRenderError(
            ErrorCode.UNAUTHORIZED,
            "Wrong bootstrap token.",
            "Use the admin token from the RunPod secret.",
        )
    with svc.db.tx(immediate=True) as c:
        if c.execute("SELECT 1 FROM bootstrap WHERE id = 1").fetchone():
            raise ArchRenderError(
                ErrorCode.CONFLICT,
                "Bootstrap was already used.",
                "Ask the first admin to create a key for you.",
            )
        c.execute(
            "INSERT INTO bootstrap(id, used_at, user_id) VALUES (1, ?, 'pending')", (now_iso(),)
        )
    user = create_user(svc.db, body.name, "admin")
    svc.db.execute("UPDATE bootstrap SET user_id = ? WHERE id = 1", (user.id,))
    key = issue_key(svc.db, user.id, "bootstrap", svc.settings.pepper())
    audit(svc.db, user.id, "auth.bootstrap", user.id, ip=ip)
    return {"user": {"id": user.id, "name": user.name, "role": user.role}, "api_key": key}


@router.post("/auth/session")
def login(body: SessionIn, request: Request, response: Response, svc: Svc) -> dict[str, object]:
    lim = limiter(request)
    ip = client_ip(request)
    lim.check(ip)
    user = user_from_key(svc.db, body.api_key, svc.settings.pepper())
    if user is None:
        lim.fail(ip)
        audit(svc.db, None, "auth.login_failed", None, ip=ip)
        raise ArchRenderError(
            ErrorCode.UNAUTHORIZED, "Invalid API key.", "Paste a valid 'ark_…' key."
        )
    token, csrf = create_session(svc.db, user, svc.settings.session_ttl_s)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=svc.settings.cookie_secure,
        samesite="strict",
        max_age=svc.settings.session_ttl_s,
        path="/",
    )
    audit(svc.db, user.id, "auth.login", user.id, ip=ip)
    return {"user": {"id": user.id, "name": user.name, "role": user.role}, "csrf_token": csrf}


@router.delete("/auth/session")
def logout(request: Request, response: Response, user: CurrentUser, svc: Svc) -> dict[str, bool]:
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        delete_session(svc.db, token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(user: CurrentUser) -> dict[str, str]:
    return {"id": user.id, "name": user.name, "role": user.role}


@router.post("/users", status_code=201)
def add_user(body: UserIn, request: Request, user: CurrentUser, svc: Svc) -> dict[str, object]:
    require(user, "admin")
    new = create_user(svc.db, body.name, body.role)
    key = issue_key(svc.db, new.id, "initial", svc.settings.pepper())
    audit(svc.db, user.id, "user.create", new.id, {"role": body.role}, client_ip(request))
    return {"user": {"id": new.id, "name": new.name, "role": new.role}, "api_key": key}


@router.get("/users")
def list_users(user: CurrentUser, svc: Svc) -> list[dict[str, object]]:
    require(user, "admin")
    return [
        dict(r)
        for r in svc.db.query(
            "SELECT id, name, role, created_at, disabled FROM users ORDER BY created_at"
        )
    ]


@router.post("/users/{user_id}/keys", status_code=201)
def add_key(user_id: str, request: Request, user: CurrentUser, svc: Svc) -> dict[str, str]:
    if user.id != user_id:
        require(user, "admin")
    key = issue_key(svc.db, user_id, "additional", svc.settings.pepper())
    audit(svc.db, user.id, "key.create", user_id, ip=client_ip(request))
    return {"api_key": key}


@router.delete("/keys/{key_id}")
def revoke_key(key_id: str, request: Request, user: CurrentUser, svc: Svc) -> dict[str, bool]:
    row = svc.db.one("SELECT user_id FROM api_keys WHERE id = ?", (key_id,))
    if row is None:
        raise ArchRenderError(ErrorCode.NOT_FOUND, "Key not found.", "Check the key id.")
    if row["user_id"] != user.id:
        require(user, "admin")
    svc.db.execute("UPDATE api_keys SET revoked_at = ? WHERE id = ?", (now_iso(), key_id))
    audit(svc.db, user.id, "key.revoke", key_id, ip=client_ip(request))
    return {"ok": True}
