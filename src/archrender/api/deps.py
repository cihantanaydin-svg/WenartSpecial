"""FastAPI dependencies: services, authentication, CSRF, roles, project access."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from archrender.api.security import FailureLimiter, Role, User, user_from_key, user_from_session
from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.paths import check_id
from archrender.pipeline.services import Services

SESSION_COOKIE = "ark_session"
CSRF_HEADER = "x-csrf-token"
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


def limiter(request: Request) -> FailureLimiter:
    lim: FailureLimiter = request.app.state.auth_limiter
    return lim


def client_ip(request: Request) -> str:
    fwd = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def current_user(request: Request, svc: Annotated[Services, Depends(services)]) -> User:
    ip = client_ip(request)
    lim = limiter(request)
    lim.check(ip)
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        user = user_from_key(svc.db, auth[7:], svc.settings.pepper())
        if user is None:
            lim.fail(ip)
            raise _unauth("Invalid or revoked API key.")
        return user
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        found = user_from_session(svc.db, token)
        if found is None:
            lim.fail(ip)
            raise _unauth("Session expired.")
        user, csrf = found
        if request.method in UNSAFE and request.headers.get(CSRF_HEADER) != csrf:
            raise ArchRenderError(
                ErrorCode.FORBIDDEN,
                "Missing or wrong CSRF token.",
                "Reload the UI; it sends the token automatically.",
            )
        return user
    raise _unauth("Authentication required.")


def _unauth(msg: str) -> ArchRenderError:
    return ArchRenderError(
        ErrorCode.UNAUTHORIZED, msg, "Send 'Authorization: Bearer ark_…' or log in through the UI."
    )


CurrentUser = Annotated[User, Depends(current_user)]
Svc = Annotated[Services, Depends(services)]


def require(user: User, role: Role) -> None:
    if not user.at_least(role):
        raise ArchRenderError(
            ErrorCode.FORBIDDEN,
            f"This action needs the '{role}' role.",
            "Ask an admin to grant the role.",
        )


def project_access(svc: Services, user: User, project_id: str, role: Role = "viewer") -> None:
    """Admins see everything; others need to be the creator or a member with at least ``role``."""
    check_id(project_id, "project id")
    row = svc.db.one("SELECT created_by, purged_at FROM projects WHERE id = ?", (project_id,))
    if row is None or row["purged_at"]:
        raise not_found("Project", project_id)
    if user.role == "admin":
        return
    if row["created_by"] == user.id:
        require(user, role)
        return
    m = svc.db.one(
        "SELECT role FROM project_members WHERE project_id = ? AND user_id = ?",
        (project_id, user.id),
    )
    if m is None:
        raise not_found("Project", project_id)
    rank = {"viewer": 0, "reviewer": 1, "editor": 2, "admin": 3}
    if rank[m["role"]] < rank[role] or not user.at_least(role):
        raise ArchRenderError(
            ErrorCode.FORBIDDEN,
            f"This action needs '{role}' on the project.",
            "Ask the project owner.",
        )
