"""API keys (HMAC-hashed, peppered), sessions (httpOnly cookie + CSRF), roles, audit."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.ids import new_id, now_iso
from archrender.db.database import Database

Role = Literal["admin", "editor", "reviewer", "viewer"]
ROLE_RANK: dict[str, int] = {"viewer": 0, "reviewer": 1, "editor": 2, "admin": 3}
KEY_PREFIX = "ark"


@dataclass(frozen=True)
class User:
    id: str
    name: str
    role: Role

    def at_least(self, role: Role) -> bool:
        return ROLE_RANK[self.role] >= ROLE_RANK[role]


def hash_secret(secret: str, pepper: bytes) -> str:
    return hmac.new(pepper, secret.encode(), hashlib.sha256).hexdigest()


def make_key() -> tuple[str, str, str]:
    """Return ``(key_id, secret, token)``; the token is shown to the user exactly once."""
    key_id = secrets.token_hex(6)
    secret = secrets.token_urlsafe(32).replace("_", "x").replace("-", "y")
    return key_id, secret, f"{KEY_PREFIX}_{key_id}_{secret}"


def parse_token(token: str) -> tuple[str, str] | None:
    parts = token.strip().split("_", 2)
    if len(parts) != 3 or parts[0] != KEY_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def create_user(db: Database, name: str, role: Role) -> User:
    uid = new_id("usr")
    try:
        db.execute(
            "INSERT INTO users(id, name, role, created_at) VALUES (?,?,?,?)",
            (uid, name, role, now_iso()),
        )
    except Exception as e:  # sqlite3.IntegrityError
        raise ArchRenderError(
            ErrorCode.CONFLICT, f"User {name!r} already exists.", "Pick another name."
        ) from e
    return User(uid, name, role)


def issue_key(db: Database, user_id: str, label: str, pepper: bytes) -> str:
    key_id, secret, token = make_key()
    db.execute(
        "INSERT INTO api_keys(id, user_id, label, hash, created_at) VALUES (?,?,?,?,?)",
        (key_id, user_id, label, hash_secret(secret, pepper), now_iso()),
    )
    return token


def user_from_key(db: Database, token: str, pepper: bytes) -> User | None:
    parsed = parse_token(token)
    if parsed is None:
        return None
    key_id, secret = parsed
    row = db.one(
        "SELECT k.hash, k.revoked_at, u.id, u.name, u.role, u.disabled FROM api_keys k JOIN users u ON u.id = k.user_id"
        " WHERE k.id = ?",
        (key_id,),
    )
    if row is None or row["revoked_at"] or row["disabled"]:
        return None
    if not hmac.compare_digest(row["hash"], hash_secret(secret, pepper)):
        return None
    db.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now_iso(), key_id))
    return User(row["id"], row["name"], row["role"])


def create_session(db: Database, user: User, ttl_s: int) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    expires = datetime.now(UTC) + timedelta(seconds=ttl_s)
    db.execute(
        "INSERT INTO sessions(id_hash, user_id, csrf, created_at, expires_at) VALUES (?,?,?,?,?)",
        (hashlib.sha256(token.encode()).hexdigest(), user.id, csrf, now_iso(), expires.isoformat()),
    )
    return token, csrf


def user_from_session(db: Database, token: str) -> tuple[User, str] | None:
    row = db.one(
        "SELECT s.csrf, s.expires_at, u.id, u.name, u.role, u.disabled FROM sessions s JOIN users u ON u.id = s.user_id"
        " WHERE s.id_hash = ?",
        (hashlib.sha256(token.encode()).hexdigest(),),
    )
    if row is None or row["disabled"]:
        return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.now(UTC):
        return None
    return User(row["id"], row["name"], row["role"]), row["csrf"]


def delete_session(db: Database, token: str) -> None:
    db.execute(
        "DELETE FROM sessions WHERE id_hash = ?", (hashlib.sha256(token.encode()).hexdigest(),)
    )


def audit(
    db: Database,
    user_id: str | None,
    action: str,
    target: str | None,
    detail: dict[str, object] | None = None,
    ip: str | None = None,
) -> None:
    db.execute(
        "INSERT INTO audit_log(ts, user_id, action, target, detail_json, ip) VALUES (?,?,?,?,?,?)",
        (now_iso(), user_id, action, target, json.dumps(detail or {}, default=str), ip),
    )


class FailureLimiter:
    """Per-client sliding-window limit on authentication failures (brute-force protection)."""

    def __init__(self, limit: int = 20, window_s: float = 300.0) -> None:
        self.limit = limit
        self.window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, client: str) -> None:
        q = self._hits[client]
        now = time.monotonic()
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise ArchRenderError(
                ErrorCode.RATE_LIMITED,
                "Too many failed authentication attempts.",
                "Wait a few minutes and check the API key.",
            )

    def fail(self, client: str) -> None:
        self._hits[client].append(time.monotonic())
