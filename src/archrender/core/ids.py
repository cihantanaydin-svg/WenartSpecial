"""Identifier generation and timestamps."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime


def new_id(prefix: str) -> str:
    """Random identifier like ``job_4f1c2a9be07d3c11`` (safe as a path component)."""
    return f"{prefix}_{secrets.token_hex(8)}"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)
