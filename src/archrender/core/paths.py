"""Path sanitization. Every filesystem path derived from external input goes through here."""

from __future__ import annotations

import re
from pathlib import Path

from archrender.core.errors import ArchRenderError, ErrorCode

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def check_id(value: str, what: str = "identifier") -> str:
    """Validate an identifier used as a path component (no separators, no dots)."""
    if not _ID_RE.fullmatch(value):
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Invalid {what}: {value!r}.",
            "Identifiers contain only letters, digits, '-' and '_' (max 64 chars).",
        )
    return value


def check_sha256(value: str) -> str:
    if not _HEX_RE.fullmatch(value):
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Invalid SHA-256 digest: {value!r}.",
            "Provide a lowercase 64-character hex SHA-256 digest.",
        )
    return value


def safe_join(root: Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and guarantee the result stays inside ``root``."""
    base = root.resolve()
    candidate = base.joinpath(*parts).resolve()
    if candidate != base and base not in candidate.parents:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            "Path escapes its storage root.",
            "Remove '..', absolute paths and symlinks from the name.",
            context={"parts": list(parts)},
        )
    return candidate


_UNSAFE_NAME = re.compile(r"[^\w.\- ()+]", re.UNICODE)


def sanitize_filename(name: str, max_len: int = 180) -> str:
    """Return a display-safe base filename (keeps Unicode letters such as Turkish characters)."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1].strip()
    base = _UNSAFE_NAME.sub("_", base).lstrip(".")
    if not base:
        base = "file"
    if len(base) > max_len:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: max_len - len(ext) - 1] + "." + ext) if dot else base[:max_len]
    return base
