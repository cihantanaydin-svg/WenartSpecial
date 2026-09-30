"""Canonical JSON and hashing helpers used for content addressing and cache keys."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_CHUNK = 1 << 20


def _normalize(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _normalize(obj.model_dump(mode="json"))
    if isinstance(obj, Mapping):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float):
        # Stable float formatting: 12 significant digits removes noise from float arithmetic.
        return float(f"{obj:.12g}")
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, Sequence):
        return [_normalize(v) for v in obj]
    raise TypeError(f"cannot canonicalize object of type {type(obj).__name__}")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON encoding (sorted keys, no whitespace, normalized floats)."""
    return json.dumps(
        _normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()
