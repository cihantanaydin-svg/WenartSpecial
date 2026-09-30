"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextvars import ContextVar
from typing import Any

_context: ContextVar[dict[str, Any] | None] = ContextVar("archrender_log_context", default=None)

# API keys, Hugging Face tokens, GitHub tokens, RunPod keys, bearer headers.
_SECRET_RE = re.compile(
    r"(ark_[A-Za-z0-9]+_[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{10,}|gh[pousr]_[A-Za-z0-9]{10,}"
    r"|rpa_[A-Za-z0-9]{10,}|(?i:bearer)\s+[A-Za-z0-9._~+/=-]+)"
)


def redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


def bind(**fields: Any) -> None:
    """Attach fields (job_id, stage, project, view…) to every log record in this context."""
    current = dict(_context.get() or {})
    current.update(fields)
    _context.set(current)


def unbind(*keys: str) -> None:
    current = dict(_context.get() or {})
    for k in keys:
        current.pop(k, None)
    _context.set(current)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": redact(record.getMessage()),
        }
        payload.update(_context.get() or {})
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
