"""Run untrusted-input parsers and external tools in a separate, resource-limited process.

Two entry points:

- :func:`run_task` runs one of the Python parsers in :mod:`archrender.ingest.tasks` (PDF, images,
  DXF, Office, ZIP) as ``python -m archrender.ingest.tasks <task>`` with a JSON request on stdin
  and a JSON reply on stdout.
- :func:`run_tool` runs an external binary (LibreDWG ``dwg2dxf``, libheif ``heif-dec``).

Both apply memory / CPU / file-size limits through :mod:`archrender.ingest.limits`, use a minimal
environment and a scratch working directory, and turn every failure (crash, limit, timeout,
parser error) into an :class:`ArchRenderError` with a fix hint.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from archrender.core.config import Settings
from archrender.core.errors import ArchRenderError, ErrorCode

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _limits_prefix(settings: Settings) -> list[str]:
    return [
        sys.executable,
        "-m",
        "archrender.ingest.limits",
        "--mem-mb",
        str(settings.sandbox_memory_mb),
        "--cpu-s",
        str(settings.sandbox_cpu_s),
        "--fsize-mb",
        str(max(1024, settings.zip_max_total_bytes // (1024 * 1024))),
        "--",
    ]


def _env(workdir: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        **_THREAD_ENV,
    }
    # keep the interpreter's import path (editable dev installs) but nothing else from the parent
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return env


def _limit_error(what: str, returncode: int, settings: Settings) -> ArchRenderError | None:
    if returncode in (-signal.SIGXCPU, -signal.SIGKILL):
        return ArchRenderError(
            ErrorCode.INGEST_LIMIT_EXCEEDED,
            f"{what} exceeded the CPU-time limit ({settings.sandbox_cpu_s} s).",
            "Split the file (fewer pages/sheets per file) or ask an admin to raise "
            "ARCHRENDER_SANDBOX_CPU_S.",
        )
    if returncode == -signal.SIGXFSZ:
        return ArchRenderError(
            ErrorCode.INGEST_LIMIT_EXCEEDED,
            f"{what} tried to write more data than allowed.",
            "The file expands to an implausible size; check that it is a genuine drawing set.",
        )
    if returncode in (-signal.SIGSEGV, -signal.SIGABRT, -signal.SIGBUS):
        return ArchRenderError(
            ErrorCode.INGEST_CORRUPT,
            f"{what} crashed while reading the file (signal {-returncode}).",
            "The file is probably damaged. Re-export it from the authoring application.",
        )
    return None


def run_task(
    task: str, request: dict[str, Any], settings: Settings, workdir: Path
) -> dict[str, Any]:
    """Run ``archrender.ingest.tasks.<task>`` in a limited child process and return its result."""
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [*_limits_prefix(settings), sys.executable, "-m", "archrender.ingest.tasks", task]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(request),
            capture_output=True,
            text=True,
            cwd=workdir,
            env=_env(workdir),
            timeout=settings.sandbox_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ArchRenderError(
            ErrorCode.INGEST_LIMIT_EXCEEDED,
            f"Parsing ({task}) did not finish within {settings.sandbox_timeout_s:.0f} s.",
            "Split the file or ask an admin to raise ARCHRENDER_SANDBOX_TIMEOUT_S.",
        ) from e
    err = _limit_error(f"The {task} parser", proc.returncode, settings)
    if err is not None:
        raise err
    reply = _last_json_line(proc.stdout)
    if reply is None:
        raise ArchRenderError(
            ErrorCode.INGEST_PARSER_FAILED,
            f"The {task} parser exited with status {proc.returncode} and no result.",
            "This is a bug or an unexpected file; the log has details.",
            context={"stderr": proc.stderr[-4000:]},
        )
    if not reply.get("ok"):
        raise ArchRenderError(
            ErrorCode(reply.get("code", ErrorCode.INGEST_PARSER_FAILED)),
            str(reply.get("message", "parser failed")),
            str(reply.get("fix_hint", "")),
            context={"task": task, **reply.get("context", {})},
        )
    result = reply["result"]
    if not isinstance(result, dict):
        raise ArchRenderError(
            ErrorCode.INGEST_PARSER_FAILED, f"The {task} parser returned no result object.", ""
        )
    return result


def _last_json_line(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run_tool(
    argv: list[str], settings: Settings, workdir: Path, *, what: str, fix_hint: str
) -> subprocess.CompletedProcess[str]:
    """Run an external binary under the sandbox limits. Raises on limits/timeouts/crashes."""
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [*_limits_prefix(settings), *argv],
            capture_output=True,
            text=True,
            errors="replace",
            cwd=workdir,
            env=_env(workdir),
            timeout=settings.sandbox_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ArchRenderError(
            ErrorCode.INGEST_LIMIT_EXCEEDED,
            f"{what} did not finish within {settings.sandbox_timeout_s:.0f} s.",
            fix_hint,
        ) from e
    err = _limit_error(what, proc.returncode, settings)
    if err is not None:
        raise err
    return proc
