"""Blender runner: the only way the app talks to Blender (a subprocess; the app never imports bpy)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from archrender.core.config import Settings
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.logging import get_logger

log = get_logger(__name__)


def build_script_path() -> Path:
    return Path(str(resources.files("archrender_blender").joinpath("build.py")))


class BlenderRunner:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def command(self, args: list[str]) -> list[str]:
        script = str(build_script_path())
        s = self.settings
        if s.blender_mode == "module":
            if not s.blender_python.exists():
                raise ArchRenderError(
                    ErrorCode.BLENDER_NOT_FOUND,
                    f"Blender Python module runtime not found at {s.blender_python}.",
                    "Run `make setup-blender` (dev) or set ARCHRENDER_BLENDER_MODE=binary in the image.",
                )
            return [str(s.blender_python), script, "--", *args]
        binary = s.blender_bin if s.blender_bin.exists() else shutil.which("blender")
        if binary is None:
            raise ArchRenderError(
                ErrorCode.BLENDER_NOT_FOUND,
                f"Blender binary not found at {s.blender_bin} or on PATH.",
                "Install Blender 5.2 LTS at /opt/blender or set ARCHRENDER_BLENDER_BIN.",
            )
        return [
            str(binary),
            "-b",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            script,
            "--",
            *args,
        ]

    def run(
        self,
        args: list[str],
        *,
        cwd: Path,
        timeout_s: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> str:
        cmd = self.command(args)
        env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("CUDA", "NVIDIA", "OPTIX", "LD_", "PATH", "HOME", "LANG", "LC_", "TMP"))
        }
        env["PYTHONNOUSERSITE"] = "1"
        cache = self.settings.data_dir / "cache"
        env.setdefault("CUDA_CACHE_PATH", str(cache / "cuda"))
        env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
        env.setdefault("OPTIX_CACHE_PATH", str(cache / "optix"))
        env.setdefault("OPTIX_CACHE_MAXSIZE", str(4 * 1024**3))
        timeout = timeout_s or self.settings.blender_timeout_s
        log.info("blender start", extra={"fields": {"args": args}})
        t0 = time.time()
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        output: list[str] = []

        def _reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                output.append(line)

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        try:
            while proc.poll() is None:
                if time.time() - t0 > timeout:
                    proc.kill()
                    raise ArchRenderError(
                        ErrorCode.STAGE_TIMEOUT,
                        f"Blender exceeded {timeout:.0f} s.",
                        "Lower resolution/samples, or check the GPU device probe in /readyz.",
                        retryable=True,
                    )
                if cancelled is not None and cancelled():
                    proc.kill()
                    raise ArchRenderError(
                        ErrorCode.JOB_CANCELLED, "Render cancelled.", "Start a new run when ready."
                    )
                time.sleep(0.2)
        finally:
            reader.join(timeout=5)
        text = "".join(output)
        if proc.returncode != 0:
            tail = "\n".join(text.strip().splitlines()[-25:])
            raise ArchRenderError(
                ErrorCode.BLENDER_FAILED,
                f"Blender exited with code {proc.returncode}.",
                "See the Blender log excerpt in the job events; a scene/geometry error names the object.",
                context={"log_tail": tail, "args": args},
            )
        log.info("blender done", extra={"fields": {"seconds": round(time.time() - t0, 2)}})
        return text

    def probe(self, workdir: Path) -> dict[str, Any]:
        workdir.mkdir(parents=True, exist_ok=True)
        self.run(["probe", str(workdir)], cwd=workdir, timeout_s=300)
        data: dict[str, Any] = json.loads((workdir / "probe.json").read_text())
        return data
