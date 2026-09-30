"""Render the supervisord configuration for the active profile (only what is implemented runs)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from archrender.core.config import get_settings
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry

PROGRAM = """
[program:{name}]
command={command}
directory=/opt/archrender
autostart={autostart}
autorestart=true
startsecs=3
stopsignal=TERM
stopwaitsecs=60
stdout_logfile={logs}/{name}.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=5
redirect_stderr=true
environment=PYTHONUNBUFFERED="1"
"""

HEADER = """[supervisord]
nodaemon=true
logfile={logs}/supervisord.log
pidfile=/tmp/supervisord.pid
user=root

[unix_http_server]
file=/tmp/supervisor.sock
chmod=0700

[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock
"""


def vllm_roles(profile: HardwareProfile, registry: Registry) -> list[tuple[str, str]]:
    """(role, model) pairs served by vLLM whose runtime implementation exists in this build."""
    out = []
    for role in ("vlm", "judge2"):
        name = profile.roles.get(role)
        if name is None:
            continue
        e = registry.get(name)
        if e.runtime.startswith("vllm") and e.impl is not None:
            out.append((role, name))
    return out


def render(
    profile: HardwareProfile, registry: Registry, logs: Path, *, venv: str, litestream: str | None
) -> str:
    py = f"{venv}/bin/python"
    parts = [HEADER.format(logs=logs)]
    parts.append(
        PROGRAM.format(
            name="api", command=f"{py} -m archrender.api.app", autostart="true", logs=logs
        )
    )
    parts.append(
        PROGRAM.format(
            name="worker-gpu",
            command=f"{py} -m archrender.pipeline.worker --queue gpu",
            autostart="true",
            logs=logs,
        )
    )
    parts.append(
        PROGRAM.format(
            name="worker-cpu",
            command=f"{py} -m archrender.pipeline.worker --queue cpu",
            autostart="true",
            logs=logs,
        )
    )
    if litestream:
        parts.append(
            PROGRAM.format(
                name="litestream",
                command=f"{litestream} replicate -config /opt/archrender/deploy/litestream.yml",
                autostart="true",
                logs=logs,
            )
        )
    for i, (role, name) in enumerate(vllm_roles(profile, registry)):
        port = 8101 + i
        model_dir = f"/workspace/models/snapshots/{name}"
        cmd = (
            f"/opt/venv-vllm/bin/vllm serve {model_dir} --host 127.0.0.1 --port {port} --served-model-name {name}"
            " --enable-sleep-mode --max-model-len 32768"
        )
        parts.append(
            PROGRAM.format(
                name=f"vllm-{role}",
                command=cmd,
                autostart="true" if role == "vlm" else "false",
                logs=logs,
            )
        )
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--venv", default="/opt/venv")
    args = p.parse_args(argv)
    s = get_settings()
    text = render(
        HardwareProfile.load(s.configs_dir, s.profile),
        Registry.load(s.configs_dir),
        s.data_dir / "logs",
        venv=args.venv,
        litestream=shutil.which("litestream"),
    )
    Path(args.out).write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
