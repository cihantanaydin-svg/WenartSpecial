"""Boot preflight (entrypoint): GPU/driver/CUDA, RAM and disk checks with clear, actionable errors."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from archrender.core.config import get_settings
from archrender.db.database import Database
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry

MIN_DRIVER_MAJOR = 580  # CUDA 13.0 runtime (torch 2.14 cu130, vLLM 0.30)


def _fail(msg: str, hint: str) -> int:
    print(f"PREFLIGHT FAILED: {msg}\n  → {hint}", file=sys.stderr)
    return 1


def gpu_info() -> list[dict[str, str]] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        name, driver, mem = (x.strip() for x in line.split(","))
        gpus.append({"name": name, "driver": driver, "memory_mib": mem})
    return gpus


def mem_total_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def required_disk_gb(profile: HardwareProfile, registry: Registry) -> float:
    return sum(registry.get(n).disk_gb for n in set(profile.roles.values())) + 25.0 + 17.0


def run(migrate: bool) -> int:
    s = get_settings()
    prof = HardwareProfile.load(s.configs_dir, s.profile)
    reg = Registry.load(s.configs_dir)
    print(f"profile {prof.name}: {prof.description}")
    if prof.render.device == "GPU":
        gpus = gpu_info()
        if not gpus:
            return _fail(
                "No NVIDIA GPU visible (nvidia-smi missing or failing).",
                "Deploy on a GPU pod; locally use ARCHRENDER_PROFILE=cpu_test.",
            )
        g = gpus[0]
        print(f"gpu: {g['name']} · driver {g['driver']} · {int(g['memory_mib']) / 1024:.0f} GiB")
        if int(g["driver"].split(".")[0]) < MIN_DRIVER_MAJOR:
            return _fail(
                f"Host driver {g['driver']} is older than {MIN_DRIVER_MAJOR} (needed for CUDA 13.0).",
                "Redeploy with minCudaVersion 13.0 (deploy.py does this) so RunPod picks a newer host.",
            )
        if int(g["memory_mib"]) / 1024 < prof.vram_budget_gb:
            return _fail(
                f"GPU has {int(g['memory_mib']) / 1024:.0f} GiB but profile {prof.name} budgets "
                f"{prof.vram_budget_gb} GB.",
                "Use a smaller profile (ARCHRENDER_PROFILE) or a larger GPU type.",
            )
    ram = mem_total_gb()
    print(f"system RAM: {ram:.0f} GiB")
    if ram and ram < prof.min_system_ram_gb * 0.95:
        return _fail(
            f"System RAM {ram:.0f} GiB is below the profile minimum {prof.min_system_ram_gb:.0f} GB.",
            "Pick a GPU type with more host RAM (deploy.py sets minRamPerGpu) or a smaller profile.",
        )
    s.data_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(s.data_dir).free / 1e9
    need = required_disk_gb(prof, reg) if prof.name != "cpu_test" else 1.0
    print(f"/workspace free: {free:.0f} GB (models+assets+caches need ≈ {need:.0f} GB)")
    if free < need * 0.5:
        return _fail(
            f"Only {free:.0f} GB free on {s.data_dir}; ≈ {need:.0f} GB are needed.",
            "Grow the network volume (deploy.py resizes it) or delete old projects.",
        )
    if migrate:
        Database(s.db_path).migrate()
        print(f"database migrated: {s.db_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--migrate", action="store_true")
    return run(ap.parse_args(argv).migrate)


if __name__ == "__main__":
    sys.exit(main())
