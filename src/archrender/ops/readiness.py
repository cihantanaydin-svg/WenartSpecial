"""Readiness checks behind ``/readyz`` (and the boot self-test)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from archrender.core.errors import ArchRenderError
from archrender.pipeline.services import Services


@dataclass
class ReadinessState:
    checked_at: float = 0.0
    ready: bool = False
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    blender_probe: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


# Roles the implemented stages call (S1 OCR, S8 refine, S9 QA). A phase that adds a stage using
# another role adds it here, so readiness never blocks on models no stage uses yet.
PIPELINE_ROLES = ("ocr", "refiner", "depth", "segmenter")


def models_present(svc: Services) -> tuple[bool, list[str], list[str]]:
    """(ok, missing, degraded). A role is *missing* when its usable model is not installed; it is
    *degraded* when neither the model nor its fallbacks have a runtime in this build and the role
    runs on its mock (outputs flagged as mock, Gate D never auto-passes)."""
    missing: list[str] = []
    degraded: list[str] = []
    root = svc.settings.data_dir / "models" / "installed"
    for role in svc.profile.roles:
        if role not in PIPELINE_ROLES:
            continue
        chain = svc.models.chain(role)
        usable = next((e for e in chain if e.impl is not None), None)
        if usable is None:
            missing.append(f"{role}: no runtime in this build and no mock")
            continue
        if usable is not chain[0]:
            degraded.append(f"{role}: {chain[0].name} → {usable.name}")
        if usable.mock:
            continue
        if usable.runtime.startswith("system:"):
            # system packages have no weights to download: probe the tool itself
            try:
                svc.models.get_with_fallback(role, "readiness")
            except ArchRenderError as e:
                missing.append(f"{role}:{usable.name} ({e.message})")
            continue
        if not (root / f"{usable.name}.ok").exists():
            missing.append(f"{role}:{usable.name}")
    return not missing, missing, degraded


def evaluate(svc: Services, state: ReadinessState, *, max_age_s: float = 30.0) -> ReadinessState:
    with state.lock:
        if time.time() - state.checked_at < max_age_s and state.checks:
            return state
        checks: dict[str, dict[str, Any]] = {}
        try:
            svc.db.one("SELECT COUNT(*) FROM schema_migrations")
            checks["database"] = {"ok": True}
        except Exception as e:
            checks["database"] = {"ok": False, "error": str(e)}
        if state.blender_probe is None:
            try:
                state.blender_probe = svc.blender.probe(svc.settings.data_dir / "cache" / "probe")
            except ArchRenderError as e:
                state.blender_probe = {"error": e.message, "fix_hint": e.fix_hint}
        probe = state.blender_probe or {}
        needs_gpu = svc.profile.render.device == "GPU"
        blender_ok = "error" not in probe and (not needs_gpu or probe.get("device") == "GPU")
        checks["blender"] = {"ok": blender_ok, **probe}
        ok, missing, degraded = models_present(svc)
        checks["models"] = {"ok": ok, "missing": missing, "degraded": degraded}
        selftest = svc.settings.data_dir / "cache" / "selftest.ok"
        checks["selftest"] = {
            "ok": svc.profile.name == "cpu_test" or selftest.exists(),
            "marker": str(selftest),
        }
        state.checks = checks
        state.ready = all(c["ok"] for c in checks.values())
        state.checked_at = time.time()
        return state


def write_selftest_marker(data_dir: Path) -> None:
    p = data_dir / "cache" / "selftest.ok"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(time.time()))
