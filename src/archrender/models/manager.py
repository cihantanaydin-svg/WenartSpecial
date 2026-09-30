"""ModelManager: the GPU worker's VRAM arbiter (principle 8).

Loads the implementation for a role per the active hardware profile, after the licence gate.
Accounts VRAM with the declared estimate (replaced by the measured peak where available) and evicts
least-recently-used models to stay within the profile budget. Degradations applied on CUDA OOM
are recorded for the run manifest.
"""

from __future__ import annotations

import importlib
import time
from collections import OrderedDict
from typing import Any, cast

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.core.schemas.common import ModelRef
from archrender.core.schemas.manifest import Degradation
from archrender.models.license_gate import LicenseGate
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry, RegistryEntry
from archrender.models.roles import ModelImpl


def _import(path: str) -> type[Any]:
    module, _, attr = path.partition(":")
    return cast(type[Any], getattr(importlib.import_module(module), attr))


class ModelManager:
    def __init__(self, profile: HardwareProfile, registry: Registry, gate: LicenseGate) -> None:
        self.profile = profile
        self.registry = registry
        self.gate = gate
        self._loaded: OrderedDict[str, tuple[RegistryEntry, ModelImpl]] = OrderedDict()
        self._vram: dict[str, float] = {}
        self.degradations: list[Degradation] = []
        self.load_seconds: dict[str, float] = {}

    def entry_for(self, role: str) -> RegistryEntry:
        name = self.profile.roles.get(role)
        if name is None:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Profile {self.profile.name} assigns no model to role {role!r}.",
                "Add the role to configs/profiles/<profile>.yaml.",
            )
        entry = self.registry.get(name)
        if entry.role != role:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Registry entry {name} has role {entry.role}, not {role}.",
                "Fix the profile's role mapping.",
            )
        return entry

    def ref(self, role: str) -> ModelRef:
        return self.entry_for(role).ref()

    def used_vram_gb(self) -> float:
        return sum(self._vram.values())

    def get(self, role: str) -> Any:
        entry = self.entry_for(role)
        if entry.name in self._loaded:
            self._loaded.move_to_end(entry.name)
            return self._loaded[entry.name][1]
        self.gate.check(entry)
        if entry.impl is None:
            raise ArchRenderError(
                ErrorCode.MODEL_NOT_DOWNLOADED,
                f"Model {entry.name} has registered weights but no runtime implementation in this build.",
                "Use a profile whose role mapping points at an implemented model (see PROGRESS.md).",
            )
        self._make_room(entry.vram_gb, keep=entry.name)
        impl = cast(ModelImpl, _import(entry.impl)(entry))
        t0 = time.time()
        impl.load()
        self.load_seconds[entry.name] = round(time.time() - t0, 3)
        self._loaded[entry.name] = (entry, impl)
        self._vram[entry.name] = entry.vram_gb
        return impl

    def _make_room(self, need_gb: float, keep: str) -> None:
        budget = self.profile.vram_budget_gb
        if need_gb > budget:
            raise ArchRenderError(
                ErrorCode.VRAM_BUDGET_EXCEEDED,
                f"Model needs {need_gb:.1f} GB but the {self.profile.name} budget is {budget:.1f} GB.",
                "Use a quantized variant (after its QA A/B) or a larger GPU profile.",
            )
        while self.used_vram_gb() + need_gb > budget and self._loaded:
            name, (_entry, impl) = next(iter(self._loaded.items()))
            if name == keep:
                break
            impl.unload()
            del self._loaded[name]
            self._vram.pop(name, None)

    def record_degradation(self, stage: str, rung: str, reason: str) -> None:
        self.degradations.append(Degradation(stage=stage, rung=rung, reason=reason))

    def unload_all(self) -> None:
        for _entry, impl in self._loaded.values():
            impl.unload()
        self._loaded.clear()
        self._vram.clear()
