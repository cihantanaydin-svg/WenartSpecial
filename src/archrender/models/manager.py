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
        self._resolved: dict[str, RegistryEntry] = {}
        self._unavailable: dict[str, tuple[float, ArchRenderError]] = {}  # role → (until, error)
        self.unavailable_ttl_s = 600.0

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
        """The model actually used for ``role`` (after fallbacks), else the profile's choice."""
        return (self._resolved.get(role) or self.entry_for(role)).ref()

    def chain(self, role: str) -> list[RegistryEntry]:
        """Profile model, its fallbacks, then the role's mock (if the registry has one)."""
        out = [self.entry_for(role)]
        while out[-1].fallback and out[-1].fallback not in {e.name for e in out}:
            out.append(self.registry.get(out[-1].fallback))
        mock = f"mock-{role}"
        if mock in self.registry.names() and mock not in {e.name for e in out}:
            out.append(self.registry.get(mock))
        return out

    def any_mock_used(self) -> bool:
        return any(e.mock for e in self._resolved.values())

    def used_vram_gb(self) -> float:
        return sum(self._vram.values())

    def get(self, role: str) -> Any:
        return self._load(self.entry_for(role))

    def _load(self, entry: RegistryEntry) -> Any:
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

    def get_with_fallback(self, role: str, stage: str) -> tuple[Any, RegistryEntry]:
        """The profile's model for ``role``, else the next usable entry of :meth:`chain`.

        A model is unusable if it has no runtime in this build, is licence-blocked or fails to
        load (e.g. a missing system package). Each step down the chain, including the final step
        to the role's mock, is recorded once as a degradation; results produced by a mock are
        flagged as such downstream (QA checks, reports, manifests).
        """
        if role in self._resolved:
            entry = self._resolved[role]
            return self._load(entry), entry
        cached = self._unavailable.get(role)
        if cached and cached[0] > time.monotonic():
            raise cached[1]  # recently found unusable: do not wait for it again on every page
        tried: list[str] = []
        last: ArchRenderError | None = None
        for entry in self.chain(role):
            try:
                impl = self._load(entry)
            except ArchRenderError as e:
                if e.code not in (ErrorCode.MODEL_NOT_DOWNLOADED, ErrorCode.MODEL_LICENSE_BLOCKED):
                    raise
                tried.append(f"{entry.name} ({e.message})")
                last = e
                continue
            if tried:
                self.record_degradation(
                    stage, f"{role}: {tried[0].split(' ')[0]} → {entry.name}", "; ".join(tried)
                )
            self._resolved[role] = entry
            return impl, entry
        err = ArchRenderError(
            ErrorCode.MODEL_NOT_DOWNLOADED,
            f"No usable model for role {role!r}: tried {'; '.join(tried)}.",
            last.fix_hint if last else "Check the profile's role mapping.",
        )
        self._unavailable[role] = (time.monotonic() + self.unavailable_ttl_s, err)
        raise err

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
