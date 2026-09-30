from __future__ import annotations

import pytest

from archrender.core.config import REPO_ROOT
from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.models.license_gate import DeploymentPolicy, LicenseAllowlist, LicenseGate
from archrender.models.manager import ModelManager
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import LicenseInfo, Registry, RegistryEntry

CFG = REPO_ROOT / "configs"


def _entry(**kw: object) -> RegistryEntry:
    base: dict[str, object] = {
        "name": "m",
        "role": "depth",
        "repo": "org/m",
        "license": LicenseInfo.model_validate({"id": "apache-2.0", "class": "permissive"}),
        "commercial_ok": True,
        "runtime": "torch",
    }
    base.update(kw)
    return RegistryEntry.model_validate(base)


def _gate(**policy: object) -> LicenseGate:
    p = {"jurisdictions": ["TR"], **policy}
    return LicenseGate(DeploymentPolicy.model_validate(p), LicenseAllowlist.load(CFG))


def test_permissive_allowed() -> None:
    assert _gate().allowed(_entry())


@pytest.mark.parametrize(
    ("kw", "reason"),
    [
        ({"commercial_ok": False}, "not commercially usable"),
        ({"license": LicenseInfo.model_validate({"id": "cc-by-nc-4.0", "class": "blocked"})}, "blocked list"),
        ({"caps": {"mau": "100000000"}}, "caps"),
        ({"license": LicenseInfo.model_validate({"id": "weird-1.0", "class": "permissive"})}, "not in the allowlist"),
    ],
)
def test_blocked_reasons(kw: dict[str, object], reason: str) -> None:
    reasons = _gate().reasons_blocked(_entry(**kw))
    assert any(reason in r for r in reasons), reasons


def test_territorial_exclusion_uses_jurisdictions() -> None:
    e = _entry(territorial_exclusions=["EU", "UK", "KR"])
    assert _gate().allowed(e)  # TR is not excluded
    assert not _gate(jurisdictions=["DE"]).allowed(e)  # DE ∈ EU
    assert not _gate(jurisdictions=["EU"]).allowed(e)


def test_conditional_requires_acceptance() -> None:
    lic = LicenseInfo.model_validate({"id": "sam-license", "class": "conditional", "terms_id": "sam-license-2025-11-19"})
    e = _entry(license=lic)
    assert not _gate().allowed(e)
    assert _gate(accepted_license_terms=["sam-license-2025-11-19"]).allowed(e)


def test_commercial_licence_override() -> None:
    lic = LicenseInfo.model_validate({"id": "flux-non-commercial", "class": "blocked"})
    e = _entry(license=lic, commercial_ok=False)
    assert not _gate().allowed(e)
    assert _gate(commercial_licenses={"flux-non-commercial": "ab" * 32}).allowed(e)


def test_repo_registry_profiles_are_all_allowed_and_blocked_entries_blocked() -> None:
    reg = Registry.load(CFG)
    gate = LicenseGate.load(CFG)
    for name in ("cpu_test", "gpu48", "gpu80", "gpu96plus"):
        prof = HardwareProfile.load(CFG, name)
        for role, model in prof.roles.items():
            e = reg.get(model)
            assert e.role == role
            gate.check(e)
    for blocked in ("trellis2-4b", "flux2-dev", "qwen-image-2.1", "hunyuan3d-2.1"):
        with pytest.raises(ArchRenderError) as err:
            gate.check(reg.get(blocked))
        assert err.value.code == ErrorCode.MODEL_LICENSE_BLOCKED


def test_depth_registry_only_apache_da3_ids() -> None:
    allowed = {"depth-anything/DA3-SMALL", "depth-anything/DA3-BASE", "depth-anything/DA3MONO-LARGE", "depth-anything/DA3METRIC-LARGE"}
    for e in Registry.load(CFG).entries():
        if e.repo and e.repo.startswith("depth-anything/"):
            assert e.repo in allowed, e.repo


def test_manager_loads_mocks_and_evicts_lru() -> None:
    reg = Registry.load(CFG)
    prof = HardwareProfile.load(CFG, "cpu_test")
    mgr = ModelManager(prof, reg, LicenseGate.load(CFG))
    r = mgr.get("refiner")
    assert mgr.get("refiner") is r
    assert mgr.ref("depth").mock
    prof2 = prof.model_copy(update={"vram_budget_gb": 5.0})
    reg2 = Registry([e.model_copy(update={"vram_gb": 3.0}) if e.mock else e for e in reg.entries()])
    mgr2 = ModelManager(prof2, reg2, LicenseGate.load(CFG))
    mgr2.get("refiner")
    mgr2.get("depth")  # 3 + 3 > 5 → refiner evicted
    assert mgr2.used_vram_gb() == 3.0
