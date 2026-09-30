"""License gate (principle 6): enforced at download, at load, and in CI.

Nothing loads unless its registry entry is commercial-OK for the configured jurisdictions. The rule
is strict: any non-commercial clause, territorial exclusion that intersects the jurisdictions,
revenue/MAU cap, blocked licence id or unaccepted conditional terms blocks the model.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from archrender.core.errors import ArchRenderError, ErrorCode
from archrender.models.registry import RegistryEntry

REGION_GROUPS: dict[str, set[str]] = {
    "EU": {
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT",
        "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    },
    "EEA": {"IS", "LI", "NO"},
}


def expand_regions(codes: list[str]) -> set[str]:
    out: set[str] = set()
    for c in codes:
        c = c.upper()
        out.add(c)
        out |= REGION_GROUPS.get(c, set())
    return out


class DeploymentPolicy(BaseModel):
    jurisdictions: list[str] = Field(min_length=1)
    accepted_license_terms: list[str] = Field(default_factory=list)
    commercial_licenses: dict[str, str] = Field(default_factory=dict)  # licence id → licence file sha256
    allow_revenue_or_mau_caps: bool = False
    datacenters_preferred: list[str] = Field(default_factory=list)
    training_use_of_client_documents: bool = False

    @classmethod
    def load(cls, configs_dir: Path) -> DeploymentPolicy:
        data = yaml.safe_load((configs_dir / "deployment.yaml").read_text(encoding="utf-8"))
        return cls.model_validate(data)


class LicenseAllowlist(BaseModel):
    permissive: list[str]
    conditional: list[str]
    blocked: list[str]

    @classmethod
    def load(cls, configs_dir: Path) -> LicenseAllowlist:
        data = yaml.safe_load((configs_dir / "licenses.yaml").read_text(encoding="utf-8"))
        return cls.model_validate(data["models"])


class LicenseGate:
    def __init__(self, policy: DeploymentPolicy, allowlist: LicenseAllowlist) -> None:
        self.policy = policy
        self.allowlist = allowlist
        self.jurisdictions = expand_regions(policy.jurisdictions)

    @classmethod
    def load(cls, configs_dir: Path) -> LicenseGate:
        return cls(DeploymentPolicy.load(configs_dir), LicenseAllowlist.load(configs_dir))

    def reasons_blocked(self, entry: RegistryEntry) -> list[str]:
        if entry.mock:
            return []
        lic = entry.license
        reasons: list[str] = []
        override = lic.id in self.policy.commercial_licenses
        if lic.id in self.allowlist.blocked and not override:
            reasons.append(f"licence {lic.id} is on the blocked list")
        if not entry.commercial_ok and not override:
            reasons.append("registry marks it not commercially usable")
        if lic.license_class == "blocked" and not override:
            reasons.append("licence class is 'blocked'")
        if lic.license_class == "conditional":
            if not lic.terms_id or lic.terms_id not in self.policy.accepted_license_terms:
                reasons.append(
                    f"conditional terms {lic.terms_id!r} not accepted in configs/deployment.yaml"
                )
        known = set(self.allowlist.permissive) | set(self.allowlist.conditional)
        if lic.license_class != "proprietary" and lic.id not in known and not override:
            reasons.append(f"licence {lic.id} is not in the allowlist")
        excluded = expand_regions(entry.territorial_exclusions) & self.jurisdictions
        if excluded:
            reasons.append(f"territorial exclusion covers {sorted(excluded)}")
        if entry.caps and not self.policy.allow_revenue_or_mau_caps:
            reasons.append(f"revenue/MAU caps present: {entry.caps}")
        return reasons

    def check(self, entry: RegistryEntry) -> None:
        reasons = self.reasons_blocked(entry)
        if reasons:
            raise ArchRenderError(
                ErrorCode.MODEL_LICENSE_BLOCKED,
                f"Model {entry.name} ({entry.repo}) is blocked: " + "; ".join(reasons) + ".",
                "Pick the registered fallback, accept the terms in configs/deployment.yaml if the "
                "firm's counsel agrees, or configure a purchased commercial licence.",
                context={"model": entry.name, "reasons": reasons},
            )

    def allowed(self, entry: RegistryEntry) -> bool:
        return not self.reasons_blocked(entry)
