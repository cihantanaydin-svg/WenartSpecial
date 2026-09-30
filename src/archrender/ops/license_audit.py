"""Licence audit (principle 6): Python packages (per environment), bundled npm packages, models,
subprocess tools. Fails on anything not allow-listed and writes THIRD_PARTY_LICENSES.md.

    python -m archrender.ops.license_audit [--python /opt/venv-vllm/bin/python ...] [--prod-only]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from archrender.core.config import get_settings
from archrender.models.license_gate import LicenseGate
from archrender.models.profiles import HardwareProfile
from archrender.models.registry import Registry

CLASSIFIER_SPDX = {
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT",
    "BSD License": "BSD",
    "Apache Software License": "Apache-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "ISC License (ISCL)": "ISC",
    "The Unlicense (Unlicense)": "Unlicense",
    "Historical Permission Notice and Disclaimer (HPND)": "HPND",
    "zlib/libpng License": "Zlib",
    "Boost Software License 1.0 (BSL-1.0)": "BSL-1.0",
    "GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.1-or-later",
    "GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "GNU Library or Lesser General Public License (LGPL)": "LGPL-2.1-or-later",
    "GNU General Public License v2 (GPLv2)": "GPL-2.0",
    "GNU General Public License v3 (GPLv3)": "GPL-3.0",
    "GNU Affero General Public License v3": "AGPL-3.0",
    "CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
}

TEXT_SPDX = [
    (re.compile(r"^\s*(the\s+)?mit(\s+license)?\s*$", re.I), "MIT"),
    (re.compile(r"^\s*apache(\s+license)?(,?\s*version)?\s*2(\.0)?\s*$", re.I), "Apache-2.0"),
    (re.compile(r"^\s*apache[- ]2(\.0)?\s*$", re.I), "Apache-2.0"),
    (re.compile(r"^\s*bsd[- ]3[- ]clause\s*$", re.I), "BSD-3-Clause"),
    (re.compile(r"^\s*bsd[- ]2[- ]clause\s*$", re.I), "BSD-2-Clause"),
    (re.compile(r"^\s*(new |modified |3-clause )?bsd( license)?\s*$", re.I), "BSD-3-Clause"),
    (re.compile(r"^\s*psfl?(-2\.0)?\s*$", re.I), "PSF-2.0"),
    (re.compile(r"^\s*isc( license)?\s*$", re.I), "ISC"),
    (re.compile(r"^\s*mpl[- ]2\.0\s*$", re.I), "MPL-2.0"),
]

DUMP = r"""
import json, importlib.metadata as m
out = []
for d in m.distributions():
    md = d.metadata
    out.append({"name": (md.get("Name") or "").lower().replace("_", "-"), "version": md.get("Version"),
                "expression": md.get("License-Expression"), "license": (md.get("License") or "")[:200],
                "classifiers": [c for c in (md.get_all("Classifier") or []) if c.startswith("License ::")]})
print(json.dumps(out))
"""


@dataclass
class Pkg:
    name: str
    version: str
    license: str
    source: str
    env: str


def spdx_of(dist: dict[str, Any]) -> str:
    if dist.get("expression"):
        return str(dist["expression"])
    lic = (dist.get("license") or "").strip()
    if lic and "\n" not in lic and len(lic) < 80:
        for rx, spdx in TEXT_SPDX:
            if rx.match(lic):
                return spdx
        if re.fullmatch(r"[A-Za-z0-9.\-+ ()]+", lic):
            return lic
    for c in dist.get("classifiers", []):
        leaf = c.split("::")[-1].strip()
        if leaf in CLASSIFIER_SPDX:
            return CLASSIFIER_SPDX[leaf]
    return "UNKNOWN"


def allowed_expression(expr: str, allowed: set[str], blocked: set[str]) -> bool:
    e = expr.replace("(", " ").replace(")", " ")
    ors = [p.strip() for p in re.split(r"\s+OR\s+", e) if p.strip()]
    if len(ors) > 1:
        return any(allowed_expression(o, allowed, blocked) for o in ors)
    ands = [p.strip() for p in re.split(r"\s+AND\s+", e) if p.strip()]
    if len(ands) > 1:
        return all(allowed_expression(a, allowed, blocked) for a in ands)
    lic = e.strip()
    if lic in blocked:
        return False
    if lic in allowed:
        return True
    base = lic.split(" WITH ")[0].strip()
    return base in allowed or (base == "BSD" and "BSD-3-Clause" in allowed)


def dump_env(python: str | None) -> list[dict[str, Any]]:
    cmd = [python or sys.executable, "-c", DUMP]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120).stdout
    data: list[dict[str, Any]] = json.loads(out)
    return data


def prod_names() -> set[str] | None:
    """Runtime dependency closure (no dev group) from uv, when uv is available."""
    try:
        out = subprocess.run(
            [
                "uv",
                "export",
                "--no-dev",
                "--no-hashes",
                "--frozen",
                "--no-emit-project",
                "--format",
                "requirements-txt",
            ],
            cwd=get_settings().app_root(),
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    names = set()
    for line in out.splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.add(re.split(r"[=<>;\[ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


SUBPROCESS_EXTRA = {
    "GPL-2.0",
    "GPL-2.0-only",
    "GPL-2.0-or-later",
    "GPL-3.0",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
}


def audit(
    pythons: list[str | None], prod_only: bool, subprocess_pythons: list[str] | None = None
) -> tuple[list[Pkg], list[str], dict[str, Any]]:
    configs = get_settings().configs_dir
    cfg = yaml.safe_load((configs / "licenses.yaml").read_text())["packages"]
    allowed, blocked = set(cfg["allowed"]), set(cfg["blocked"])
    blocked_names = {n.lower() for n in cfg["blocked_names"]}
    verified = {k.lower(): v for k, v in (cfg.get("verified") or {}).items()}
    problems: list[str] = []
    pkgs: list[Pkg] = []
    keep = prod_names() if prod_only else None
    envs: list[tuple[str | None, bool]] = [(py, False) for py in pythons or [None]]
    envs += [(py, True) for py in subprocess_pythons or []]
    for py, arms_length in envs:
        env = (py or "app") + (" (subprocess only)" if arms_length else "")
        env_allowed = allowed | SUBPROCESS_EXTRA if arms_length else allowed
        env_blocked = blocked - SUBPROCESS_EXTRA if arms_length else blocked
        for d in dump_env(py):
            name = d["name"]
            if not name or name == "archrender":
                continue
            if keep is not None and py is None and name not in keep:
                continue
            lic = verified[name]["license"] if name in verified else spdx_of(d)
            pkgs.append(
                Pkg(
                    name,
                    d["version"] or "?",
                    lic,
                    "verified" if name in verified else "metadata",
                    env,
                )
            )
            if name in blocked_names:
                problems.append(f"{env}: {name} {d['version']} is explicitly blocked ({lic})")
            elif lic == "UNKNOWN":
                problems.append(
                    f"{env}: {name} {d['version']} has no machine-readable licence; verify and add to licenses.yaml"
                )
            elif not allowed_expression(lic, env_allowed, env_blocked):
                problems.append(f"{env}: {name} {d['version']} licence '{lic}' is not allowed")
    npm: list[Pkg] = []
    lock = get_settings().app_root() / "ui" / "package-lock.json"
    if lock.exists():
        data = json.loads(lock.read_text())
        for path, meta in data.get("packages", {}).items():
            if not path or meta.get("dev"):
                continue
            name = path.split("node_modules/")[-1]
            lic = meta.get("license", "UNKNOWN")
            npm.append(Pkg(name, meta.get("version", "?"), lic, "package-lock", "ui"))
            if not allowed_expression(lic, allowed, blocked):
                problems.append(f"ui: npm {name} licence '{lic}' is not allowed")
    registry = Registry.load(configs)
    gate = LicenseGate.load(configs)
    used: dict[str, list[str]] = {}
    for prof in ("cpu_test", "gpu48", "gpu80", "gpu96plus"):
        for role, name in HardwareProfile.load(configs, prof).roles.items():
            used.setdefault(name, []).append(f"{prof}:{role}")
    models = []
    for e in registry.entries():
        reasons = gate.reasons_blocked(e)
        models.append({"entry": e, "used_by": used.get(e.name, []), "blocked": reasons})
        if e.name in used and reasons:
            problems.append(f"model {e.name} is used by {used[e.name]} but blocked: {reasons}")
    return pkgs + npm, problems, {"models": models, "tools": cfg.get("subprocess_tools", {})}


def write_report(pkgs: list[Pkg], extra: dict[str, Any], path: Path) -> None:
    lines = [
        "# Third-party licences",
        "",
        "Generated by `python -m archrender.ops.license_audit`. Do not edit by hand.",
        "",
        "## Models",
        "",
        "| Model | Repo | Licence | Class | Status | Used by | Evidence | Obligations |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in extra["models"]:
        e = m["entry"]
        status = "blocked: " + "; ".join(m["blocked"]) if m["blocked"] else "allowed"
        lines.append(
            f"| {e.name} | {e.repo or '-'} | {e.license.id} | {e.license.license_class} | {status} | "
            f"{', '.join(m['used_by']) or '-'} | {e.license.evidence_url or '-'} | "
            f"{'; '.join(e.license.obligations) or '-'} |"
        )
    lines += ["", "## Tools run as separate processes", "", "| Tool | Licence |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in extra["tools"].items()]
    lines += [
        "",
        "## Packages",
        "",
        "| Environment | Package | Version | Licence | Source |",
        "|---|---|---|---|---|",
    ]
    for p in sorted(pkgs, key=lambda x: (x.env, x.name)):
        lines.append(f"| {p.env} | {p.name} | {p.version} | {p.license} | {p.source} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--python", action="append", default=[], help="extra interpreters (vLLM, Blender venvs)"
    )
    ap.add_argument(
        "--subprocess-python",
        action="append",
        default=[],
        help="interpreters that only ever run as separate processes (Blender): GPL allowed, AGPL not",
    )
    ap.add_argument(
        "--prod-only", action="store_true", help="skip dev-only packages of the app env"
    )
    ap.add_argument("--out", default=str(get_settings().app_root() / "THIRD_PARTY_LICENSES.md"))
    args = ap.parse_args(argv)
    pythons: list[str | None] = [None, *args.python]
    pkgs, problems, extra = audit(pythons, args.prod_only, args.subprocess_python)
    write_report(pkgs, extra, Path(args.out))
    print(f"audited {len(pkgs)} packages and {len(extra['models'])} models → {args.out}")
    for p in problems:
        print(f"LICENCE VIOLATION: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
