#!/usr/bin/env python3
"""Export JSON Schema (draft 2020-12) for every public pydantic schema into schemas/.

    python scripts/export_schemas.py            # write schemas/*.schema.json
    python scripts/export_schemas.py --check    # exit 1 if the committed files are stale

The pydantic models in ``archrender.core.schemas`` are the single source of truth; the exported
files are for external consumers (the UI, integrators) and are checked for freshness in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from archrender.core.schemas import EXPORTED_SCHEMAS  # noqa: E402

OUT = ROOT / "schemas"


def render() -> dict[str, str]:
    files = {}
    for name, model in sorted(EXPORTED_SCHEMAS.items()):
        schema = model.model_json_schema(mode="serialization")
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://archrender.local/schemas/{name}.schema.json",
            **schema,
        }
        files[f"{name}.schema.json"] = json.dumps(schema, indent=2, ensure_ascii=False) + "\n"
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if schemas/ is out of date")
    args = ap.parse_args(argv)
    files = render()
    existing = {p.name for p in OUT.glob("*.schema.json")} if OUT.exists() else set()
    stale = sorted(
        n
        for n, text in files.items()
        if not (OUT / n).exists() or (OUT / n).read_text(encoding="utf-8") != text
    )
    extra = sorted(existing - files.keys())
    if args.check:
        for n in stale:
            print(f"stale: schemas/{n}", file=sys.stderr)
        for n in extra:
            print(f"no longer exported: schemas/{n}", file=sys.stderr)
        if stale or extra:
            print("run `make schemas`", file=sys.stderr)
            return 1
        print(f"{len(files)} schemas up to date")
        return 0
    OUT.mkdir(exist_ok=True)
    for n, text in files.items():
        (OUT / n).write_text(text, encoding="utf-8")
    for n in extra:
        (OUT / n).unlink()
    print(f"wrote {len(files)} schemas to {OUT.relative_to(ROOT)}/ ({len(stale)} changed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
