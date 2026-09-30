#!/usr/bin/env python3
"""Fail if production code contains stub markers.

Mocks are allowed only where they are declared as mock implementations of a model role
(``archrender/models/mocks``) or under ``tests/``; everywhere else ``TODO``/``FIXME``/``XXX`` and
``NotImplementedError`` placeholders are rejected.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"\b(TODO|FIXME|XXX)\b|raise NotImplementedError\b")
SCAN = [ROOT / "src", ROOT / "deploy", ROOT / "scripts", ROOT / "ui" / "src"]
SUFFIXES = {".py", ".ts", ".tsx", ".sh", ".yaml", ".yml"}


def main() -> int:
    bad: list[str] = []
    for base in SCAN:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix not in SUFFIXES or not path.is_file() or path.name == Path(__file__).name:
                continue
            for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if PATTERN.search(line):
                    bad.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
