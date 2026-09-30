"""Hourly consistent DB snapshot to the volume (second recovery path next to Litestream, ADR-S03)."""

from __future__ import annotations

import sys
import time

from archrender.core.config import get_settings
from archrender.db.database import Database

KEEP = 48


def main() -> int:
    s = get_settings()
    snaps = s.data_dir / "db" / "snapshots"
    dest = snaps / f"archrender-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.sqlite"
    Database(s.db_path).snapshot_to(dest)
    for old in sorted(snaps.glob("archrender-*.sqlite"))[:-KEEP]:
        old.unlink(missing_ok=True)
    print(f"snapshot {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
