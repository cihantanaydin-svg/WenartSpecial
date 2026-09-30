"""SQLite access: WAL mode, per-thread connections, explicit transactions, numbered migrations."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

_MIGRATIONS_PKG = "archrender.db.migrations"


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)

    def conn(self) -> sqlite3.Connection:
        c: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=True)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """A transaction. ``immediate=True`` takes the write lock up front (job leasing)."""
        c = self.conn()
        c.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield c
        except BaseException:
            c.execute("ROLLBACK")
            raise
        else:
            c.execute("COMMIT")

    def query(self, sql: str, params: tuple[object, ...] | dict[str, object] = ()) -> list[sqlite3.Row]:
        return list(self.conn().execute(sql, params).fetchall())

    def one(self, sql: str, params: tuple[object, ...] | dict[str, object] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn().execute(sql, params).fetchone()
        return row

    def execute(self, sql: str, params: tuple[object, ...] | dict[str, object] = ()) -> int:
        cur = self.conn().execute(sql, params)
        return cur.rowcount

    def migrate(self) -> list[int]:
        c = self.conn()
        c.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)")
        applied = {r[0] for r in c.execute("SELECT version FROM schema_migrations")}
        done: list[int] = []
        files = sorted(
            (f for f in resources.files(_MIGRATIONS_PKG).iterdir() if f.name.endswith(".sql")),
            key=lambda f: f.name,
        )
        for f in files:
            version = int(f.name.split("_", 1)[0])
            if version in applied:
                continue
            sql = f.read_text(encoding="utf-8")
            with self.tx(immediate=True) as tx:
                for statement in _split_sql(sql):
                    tx.execute(statement)
                tx.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
            done.append(version)
        return done

    def snapshot_to(self, dest: Path) -> None:
        """Consistent online snapshot (``VACUUM INTO``), used for hourly volume snapshots."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.unlink(missing_ok=True)
        self.conn().execute("VACUUM INTO ?", (str(tmp),))
        tmp.replace(dest)

    def close(self) -> None:
        c: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None


def _split_sql(sql: str) -> list[str]:
    lines = [ln for ln in sql.splitlines() if not ln.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
