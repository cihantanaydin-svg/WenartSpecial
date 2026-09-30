"""Durable job queue on SQLite with leases, heartbeats, cancellation and an event log (SSE source)."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from archrender.core.errors import ErrorCode, ErrorInfo, not_found
from archrender.core.ids import new_id, now_iso, parse_iso
from archrender.core.schemas.jobs import Job, JobEvent, JobKind, JobStatus, Queue
from archrender.db.database import Database


def _job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        project_id=row["project_id"],
        kind=JobKind(row["kind"]),
        queue=row["queue"],
        status=JobStatus(row["status"]),
        priority=row["priority"],
        payload=json.loads(row["payload_json"]),
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=ErrorInfo.model_validate_json(row["error_json"]) if row["error_json"] else None,
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        progress=row["progress"],
        stage=row["stage"],
        created_at=parse_iso(row["created_at"]),
        started_at=parse_iso(row["started_at"]) if row["started_at"] else None,
        finished_at=parse_iso(row["finished_at"]) if row["finished_at"] else None,
        created_by=row["created_by"],
    )


class LeaseLost(Exception):
    """Raised when a worker no longer holds the lease on its job (expired or cancelled)."""


class JobQueue:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ---- producers -------------------------------------------------------------------------
    def enqueue(
        self,
        project_id: str,
        kind: JobKind,
        queue: Queue,
        payload: dict[str, Any],
        *,
        created_by: str | None,
        priority: int = 0,
        max_attempts: int = 3,
        job_id: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> str:
        """Insert a queued job. Pass ``conn`` to join the caller's transaction (atomic with it)."""
        jid = job_id or new_id("job")

        def _insert(c: sqlite3.Connection) -> None:
            c.execute(
                "INSERT INTO jobs(id, project_id, kind, queue, status, priority, payload_json,"
                " max_attempts, created_by, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    project_id,
                    kind.value,
                    queue,
                    JobStatus.QUEUED.value,
                    priority,
                    json.dumps(payload),
                    max_attempts,
                    created_by,
                    now_iso(),
                ),
            )
            self._emit(c, jid, "status", {"status": JobStatus.QUEUED.value})

        if conn is not None:
            _insert(conn)
        else:
            with self.db.tx(immediate=True) as c:
                _insert(c)
        return jid

    def get(self, job_id: str) -> Job:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise not_found("Job", job_id)
        return _job(row)

    def list_for_project(self, project_id: str, limit: int = 50) -> list[Job]:
        rows = self.db.query(
            "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
            (project_id, limit),
        )
        return [_job(r) for r in rows]

    def request_cancel(self, job_id: str) -> Job:
        with self.db.tx(immediate=True) as c:
            row = c.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise not_found("Job", job_id)
            status = JobStatus(row["status"])
            if status in (JobStatus.QUEUED, JobStatus.WAITING_GATE):
                c.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, lease_owner = NULL, error_json = ?"
                    " WHERE id = ?",
                    (
                        JobStatus.CANCELLED.value,
                        now_iso(),
                        _cancel_error().model_dump_json(),
                        job_id,
                    ),
                )
                self._emit(c, job_id, "status", {"status": JobStatus.CANCELLED.value})
            elif status == JobStatus.RUNNING:
                c.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                self._emit(c, job_id, "log", {"message": "cancellation requested"})
        return self.get(job_id)

    def resume(self, job_id: str) -> None:
        """Re-queue a job parked at a gate (called when the gate is approved)."""
        with self.db.tx(immediate=True) as c:
            n = c.execute(
                "UPDATE jobs SET status = ?, lease_owner = NULL, lease_until = NULL"
                " WHERE id = ? AND status = ?",
                (JobStatus.QUEUED.value, job_id, JobStatus.WAITING_GATE.value),
            ).rowcount
            if n:
                self._emit(c, job_id, "status", {"status": JobStatus.QUEUED.value})

    # ---- workers ---------------------------------------------------------------------------
    def lease(self, queues: list[Queue], owner: str, lease_s: float) -> Job | None:
        """Atomically lease the next job. Jobs whose lease expired (crashed worker) are re-leased."""
        now = time.time()
        placeholders = ",".join("?" for _ in queues)
        with self.db.tx(immediate=True) as c:
            row = c.execute(
                f"SELECT * FROM jobs WHERE queue IN ({placeholders}) AND ("  # noqa: S608
                "  status = 'queued' OR (status = 'running' AND lease_until < ?))"
                " ORDER BY priority DESC, created_at ASC LIMIT 1",
                (*queues, now),
            ).fetchone()
            if row is None:
                return None
            if row["attempts"] >= row["max_attempts"]:
                err = ErrorInfo(
                    code=ErrorCode.STAGE_FAILED,
                    message=f"Job exhausted {row['max_attempts']} attempts (worker lost or crashed).",
                    fix_hint="Check the worker logs for crashes (OOM kill, timeout) and re-run.",
                )
                c.execute(
                    "UPDATE jobs SET status = 'failed', finished_at = ?, error_json = ?,"
                    " lease_owner = NULL WHERE id = ?",
                    (now_iso(), err.model_dump_json(), row["id"]),
                )
                self._emit(c, row["id"], "error", err.model_dump(mode="json"))
                return None
            c.execute(
                "UPDATE jobs SET status = 'running', lease_owner = ?, lease_until = ?,"
                " heartbeat_at = ?, attempts = attempts + 1, started_at = COALESCE(started_at, ?)"
                " WHERE id = ?",
                (owner, now + lease_s, now, now_iso(), row["id"]),
            )
            self._emit(c, row["id"], "status", {"status": "running", "worker": owner})
        return self.get(row["id"])

    def heartbeat(self, job_id: str, owner: str, lease_s: float) -> None:
        """Extend the lease. Raises :class:`LeaseLost` if cancelled or taken over."""
        with self.db.tx(immediate=True) as c:
            row = c.execute(
                "SELECT lease_owner, status, cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None or row["lease_owner"] != owner or row["status"] != "running":
                raise LeaseLost(job_id)
            if row["cancel_requested"]:
                raise LeaseLost(job_id)
            now = time.time()
            c.execute(
                "UPDATE jobs SET lease_until = ?, heartbeat_at = ? WHERE id = ?",
                (now + lease_s, now, job_id),
            )

    def cancel_requested(self, job_id: str) -> bool:
        row = self.db.one("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,))
        return bool(row and row["cancel_requested"])

    def progress(self, job_id: str, progress: float, stage: str | None, message: str = "") -> None:
        with self.db.tx(immediate=True) as c:
            c.execute(
                "UPDATE jobs SET progress = ?, stage = COALESCE(?, stage) WHERE id = ?",
                (max(0.0, min(1.0, progress)), stage, job_id),
            )
            self._emit(
                c, job_id, "progress", {"progress": round(progress, 4), "stage": stage, "message": message}
            )

    def emit(self, job_id: str, type_: str, data: dict[str, Any]) -> None:
        with self.db.tx(immediate=True) as c:
            self._emit(c, job_id, type_, data)

    def complete(self, job_id: str, owner: str, result: dict[str, Any]) -> None:
        with self.db.tx(immediate=True) as c:
            n = c.execute(
                "UPDATE jobs SET status = 'succeeded', result_json = ?, finished_at = ?, progress = 1,"
                " lease_owner = NULL, lease_until = NULL WHERE id = ? AND lease_owner = ?",
                (json.dumps(result), now_iso(), job_id, owner),
            ).rowcount
            if n == 0:
                raise LeaseLost(job_id)
            self._emit(c, job_id, "result", result)
            self._emit(c, job_id, "status", {"status": "succeeded"})

    def fail(self, job_id: str, owner: str, error: ErrorInfo, *, retry: bool) -> None:
        with self.db.tx(immediate=True) as c:
            row = c.execute(
                "SELECT attempts, max_attempts, cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return
            if row["cancel_requested"]:
                status = JobStatus.CANCELLED
                error = _cancel_error()
            elif retry and row["attempts"] < row["max_attempts"]:
                status = JobStatus.QUEUED
            else:
                status = JobStatus.FAILED
            c.execute(
                "UPDATE jobs SET status = ?, error_json = ?, lease_owner = NULL, lease_until = NULL,"
                " finished_at = CASE WHEN ? THEN ? ELSE finished_at END WHERE id = ? AND lease_owner = ?",
                (
                    status.value,
                    error.model_dump_json(),
                    status.terminal,
                    now_iso(),
                    job_id,
                    owner,
                ),
            )
            self._emit(c, job_id, "error", error.model_dump(mode="json"))
            self._emit(c, job_id, "status", {"status": status.value})

    def park_at_gate(self, job_id: str, owner: str, gate: str, evidence: dict[str, Any]) -> None:
        with self.db.tx(immediate=True) as c:
            c.execute(
                "UPDATE jobs SET status = 'waiting_gate', stage = ?, lease_owner = NULL,"
                " lease_until = NULL, attempts = 0 WHERE id = ? AND lease_owner = ?",
                (f"gate:{gate}", job_id, owner),
            )
            self._emit(c, job_id, "gate", {"gate": gate, "evidence": evidence})
            self._emit(c, job_id, "status", {"status": "waiting_gate", "gate": gate})

    # ---- events ----------------------------------------------------------------------------
    def events(self, job_id: str, after_id: int = 0, limit: int = 500) -> list[JobEvent]:
        rows = self.db.query(
            "SELECT * FROM job_events WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
            (job_id, after_id, limit),
        )
        return [
            JobEvent(
                id=r["id"],
                job_id=r["job_id"],
                ts=parse_iso(r["ts"]),
                type=r["type"],
                data=json.loads(r["data_json"]),
            )
            for r in rows
        ]

    @staticmethod
    def _emit(c: sqlite3.Connection, job_id: str, type_: str, data: dict[str, Any]) -> None:
        c.execute(
            "INSERT INTO job_events(job_id, ts, type, data_json) VALUES (?,?,?,?)",
            (job_id, now_iso(), type_, json.dumps(data, default=str)),
        )


def _cancel_error() -> ErrorInfo:
    return ErrorInfo(
        code=ErrorCode.JOB_CANCELLED,
        message="The job was cancelled.",
        fix_hint="Start a new run when ready; completed stages are reused from the cache.",
    )
