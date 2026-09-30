from __future__ import annotations

import time

import pytest

from archrender.core.errors import ErrorCode, ErrorInfo
from archrender.core.schemas.jobs import JobKind, JobStatus
from archrender.db.database import Database
from archrender.pipeline.queue import JobQueue, LeaseLost


def _err() -> ErrorInfo:
    return ErrorInfo(code=ErrorCode.STAGE_FAILED, message="boom", fix_hint="retry")


def test_lease_complete_and_events(db: Database, project_id: str) -> None:
    q = JobQueue(db)
    jid = q.enqueue(project_id, JobKind.RUN, "gpu", {"x": 1}, created_by="usr_test")
    assert q.lease(["cpu"], "w1", 10) is None  # wrong queue
    job = q.lease(["gpu"], "w1", 10)
    assert job is not None and job.id == jid and job.status == JobStatus.RUNNING
    assert q.lease(["gpu"], "w2", 10) is None  # already leased
    q.progress(jid, 0.5, "S7")
    q.complete(jid, "w1", {"ok": True})
    job = q.get(jid)
    assert job.status == JobStatus.SUCCEEDED and job.result == {"ok": True}
    types = [e.type for e in q.events(jid)]
    assert types[0] == "status" and "progress" in types and "result" in types
    last = q.events(jid)[-1].id
    assert q.events(jid, after_id=last) == []


def test_expired_lease_is_reclaimed(db: Database, project_id: str) -> None:
    q = JobQueue(db)
    jid = q.enqueue(project_id, JobKind.RUN, "gpu", {}, created_by=None)
    assert q.lease(["gpu"], "w1", 0.01) is not None
    time.sleep(0.05)
    job = q.lease(["gpu"], "w2", 10)
    assert job is not None and job.id == jid and job.attempts == 2
    with pytest.raises(LeaseLost):
        q.heartbeat(jid, "w1", 10)  # old owner lost it
    q.heartbeat(jid, "w2", 10)


def test_retry_then_fail(db: Database, project_id: str) -> None:
    q = JobQueue(db)
    jid = q.enqueue(project_id, JobKind.RUN, "cpu", {}, created_by=None, max_attempts=2)
    q.lease(["cpu"], "w", 10)
    q.fail(jid, "w", _err(), retry=True)
    assert q.get(jid).status == JobStatus.QUEUED
    q.lease(["cpu"], "w", 10)
    q.fail(jid, "w", _err(), retry=True)
    job = q.get(jid)
    assert job.status == JobStatus.FAILED and job.error is not None


def test_cancel_running_and_queued(db: Database, project_id: str) -> None:
    q = JobQueue(db)
    a = q.enqueue(project_id, JobKind.RUN, "cpu", {}, created_by=None)
    b = q.enqueue(project_id, JobKind.RUN, "cpu", {}, created_by=None)
    q.lease(["cpu"], "w", 10)
    assert q.request_cancel(b).status == JobStatus.CANCELLED
    q.request_cancel(a)
    with pytest.raises(LeaseLost):
        q.heartbeat(a, "w", 10)
    q.fail(a, "w", _err(), retry=True)
    assert q.get(a).status == JobStatus.CANCELLED


def test_gate_park_and_resume(db: Database, project_id: str) -> None:
    q = JobQueue(db)
    jid = q.enqueue(project_id, JobKind.RUN, "gpu", {}, created_by=None)
    q.lease(["gpu"], "w", 10)
    q.park_at_gate(jid, "w", "A_plan", {"issues": 1})
    assert q.get(jid).status == JobStatus.WAITING_GATE
    assert q.lease(["gpu"], "w", 10) is None
    q.resume(jid)
    assert q.lease(["gpu"], "w", 10) is not None
