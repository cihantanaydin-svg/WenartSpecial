"""Worker process: leases jobs, heartbeats, dispatches by kind, parks at gates.

``python -m archrender.pipeline.worker --queue gpu`` (exactly one per pod) or ``--queue cpu``.
``--queue all`` serves both (development, tests).
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import threading
import time
from typing import Any

from archrender.core.config import Settings, get_settings
from archrender.core.errors import ArchRenderError, ErrorCode, ErrorInfo
from archrender.core.logging import bind, configure_logging, get_logger, unbind
from archrender.core.schemas.jobs import Job, JobKind, Queue
from archrender.ingest.uploads import run_intake
from archrender.pipeline.engine import StageContext
from archrender.pipeline.gates import GateWait
from archrender.pipeline.queue import LeaseLost
from archrender.pipeline.run import RunOrchestrator
from archrender.pipeline.services import Services

log = get_logger(__name__)


class _Heartbeat(threading.Thread):
    def __init__(self, svc: Services, job_id: str, owner: str) -> None:
        super().__init__(daemon=True)
        self.svc = svc
        self.job_id = job_id
        self.owner = owner
        self.stop_event = threading.Event()
        self.lost = threading.Event()

    def run(self) -> None:
        interval = max(0.5, self.svc.settings.job_lease_s / 3)
        while not self.stop_event.wait(interval):
            try:
                self.svc.queue.heartbeat(self.job_id, self.owner, self.svc.settings.job_lease_s)
            except LeaseLost:
                self.lost.set()
                return


class Worker:
    def __init__(self, svc: Services, queues: list[Queue], name: str | None = None) -> None:
        self.svc = svc
        self.queues = queues
        self.name = name or f"{socket.gethostname()}-{os.getpid()}-{'+'.join(queues)}"
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        log.info("worker started", extra={"fields": {"queues": self.queues, "worker": self.name}})
        while not self._stop.is_set():
            if not self.run_once():
                self._stop.wait(self.svc.settings.worker_poll_s)

    def run_until_idle(self, max_jobs: int = 1000) -> int:
        n = 0
        while n < max_jobs and self.run_once():
            n += 1
        return n

    def run_once(self) -> bool:
        job = self.svc.queue.lease(self.queues, self.name, self.svc.settings.job_lease_s)
        if job is None:
            return False
        self.handle(job)
        return True

    def handle(self, job: Job) -> None:
        q = self.svc.queue
        hb = _Heartbeat(self.svc, job.id, self.name)
        hb.start()
        bind(job_id=job.id, project=job.project_id)
        try:
            result = self._dispatch(job, hb)
            q.complete(job.id, self.name, result)
        except GateWait as gw:
            q.park_at_gate(job.id, self.name, gw.gate.value, gw.evidence)
            self.svc.db.execute("UPDATE runs SET status = 'waiting_gate' WHERE job_id = ?", (job.id,))
        except LeaseLost:
            log.warning("lease lost; abandoning job")
        except ArchRenderError as e:
            if e.code == ErrorCode.JOB_CANCELLED or hb.lost.is_set():
                q.fail(job.id, self.name, e.to_info(), retry=False)
            else:
                log.error("job failed", extra={"fields": {"code": e.code.value, "error": e.message}})
                q.fail(job.id, self.name, e.to_info(), retry=e.retryable)
            self._mark_run_failed(job)
        except InterruptedError:
            q.fail(job.id, self.name, ErrorInfo(code=ErrorCode.JOB_CANCELLED, message="Cancelled.", fix_hint="Start a new run."), retry=False)
            self._mark_run_failed(job)
        except Exception as e:  # noqa: BLE001 - last-resort guard: record a loud, actionable failure
            log.exception("unexpected worker error")
            q.fail(
                job.id,
                self.name,
                ErrorInfo(
                    code=ErrorCode.INTERNAL,
                    message=f"Unexpected {type(e).__name__}: {e}",
                    fix_hint="This is a bug; the traceback is in the worker log. Retrying reuses cached stages.",
                ),
                retry=False,
            )
            self._mark_run_failed(job)
        finally:
            hb.stop_event.set()
            unbind("job_id", "project")

    def _mark_run_failed(self, job: Job) -> None:
        if job.kind == JobKind.RUN:
            status = self.svc.queue.get(job.id).status.value
            self.svc.db.execute("UPDATE runs SET status = ? WHERE job_id = ?", (status, job.id))

    def _dispatch(self, job: Job, hb: _Heartbeat) -> dict[str, Any]:
        def cancelled() -> bool:
            return hb.lost.is_set() or self.svc.queue.cancel_requested(job.id)

        def progress(fraction: float, message: str) -> None:
            if cancelled():
                raise ArchRenderError(ErrorCode.JOB_CANCELLED, "The job was cancelled.", "Start a new run when ready.")
            self.svc.queue.progress(job.id, fraction, message.split(" ", 1)[0], message)

        if job.kind == JobKind.INTAKE:
            return run_intake(self.svc, str(job.payload["upload_id"]))
        if job.kind == JobKind.RUN:
            ctx = StageContext(
                project_id=job.project_id,
                store=self.svc.store(job.project_id),
                progress=progress,
                cancelled=cancelled,
            )
            return RunOrchestrator(self.svc, job.id, str(job.payload["run_id"]), ctx).execute()
        raise ArchRenderError(
            ErrorCode.VALIDATION, f"Unknown job kind {job.kind}.", "Upgrade the worker to match the API version."
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ArchRender worker")
    parser.add_argument("--queue", choices=["cpu", "gpu", "all"], required=True)
    args = parser.parse_args(argv)
    configure_logging(os.environ.get("ARCHRENDER_LOG_LEVEL", "INFO"))
    settings: Settings = get_settings()
    svc = Services.create(settings)
    queues: list[Queue] = ["cpu", "gpu"] if args.queue == "all" else [args.queue]
    worker = Worker(svc, queues)
    signal.signal(signal.SIGTERM, lambda *_: worker.stop())
    signal.signal(signal.SIGINT, lambda *_: worker.stop())
    worker.run_forever()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
