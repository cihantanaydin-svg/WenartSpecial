"""Human gates A–D (principle 4). Decisions are persisted per run and audited."""

from __future__ import annotations

import json
from typing import Any, Literal

from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.ids import now_iso
from archrender.core.schemas.jobs import GateName, GateStatus
from archrender.db.database import Database

GatePolicy = Literal["always", "on_low_confidence", "never"]


class GateWait(Exception):
    """Raised inside a run when a gate needs a human decision; the worker parks the job."""

    def __init__(self, gate: GateName, evidence: dict[str, Any]) -> None:
        super().__init__(gate.value)
        self.gate = gate
        self.evidence = evidence


class Gates:
    def __init__(self, db: Database, run_id: str, policy: GatePolicy) -> None:
        self.db = db
        self.run_id = run_id
        self.policy = policy

    def status(self, gate: GateName) -> GateStatus | None:
        row = self.db.one(
            "SELECT status FROM gates WHERE run_id = ? AND gate = ?", (self.run_id, gate.value)
        )
        return GateStatus(row["status"]) if row else None

    def _refuse_if_rejected(self, gate: GateName, current: GateStatus | None) -> None:
        if current == GateStatus.REJECTED:
            raise ArchRenderError(
                ErrorCode.GATE_REJECTED,
                f"Gate {gate.value} was rejected by a reviewer.",
                "Address the reviewer's notes (edit plan/brief/cameras) and start a new run.",
            )

    def record_approval(
        self, gate: GateName, decided_by: str | None, evidence: dict[str, Any]
    ) -> GateStatus:
        """The gate's subject was already approved by a person outside this run (e.g. a plan
        version approved at Gate A of an earlier run): record that decision for this run."""
        current = self.status(gate)
        if current in (GateStatus.APPROVED, GateStatus.AUTO_PASSED):
            return current
        self._refuse_if_rejected(gate, current)
        with self.db.tx(immediate=True) as c:
            c.execute(
                "INSERT OR REPLACE INTO gates(run_id, gate, status, policy, evidence_json,"
                " decided_by, decided_at, notes) VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.run_id,
                    gate.value,
                    GateStatus.APPROVED.value,
                    self.policy,
                    json.dumps(evidence, default=str),
                    decided_by,
                    now_iso(),
                    "approved before this run",
                ),
            )
        return GateStatus.APPROVED

    def check(
        self, gate: GateName, *, auto_ok: bool, evidence: dict[str, Any], mandatory: bool = False
    ) -> GateStatus:
        """Pass, auto-pass, or raise :class:`GateWait` / GATE_REJECTED."""
        current = self.status(gate)
        if current in (GateStatus.APPROVED, GateStatus.AUTO_PASSED):
            return current
        self._refuse_if_rejected(gate, current)
        policy = "always" if mandatory else self.policy
        auto = policy == "never" or (policy == "on_low_confidence" and auto_ok)
        status = GateStatus.AUTO_PASSED if auto else GateStatus.PENDING
        with self.db.tx(immediate=True) as c:
            c.execute(
                "INSERT OR REPLACE INTO gates(run_id, gate, status, policy, evidence_json, decided_at)"
                " VALUES (?,?,?,?,?,?)",
                (
                    self.run_id,
                    gate.value,
                    status.value,
                    policy,
                    json.dumps(evidence, default=str),
                    now_iso() if auto else None,
                ),
            )
        if not auto:
            raise GateWait(gate, evidence)
        return status


def decide(
    db: Database, run_id: str, gate: str, *, approve: bool, user_id: str, notes: str | None
) -> None:
    row = db.one("SELECT status FROM gates WHERE run_id = ? AND gate = ?", (run_id, gate))
    if row is None:
        raise not_found("Gate", f"{run_id}/{gate}")
    if row["status"] != GateStatus.PENDING.value:
        raise ArchRenderError(
            ErrorCode.CONFLICT,
            f"Gate {gate} is already {row['status']}.",
            "Refresh the run; only pending gates can be decided.",
        )
    status = GateStatus.APPROVED if approve else GateStatus.REJECTED
    with db.tx(immediate=True) as c:
        c.execute(
            "UPDATE gates SET status = ?, decided_by = ?, decided_at = ?, notes = ? WHERE run_id = ? AND gate = ?",
            (status.value, user_id, now_iso(), notes, run_id, gate),
        )
