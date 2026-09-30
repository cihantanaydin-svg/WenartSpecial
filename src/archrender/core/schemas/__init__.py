"""Pydantic schemas: the single source of truth for every data structure (JSON Schema exported)."""

from archrender.core.schemas.brief import DesignBrief
from archrender.core.schemas.jobs import GateName, GateStatus, Job, JobEvent, JobKind, JobStatus
from archrender.core.schemas.manifest import RunManifest
from archrender.core.schemas.plan import PlanGraph
from archrender.core.schemas.qa import CandidateQA, CheckResult, ViewOutcome
from archrender.core.schemas.scene import SceneSpec
from archrender.core.schemas.section import Section

EXPORTED_SCHEMAS = {
    "PlanGraph": PlanGraph,
    "Section": Section,
    "DesignBrief": DesignBrief,
    "SceneSpec": SceneSpec,
    "CheckResult": CheckResult,
    "CandidateQA": CandidateQA,
    "ViewOutcome": ViewOutcome,
    "RunManifest": RunManifest,
    "Job": Job,
    "JobEvent": JobEvent,
}

__all__ = [
    "EXPORTED_SCHEMAS",
    "CandidateQA",
    "CheckResult",
    "DesignBrief",
    "GateName",
    "GateStatus",
    "Job",
    "JobEvent",
    "JobKind",
    "JobStatus",
    "PlanGraph",
    "RunManifest",
    "SceneSpec",
    "Section",
    "ViewOutcome",
]
