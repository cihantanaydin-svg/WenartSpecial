"""Pydantic schemas: the single source of truth for every data structure (JSON Schema exported)."""

from archrender.core.schemas.brief import DesignBrief
from archrender.core.schemas.document import IntakeResult, PageRef
from archrender.core.schemas.jobs import GateName, GateStatus, Job, JobEvent, JobKind, JobStatus
from archrender.core.schemas.manifest import RunManifest
from archrender.core.schemas.plan import PlanGraph
from archrender.core.schemas.qa import CandidateQA, CheckResult, ViewOutcome
from archrender.core.schemas.scene import SceneSpec
from archrender.core.schemas.section import Section
from archrender.core.schemas.understanding import Classification, NorthArrow, Schedule, TitleBlock

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
    "IntakeResult": IntakeResult,
    "PageRef": PageRef,
    "Classification": Classification,
    "TitleBlock": TitleBlock,
    "NorthArrow": NorthArrow,
    "Schedule": Schedule,
}

__all__ = [
    "EXPORTED_SCHEMAS",
    "CandidateQA",
    "CheckResult",
    "Classification",
    "DesignBrief",
    "GateName",
    "GateStatus",
    "IntakeResult",
    "Job",
    "JobEvent",
    "JobKind",
    "JobStatus",
    "NorthArrow",
    "PageRef",
    "PlanGraph",
    "RunManifest",
    "SceneSpec",
    "Schedule",
    "Section",
    "TitleBlock",
    "ViewOutcome",
]
