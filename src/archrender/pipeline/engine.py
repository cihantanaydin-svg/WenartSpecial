"""Cached, typed stage execution (principle 7, ADR-S05).

A stage is a pure function of its typed input, the declared config subset and the resolved model
identities. Its cache key hashes exactly those. Outputs are pydantic models whose blobs are
:class:`CasRef` s in the project CAS. On a cache hit every referenced blob is checked, and a
missing blob forces recomputation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from archrender.core.cas import CasRef, ProjectStore
from archrender.core.errors import ArchRenderError
from archrender.core.hashing import sha256_json
from archrender.core.ids import now_iso
from archrender.core.logging import bind, get_logger, unbind
from archrender.core.schemas.common import ModelRef
from archrender.core.schemas.manifest import StageTiming
from archrender.db.database import Database

log = get_logger(__name__)


@dataclass
class StageContext:
    project_id: str
    store: ProjectStore
    progress: Callable[[float, str], None]
    cancelled: Callable[[], bool]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageDef[I: BaseModel, O: BaseModel]:
    id: str
    version: str
    output: type[O]
    fn: Callable[[I, StageContext], O]
    config: Callable[[], dict[str, Any]] = lambda: {}
    models: Callable[[], list[ModelRef]] = lambda: []


def iter_cas_refs(obj: Any) -> Iterator[CasRef]:
    if isinstance(obj, CasRef):
        yield obj
    elif isinstance(obj, BaseModel):
        for name in type(obj).model_fields:
            yield from iter_cas_refs(getattr(obj, name))
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from iter_cas_refs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from iter_cas_refs(v)


class StageEngine:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.timings: list[StageTiming] = []

    def cache_key(self, stage: StageDef[Any, Any], inp: BaseModel) -> str:
        return sha256_json(
            {
                "stage": stage.id,
                "version": stage.version,
                "input": inp,
                "config": stage.config(),
                "models": [m.model_dump(mode="json") for m in stage.models()],
            }
        )

    def run[I: BaseModel, O: BaseModel](self, stage: StageDef[I, O], inp: I, ctx: StageContext) -> O:
        key = self.cache_key(stage, inp)
        row = self.db.one(
            "SELECT output_json FROM stage_runs WHERE project_id = ? AND cache_key = ?",
            (ctx.project_id, key),
        )
        if row is not None:
            out = stage.output.model_validate_json(row["output_json"])
            if all(ctx.store.exists(r.sha256) for r in iter_cas_refs(out)):
                self.timings.append(StageTiming(stage=stage.id, key=key[:16], cached=True, seconds=0.0))
                return out
            log.warning("cache entry references missing blobs; recomputing", extra={"fields": {"stage": stage.id}})
        bind(stage=stage.id)
        t0 = time.time()
        try:
            out = stage.fn(inp, ctx)
        except ArchRenderError as e:
            if e.stage is None:
                e.stage = stage.id
            raise
        finally:
            unbind("stage")
        seconds = time.time() - t0
        with self.db.tx(immediate=True) as c:
            c.execute(
                "INSERT OR REPLACE INTO stage_runs(cache_key, project_id, stage, stage_version, output_json,"
                " manifest_json, created_at, seconds) VALUES (?,?,?,?,?,?,?,?)",
                (
                    key,
                    ctx.project_id,
                    stage.id,
                    stage.version,
                    out.model_dump_json(),
                    json.dumps({"models": [m.model_dump(mode="json") for m in stage.models()]}),
                    now_iso(),
                    seconds,
                ),
            )
        self.timings.append(StageTiming(stage=stage.id, key=key[:16], cached=False, seconds=round(seconds, 3)))
        return out
