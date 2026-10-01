"""Plan versions and Gate A (ARCHITECTURE §S3).

S2 stores each new extraction as a draft version. Gate A edits are RFC 6902 JSON Patches on the
PlanGraph; every edit makes a new draft (the edited facts carry ``user`` provenance), and an
approved version never changes again (a later edit starts a new draft from it). Resolving a
scale conflict rescales the plan to the chosen estimate. Approval is refused while blocking
issues remain. On approval, the corrections from the extraction to the approved plan are kept as
a training example; whether they may ever be used for training is the owner's decision, so they
are stored with ``training_use_allowed = false`` (owner decision, Phase 0).
"""

from __future__ import annotations

import copy
import json
from typing import Any

from pydantic import ValidationError

from archrender.core.cas import CasRef
from archrender.core.errors import ArchRenderError, ErrorCode, not_found
from archrender.core.hashing import sha256_json
from archrender.core.ids import new_id, now_iso
from archrender.core.schemas.plan import PlanGraph
from archrender.pipeline.services import Services
from archrender.plan.validate import blocking, validate_plan

# ---------------------------------------------------------------------------------------------
# JSON Patch (RFC 6902)
# ---------------------------------------------------------------------------------------------


def _tokens(path: str) -> list[str]:
    if path == "":
        return []
    if not path.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {path!r}")
    return [t.replace("~1", "/").replace("~0", "~") for t in path[1:].split("/")]


def _parent(doc: Any, path: str) -> tuple[Any, str]:
    toks = _tokens(path)
    if not toks:
        raise ValueError("the operation needs a member path, not the whole document")
    cur = doc
    for t in toks[:-1]:
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur, toks[-1]


def _get(doc: Any, path: str) -> Any:
    cur = doc
    for t in _tokens(path):
        cur = cur[int(t)] if isinstance(cur, list) else cur[t]
    return cur


def apply_patch(doc: Any, ops: list[dict[str, Any]]) -> Any:
    """Apply a JSON Patch to a copy of ``doc`` (all operations or none)."""
    out = copy.deepcopy(doc)
    for i, op in enumerate(ops):
        kind = op.get("op")
        try:
            if kind == "test":
                if _get(out, op["path"]) != op["value"]:
                    raise ValueError("test failed")
                continue
            if kind in ("move", "copy"):
                value = copy.deepcopy(_get(out, op["from"]))
                if kind == "move":
                    out = apply_patch(out, [{"op": "remove", "path": op["from"]}])
                out = apply_patch(out, [{"op": "add", "path": op["path"], "value": value}])
                continue
            parent, key = _parent(out, op["path"])
            if kind == "add":
                if isinstance(parent, list):
                    parent.insert(len(parent) if key == "-" else int(key), op["value"])
                else:
                    parent[key] = op["value"]
            elif kind == "remove":
                if isinstance(parent, list):
                    parent.pop(int(key))
                else:
                    del parent[key]
            elif kind == "replace":
                if isinstance(parent, list):
                    parent[int(key)] = op["value"]
                else:
                    if key not in parent:
                        raise KeyError(key)
                    parent[key] = op["value"]
            else:
                raise ValueError(f"unknown operation {kind!r}")
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                f"Edit operation {i} ({kind} {op.get('path')}) cannot be applied: {e}.",
                "Reload the plan (it may have changed) and repeat the edit.",
            ) from e
    return out


# ---------------------------------------------------------------------------------------------
# provenance of edited values
# ---------------------------------------------------------------------------------------------


def _user_fact(value: Any, unit: Any, note: str) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "provenance": [
            {"method": "user", "confidence": 1.0, "note": note, "created_at": now_iso()}
        ],
        "status": "user_confirmed",
    }


def _is_fact(x: Any) -> bool:
    return isinstance(x, dict) and "value" in x and "provenance" in x


def _mark_user(old: Any, new: Any, note: str) -> Any:
    """Facts whose value changed (or that are new) become user facts; elements are matched by
    id inside lists so reordering does not count as an edit."""
    if _is_fact(new):
        if not _is_fact(old) or old["value"] != new["value"]:
            return _user_fact(new["value"], new.get("unit"), note)
        return new
    if isinstance(new, dict):
        return {
            k: _mark_user(old.get(k) if isinstance(old, dict) else None, v, note)
            for k, v in new.items()
        }
    if isinstance(new, list):
        by_id = (
            {o.get("id"): o for o in old if isinstance(o, dict) and "id" in o}
            if isinstance(old, list)
            else {}
        )
        out = []
        for i, v in enumerate(new):
            if isinstance(v, dict) and "id" in v:
                out.append(_mark_user(by_id.get(v["id"]), v, note))
            else:
                ov = old[i] if isinstance(old, list) and i < len(old) else None
                out.append(_mark_user(ov, v, note))
        return out
    return new


def content_sha(plan: PlanGraph) -> str:
    """Hash of the plan's content (no version id, no provenance timestamps)."""

    def strip(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in ("created_at", "version")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x

    return sha256_json(strip(plan.model_dump(mode="json")))


# ---------------------------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------------------------


def get_row(svc: Services, project_id: str, version_id: str) -> dict[str, Any]:
    row = svc.db.one(
        "SELECT * FROM plan_versions WHERE id = ? AND project_id = ?", (version_id, project_id)
    )
    if row is None:
        raise not_found("Plan version", version_id)
    return dict(row)


def load(svc: Services, project_id: str, version_id: str) -> PlanGraph:
    row = get_row(svc, project_id, version_id)
    ref = CasRef.model_validate_json(row["plan_ref_json"])
    return PlanGraph.model_validate_json(svc.store(project_id).read_bytes(ref))


def summary(row: dict[str, Any]) -> dict[str, Any]:
    issues = json.loads(row["issues_json"])
    return {
        "id": row["id"],
        "number": row["number"],
        "parent_id": row["parent_id"],
        "root_id": row["root_id"],
        "status": row["status"],
        "origin": row["origin"],
        "plan_sha256": row["plan_sha256"],
        "issues": issues,
        "blocking": sum(1 for i in issues if i["severity"] in ("error", "blocker")),
        "extraction": json.loads(row["extraction_json"]),
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "approved_by": row["approved_by"],
        "approved_at": row["approved_at"],
    }


def list_versions(svc: Services, project_id: str) -> list[dict[str, Any]]:
    rows = svc.db.query(
        "SELECT * FROM plan_versions WHERE project_id = ? ORDER BY number DESC", (project_id,)
    )
    return [summary(dict(r)) for r in rows]


def latest(svc: Services, project_id: str, *, approved: bool = False) -> dict[str, Any] | None:
    sql = "SELECT * FROM plan_versions WHERE project_id = ? AND status = 'approved'"
    if not approved:
        sql = "SELECT * FROM plan_versions WHERE project_id = ? AND status != 'superseded'"
    row = svc.db.one(sql + " ORDER BY number DESC LIMIT 1", (project_id,))
    return dict(row) if row else None


def head(svc: Services, project_id: str, root_id: str) -> dict[str, Any]:
    """The newest version descending from the extraction ``root_id`` (the latest edit)."""
    row = svc.db.one(
        "SELECT * FROM plan_versions WHERE project_id = ? AND root_id = ? ORDER BY number DESC"
        " LIMIT 1",
        (project_id, root_id),
    )
    if row is None:
        raise not_found("Plan version", root_id)
    return dict(row)


def working(svc: Services, project_id: str, root_id: str) -> dict[str, Any]:
    """The version a run uses for the extraction ``root_id``: the approved one when there is one
    (later drafts take effect only once approved), otherwise the latest edit."""
    row = svc.db.one(
        "SELECT * FROM plan_versions WHERE project_id = ? AND root_id = ? AND status = 'approved'",
        (project_id, root_id),
    )
    return dict(row) if row else head(svc, project_id, root_id)


def _insert(
    svc: Services,
    project_id: str,
    plan: PlanGraph,
    *,
    origin: str,
    parent: str | None,
    root: str | None,
    extraction: dict[str, Any],
    user_id: str | None,
) -> tuple[str, PlanGraph]:
    vid = new_id("plv")
    plan = plan.model_copy(update={"version": vid, "issues": validate_plan(plan)})
    plan = PlanGraph.model_validate(plan.model_dump())
    store = svc.store(project_id)
    ref = store.put_bytes(
        plan.model_dump_json(indent=1).encode(), "application/json", f"plan_{vid}.json"
    )
    with svc.db.tx(immediate=True) as c:
        row = c.execute(
            "SELECT COALESCE(MAX(number), 0) AS n FROM plan_versions WHERE project_id = ?",
            (project_id,),
        ).fetchone()
        c.execute(
            "INSERT INTO plan_versions(id, project_id, number, parent_id, root_id, status, origin,"
            " plan_sha256, plan_ref_json, issues_json, extraction_json, created_by, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                vid,
                project_id,
                int(row["n"]) + 1,
                parent,
                root or vid,
                "draft",
                origin,
                content_sha(plan),
                ref.model_dump_json(),
                json.dumps([i.model_dump(mode="json") for i in plan.issues]),
                json.dumps(extraction, default=str),
                user_id,
                now_iso(),
            ),
        )
    return vid, plan


def record_extraction(
    svc: Services, project_id: str, plan: PlanGraph, extraction: dict[str, Any]
) -> str:
    """A draft version for a new extraction; an unchanged extraction reuses its version."""
    sha = content_sha(plan.model_copy(update={"issues": validate_plan(plan)}))
    row = svc.db.one(
        "SELECT id FROM plan_versions WHERE project_id = ? AND origin = 'extraction' AND plan_sha256 = ?"
        " ORDER BY number DESC LIMIT 1",
        (project_id, sha),
    )
    if row is not None:
        return str(row["id"])
    vid, _ = _insert(
        svc,
        project_id,
        plan,
        origin="extraction",
        parent=None,
        root=None,
        extraction=extraction,
        user_id=None,
    )
    return vid


def edit(
    svc: Services,
    project_id: str,
    version_id: str,
    ops: list[dict[str, Any]],
    *,
    user_id: str,
    note: str = "",
    rederive_rooms: bool = False,
) -> str:
    """``rederive_rooms``: recompute the rooms from the edited walls (after moving, adding or
    deleting walls); rooms keep their names and labels where they overlap the old ones."""
    base_row = get_row(svc, project_id, version_id)
    base = load(svc, project_id, version_id)
    old = base.model_dump(mode="json")
    new = apply_patch(old, ops)
    new = _mark_user(old, new, note or "Gate A edit")
    try:
        plan = PlanGraph.model_validate(new)
    except ValidationError as e:
        err = e.errors()[0]
        msg = str(err["msg"]).removeprefix("Value error, ")
        where = "/".join(map(str, err["loc"]))
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"The edited plan is not valid: {msg}" + (f" (at /{where})." if where else "."),
            "Undo the last change; every opening needs an existing host wall, every room ≥ 3 points.",
        ) from e
    plan.source = base.source if base.source != "mock" else "user"
    if rederive_rooms:
        from archrender.plan.rooms import rederive_rooms as derive

        plan = derive(plan)
    vid, _ = _insert(
        svc,
        project_id,
        plan,
        origin="edit",
        parent=version_id,
        root=base_row["root_id"],
        extraction=json.loads(base_row["extraction_json"]),
        user_id=user_id,
    )
    svc.db.execute(
        "INSERT INTO plan_edits(id, project_id, from_version, to_version, patch_json, summary, created_by,"
        " created_at) VALUES (?,?,?,?,?,?,?,?)",
        (new_id("ple"), project_id, version_id, vid, json.dumps(ops), note, user_id, now_iso()),
    )
    return vid


def rescale(
    plan: PlanGraph, k: float, *, level: str | None = None, page_id: str | None = None
) -> PlanGraph:
    """Lengths × k about the plan origin (a different scale for the same sheet): the elements of
    ``level`` (all when None) and the document transform of ``page_id`` (all when None).
    Heights and stated values (area labels) are not plan measurements and stay."""
    d = plan.model_dump(mode="json")

    def pt(p: dict[str, Any]) -> None:
        p["x"], p["y"] = p["x"] * k, p["y"] * k

    def on(el: dict[str, Any]) -> bool:
        return level is None or el["level"] == level

    walls = {w["id"] for w in d["walls"] if on(w)}
    for w in d["walls"]:
        if w["id"] not in walls:
            continue
        cl = w["centerline"]
        if cl["kind"] == "segment":
            pt(cl["a"])
            pt(cl["b"])
        else:
            pt(cl["center"])
            cl["radius"] *= k
        w["thickness_m"]["value"] *= k
    for o in d["openings"]:
        if o["host_wall"] in walls:
            o["offset_m"]["value"] *= k
            o["width_m"]["value"] *= k
    for r in d["rooms"]:
        if on(r):
            for p in r["polygon"]:
                pt(p)
            for h in r["holes"]:
                for p in h:
                    pt(p)
    for c in d["columns"]:
        if on(c):
            for p in c["footprint"]:
                pt(p)
    for t in d["doc_transforms"]:
        if page_id is None or f"{t['doc_id']}_p{t['page']}" == page_id:
            t["matrix"] = [[v * k for v in row] for row in t["matrix"]]
    return PlanGraph.model_validate(d)


def _fact_from(value: float, note: str) -> Any:
    from archrender.core.schemas.provenance import Fact

    return Fact.model_validate(_user_fact(value, None, note))


def resolve_conflict(
    svc: Services, project_id: str, version_id: str, key: str, choice: int, *, user_id: str
) -> str:
    base_row = get_row(svc, project_id, version_id)
    plan = load(svc, project_id, version_id)
    idx = next((i for i, c in enumerate(plan.conflicts) if c.key == key), None)
    if idx is None:
        raise not_found("Conflict", key)
    c = plan.conflicts[idx]
    if not 0 <= choice < len(c.candidates):
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Choice {choice} is not a candidate of {key}.",
            "Pick one of the listed candidates.",
        )
    if key.startswith("scale/"):
        used = float(c.candidates[c.proposed]["m_per_unit"])
        chosen = float(c.candidates[choice]["m_per_unit"])
        if abs(chosen - used) / used > 1e-9:
            page_id = key.split("/", 1)[1]
            sources = json.loads(base_row["extraction_json"]).get("sources", [])
            level_name = next((s["level"] for s in sources if s["page"] == page_id), None)
            level = (
                None
                if len(plan.levels) == 1
                else next((lv.id for lv in plan.levels if lv.name == level_name), None)
            )
            if len(plan.levels) > 1 and level is None:
                raise ArchRenderError(
                    ErrorCode.CONFLICT,
                    f"The level drawn on page {page_id} is not in this plan version.",
                    "Re-extract the plan, then resolve the scale again.",
                )
            plan = rescale(plan, chosen / used, level=level, page_id=page_id)
    elif key.startswith("opening/") and key.endswith("/width"):
        oid = key.split("/")[1]
        o = next((x for x in plan.openings if x.id == oid), None)
        if o is None:
            raise not_found("Opening", oid)
        cand = c.candidates[choice]
        o.width_m = _fact_from(
            float(cand["value"]), f"Gate A: {cand.get('method', '?')} width chosen ({key})"
        )
    plan.conflicts[idx] = c.model_copy(update={"resolved_by": user_id, "resolution": choice})
    vid, _ = _insert(
        svc,
        project_id,
        plan,
        origin="edit",
        parent=version_id,
        root=base_row["root_id"],
        extraction=json.loads(base_row["extraction_json"]),
        user_id=user_id,
    )
    svc.db.execute(
        "INSERT INTO plan_edits(id, project_id, from_version, to_version, patch_json, summary, created_by,"
        " created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            new_id("ple"),
            project_id,
            version_id,
            vid,
            json.dumps([{"op": "resolve", "key": key, "choice": choice}]),
            f"conflict {key}: candidate {choice}",
            user_id,
            now_iso(),
        ),
    )
    return vid


def approve(svc: Services, project_id: str, version_id: str, *, user_id: str) -> dict[str, Any]:
    """Make ``version_id`` the project's approved plan (the previous one becomes superseded).
    Refused while the plan has blocking issues. A superseded version may be approved again (the
    documents went back to an earlier state)."""
    row = get_row(svc, project_id, version_id)
    if row["status"] == "approved":
        return summary(row)
    plan = load(svc, project_id, version_id)
    blockers = blocking(validate_plan(plan))
    if blockers:
        what = [f"{i.code}: {i.message}" for i in blockers[:5]]
        raise ArchRenderError(
            ErrorCode.PLAN_INVALID,
            "The plan cannot be approved yet: " + "; ".join(what),
            "Fix the listed issues in the plan editor (or resolve the scale conflict), then approve.",
        )
    now = now_iso()
    with svc.db.tx(immediate=True) as c:
        c.execute(
            "UPDATE plan_versions SET status = 'superseded' WHERE project_id = ? AND status = 'approved'",
            (project_id,),
        )
        c.execute(
            "UPDATE plan_versions SET status = 'approved', approved_by = ?, approved_at = ? WHERE id = ?",
            (user_id, now, version_id),
        )
    _training_example(svc, project_id, version_id)
    return summary(get_row(svc, project_id, version_id))


def _training_example(svc: Services, project_id: str, version_id: str) -> None:
    """The Gate A corrections between the extraction and the approved plan."""
    row = get_row(svc, project_id, version_id)
    root = str(row["root_id"])
    if root == version_id:
        return  # approved as extracted: nothing was corrected
    if svc.db.one(
        "SELECT 1 FROM training_examples WHERE plan_version = ? AND kind = 'plan_correction'",
        (version_id,),
    ):
        return  # approved before (and superseded since)
    chain = []
    cur = version_id
    while cur != root:
        e = svc.db.one(
            "SELECT from_version, patch_json, summary, created_by FROM plan_edits"
            " WHERE to_version = ?",
            (cur,),
        )
        if e is None:
            break
        chain.append(
            {
                "from": e["from_version"],
                "to": cur,
                "patch": json.loads(e["patch_json"]),
                "summary": e["summary"],
                "by": e["created_by"],
            }
        )
        cur = e["from_version"]
    extraction = json.loads(get_row(svc, project_id, root)["extraction_json"])
    pages = [s["page"] for s in extraction.get("sources", [])]
    svc.db.execute(
        "INSERT INTO training_examples(id, project_id, kind, source_page, plan_version, payload_json,"
        " training_use_allowed, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            new_id("trx"),
            project_id,
            "plan_correction",
            pages[0] if pages else None,
            version_id,
            json.dumps(
                {
                    "extraction_version": root,
                    "approved_version": version_id,
                    "edits": list(reversed(chain)),
                    "pages": pages,
                }
            ),
            0,
            now_iso(),
        ),
    )


def training_examples(svc: Services, project_id: str) -> list[dict[str, Any]]:
    rows = svc.db.query(
        "SELECT * FROM training_examples WHERE project_id = ? ORDER BY created_at", (project_id,)
    )
    return [
        {
            "id": r["id"],
            "kind": r["kind"],
            "source_page": r["source_page"],
            "plan_version": r["plan_version"],
            "payload": json.loads(r["payload_json"]),
            "training_use_allowed": bool(r["training_use_allowed"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------------------------
# VLM assist at Gate A: confirmations and suggestion decisions (ADR-S19)
# ---------------------------------------------------------------------------------------------
def _derived_version(
    svc: Services,
    project_id: str,
    version_id: str,
    plan: PlanGraph,
    patch: list[dict[str, Any]],
    note: str,
    user_id: str,
) -> str:
    base_row = get_row(svc, project_id, version_id)
    vid, _ = _insert(
        svc,
        project_id,
        plan,
        origin="edit",
        parent=version_id,
        root=base_row["root_id"],
        extraction=json.loads(base_row["extraction_json"]),
        user_id=user_id,
    )
    svc.db.execute(
        "INSERT INTO plan_edits(id, project_id, from_version, to_version, patch_json, summary, created_by,"
        " created_at) VALUES (?,?,?,?,?,?,?,?)",
        (new_id("ple"), project_id, version_id, vid, json.dumps(patch), note, user_id, now_iso()),
    )
    return vid


def _assist_example(
    svc: Services, project_id: str, version_id: str, page: str | None, payload: dict[str, Any]
) -> None:
    svc.db.execute(
        "INSERT INTO training_examples(id, project_id, kind, source_page, plan_version, payload_json,"
        " training_use_allowed, created_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            new_id("trx"),
            project_id,
            "assist_decision",
            page,
            version_id,
            json.dumps(payload),
            0,
            now_iso(),
        ),
    )


def suggestions(svc: Services, project_id: str, version_id: str) -> list[dict[str, Any]]:
    """The VLM hints of this version's extraction that had no evidence (shown dashed at Gate A),
    with the decision taken on each so far (``None`` while open)."""
    row = get_row(svc, project_id, version_id)
    extraction = json.loads(row["extraction_json"])
    decided: dict[str, str] = {}
    for r in svc.db.query(
        "SELECT payload_json FROM training_examples WHERE project_id = ? AND kind = 'assist_decision'",
        (project_id,),
    ):
        p = json.loads(r["payload_json"])
        if p.get("root") == row["root_id"] and "suggestion" in p:
            decided[p["suggestion"]] = p["action"]
    return [
        {**sg, "decision": decided.get(sg["id"])}
        for src in extraction.get("sources", [])
        for sg in ((src.get("assist") or {}).get("suggestions") or [])
    ]


def confirm_assists(
    svc: Services, project_id: str, version_id: str, element_ids: list[str], *, user_id: str
) -> str:
    """A new version in which a person confirmed elements measured from VLM hints."""
    row = get_row(svc, project_id, version_id)
    d = load(svc, project_id, version_id).model_dump(mode="json")
    found: set[str] = set()
    for el, keys in [(w, ("thickness_m",)) for w in d["walls"]] + [
        (o, ("offset_m", "width_m")) for o in d["openings"]
    ]:
        if el["id"] not in element_ids:
            continue
        for k in keys:
            for prov in el[k]["provenance"]:
                if prov["method"] == "vlm_assisted" and prov.get("assist"):
                    prov["assist"]["user_confirmed"] = True
                    found.add(el["id"])
    missing = sorted(set(element_ids) - found)
    if missing:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Not elements placed with VLM help in this version: {', '.join(missing)}.",
            "Confirm only the highlighted (vlm_assisted) elements.",
        )
    vid = _derived_version(
        svc,
        project_id,
        version_id,
        PlanGraph.model_validate(d),
        [{"op": "confirm_assist", "ids": sorted(found)}],
        f"confirmed {len(found)} VLM-assisted element(s)",
        user_id,
    )
    _assist_example(
        svc,
        project_id,
        vid,
        None,
        {"root": row["root_id"], "action": "confirmed", "elements": sorted(found)},
    )
    return vid


def decide_suggestion(
    svc: Services,
    project_id: str,
    version_id: str,
    suggestion_id: str,
    *,
    accept: bool,
    user_id: str,
    thickness_m: float | None = None,
) -> str | None:
    """Accept (insert as the user's own element; rooms re-derived) or reject a suggestion.
    Returns the new version id on accept, None on reject."""
    import numpy as np

    from archrender.core.schemas.common import Point2
    from archrender.core.schemas.plan import Opening, Segment, Wall
    from archrender.core.schemas.provenance import fact
    from archrender.plan.assist import _host
    from archrender.plan.rooms import rederive_rooms

    row = get_row(svc, project_id, version_id)
    sg = next(
        (s for s in suggestions(svc, project_id, version_id) if s["id"] == suggestion_id), None
    )
    if sg is None:
        raise not_found("Suggestion", suggestion_id)
    if sg["decision"] is not None:
        raise ArchRenderError(
            ErrorCode.CONFLICT,
            f"Suggestion {suggestion_id} was already {sg['decision']}.",
            "Refresh the plan.",
        )
    payload = {"root": row["root_id"], "suggestion": suggestion_id, "hint": sg}
    if not accept:
        _assist_example(svc, project_id, version_id, sg["page"], {**payload, "action": "rejected"})
        return None
    plan = load(svc, project_id, version_id)
    t = next((t for t in plan.doc_transforms if f"{t.doc_id}_p{t.page}" == sg["page"]), None)
    if t is None:
        raise ArchRenderError(
            ErrorCode.CONFLICT,
            f"Page {sg['page']} is not a source of this plan version.",
            "Re-extract the plan.",
        )
    m = np.array(t.matrix, float)
    a, b = np.array(sg["hint_px"], float) @ m[:, :2].T + m[:, 2]  # the version's current frame
    page_level = next(
        (
            s["level"]
            for s in json.loads(row["extraction_json"])["sources"]
            if s["page"] == sg["page"]
        ),
        None,
    )
    level = next((lv.id for lv in plan.levels if lv.name == page_level), plan.levels[0].id)
    note = f"accepted VLM suggestion {suggestion_id} (no drawing evidence found)"
    d = plan.model_copy(deep=True)
    if sg["kind"] == "wall":
        inner = [w.thickness_m.value for w in plan.walls if w.kind != "exterior"]
        tk = thickness_m or (float(np.median(inner)) if inner else 0.12)
        heights = [w.height_m.value for w in plan.walls]
        d.walls.append(
            Wall(
                id=f"WU{len(plan.walls) + 1}",
                level=level,
                centerline=Segment(
                    a=Point2(x=float(a[0]), y=float(a[1])), b=Point2(x=float(b[0]), y=float(b[1]))
                ),
                thickness_m=fact(tk, "user", 1.0, note=note),
                height_m=fact(float(np.median(heights)) if heights else 2.7, "default", 0.5),
                kind="interior",
            )
        )
        d = rederive_rooms(PlanGraph.model_validate(d.model_dump()))
    else:
        span = float(np.hypot(*(b - a)))
        host = _host(plan, (a + b) / 2, (b - a) / span if span > 0.2 else None)
        if host is None or not isinstance(host.centerline, Segment):
            raise ArchRenderError(
                ErrorCode.VALIDATION,
                "No straight wall near the suggested opening.",
                "Draw the opening in the editor instead.",
            )
        h0 = np.array([host.centerline.a.x, host.centerline.a.y])
        h1 = np.array([host.centerline.b.x, host.centerline.b.y])
        u = (h1 - h0) / float(np.hypot(*(h1 - h0)))
        s0, s1 = sorted((float(np.dot(a - h0, u)), float(np.dot(b - h0, u))))
        door = sg["kind"] != "window"
        d.openings.append(
            Opening(
                id=f"OU{len(plan.openings) + 1}",
                host_wall=host.id,
                offset_m=fact((s0 + s1) / 2, "user", 1.0, note=note),
                width_m=fact(s1 - s0, "user", 1.0, note=note),
                height_m=fact(2.1 if door else 1.2, "default", 0.5),
                sill_m=fact(0.0 if door else 0.9, "default", 0.5),
                type=sg["kind"],
            )
        )
    vid = _derived_version(
        svc,
        project_id,
        version_id,
        PlanGraph.model_validate(d.model_dump()),
        [{"op": "accept_suggestion", "id": suggestion_id}],
        note,
        user_id,
    )
    _assist_example(svc, project_id, vid, sg["page"], {**payload, "action": "accepted"})
    return vid


def calibrate(
    svc: Services,
    project_id: str,
    version_id: str,
    a: tuple[float, float],
    b: tuple[float, float],
    length_m: float,
    *,
    user_id: str,
    level: str | None = None,
) -> str:
    """Gate A scale calibration: the distance a–b (plan metres in this version) is really
    ``length_m``. The plan (or ``level``) is rescaled about the origin; the user's measurement
    becomes the scale's provenance."""
    import math

    plan = load(svc, project_id, version_id)
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    if d < 0.05 or not 0.05 <= length_m <= 500:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"Calibration needs two distinct points and a real length (got {d:.3f} m → {length_m} m).",
            "Click two points on a known dimension and type its length in metres.",
        )
    k = length_m / d
    if not 0.2 <= k <= 5:
        raise ArchRenderError(
            ErrorCode.VALIDATION,
            f"The calibration would rescale the plan by ×{k:.3f}.",
            "Check the clicked points and the typed length (metres, not centimetres).",
        )
    if level is not None and level not in {lv.id for lv in plan.levels}:
        raise not_found("Level", level)
    scaled = rescale(plan, k, level=level if len(plan.levels) > 1 else None)
    note = f"scale calibrated by the user: {d:.3f} m measured is {length_m} m (×{k:.4f})"
    for t in scaled.doc_transforms:
        t.method = "user_calibration"
    # the user's measurement settles any open scale question
    scaled.conflicts = [
        c.model_copy(update={"resolved_by": user_id, "resolution": c.proposed})
        if c.key.startswith("scale/") and c.resolution is None
        else c
        for c in scaled.conflicts
    ]
    scaled.assumptions = [x for x in scaled.assumptions if x.key != "user/scale_calibration"]
    from archrender.core.schemas.provenance import Assumption

    scaled.assumptions.append(
        Assumption(key="user/scale_calibration", value=k, reason=note, stage="S3", overridable=True)
    )
    return _derived_version(
        svc,
        project_id,
        version_id,
        scaled,
        [{"op": "calibrate", "a": list(a), "b": list(b), "length_m": length_m}],
        note,
        user_id,
    )


def backgrounds(svc: Services, project_id: str, plan: PlanGraph) -> list[dict[str, Any]]:
    """The source pages under the plan in the editor: image, size and page px → plan matrix
    (raster and PDF pages; a phone photo shows its rectified sheet). DXF/IFC have none."""
    from archrender.ingest.intake import page_refs

    pages = {p.id: p for p in page_refs(svc, project_id)}
    out = []
    for t in plan.doc_transforms:
        page = pages.get(f"{t.doc_id}_p{t.page}")
        if page is None or page.kind in ("dxf", "ifc") or page.raster is None:
            continue
        image, w, h = page.raster, page.width_px, page.height_px
        row = svc.db.one("SELECT analysis_json FROM page_analysis WHERE page_id = ?", (page.id,))
        if row is not None:
            a = json.loads(row["analysis_json"])
            if a.get("rectified") and a.get("rectification"):
                image = CasRef.model_validate(a["rectified"])
                w, h = a["rectification"]["width_px"], a["rectification"]["height_px"]
        out.append(
            {
                "page_id": page.id,
                "image": image.model_dump(mode="json"),
                "width_px": w,
                "height_px": h,
                "matrix": [list(r) for r in t.matrix],
            }
        )
    return out
