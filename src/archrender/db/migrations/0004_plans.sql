-- S2 plan versions (immutable once approved at Gate A), the edits between them, and the
-- corrections captured as training examples (never used for training unless the owner allows it).
CREATE TABLE plan_versions (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    number INTEGER NOT NULL,
    parent_id TEXT REFERENCES plan_versions(id),
    root_id TEXT NOT NULL,             -- the extraction this version descends from (itself for one)
    status TEXT NOT NULL,              -- draft | approved | superseded
    origin TEXT NOT NULL,              -- extraction | edit
    plan_sha256 TEXT NOT NULL,         -- PlanGraph JSON in the project CAS
    plan_ref_json TEXT NOT NULL,       -- CasRef
    issues_json TEXT NOT NULL DEFAULT '[]',
    extraction_json TEXT NOT NULL DEFAULT '{}',  -- sources, scale estimates, notes, assist
    created_by TEXT REFERENCES users(id),
    created_at TEXT NOT NULL,
    approved_by TEXT REFERENCES users(id),
    approved_at TEXT,
    UNIQUE (project_id, number)
);
CREATE INDEX plan_versions_project ON plan_versions(project_id, number);
CREATE INDEX plan_versions_root ON plan_versions(root_id, status);

CREATE TABLE plan_edits (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    from_version TEXT NOT NULL REFERENCES plan_versions(id),
    to_version TEXT NOT NULL REFERENCES plan_versions(id),
    patch_json TEXT NOT NULL,          -- RFC 6902 JSON Patch on the PlanGraph
    summary TEXT NOT NULL DEFAULT '',
    created_by TEXT REFERENCES users(id),
    created_at TEXT NOT NULL
);

CREATE TABLE training_examples (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                -- plan_correction | assist_decision
    source_page TEXT REFERENCES pages(id) ON DELETE SET NULL,
    plan_version TEXT REFERENCES plan_versions(id),
    payload_json TEXT NOT NULL,
    training_use_allowed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX training_examples_project ON training_examples(project_id);

-- the plan version a run renders (fixed when the run first reaches S2, or by the Gate A decision)
ALTER TABLE runs ADD COLUMN plan_version TEXT REFERENCES plan_versions(id);
