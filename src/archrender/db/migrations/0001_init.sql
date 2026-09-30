-- ArchRender initial schema. SQLite (WAL) on local disk, replicated by Litestream (ADR-S03).

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'reviewer', 'viewer')),
    created_at TEXT NOT NULL,
    disabled INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,                 -- public key id (part of the token)
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    hash TEXT NOT NULL,                  -- HMAC-SHA256(secret, pepper), hex
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT
);

CREATE TABLE sessions (
    id_hash TEXT PRIMARY KEY,            -- SHA-256 of the session token
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE bootstrap (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    used_at TEXT NOT NULL,
    user_id TEXT NOT NULL
);

CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    timezone TEXT NOT NULL DEFAULT 'Europe/Istanbul',
    units TEXT NOT NULL DEFAULT 'metric',
    meta_json TEXT NOT NULL DEFAULT '{}',
    purged_at TEXT
);

CREATE TABLE project_members (
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('editor', 'reviewer', 'viewer')),
    PRIMARY KEY (project_id, user_id)
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size INTEGER NOT NULL,
    parent_id TEXT REFERENCES documents(id),
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (project_id, sha256)
);

CREATE TABLE uploads (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'complete', 'aborted')),
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    document_id TEXT
);

CREATE TABLE upload_chunks (
    upload_id TEXT NOT NULL REFERENCES uploads(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    PRIMARY KEY (upload_id, idx)
);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    queue TEXT NOT NULL CHECK (queue IN ('cpu', 'gpu')),
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    error_json TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    lease_owner TEXT,
    lease_until REAL,
    heartbeat_at REAL,
    progress REAL NOT NULL DEFAULT 0,
    stage TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_by TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX jobs_queue_status ON jobs(queue, status, priority DESC, created_at);
CREATE INDEX jobs_project ON jobs(project_id, created_at);

CREATE TABLE job_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    type TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX job_events_job ON job_events(job_id, id);

CREATE TABLE stage_runs (
    cache_key TEXT NOT NULL,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    stage_version TEXT NOT NULL,
    output_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    seconds REAL NOT NULL,
    vram_peak_mb REAL,
    PRIMARY KEY (project_id, cache_key)
);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE gates (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    gate TEXT NOT NULL,
    status TEXT NOT NULL,
    policy TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    decided_by TEXT,
    decided_at TEXT,
    notes TEXT,
    PRIMARY KEY (run_id, gate)
);

CREATE TABLE bundles (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    job_id TEXT NOT NULL,
    sha256 TEXT,
    size INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    user_id TEXT,
    action TEXT NOT NULL,
    target TEXT,
    detail_json TEXT NOT NULL DEFAULT '{}',
    ip TEXT
);
CREATE INDEX audit_ts ON audit_log(ts);
