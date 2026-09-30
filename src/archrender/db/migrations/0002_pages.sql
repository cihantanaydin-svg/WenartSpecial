-- S0 output: pages of ingested documents (PageRef JSON), and the S1 review queue.
CREATE TABLE pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    kind TEXT NOT NULL,
    page_json TEXT NOT NULL,
    UNIQUE (document_id, idx)
);
CREATE INDEX pages_project ON pages(project_id);

CREATE TABLE review_items (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,              -- page_class | ocr_conflict | schedule_link | …
    subject_id TEXT NOT NULL,        -- page id, schedule row id, …
    status TEXT NOT NULL DEFAULT 'open',  -- open | resolved | dismissed
    payload_json TEXT NOT NULL DEFAULT '{}',
    resolution_json TEXT,
    created_at TEXT NOT NULL,
    resolved_by TEXT REFERENCES users(id),
    resolved_at TEXT,
    UNIQUE (kind, subject_id)
);
CREATE INDEX review_items_project ON review_items(project_id, status);
