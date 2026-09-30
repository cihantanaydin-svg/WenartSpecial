-- S1 results per page (latest analysis), user overrides, and project schedules.
CREATE TABLE page_analysis (
    page_id TEXT PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    needs_review INTEGER NOT NULL DEFAULT 0,
    override_label TEXT,
    override_by TEXT REFERENCES users(id),
    analysis_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX page_analysis_project ON page_analysis(project_id);

CREATE TABLE schedules (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_page TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    schedule_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX schedules_project ON schedules(project_id);
