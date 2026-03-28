ALTER TABLE sessions ADD COLUMN expires_at TEXT;
ALTER TABLE sessions ADD COLUMN source_storage TEXT;

ALTER TABLE models ADD COLUMN artifact_version TEXT;
ALTER TABLE models ADD COLUMN training_mode TEXT;
ALTER TABLE models ADD COLUMN source_storage TEXT;

UPDATE sessions
SET source_storage = 'd1'
WHERE source_storage IS NULL;

UPDATE models
SET artifact_version = '2026-03-28',
    training_mode = 'cloudflare-approx',
    source_storage = 'd1'
WHERE artifact_version IS NULL
   OR training_mode IS NULL
   OR source_storage IS NULL;

CREATE TABLE IF NOT EXISTS job_logs (
  id TEXT PRIMARY KEY,
  session_id TEXT,
  model_id TEXT,
  job_type TEXT NOT NULL,
  status TEXT NOT NULL,
  detail TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
  FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_job_logs_session_id ON job_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_job_logs_model_id ON job_logs(model_id);
