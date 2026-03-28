CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  file_keys TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  r2_key TEXT NOT NULL,
  weights_json TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_models_session_id ON models(session_id);

CREATE TABLE IF NOT EXISTS constraints (
  id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  from_skill TEXT NOT NULL,
  to_skill TEXT NOT NULL,
  constraint_type TEXT NOT NULL,
  value REAL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_constraints_session_id ON constraints(session_id);
