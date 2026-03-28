CREATE TABLE IF NOT EXISTS storage_objects (
  object_key TEXT PRIMARY KEY,
  json_value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
