CREATE TABLE IF NOT EXISTS human_sessions (
  session_uuid TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  env_version TEXT NOT NULL,
  tier INTEGER NOT NULL,
  role TEXT NOT NULL,
  split TEXT NOT NULL,
  seed_index INTEGER NOT NULL,
  seed TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  variant TEXT NOT NULL,
  prior_beer_game_experience TEXT NOT NULL,
  actions_json TEXT NOT NULL,
  weekly_json TEXT NOT NULL,
  final_total_cost REAL,
  base_stock_cost REAL,
  episode_reward REAL,
  status TEXT NOT NULL CHECK (status IN ('completed', 'abandoned')),
  completed INTEGER NOT NULL CHECK (completed IN (0, 1))
);

CREATE INDEX IF NOT EXISTS human_sessions_completed_idx
  ON human_sessions (completed, timestamp);
