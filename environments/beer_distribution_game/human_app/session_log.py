"""Fail-soft anonymous session logging to a Hugging Face Dataset repo."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "beer-game-human-sessions"
DEFAULT_INTERVAL_MINUTES = 5
SESSIONS_FILENAME = "sessions.jsonl"
ALLOWED_RECORD_KEYS = frozenset(
    {
        "session_uuid",
        "timestamp",
        "env_version",
        "tier",
        "role",
        "split",
        "seed_index",
        "seed",
        "scenario_id",
        "variant",
        "prior_beer_game_experience",
        "status",
        "actions",
        "weekly",
        "final_total_cost",
        "base_stock_cost",
        "episode_reward",
    }
)


def resolve_dataset_repo(token: str | None, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    if not token:
        return None
    try:
        from huggingface_hub import whoami

        info = whoami(token=token)
        name = info.get("name")
        if not name:
            return None
        return f"{name}/{DEFAULT_DATASET_NAME}"
    except Exception:
        logger.exception("failed to resolve HF dataset repo from token identity")
        return None


class SessionLogger:
    """Append JSON lines locally; optionally sync via CommitScheduler."""

    def __init__(
        self,
        *,
        token: str | None = None,
        dataset_repo: str | None = None,
        folder: str | Path | None = None,
        every: int = DEFAULT_INTERVAL_MINUTES,
        fallback_path: str | Path | None = None,
    ):
        self.token = token or os.environ.get("HF_TOKEN") or None
        self.dataset_repo = dataset_repo or os.environ.get("HF_DATASET_REPO") or None
        self.every = every
        self._scheduler = None
        self._lock = None
        base = Path(folder) if folder is not None else Path("human_session_logs")
        self.folder = base
        self.sessions_path = self.folder / SESSIONS_FILENAME
        self.fallback_path = Path(fallback_path or "human_sessions.jsonl")
        self._init_scheduler()

    def _init_scheduler(self) -> None:
        repo_id = resolve_dataset_repo(self.token, self.dataset_repo)
        self.dataset_repo = repo_id
        if not self.token or not repo_id:
            try:
                self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("failed to prepare local fallback log path")
            return
        try:
            from huggingface_hub import CommitScheduler, HfApi

            self.folder.mkdir(parents=True, exist_ok=True)
            if not self.sessions_path.exists():
                self.sessions_path.touch()
            api = HfApi(token=self.token)
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
                exist_ok=True,
            )
            self._scheduler = CommitScheduler(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(self.folder),
                every=self.every,
                token=self.token,
            )
            self._lock = self._scheduler.lock
        except Exception:
            logger.exception("CommitScheduler init failed; using local fallback only")
            self._scheduler = None
            self._lock = None
            try:
                self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("failed to prepare local fallback log path")

    def append(self, record: dict[str, Any]) -> bool:
        """Append one session record. Never raises into gameplay."""
        try:
            payload = {key: record[key] for key in ALLOWED_RECORD_KEYS if key in record}
            line = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        except Exception:
            logger.exception("failed to serialize session record")
            return False

        if self._scheduler is not None and self._lock is not None:
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                with self._lock:
                    with self.sessions_path.open("a", encoding="utf-8") as handle:
                        handle.write(line)
                return True
            except Exception:
                logger.exception("failed to append via CommitScheduler folder")

        try:
            self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with self.fallback_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
            return True
        except Exception:
            logger.exception("failed to append session record to fallback path")
            return False

    def log_session(self, session: Any) -> bool:
        if getattr(session, "_logged", False):
            return True
        try:
            record = session.to_log_record()
            if record.get("status") == "in_progress":
                return False
            ok = self.append(record)
            if ok:
                session._logged = True
            return ok
        except Exception:
            logger.exception("failed to log human session")
            return False
