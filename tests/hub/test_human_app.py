"""Tests for the Hub-parity human baseline Gradio session stack."""

from __future__ import annotations

import json
from pathlib import Path
import random
import types
from typing import Any

import pytest

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import SPLIT_SIZES, master_seed_hex, scenario_for
from human_app.logging import ALLOWED_RECORD_KEYS, SessionLogger, resolve_dataset_repo
from human_app.session import (
    EVAL_SEED_POOL,
    FIXED_ROLE,
    FIXED_TIER,
    FIXED_VARIANT,
    HUMAN_SPLITS,
    HumanSession,
)
from human_app.ui_obs import (
    OBSERVATION_KEYS,
    assert_observation_parity,
    format_observation_markdown,
    observation_field_keys,
)


def test_eval_seed_pool_is_dev_plus_validation_only() -> None:
    assert len(EVAL_SEED_POOL) == 8
    assert set(HUMAN_SPLITS) == {"development", "validation"}
    expected = {
        (split, index)
        for split in ("development", "validation")
        for index in range(SPLIT_SIZES[split])
    }
    assert set(EVAL_SEED_POOL) == expected
    for split, index in EVAL_SEED_POOL:
        assert split != "test"
        seed = master_seed_hex(split, index)
        spec = scenario_for(FIXED_TIER, split, index, variant=FIXED_VARIANT)
        assert spec.tier == 5
        assert spec.horizon == 36
        assert spec.order_cap == 128
        assert spec.master_seed_hex == seed


def test_session_draws_only_from_eval_seed_pool() -> None:
    seen: set[tuple[str, int]] = set()
    for i in range(40):
        session = HumanSession(
            prior_beer_game_experience="no",
            rng=random.Random(i),
        )
        seen.add((session.split, session.seed_index))
        assert (session.split, session.seed_index) in EVAL_SEED_POOL
        assert session.seed == master_seed_hex(session.split, session.seed_index)
        assert session.episode.controlled_role == FIXED_ROLE
        assert session.episode.spec.tier == FIXED_TIER
    assert seen


def test_observation_renderer_keys_subset_of_hub_observation() -> None:
    session = HumanSession(
        prior_beer_game_experience="unsure",
        rng=random.Random(0),
    )
    assert session.observation is not None
    keys = observation_field_keys(session.observation)
    assert keys <= OBSERVATION_KEYS
    assert_observation_parity(session.observation)
    md = format_observation_markdown(session.observation)
    assert "Inventory on hand" in md
    assert "demand history chart" not in md.lower()
    # FOW: do not surface other roles' private state labels.
    assert "retailer_a inventory" not in md.lower()
    assert "downstream" not in md.lower()


def test_completed_session_reward_matches_grade_episode() -> None:
    session = HumanSession(
        prior_beer_game_experience="yes",
        rng=random.Random(1),
    )
    actions: list[int] = []
    while session.status == "in_progress":
        qty = 8
        session.place_order(qty)
        actions.append(qty)

    assert session.status == "completed"
    assert len(actions) == 36
    record = session.to_log_record()
    assert record["status"] == "completed"
    assert record["actions"] == actions
    assert len(record["weekly"]) == 36
    assert set(record) <= ALLOWED_RECORD_KEYS

    spec = scenario_for(
        FIXED_TIER,
        session.split,
        session.seed_index,
        variant=FIXED_VARIANT,
    )
    episode = BeerEpisode(spec, FIXED_ROLE)
    episode.start()
    for qty in actions:
        episode.place_order(qty)
    assert episode.outcome is not None
    grade = episode.outcome["grade"]
    assert record["final_total_cost"] == grade["primary"]["local_total_cost"]
    assert (
        record["base_stock_cost"]
        == grade["primary"]["paired_base_stock_local_total_cost"]
    )
    assert record["episode_reward"] == grade["episode_reward"]


def test_abandon_logs_partial_without_reward(tmp_path: Path) -> None:
    session = HumanSession(
        prior_beer_game_experience="no",
        rng=random.Random(2),
    )
    session.place_order(10)
    session.place_order(12)
    session.abandon()
    record = session.to_log_record()
    assert record["status"] == "abandoned"
    assert record["actions"] == [10, 12]
    assert len(record["weekly"]) == 2
    assert record["final_total_cost"] is None
    assert record["base_stock_cost"] is None
    assert record["episode_reward"] is None

    logger = SessionLogger(
        token=None,
        folder=tmp_path / "sched",
        fallback_path=tmp_path / "fallback.jsonl",
    )
    assert logger.log_session(session) is True
    lines = (tmp_path / "fallback.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["session_uuid"] == session.session_uuid
    assert set(loaded) <= ALLOWED_RECORD_KEYS
    for banned in ("name", "email", "ip", "free_text", "user"):
        assert banned not in loaded


def test_logger_failure_does_not_break_gameplay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = types.ModuleType("huggingface_hub")

    class BoomApi:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def create_repo(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("nope")

    def boom_scheduler(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("nope")

    fake.HfApi = BoomApi  # type: ignore[attr-defined]
    fake.CommitScheduler = boom_scheduler  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake)

    logger = SessionLogger(
        token="fake-token",
        dataset_repo="user/beer-game-human-sessions",
        folder=tmp_path / "sched",
        fallback_path=tmp_path / "ok.jsonl",
    )
    session = HumanSession(
        prior_beer_game_experience="unsure",
        rng=random.Random(3),
    )
    session.place_order(5)
    assert session.status == "in_progress"
    session.abandon()
    assert logger.log_session(session) is True
    assert (tmp_path / "ok.jsonl").exists()


def test_resolve_dataset_repo_explicit_and_whoami(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert resolve_dataset_repo(None, "org/custom") == "org/custom"
    assert resolve_dataset_repo(None, None) is None

    fake = types.ModuleType("huggingface_hub")
    fake.whoami = lambda token: {"name": "alice"}  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake)
    assert resolve_dataset_repo("tok", None) == "alice/beer-game-human-sessions"


def test_order_bounds_enforced() -> None:
    session = HumanSession(
        prior_beer_game_experience="no",
        rng=random.Random(4),
    )
    with pytest.raises(ValueError):
        session.place_order(-1)
    with pytest.raises(ValueError):
        session.place_order(129)
    with pytest.raises(ValueError):
        session.place_order(True)  # type: ignore[arg-type]
    session.place_order(0)
    session.place_order(128)
    assert session.actions == [0, 128]
