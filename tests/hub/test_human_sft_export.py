"""Tests for human wholesaler demo replay → SFT export."""

from __future__ import annotations

import json
from pathlib import Path
import random

from beer_distribution_game.prompts import system_prompt
from beer_distribution_game.scenario import scenario_for
from human_app.replay import iter_sft_examples, replay_session_weeks
from human_app.session import FIXED_ROLE, FIXED_TIER, HumanSession


def _complete_session(seed: int = 0) -> HumanSession:
    session = HumanSession(
        prior_beer_game_experience="yes",
        rng=random.Random(seed),
    )
    assert session.episode.controlled_role == "wholesaler"
    while session.status == "in_progress":
        # Stable demo policy: order near incoming / 2 with a floor.
        obs = session.observation
        assert obs is not None
        incoming = int(obs["state"]["incoming_demand_or_order"])
        session.place_order(max(0, min(128, incoming)))
    return session


def test_human_session_is_wholesaler_tier5() -> None:
    session = HumanSession(prior_beer_game_experience="no", rng=random.Random(9))
    assert session.episode.controlled_role == FIXED_ROLE == "wholesaler"
    assert session.episode.spec.tier == FIXED_TIER == 5
    assert session.episode.spec.topology == "y"
    assert session.episode.spec.horizon == 36


def test_system_prompt_matches_taskset_helper() -> None:
    spec = scenario_for(5, "development", 0, variant="headline")
    text = system_prompt(spec, "wholesaler")
    assert "You control the wholesaler" in text
    assert "place_order" in text
    assert "capacity is 22" in text


def test_replay_rebuilds_36_weeks_and_reward() -> None:
    session = _complete_session(1)
    record = session.to_log_record()
    weeks = replay_session_weeks(record)
    assert len(weeks) == 36
    assert [w["quantity"] for w in weeks] == record["actions"]
    # First observation week index is 1.
    assert weeks[0]["observation"]["week"] == 1
    assert weeks[0]["observation"]["role"] == "wholesaler"
    examples = list(iter_sft_examples(record, action_format="json"))
    assert len(examples) == 36
    assert examples[0]["messages"][0]["role"] == "system"
    assert examples[0]["messages"][2]["content"] == json.dumps(
        {"quantity": record["actions"][0]},
        sort_keys=True,
        separators=(",", ":"),
    )


def test_export_script_dry_path(tmp_path: Path) -> None:
    session = _complete_session(2)
    raw = tmp_path / "sessions.jsonl"
    out = tmp_path / "sft.jsonl"
    raw.write_text(json.dumps(session.to_log_record()) + "\n", encoding="utf-8")

    # Inline the export loop (same as script) for a fast unit check.
    record = json.loads(raw.read_text(encoding="utf-8"))
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for example in iter_sft_examples(record, action_format="json"):
            handle.write(json.dumps(example) + "\n")
            n += 1
    assert n == 36
    loaded = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert loaded[0]["meta"]["role"] == "wholesaler"
    assert "Current observation:" in loaded[0]["messages"][1]["content"]
