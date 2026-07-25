"""Replay logged human sessions into (observation, action) weeks for SFT."""

from __future__ import annotations

from typing import Any, Iterator

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import ActionFormat, sft_chat_example
from beer_distribution_game.scenario import Variant, scenario_for

from .session import FIXED_ROLE, FIXED_TIER, FIXED_VARIANT


def replay_session_weeks(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministically rebuild per-week observations from seed + actions."""
    if record.get("status") != "completed":
        raise ValueError("only completed sessions can be replayed for SFT")
    role = record.get("role", FIXED_ROLE)
    tier = int(record.get("tier", FIXED_TIER))
    variant_raw = record.get("variant", FIXED_VARIANT)
    if variant_raw not in ("headline", "t5_control_base_rival", "t5_control_uniform"):
        raise ValueError(f"unsupported variant: {variant_raw!r}")
    variant: Variant = variant_raw
    split = record["split"]
    seed_index = int(record["seed_index"])
    actions = list(record["actions"])
    if len(actions) != 36:
        raise ValueError(f"expected 36 actions, found {len(actions)}")

    spec = scenario_for(tier, split, seed_index, variant=variant)
    if spec.master_seed_hex != record.get("seed"):
        raise ValueError("logged seed does not match scenario_for derivation")
    if role != FIXED_ROLE:
        raise ValueError(f"expected role {FIXED_ROLE!r}, found {role!r}")

    episode = BeerEpisode(spec, role)
    observation = episode.start()
    weeks: list[dict[str, Any]] = []
    for quantity in actions:
        weeks.append(
            {
                "observation": observation,
                "quantity": int(quantity),
                "spec": spec,
                "role": role,
            }
        )
        result = episode.place_order(int(quantity))
        if result["done"]:
            observation = None
        else:
            observation = result["next_observation"]
    if not episode.done or episode.outcome is None:
        raise RuntimeError("replay did not complete the episode")
    return weeks


def iter_sft_examples(
    record: dict[str, Any],
    *,
    action_format: ActionFormat = "json",
    require_beat_base_stock: bool = False,
) -> Iterator[dict[str, Any]]:
    if require_beat_base_stock:
        final_cost = record.get("final_total_cost")
        base_cost = record.get("base_stock_cost")
        if final_cost is None or base_cost is None or not (final_cost < base_cost):
            return
    for week in replay_session_weeks(record):
        example = sft_chat_example(
            week["spec"],
            week["role"],
            week["observation"],
            week["quantity"],
            action_format=action_format,
        )
        example["meta"]["session_uuid"] = record.get("session_uuid")
        example["meta"]["episode_reward"] = record.get("episode_reward")
        example["meta"]["prior_beer_game_experience"] = record.get(
            "prior_beer_game_experience"
        )
        yield example
