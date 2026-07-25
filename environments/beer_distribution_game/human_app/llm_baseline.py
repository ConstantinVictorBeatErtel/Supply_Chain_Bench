"""Same-seed LLM eval baselines for human debrief comparison."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_for

from human_app.session import FIXED_ROLE, FIXED_TIER, FIXED_VARIANT

_BASELINE_PATH = Path(__file__).with_name("llm_baseline.json")


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def llm_model_name() -> str:
    return str(_catalog().get("model", "LLM baseline"))


def lookup_llm_baseline(split: str, seed_index: int) -> dict[str, Any] | None:
    key = f"{split}:{seed_index}"
    row = _catalog().get("by_seed", {}).get(key)
    if not row:
        return None
    out: dict[str, Any] = {
        "local_total_cost": float(row["local_total_cost"]),
        "reward": float(row["reward"]),
        "master_seed_hex": str(row.get("master_seed_hex", "")),
    }
    actions = row.get("actions")
    if isinstance(actions, list):
        out["actions"] = [int(a) for a in actions]
    return out


def llm_weekly_costs(split: str, seed_index: int) -> list[float] | None:
    """Replay stored LLM orders on the same seed to recover per-week costs."""
    baseline = lookup_llm_baseline(split, seed_index)
    if baseline is None or "actions" not in baseline:
        return None
    spec = scenario_for(
        FIXED_TIER,
        split,  # type: ignore[arg-type]
        seed_index,
        variant=FIXED_VARIANT,
    )
    episode = BeerEpisode(spec, FIXED_ROLE)
    episode.start()
    costs: list[float] = []
    for qty in baseline["actions"]:
        episode.place_order(int(qty))
        costs.append(
            float(episode.operational_transitions[-1]["local_costs"][FIXED_ROLE])
        )
    return costs
