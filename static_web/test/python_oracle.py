"""Read-only JSON oracle over the frozen Hub environment."""

from __future__ import annotations

import dataclasses
import json
import sys

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_for


def run_case(case: dict) -> dict:
    if case.get("training_v2"):
        from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2.environment import (
            training_spec,
        )

        spec = training_spec(case["seed"], index=int(case["seed_index"]))
    elif case.get("research_bucket"):
        from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
            research_spec,
        )

        spec = research_spec(
            case["seed"],
            bucket=case["research_bucket"],
            index=int(case["seed_index"]),
        )
    else:
        spec = scenario_for(5, case["split"], int(case["seed_index"]))
    if case.get("capacity") is not None:
        # Public play raises Tier-5 capacity above the research value.
        spec = dataclasses.replace(spec, capacity=int(case["capacity"]))
    episode = BeerEpisode(spec, "wholesaler")
    observation = episode.start()
    observations = [observation]
    for quantity in case["actions"]:
        result = episode.place_order(int(quantity))
        if not result["done"]:
            observation = result["next_observation"]
            observations.append(observation)
    if not episode.done or episode.outcome is None:
        raise RuntimeError("oracle action sequence did not complete")
    return {
        "episode_id": episode.episode_id,
        "observations": observations,
        "operational_transitions": episode.operational_transitions,
        "settlement_transitions": episode.settlement_transitions,
        "terminal_inventory_positions": episode.outcome["terminal_inventory_positions"],
        "final_state": episode.outcome["final_state"],
        "grade": episode.outcome["grade"],
    }


def main() -> None:
    request = json.load(sys.stdin)
    json.dump(
        {"cases": [run_case(case) for case in request["cases"]]},
        sys.stdout,
        allow_nan=False,
        separators=(",", ":"),
    )


if __name__ == "__main__":
    main()
