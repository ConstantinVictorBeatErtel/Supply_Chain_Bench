"""Read-only JSON oracle over the frozen Hub environment."""

from __future__ import annotations

import json
import sys

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_for


def run_case(case: dict) -> dict:
    spec = scenario_for(5, case["split"], int(case["seed_index"]))
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
