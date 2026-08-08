#!/usr/bin/env python3
"""Freeze research evaluation buckets and run CPU reference policies.

Model evaluation is intentionally a separate GPU step; this command can still
validate all four demand buckets and produce the paired reference rows locally.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_game.scenario import scenario_from_dict
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import research_spec


def play(seed: str, bucket: str, policy_name: str) -> dict:
    episode = BeerEpisode(research_spec(seed, bucket=bucket), "wholesaler", include_reference=False)
    observation = episode.start()
    policy = adaptive_policy(episode.spec, "wholesaler")
    actions = []
    while not episode.done:
        quantity = 8 if policy_name == "naive_base_stock" else policy.act(observation)
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    return {
        "seed": seed,
        "bucket": bucket,
        "model": policy_name,
        "local_total_cost": grade["primary"]["local_total_cost"],
        "system_total_cost": grade["costs"]["system_total_cost"],
        "protocol_clean": grade["protocol_clean"],
        "bullwhip_ratio": grade["stability"]["bullwhip_ratio"],
        "normalized_order_volatility": grade["stability"]["normalized_order_volatility"],
        "actions": actions,
        "demand": episode.outcome["research_exogenous"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((ROOT / "experiments/live_y_domain_randomized_grpo_v1/seed_manifest.json").read_text())
    rows = []
    for bucket, seeds in manifest["evaluation"].items():
        for seed in seeds:
            for policy in ("naive_base_stock", "adaptive_base_stock"):
                rows.append(play(seed, bucket, policy))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"evaluation_kind": "CPU reference-only; no model checkpoint selection", "rows": rows}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
