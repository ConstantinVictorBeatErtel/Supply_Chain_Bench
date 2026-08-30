#!/usr/bin/env python3
"""Hindsight-perfect wholesaler costs for the fixed live-Y research seeds.

Demand and scripted counterparties are CRN-fixed by seed, so each evaluation
episode is a deterministic open-loop control problem. Exact enumeration of
0..128^36 is intractable; this script finds the best feasible wholesaler
action sequence via:

1. adaptive / incoming / constant order-up-to grids
2. warm starts from recorded model action traces (when available)
3. coordinate-descent local search

The returned cost is a feasible upper bound on the true optimum (true optimum
≤ reported perfect). Adaptive base-stock is not assumed optimal.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import AdaptiveBaseStockPolicy, adaptive_policy, initial_forecast
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    research_spec,
)

ROLE = "wholesaler"
EVAL_DIR = ROOT / "artifacts/live_y_domain_randomized_grpo_v1/evaluations"
WARM_START_FILES = (
    "untrained_base_research_prompt.json",
    "final_per_turn_grpo_research_prompt.json",
    "openrouter_openai_gpt-5.6-luna.json",
    "openrouter_deepseek_deepseek-v4-flash-0731.json",
    "openrouter_poolside_laguna-s-2.1_free.json",
    "openrouter_z-ai_glm-5.3-flash.json",
)


def play_actions(seed: str, bucket: str, actions: list[int]) -> dict[str, Any]:
    episode = BeerEpisode(research_spec(seed, bucket=bucket), ROLE, include_reference=False)
    observation = episode.start()
    placed: list[int] = []
    for quantity in actions:
        placed.append(int(quantity))
        result = episode.place_order(int(quantity))
        if result["done"]:
            break
        observation = result["next_observation"]
    if not episode.done:
        raise RuntimeError(f"action sequence too short for {bucket}/{seed}")
    grade = episode.outcome["grade"]
    return {
        "actions": placed,
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
        "demand": episode.outcome.get("research_exogenous"),
    }


def play_policy(seed: str, bucket: str, policy_name: str, **kwargs: Any) -> dict[str, Any]:
    episode = BeerEpisode(research_spec(seed, bucket=bucket), ROLE, include_reference=False)
    observation = episode.start()
    actions: list[int] = []
    if policy_name == "adaptive":
        policy: Any = adaptive_policy(episode.spec, ROLE)
    elif policy_name == "adaptive_tuned":
        policy = AdaptiveBaseStockPolicy(
            forecast=float(kwargs["forecast"]),
            alpha=float(kwargs["alpha"]),
            order_cap=episode.spec.order_cap,
            replenishment_delay=int(kwargs["delay"]),
        )
    else:
        policy = None
    while not episode.done:
        if policy_name == "incoming":
            quantity = int(observation["state"]["incoming_demand_or_order"])
        elif policy_name == "zero":
            quantity = 0
        elif policy_name == "constant_s":
            ip = int(observation["state"]["inventory_position"])
            quantity = min(episode.spec.order_cap, max(0, int(kwargs["target"]) - ip))
        else:
            quantity = int(policy.act(observation))
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    return {
        "method": policy_name if policy_name != "constant_s" else f"constant_s_{kwargs['target']}",
        "actions": actions,
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
        "demand": episode.outcome.get("research_exogenous"),
    }


def load_warm_starts(seed: str, bucket: str) -> list[list[int]]:
    sequences: list[list[int]] = []
    for name in WARM_START_FILES:
        path = EVAL_DIR / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            if row.get("seed") == seed and row.get("bucket") == bucket and row.get("protocol_clean"):
                actions = row.get("actions") or []
                if len(actions) == 36:
                    sequences.append([int(x) for x in actions])
    return sequences


def coordinate_descent(
    seed: str,
    bucket: str,
    actions: list[int],
    *,
    radius: int = 24,
    passes: int = 3,
) -> dict[str, Any]:
    best_actions = [int(x) for x in actions]
    best = play_actions(seed, bucket, best_actions)
    for _ in range(passes):
        improved = False
        for week in range(len(best_actions)):
            current = best_actions[week]
            best_qty = current
            best_cost = best["local_total_cost"]
            lo = max(0, current - radius)
            hi = min(128, current + radius)
            for quantity in range(lo, hi + 1):
                if quantity == current:
                    continue
                trial = list(best_actions)
                trial[week] = quantity
                row = play_actions(seed, bucket, trial)
                if row["local_total_cost"] + 1e-12 < best_cost:
                    best_cost = row["local_total_cost"]
                    best_qty = quantity
                    best = row
                    best_actions = trial
                    improved = True
            best_actions[week] = best_qty
        if not improved:
            break
    best["method"] = "coordinate_descent"
    best["actions"] = best_actions
    return best


def optimize_seed(seed: str, bucket: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    candidates: list[dict[str, Any]] = [
        play_policy(seed, bucket, "adaptive"),
        play_policy(seed, bucket, "incoming"),
        play_policy(seed, bucket, "zero"),
    ]
    for target in range(0, 97):
        candidates.append(play_policy(seed, bucket, "constant_s", target=target))
    for alpha in (0.1, 0.25, 0.4, 0.6):
        for delay in (2, 3, 4, 5):
            for forecast in (4.0, 7.5, 8.0, 10.0, 12.0, 15.0, 20.0):
                candidates.append(
                    play_policy(
                        seed,
                        bucket,
                        "adaptive_tuned",
                        alpha=alpha,
                        delay=delay,
                        forecast=forecast,
                    )
                )
    for actions in load_warm_starts(seed, bucket):
        candidates.append({"method": "warm_start", **play_actions(seed, bucket, actions)})

    candidates.sort(key=lambda row: row["local_total_cost"])
    # Refine the best few starts with a cheaper local search.
    refined: list[dict[str, Any]] = []
    for starter in candidates[:3]:
        refined.append(
            coordinate_descent(seed, bucket, starter["actions"], radius=12, passes=2)
        )
    best = min(candidates + refined, key=lambda row: row["local_total_cost"])
    adaptive = next(row for row in candidates if row["method"] == "adaptive")
    return {
        "seed": seed,
        "bucket": bucket,
        "perfect_local_cost": best["local_total_cost"],
        "perfect_system_cost": best["system_total_cost"],
        "perfect_actions": best["actions"],
        "perfect_method": best.get("method"),
        "adaptive_local_cost": adaptive["local_total_cost"],
        "adaptive_gap": adaptive["local_total_cost"] - best["local_total_cost"],
        "top_heuristic_local_cost": candidates[0]["local_total_cost"],
        "top_heuristic_method": candidates[0]["method"],
        "demand": best.get("demand"),
        "elapsed_s": time.perf_counter() - t0,
        "note": (
            "Feasible hindsight minimum under fixed CRN demand and scripted "
            "counterparties. True optimum is ≤ this value."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "hindsight_perfect_costs.json",
    )
    args = parser.parse_args()
    manifest = json.loads(
        (ROOT / "experiments/live_y_domain_randomized_grpo_v1/seed_manifest.json").read_text()
    )
    jobs = [
        (seed, bucket)
        for bucket, seeds in manifest["evaluation"].items()
        for seed in seeds
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(optimize_seed, seed, bucket): (bucket, seed) for seed, bucket in jobs
        }
        for future in as_completed(futures):
            bucket, seed = futures[future]
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "bucket": bucket,
                        "seed": seed,
                        "perfect": row["perfect_local_cost"],
                        "adaptive": row["adaptive_local_cost"],
                        "gap": row["adaptive_gap"],
                        "method": row["perfect_method"],
                        "elapsed_s": round(row["elapsed_s"], 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    rows.sort(key=lambda row: (row["bucket"], row["seed"]))
    perfect_mean = sum(row["perfect_local_cost"] for row in rows) / len(rows)
    adaptive_mean = sum(row["adaptive_local_cost"] for row in rows) / len(rows)
    payload = {
        "protocol_id": "live-y-domain-randomized-grpo-v1",
        "reference": "hindsight_search_perfect",
        "definition": (
            "Minimum wholesaler local_total_cost found by order-up-to / tuned-adaptive "
            "grids, model warm starts, and coordinate descent on each fixed CRN seed. "
            "Feasible upper bound on the true hindsight optimum."
        ),
        "summary": {
            "n": len(rows),
            "perfect_mean_local_cost": perfect_mean,
            "adaptive_mean_local_cost": adaptive_mean,
            "adaptive_over_perfect": adaptive_mean / perfect_mean if perfect_mean else None,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
