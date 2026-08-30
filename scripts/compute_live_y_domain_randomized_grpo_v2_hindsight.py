#!/usr/bin/env python3
"""Find a strong feasible hindsight reference on frozen v2 evaluation seeds.

The 36-step integer control problem is too large to enumerate exactly. This
script searches policy grids and then performs coordinate descent. Its result
is therefore a best-found feasible hindsight cost, not a proof of optimality.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import AdaptiveBaseStockPolicy, adaptive_policy
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2.environment import training_spec

ROLE = "wholesaler"
MANIFEST = ROOT / "experiments/live_y_domain_randomized_grpo_v2/seed_manifest.json"
EVAL_DIR = ROOT / "artifacts/live_y_domain_randomized_grpo_v2/evaluations"


def spec_for(seed: str, index: int):
    # Deliberately use the training constructor: only the held-out seed differs.
    return training_spec(seed, index=index)


def play_actions(seed: str, index: int, actions: list[int]) -> dict[str, Any]:
    episode = BeerEpisode(spec_for(seed, index), ROLE, include_reference=False)
    episode.start()
    placed: list[int] = []
    for quantity in actions:
        placed.append(int(quantity))
        result = episode.place_order(int(quantity))
        if result["done"]:
            break
    if not episode.done:
        raise RuntimeError(f"action sequence too short for evaluation seed {index}")
    grade = episode.outcome["grade"]
    return {
        "actions": placed,
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
        "demand": episode.outcome.get("research_exogenous"),
    }


def play_policy(seed: str, index: int, policy_name: str, **kwargs: Any) -> dict[str, Any]:
    episode = BeerEpisode(spec_for(seed, index), ROLE, include_reference=False)
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
            quantity = min(
                episode.spec.order_cap,
                max(0, int(kwargs["target"]) - int(observation["state"]["inventory_position"])),
            )
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


def model_warm_starts(seed: str) -> list[list[int]]:
    sequences: list[list[int]] = []
    for path in EVAL_DIR.glob("*.json"):
        payload = json.loads(path.read_text())
        for row in payload.get("rows", []):
            if row.get("seed") == seed and row.get("protocol_clean"):
                actions = row.get("actions") or []
                if len(actions) == 36:
                    sequences.append([int(x) for x in actions])
    return sequences


def coordinate_descent(
    seed: str, index: int, actions: list[int], *, radius: int = 12, passes: int = 2
) -> dict[str, Any]:
    best_actions = [int(x) for x in actions]
    best = play_actions(seed, index, best_actions)
    for _ in range(passes):
        improved = False
        for week in range(len(best_actions)):
            current = best_actions[week]
            best_qty = current
            best_cost = best["local_total_cost"]
            for quantity in range(max(0, current - radius), min(128, current + radius) + 1):
                if quantity == current:
                    continue
                trial = list(best_actions)
                trial[week] = quantity
                row = play_actions(seed, index, trial)
                if row["local_total_cost"] + 1e-12 < best_cost:
                    best_cost = row["local_total_cost"]
                    best_qty = quantity
                    best = row
                    best_actions = trial
                    improved = True
            best_actions[week] = best_qty
        if not improved:
            break
    return {"method": "coordinate_descent", **best, "actions": best_actions}


def optimize_seed(seed: str, index: int) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = [
        play_policy(seed, index, "adaptive"),
        play_policy(seed, index, "incoming"),
        play_policy(seed, index, "zero"),
    ]
    for target in range(0, 97):
        candidates.append(play_policy(seed, index, "constant_s", target=target))
    for alpha in (0.1, 0.25, 0.4, 0.6):
        for delay in (2, 3, 4, 5):
            for forecast in (4.0, 7.5, 8.0, 10.0, 12.0, 15.0, 20.0):
                candidates.append(
                    play_policy(
                        seed,
                        index,
                        "adaptive_tuned",
                        alpha=alpha,
                        delay=delay,
                        forecast=forecast,
                    )
                )
    for actions in model_warm_starts(seed):
        candidates.append({"method": "model_warm_start", **play_actions(seed, index, actions)})

    candidates.sort(key=lambda row: row["local_total_cost"])
    refined = [
        coordinate_descent(seed, index, starter["actions"])
        for starter in candidates[:3]
    ]
    best = min(candidates + refined, key=lambda row: row["local_total_cost"])
    adaptive = next(row for row in candidates if row["method"] == "adaptive")
    return {
        "index": index,
        "seed": seed,
        "best_found_hindsight_local_cost": best["local_total_cost"],
        "best_found_hindsight_system_cost": best["system_total_cost"],
        "best_found_hindsight_actions": best["actions"],
        "best_found_hindsight_method": best["method"],
        "adaptive_local_cost": adaptive["local_total_cost"],
        "adaptive_gap": adaptive["local_total_cost"] - best["local_total_cost"],
        "demand": best.get("demand"),
        "elapsed_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "hindsight_reference.json")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    seeds = manifest["evaluation"]["in_distribution"]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(optimize_seed, seed, index): (index, seed)
            for index, seed in enumerate(seeds)
        }
        for future in as_completed(futures):
            index, seed = futures[future]
            row = future.result()
            rows.append(row)
            print(json.dumps({
                "index": index,
                "seed": seed,
                "best_found": row["best_found_hindsight_local_cost"],
                "adaptive": row["adaptive_local_cost"],
                "elapsed_s": round(row["elapsed_s"], 2),
            }, sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["index"])
    total_best = sum(row["best_found_hindsight_local_cost"] for row in rows)
    total_adaptive = sum(row["adaptive_local_cost"] for row in rows)
    payload = {
        "protocol_id": "live-y-domain-randomized-grpo-v2",
        "reference": "best_found_feasible_hindsight_v1",
        "definition": (
            "Minimum wholesaler local_total_cost found by policy grids and coordinate descent "
            "on each fixed held-out v2 seed. This is feasible but not a proof of the true optimum."
        ),
        "summary": {
            "n": len(rows),
            "best_found_hindsight_total_cost": total_best,
            "best_found_hindsight_mean_cost": total_best / len(rows),
            "adaptive_total_cost": total_adaptive,
            "adaptive_mean_cost": total_adaptive / len(rows),
            "adaptive_score": 100.0 * total_best / total_adaptive,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
