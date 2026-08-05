#!/usr/bin/env python3
"""Token-efficient Claude Opus 5 evaluation on the live-Y eval split.

The runner preserves every decision-relevant observation field while replacing
verbose JSON keys with a documented compact tuple format.  It uses a fresh
request per week, forced tool use, no reasoning, a 16-token output ceiling,
parallel episodes, per-week checkpoints, and a hard usage-cost guard.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import ssl
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "beer_distribution_game"
import sys

for path in (ROOT, ENV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_for

import scripts.run_frontier_t5_y_36w as benchmark


MODEL_KEY = "claude_opus_5"
MODEL_ID = "anthropic/claude-opus-5"
DEFAULT_OUTPUT = ROOT / "artifacts" / "live_y_qwen35_4b_rl" / "claude_opus_5_efficient"
SPLIT_PATH = ROOT / "experiments" / "live_y_qwen_rl" / "splits.json"

SYSTEM = (
    "Control wholesaler in a 36-week Y beer supply chain. Minimize only local "
    "holding/backlog cost (0.5/1 per unit-week). Order lead=1 week; shipment "
    "lead=2. Two retailers have persistent stochastic demand, mean 7.5 each. "
    "Factory cap=22; shortages use proportional allocation; a hidden scripted "
    "rival may order aggressively. Call place_order exactly once with integer "
    "q in [0,128]. No prose."
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "place_order",
        "description": "Place this week's replenishment order.",
        "parameters": {
            "type": "object",
            "properties": {"quantity": {"type": "integer", "minimum": 0, "maximum": 128}},
            "required": ["quantity"],
            "additionalProperties": False,
        },
    },
}]


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def compact_observation(observation: dict[str, Any]) -> str:
    """Information-preserving compact encoding of the decision state.

    s = inventory, backlog, inventory position, on order, received, incoming,
        units filled, last order
    c = current local cost, cumulative local cost through the prior week
    hist rows = week, incoming, received, order, ending inventory, ending
        backlog, local cost
    """

    state = observation["state"]
    costs = observation["costs"]
    history = [
        [
            row["week"],
            row["incoming_demand_or_order"],
            row["shipment_received"],
            row["order_placed"],
            row["ending_inventory"],
            row["ending_backlog"],
            row["local_cost"],
        ]
        for row in observation["recent_history"]
    ]
    payload = {
        "w": observation["week"],
        "left": observation["weeks_remaining"],
        "mode": observation["observation_mode"],
        "s": [
            state["inventory_on_hand"],
            state["backlog"],
            state["inventory_position"],
            state["on_order"],
            state["shipment_received"],
            state["incoming_demand_or_order"],
            state["units_filled"],
            state["last_order_placed"],
        ],
        "c": [
            costs["current_inventory_backlog_cost"],
            costs["cumulative_local_cost_through_previous_week"],
        ],
        "hist": history,
    }
    return (
        "Fields: s=[inv,backlog,inv_pos,on_order,received,incoming,filled,last]; "
        "c=[week_cost,cumulative_prior]; hist=[w,in,recv,order,end_inv,end_backlog,cost].\n"
        + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    )


def parse_action(response: dict[str, Any]) -> tuple[int | None, str | None]:
    return benchmark.parse_tool_action(response)


class Client:
    def __init__(self, output: Path, budget_usd: float):
        self.output = output
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.budget_usd = budget_usd
        self.lock = threading.Lock()
        self.cost_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        for path in sorted((output / "state").glob("*.json")):
            row = json.loads(path.read_text())
            self.cost_usd += float(row.get("usage_cost_usd", 0.0))
            self.prompt_tokens += int(row.get("prompt_tokens", 0))
            self.completion_tokens += int(row.get("completion_tokens", 0))

    def call(self, observation: dict[str, Any], seed_index: int, week: int) -> tuple[int | None, str | None, dict[str, Any]]:
        with self.lock:
            if self.cost_usd >= self.budget_usd:
                return None, "budget_exhausted", {}
        body = {
            "model": MODEL_ID,
            "temperature": 0,
            "max_tokens": 32,
            "reasoning": {"effort": "none"},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": compact_observation(observation)},
            ],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "place_order"}},
            "parallel_tool_calls": False,
            "provider": {"sort": "price"},
            "usage": {"include": True},
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body, separators=(",", ":")).encode(),
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ConstantinVictorBeatErtel/beer_distribution_RL",
                "X-Title": "Beer Distribution Claude Opus 5 Efficient Evaluation",
            },
            method="POST",
        )
        response: dict[str, Any] = {}
        error: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=120, context=ssl_context()) as handle:
                    response = json.loads(handle.read().decode())
                break
            except urllib.error.HTTPError as exc:
                try:
                    error = {"status": exc.code, "body": json.loads(exc.read().decode())}
                except Exception:
                    error = {"status": exc.code}
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            time.sleep(2**attempt)
        if not response:
            response = {"_request_error": error or {"message": "request failed"}}
        usage = response.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        with self.lock:
            self.cost_usd += cost
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            with (self.output / "responses.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "seed_index": seed_index,
                    "week": week,
                    "usage": usage,
                    "response": response,
                }, sort_keys=True) + "\n")
        quantity, parse_error = parse_action(response)
        return quantity, parse_error, {
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def run_episode(client: Client, seed_hex: str, seed_index: int) -> dict[str, Any]:
    template = scenario_for(5, "test", 0, "headline")
    spec = replace(
        template,
        split="research_eval",
        seed_index=seed_index,
        master_seed_hex=seed_hex,
    )
    episode = BeerEpisode(spec, benchmark.ROLE, include_reference=False)
    observation = episode.start()
    state_path = client.output / "state" / f"{seed_index:03d}.json"
    state = {
        "seed_index": seed_index,
        "master_seed_hex": seed_hex,
        "actions": [],
        "format_failures": 0,
        "usage_cost_usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "status": "running",
    }
    if state_path.is_file():
        loaded = json.loads(state_path.read_text())
        if loaded.get("master_seed_hex") != seed_hex:
            raise ValueError(f"seed mismatch in {state_path}")
        state.update(loaded)
        for quantity in state["actions"]:
            result = episode.place_order(int(quantity))
            if not result["done"]:
                observation = result["next_observation"]
    if state.get("status") == "complete":
        return state
    while not episode.done:
        quantity, error, usage = client.call(observation, seed_index, int(observation["week"]))
        state["usage_cost_usd"] += float(usage.get("cost", 0.0))
        state["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        state["completion_tokens"] += int(usage.get("completion_tokens", 0))
        if error is not None:
            state["format_failures"] += 1
            state["status"] = error
            save_state(state_path, state)
            return state
        assert quantity is not None
        result = episode.place_order(quantity)
        state["actions"].append(quantity)
        save_state(state_path, state)
        if result["done"]:
            break
        observation = result["next_observation"]
    assert episode.outcome is not None
    grade = episode.outcome["grade"]
    state.update({
        "episode_id": episode.episode_id,
        "status": "complete",
        "protocol_clean": bool(grade.get("protocol_clean")),
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
        "grade": grade,
    })
    save_state(state_path, state)
    return state


def aggregate(rows: list[dict[str, Any]], naive_cost: float) -> dict[str, Any]:
    complete = [row for row in rows if row.get("status") == "complete"]
    costs = [float(row["local_total_cost"]) for row in complete]
    mean_cost = statistics.fmean(costs) if costs else None
    stderr = statistics.stdev(costs) / math.sqrt(len(costs)) if len(costs) > 1 else 0.0 if costs else None
    score = 100.0 * naive_cost / (naive_cost + mean_cost) if mean_cost is not None else None
    return {
        "model_id": MODEL_ID,
        "episodes": len(rows),
        "completed_episodes": len(complete),
        "protocol_clean_rate": len(complete) / len(rows) if rows else 0.0,
        "format_failures": sum(int(row.get("format_failures", 0)) for row in rows),
        "mean_local_wholesaler_cost": mean_cost,
        "stderr": stderr,
        "score_0_to_100": score,
        "usage_cost_usd": sum(float(row.get("usage_cost_usd", 0.0)) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--budget-usd", type=float, default=6.5)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    split = json.loads(SPLIT_PATH.read_text())
    seeds = [str(seed) for seed in split["eval"]][: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    client = Client(args.output, args.budget_usd)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_episode, client, seed, index): index
            for index, seed in enumerate(seeds)
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(f"seed {row['seed_index']:03d}: {row['status']}", flush=True)
    rows.sort(key=lambda row: int(row["seed_index"]))
    specs = [
        replace(
            scenario_for(5, "test", 0, "headline"),
            split="research_eval",
            seed_index=index,
            master_seed_hex=seed,
        )
        for index, seed in enumerate(seeds)
    ]
    naive_rows = [benchmark.run_policy(spec, "naive") for spec in specs]
    oracle_rows = [benchmark.run_policy(spec, "oracle") for spec in specs]
    naive_cost = statistics.fmean(float(row["local_total_cost"]) for row in naive_rows)
    oracle_cost = statistics.fmean(float(row["local_total_cost"]) for row in oracle_rows)
    result = {
        "protocol": {
            "environment_version": benchmark.ENVIRONMENT_VERSION,
            "scenario_id": "t5-strategic-y-v2",
            "controlled_role": benchmark.ROLE,
            "horizon_operational_weeks": 36,
            "split": str(SPLIT_PATH.relative_to(ROOT)),
            "prompt": "compact information-preserving state; stateless weekly calls",
            "reasoning": "disabled",
            "max_tokens": 32,
            "tool_choice": "forced place_order",
            "budget_usd": args.budget_usd,
        },
        "baselines": {
            "naive_mean_local_cost": naive_cost,
            "adaptive_base_stock_mean_local_cost": oracle_cost,
        },
        "metrics": aggregate(rows, naive_cost),
        "episodes": rows,
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
