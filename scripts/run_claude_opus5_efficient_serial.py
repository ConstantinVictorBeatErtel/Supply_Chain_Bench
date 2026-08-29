#!/usr/bin/env python3
"""Cost-controlled Claude Opus 5 run on the frozen serial Qwen split.

Every policy sees the same native 20-week state. The runner sends only the
decision-relevant state tuple, constrains output to a tiny JSON object, disables
reasoning, deduplicates identical weekly prompts, and enables OpenRouter's
response cache so an interrupted deterministic rerun does not repay requests.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
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
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beer_distribution_rl.env.core import Role
from eval.run_eval import HORIZON, ORDER_CAP, base_orders, make_core, read_held_out_seeds, score_against_naive


MODEL = "anthropic/claude-opus-5"
OUTPUT = ROOT / "artifacts" / "serial_claude_opus_5_efficient"


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def compact_prompt(observation: dict[str, Any]) -> str:
    fields = (
        "t", "inventory", "backlog", "inventory_position", "on_order",
        "ship_pipeline", "order_pipeline", "last_demand_or_order",
        "last_shipment_received", "last_order_placed",
    )
    values = ",".join(str(observation[name]) for name in fields)
    return (
        "20-week serial Beer Game. Control wholesaler; minimize total supply-chain "
        "holding+backlog cost. Other roles are fixed base-stock. Return JSON only "
        "as {\"q\":integer 0..128}. "
        "s=[week,inv,backlog,inv_pos,on_order,ship_pipe,order_pipe,last_in,last_recv,last_order]="
        f"[{values}]"
    )


class Client:
    def __init__(self, output: Path, budget_usd: float, workers: int):
        self.output = output
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.budget_usd = budget_usd
        self.workers = workers
        self.lock = threading.Lock()
        self.cost = 0.0
        self.calls = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def one(self, prompt: str) -> tuple[int, bool]:
        with self.lock:
            if self.cost >= self.budget_usd:
                return 0, False
        body = {
            "model": MODEL,
            "max_tokens": 16,
            "reasoning": {"effort": "none", "exclude": True},
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "wholesaler_order",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"q": {"type": "integer"}},
                        "required": ["q"],
                        "additionalProperties": False,
                    },
                },
            },
            "provider": {"sort": "price"},
            "usage": {"include": True},
        }
        headers = {
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ConstantinVictorBeatErtel/Supply_Chain_Bench",
                "X-Title": "Beer Distribution Serial Opus 5 Evaluation",
                "X-OpenRouter-Cache": "true",
                "X-OpenRouter-Cache-TTL": "86400",
            }
        def send(payload: dict[str, Any]) -> dict[str, Any]:
            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload, separators=(",", ":")).encode(),
                headers=headers,
                method="POST",
            )
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=120, context=ssl_context()) as handle:
                        return json.loads(handle.read().decode())
                except urllib.error.HTTPError as error:
                    try:
                        error_payload = json.loads(error.read().decode())
                    except json.JSONDecodeError:
                        if attempt == 3:
                            return {}
                        time.sleep(2**attempt)
                        continue
                    if error.code != 400 and attempt < 3:
                        time.sleep(2**attempt)
                        continue
                    return error_payload
                except (urllib.error.URLError, TimeoutError):
                    if attempt == 3:
                        return {}
                    time.sleep(2**attempt)
            return {}
        response: dict[str, Any] = {}
        response = send(body)
        if "structured_outputs not supported" in str(response.get("error", {}).get("message", "")):
            fallback = copy.deepcopy(body)
            fallback.pop("response_format")
            response = send(fallback)
        usage = response.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        with self.lock:
            self.calls += 1
            self.cost += float(usage.get("cost") or 0.0)
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.cache_hits += int(bool(usage.get("total_tokens") == 0 or details.get("cached_tokens")))
            with (self.output / "responses.jsonl").open("a") as handle:
                handle.write(json.dumps({"prompt": prompt, "response": response}, sort_keys=True) + "\n")
        try:
            content = str(response["choices"][0]["message"]["content"])
            quantity = int(json.loads(content)["q"])
            return quantity, 0 <= quantity <= ORDER_CAP
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return 0, False

    def orders(self, observations: list[dict[str, Any]]) -> tuple[list[int], int]:
        groups: dict[str, list[int]] = {}
        for index, observation in enumerate(observations):
            groups.setdefault(compact_prompt(observation), []).append(index)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            responses = list(pool.map(self.one, groups))
        orders = [0] * len(observations)
        failures = 0
        for indexes, (quantity, valid) in zip(groups.values(), responses, strict=True):
            if not valid:
                failures += len(indexes)
            for index in indexes:
                orders[index] = quantity if valid else 0
        return orders, failures


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--budget-usd", type=float, default=4.5)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    seeds = read_held_out_seeds()[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    state_path = args.output / "state.json"
    state = {"orders": [[] for _ in seeds], "failures": 0}
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if len(state["orders"]) != len(seeds):
            raise ValueError("state seed count does not match this run")
    completed = {len(actions) for actions in state["orders"]}
    if len(completed) != 1:
        raise ValueError("checkpoint contains unequal episode lengths")
    completed_weeks = completed.pop()
    cores = [make_core(seed) for seed in seeds]
    costs = [0.0 for _ in cores]
    retailer_orders: list[int] = []
    customer_demands: list[int] = []
    for week in range(completed_weeks):
        for index, core in enumerate(cores):
            _, _, _, info = core.step(base_orders(core, int(state["orders"][index][week])))
            costs[index] += float(info.system_cost)
            retailer_orders.append(int(info.orders_placed[Role.RETAILER]))
            customer_demands.append(int(info.customer_demand or 0))
    client = Client(args.output, args.budget_usd, args.workers)
    for week in range(completed_weeks, HORIZON):
        observations = [core.observe(Role.WHOLESALER) for core in cores]
        orders, failures = client.orders(observations)
        state["failures"] += failures
        for index, (core, order) in enumerate(zip(cores, orders, strict=True)):
            _, _, _, info = core.step(base_orders(core, order))
            state["orders"][index].append(order)
            costs[index] += float(info.system_cost)
            retailer_orders.append(int(info.orders_placed[Role.RETAILER]))
            customer_demands.append(int(info.customer_demand or 0))
        save_json(state_path, state)
        print(f"week {week + 1}/{HORIZON}; requests={client.calls}; cost=${client.cost:.4f}", flush=True)
        if client.cost >= args.budget_usd:
            break
    if len(state["orders"][0]) != HORIZON:
        print("budget reached before full benchmark completion", flush=True)
        return
    naive_cost = 1730.82
    mean_cost = statistics.fmean(costs)
    demand_variance = statistics.pvariance(customer_demands)
    payload = {
        "protocol": {
            "model": MODEL,
            "split": "eval/held_out_seeds.json",
            "seed_count": len(seeds),
            "horizon_weeks": HORIZON,
            "prompting": "compact stateless state tuple; JSON schema; max 16 tokens; reasoning disabled",
            "cost_controls": "deduplicated same-week prompts; OpenRouter response caching enabled; resume checkpoint",
        },
        "metrics": {
            "mean_total_supply_chain_cost": mean_cost,
            "stderr": statistics.stdev(costs) / math.sqrt(len(costs)) if len(costs) > 1 else 0.0,
            "bullwhip_ratio": statistics.pvariance(retailer_orders) / demand_variance,
            "order_format_failure_rate": state["failures"] / (len(seeds) * HORIZON),
            "episodes": len(seeds),
            "benchmark_score_0_to_100": score_against_naive(mean_cost, naive_cost),
            "api_usage_cost_usd": client.cost,
            "api_requests": client.calls,
            "response_cache_hits": client.cache_hits,
            "prompt_tokens": client.prompt_tokens,
            "completion_tokens": client.completion_tokens,
        },
    }
    save_json(args.output / "results.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
