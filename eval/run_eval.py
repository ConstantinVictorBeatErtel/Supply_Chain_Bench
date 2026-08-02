#!/usr/bin/env python3
"""Frozen 20-week wholesaler benchmark over the native BeerGameCore."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from beer_distribution_rl.agents.baselines import base_stock_order
from beer_distribution_rl.env.core import BeerGameCore, Role, classic_env_config
from beer_distribution_rl.env.demand import AR1Demand

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT_PATH = ROOT / "eval" / "held_out_seeds.json"
RESULT_PATH = ROOT / "results" / "baseline.json"
HORIZON = 20
ORDER_CAP = 128
TRAIN_RNG_SEED = 20260803


class BatchPolicy(Protocol):
    name: str
    failures: int
    calls: int

    def orders(self, observations: list[dict[str, Any]]) -> list[int]: ...


def read_eval_seeds(path: Path = HELD_OUT_PATH) -> list[int]:
    """Read a frozen 100-seed evaluation split without sampling at runtime."""
    payload = json.loads(path.read_text())
    seeds = [int(seed) for seed in payload["seeds"]]
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError(f"evaluation split {path} must contain exactly 100 unique seeds")
    return seeds


def read_held_out_seeds() -> list[int]:
    return read_eval_seeds(HELD_OUT_PATH)


def training_seeds(n: int = 100) -> list[int]:
    import random

    rng = random.Random(TRAIN_RNG_SEED)
    seeds = [rng.randrange(1, 2**63) for _ in range(n)]
    held_out = set(read_held_out_seeds())
    overlap = held_out.intersection(seeds)
    if overlap:
        raise AssertionError(f"training seed appears in held-out split: {sorted(overlap)!r}")
    return seeds


def make_core(seed: int) -> BeerGameCore:
    cfg = classic_env_config(
        horizon=HORIZON,
        order_cap=ORDER_CAP,
        demand=AR1Demand(mu=7.5, phi=0.7, sigma=2.0, x0=7.5),
        seed=int(seed),
        signaling_enabled=False,
        regime="A",
    )
    core = BeerGameCore(cfg)
    core.reset(seed)
    return core


def prompt_for(observation: dict[str, Any]) -> str:
    state = {
        key: observation[key]
        for key in (
            "t",
            "inventory",
            "backlog",
            "inventory_position",
            "on_order",
            "ship_pipeline",
            "order_pipeline",
            "last_demand_or_order",
            "last_shipment_received",
            "last_order_placed",
        )
    }
    return (
        "You are the wholesaler in a 20-week serial Beer Distribution Game. "
        "Minimize total supply-chain holding plus backlog cost. Retailer, distributor, "
        "and factory use fixed base-stock policies. Reasoning is disabled. "
        "Return exactly one integer order quantity from 0 through 128, with no other text.\n"
        "Wholesaler observation: "
        + json.dumps(state, separators=(",", ":"))
    )


@dataclass
class NaivePolicy:
    name: str = "naive"
    failures: int = 0
    calls: int = 0

    def orders(self, observations: list[dict[str, Any]]) -> list[int]:
        self.calls += len(observations)
        return [max(0, min(ORDER_CAP, int(obs["last_demand_or_order"]))) for obs in observations]


@dataclass
class BaseStockPolicy:
    level: int
    name: str = "base_stock"
    failures: int = 0
    calls: int = 0

    def orders(self, observations: list[dict[str, Any]]) -> list[int]:
        self.calls += len(observations)
        return [
            max(0, min(ORDER_CAP, int(self.level - int(obs["inventory_position"]))))
            for obs in observations
        ]


@dataclass
class OpenRouterPolicy:
    model: str
    workers: int = 16
    name: str = "large_model"
    failures: int = 0
    calls: int = 0

    def _one(self, observation: dict[str, Any]) -> tuple[int, bool]:
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 8,
            "reasoning": {"effort": "none"},
            "messages": [{"role": "user", "content": prompt_for(observation)}],
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/ConstantinVictorBeatErtel/beer_distribution_RL",
                "X-Title": "Beer Distribution Benchmark",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode())
            content = str(data["choices"][0]["message"].get("content", "")).strip()
            match = re.fullmatch(r"\s*(\d{1,3})\s*", content)
            if not match:
                return 0, False
            quantity = int(match.group(1))
            return quantity, 0 <= quantity <= ORDER_CAP
        except (KeyError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
            return 0, False

    def orders(self, observations: list[dict[str, Any]]) -> list[int]:
        self.calls += len(observations)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            outputs = list(pool.map(self._one, observations))
        self.failures += sum(not ok for _, ok in outputs)
        return [order if ok else 0 for order, ok in outputs]


class LocalQwenPolicy:
    name = "qwen_base"

    def __init__(
        self, model_name: str, name: str = "qwen_base", adapter_path: str | None = None
    ) -> None:
        self.name = name
        import torch
        from unsloth import FastLanguageModel

        self.torch = torch
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        FastLanguageModel.for_inference(self.model)
        self.failures = 0
        self.calls = 0

    def orders(self, observations: list[dict[str, Any]]) -> list[int]:
        self.calls += len(observations)
        chats = [
            [
                {
                    "role": "user",
                    "content": prompt_for(observation),
                }
            ]
            for observation in observations
        ]
        rendered = [
            self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for chat in chats
        ]
        inputs = self.tokenizer(text=rendered, return_tensors="pt", padding=True).to("cuda")
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        raw = self.tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        orders: list[int] = []
        for text in raw:
            match = re.fullmatch(r"\s*(\d{1,3})\s*", text)
            if not match or not 0 <= int(match.group(1)) <= ORDER_CAP:
                self.failures += 1
                orders.append(0)
            else:
                orders.append(int(match.group(1)))
        return orders


def base_orders(core: BeerGameCore, wholesaler_order: int) -> dict[Role, int]:
    levels = {
        Role.RETAILER: 9,
        Role.WHOLESALER: 5,
        Role.DISTRIBUTOR: 3,
        Role.FACTORY: 1,
    }
    orders = {
        role: base_stock_order(core._states[role], levels[role], order_cap=ORDER_CAP)
        for role in core.roles
    }
    orders[Role.WHOLESALER] = int(wholesaler_order)
    return orders


def evaluate(policy: BatchPolicy, seeds: list[int]) -> dict[str, float | int]:
    cores = [make_core(seed) for seed in seeds]
    total_costs = [0.0 for _ in cores]
    retailer_orders: list[int] = []
    customer_demands: list[int] = []
    for _week in range(HORIZON):
        observations = [core.observe(Role.WHOLESALER) for core in cores]
        orders = policy.orders(observations)
        if len(orders) != len(cores):
            raise ValueError("policy returned the wrong batch size")
        for index, (core, wholesaler_order) in enumerate(zip(cores, orders, strict=True)):
            _, _, _, info = core.step(base_orders(core, wholesaler_order))
            total_costs[index] += float(info.system_cost)
            retailer_orders.append(int(info.orders_placed[Role.RETAILER]))
            customer_demands.append(int(info.customer_demand or 0))
    mean = statistics.fmean(total_costs)
    stderr = statistics.stdev(total_costs) / math.sqrt(len(total_costs)) if len(total_costs) > 1 else 0.0
    demand_var = statistics.pvariance(customer_demands)
    bullwhip = statistics.pvariance(retailer_orders) / demand_var if demand_var else float("nan")
    return {
        "mean_total_supply_chain_cost": mean,
        "stderr": stderr,
        "bullwhip_ratio": bullwhip,
        "order_format_failure_rate": policy.failures / policy.calls if policy.calls else 0.0,
        "episodes": len(seeds),
    }


def tune_base_stock() -> int:
    seeds = training_seeds()
    best: tuple[float, int] | None = None
    for level in range(0, ORDER_CAP + 1):
        metric = evaluate(BaseStockPolicy(level), seeds)
        candidate = (float(metric["mean_total_supply_chain_cost"]), level)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1]


def score_against_naive(policy_cost: float, naive_cost: float) -> float:
    if naive_cost <= 0:
        return 0.0
    return 100.0 * naive_cost / (naive_cost + max(0.0, policy_cost))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--qwen-name", default="qwen_base")
    parser.add_argument("--qwen-adapter", default=None)
    parser.add_argument("--large-model", default="openai/gpt-5.6-terra")
    parser.add_argument("--seed-file", default=str(HELD_OUT_PATH.relative_to(ROOT)))
    parser.add_argument("--result-file", default=str(RESULT_PATH.relative_to(ROOT)))
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-large", action="store_true")
    args = parser.parse_args()

    seed_path = Path(args.seed_file)
    if not seed_path.is_absolute():
        seed_path = ROOT / seed_path
    result_path = Path(args.result_file)
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    seeds = read_eval_seeds(seed_path)
    overlap = set(seeds).intersection(training_seeds())
    if overlap:
        raise AssertionError(f"training seed appears in evaluation split: {sorted(overlap)!r}")
    level = tune_base_stock()
    policies: list[BatchPolicy] = [NaivePolicy(), BaseStockPolicy(level)]
    if not args.skip_qwen:
        policies.append(LocalQwenPolicy(args.qwen_model, args.qwen_name, args.qwen_adapter))
    if not args.skip_large:
        policies.append(OpenRouterPolicy(args.large_model))

    results: dict[str, dict[str, float | int]] = {}
    for policy in policies:
        results[policy.name] = evaluate(policy, seeds)
    naive_cost = float(results["naive"]["mean_total_supply_chain_cost"])
    for metrics in results.values():
        metrics["benchmark_score_0_to_100"] = score_against_naive(
            float(metrics["mean_total_supply_chain_cost"]), naive_cost
        )
    payload: dict[str, Any] = {
        "protocol": {
            "controlled_role": "wholesaler",
            "topology": "serial",
            "horizon_weeks": HORIZON,
            "held_out_seed_file": str(seed_path.relative_to(ROOT)),
            "held_out_seed_count": len(seeds),
            "benchmark_score": "100 * naive_mean_cost / (naive_mean_cost + policy_mean_cost); naive scores 50, zero cost approaches 100",
            "large_model": args.large_model,
            "qwen_model": args.qwen_model,
            "base_stock_level_tuned_on_train_seeds_only": level,
        },
        "results": results,
    }
    result_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
