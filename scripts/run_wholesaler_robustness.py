#!/usr/bin/env python3
"""Independent frozen-seed replication for the published wholesaler benchmark."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from eval.run_eval import (
    BaseStockPolicy,
    LocalQwenPolicy,
    NaivePolicy,
    OpenRouterPolicy,
    evaluate,
    read_eval_seeds,
    score_against_naive,
    training_seeds,
    tune_base_stock,
)


ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "eval" / "robustness_seeds.json"
RESULT_PATH = ROOT / "results" / "robustness.json"
QWEN = "Qwen/Qwen3.5-4B"
ADAPTER = "/workspace/outputs/beer-wholesaler-qwen35-4b-lora/adapter"
TERRA = "openai/gpt-5.6-terra"


def run(policy: Any, seeds: list[int], results: dict[str, dict[str, float | int]]) -> None:
    results[policy.name] = evaluate(policy, seeds)
    del policy
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    seeds = read_eval_seeds(SEED_PATH)
    overlap = set(seeds).intersection(training_seeds())
    if overlap:
        raise AssertionError(f"training seed appears in robustness split: {sorted(overlap)!r}")

    level = tune_base_stock()
    results: dict[str, dict[str, float | int]] = {}
    run(NaivePolicy(), seeds, results)
    run(BaseStockPolicy(level), seeds, results)
    run(LocalQwenPolicy(QWEN, "qwen_base"), seeds, results)
    run(LocalQwenPolicy(QWEN, "qwen_lora", ADAPTER), seeds, results)
    run(OpenRouterPolicy(TERRA), seeds, results)

    naive_cost = float(results["naive"]["mean_total_supply_chain_cost"])
    for metrics in results.values():
        metrics["benchmark_score_0_to_100"] = score_against_naive(
            float(metrics["mean_total_supply_chain_cost"]), naive_cost
        )
    payload = {
        "protocol": {
            "name": "independent post-training robustness replication",
            "controlled_role": "wholesaler",
            "topology": "serial",
            "horizon_weeks": 20,
            "seed_file": str(SEED_PATH.relative_to(ROOT)),
            "seed_count": len(seeds),
            "seed_rule": "fixed before evaluation; never used for training or base-stock tuning",
            "benchmark_score": "100 * naive_mean_cost / (naive_mean_cost + policy_mean_cost)",
            "base_stock_level_tuned_on_train_seeds_only": level,
            "qwen_model": QWEN,
            "qwen_lora_adapter": ADAPTER,
            "large_model": TERRA,
        },
        "results": results,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
