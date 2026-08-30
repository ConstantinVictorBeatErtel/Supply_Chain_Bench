#!/usr/bin/env python3
"""Evaluate the local Qwen base or adapter on frozen v2 seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_colab_grpo_wholesaler as pilot
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_rl.agents.llm.prefix_kv_cache import PrefixKVCache
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
    research_system_prompt,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2.environment import training_spec
from scripts.eval_live_y_domain_randomized_grpo_v1_models import (
    default_model_name,
    load_policy,
)

MANIFEST = ROOT / "experiments/live_y_domain_randomized_grpo_v2/seed_manifest.json"
REFERENCE = ROOT / "artifacts/live_y_domain_randomized_grpo_v2/evaluations/hindsight_reference.json"


def prompt_text(system: str, observation: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": research_observation_user_message(observation)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_episode(model, tokenizer, seed: str, index: int, label: str, gen_args) -> dict:
    import torch

    torch.manual_seed(int(seed, 16) & 0x7FFFFFFF)
    spec = training_spec(seed, index=index)
    system = research_system_prompt(spec, "wholesaler")
    episode = BeerEpisode(spec, "wholesaler", include_reference=False)
    observation = episode.start()
    raw_outputs: list[str] = []
    actions: list[int] = []
    while not episode.done:
        prompt = prompt_text(system, observation, tokenizer)
        _, _, raw = pilot.generate_batch(model, tokenizer, [prompt], gen_args, sample=True)[0]
        raw_outputs.append(raw)
        quantity = parse_completion(raw, tokenizer=tokenizer)
        if quantity is None:
            episode.protocol_failure_outcome(error_count=1, category="invalid_protocol")
            break
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    return {
        "model": label,
        "index": index,
        "seed": seed,
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "format_failures": int(not grade.get("protocol_clean", False)),
        "attempted_actions": len(raw_outputs),
        "completed_weeks": len(actions),
        "local_total_cost": grade.get("primary", {}).get("local_total_cost"),
        "system_total_cost": grade.get("costs", {}).get("system_total_cost"),
        "bullwhip_ratio": grade.get("stability", {}).get("bullwhip_ratio"),
        "normalized_order_volatility": grade.get("stability", {}).get("normalized_order_volatility"),
        "actions": actions,
        "raw_outputs": raw_outputs,
        "demand": episode.outcome.get("research_exogenous"),
    }


def adaptive_cost(seed: str, index: int) -> float:
    episode = BeerEpisode(training_spec(seed, index=index), "wholesaler", include_reference=False)
    observation = episode.start()
    policy = adaptive_policy(episode.spec, "wholesaler")
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    return float(episode.outcome["grade"]["primary"]["local_total_cost"])


def summarize(rows: list[dict], references: dict[str, dict]) -> dict:
    clean = [row for row in rows if row.get("protocol_clean")]
    model_costs = [float(row["local_total_cost"]) for row in clean]
    best_costs = [float(references[row["seed"]]["best_found_hindsight_local_cost"]) for row in clean]
    adaptive_costs = [float(references[row["seed"]]["adaptive_local_cost"]) for row in clean]
    return {
        "n_scheduled": len(rows),
        "n_protocol_clean": len(clean),
        "coverage": len(clean) / len(rows) if rows else None,
        "protocol_failure_rate": 1.0 - len(clean) / len(rows) if rows else None,
        "model_mean_cost": statistics.mean(model_costs) if model_costs else None,
        "model_total_cost": sum(model_costs),
        "best_found_hindsight_total_cost": sum(best_costs),
        "adaptive_total_cost": sum(adaptive_costs),
        "primary_score": 100.0 * sum(best_costs) / sum(model_costs) if model_costs else None,
        "adaptive_score": 100.0 * sum(best_costs) / sum(adaptive_costs) if adaptive_costs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default=default_model_name())
    parser.add_argument("--disable-prefix-kv-cache", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    seeds = manifest["evaluation"]["in_distribution"]
    reference_payload = json.loads(REFERENCE.read_text())
    references = {row["seed"]: row for row in reference_payload["rows"]}
    model, tokenizer = load_policy(args.adapter, args.model_name)
    gen_args = SimpleNamespace(
        prompt_max_tokens=2048,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        _prefix_kv_cache=None,
    )
    if not args.disable_prefix_kv_cache:
        gen_args._prefix_kv_cache = PrefixKVCache(model, min_prefix_tokens=64)

    rows: list[dict] = []
    if args.output.exists():
        prior = json.loads(args.output.read_text())
        if prior.get("model") == args.label:
            rows = list(prior.get("rows") or [])
    completed = {row["seed"] for row in rows}

    def checkpoint() -> None:
        payload = {
            "evaluation_kind": "fixed held-out v2 in-distribution evaluation",
            "protocol_id": "live-y-domain-randomized-grpo-v2",
            "environment_constructor": "training_spec",
            "model": args.label,
            "adapter": args.adapter,
            "model_name": args.model_name,
            "prompt": "research_observation_user_message",
            "decoding": {"temperature": 0.7, "top_p": 0.95, "max_completion_tokens": 64},
            "summary": summarize(rows, references),
            "rows": sorted(rows, key=lambda row: row["index"]),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for index, seed in enumerate(seeds):
        if seed in completed:
            continue
        print(f"[{args.label}] {index + 1}/{len(seeds)} {seed}", flush=True)
        row = evaluate_episode(model, tokenizer, seed, index, args.label, gen_args)
        rows.append(row)
        checkpoint()
        print(json.dumps({
            "model": args.label,
            "index": index,
            "clean": row["protocol_clean"],
            "cost": row["local_total_cost"],
        }, sort_keys=True), flush=True)
    checkpoint()
    print(json.dumps(summarize(rows, references), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
