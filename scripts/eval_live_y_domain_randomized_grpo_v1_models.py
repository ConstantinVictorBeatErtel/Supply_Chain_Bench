#!/usr/bin/env python3
"""Evaluate one Qwen base/adapter against all fixed research buckets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_colab_grpo_wholesaler as pilot
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_from_dict
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    EVAL_PROCESSES,
    ResearchTask,
    research_spec,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion


def aggregate(rows: list[dict]) -> dict:
    def mean_stderr(key: str):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return {
            "mean": statistics.mean(values) if values else None,
            "stderr": statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0 if values else None,
        }

    out = {key: mean_stderr(key) for key in ("local_total_cost", "system_total_cost", "score", "bullwhip_ratio", "normalized_order_volatility")}
    out["n"] = len(rows)
    out["protocol_failure_rate"] = sum(not row["protocol_clean"] for row in rows) / len(rows) if rows else None
    out["format_failure_rate"] = sum(row["format_failures"] for row in rows) / max(sum(row["attempted_actions"] for row in rows), 1)
    return out


def load_policy(adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    return model, tokenizer


def evaluate_episode(model, tokenizer, seed: str, bucket: str, label: str) -> dict:
    import torch

    torch.manual_seed(int(seed, 16) & 0x7FFFFFFF)
    task = ResearchTask(
        name=f"live-y-domain-randomized-grpo-v1:eval:{bucket}:{seed}",
        scenario=research_spec(seed, bucket=bucket).to_dict(),
    )
    spec = scenario_from_dict(task.scenario)
    episode = BeerEpisode(spec, "wholesaler", include_reference=False)
    observation = episode.start()
    args = SimpleNamespace(
        prompt_max_tokens=2048,
        max_new_tokens=192,
        temperature=0.7,
        top_p=0.95,
    )
    raw_outputs: list[str] = []
    actions: list[int] = []
    format_failures = 0
    while not episode.done:
        prompt = pilot.prompt_text(SimpleNamespace(data=task), observation, tokenizer)
        _, _, raw = pilot.generate_batch(model, tokenizer, [prompt], args, sample=True)[0]
        raw_outputs.append(raw)
        quantity = parse_completion(raw, tokenizer=tokenizer)
        if quantity is None:
            format_failures += 1
            episode.protocol_failure_outcome(error_count=1, category="invalid_protocol")
            break
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    naive = pilot_reference(seed, bucket, "naive_base_stock")
    local = grade.get("primary", {}).get("local_total_cost")
    score = None if local is None else 100.0 * naive / (naive + float(local))
    return {
        "model": label,
        "bucket": bucket,
        "seed": seed,
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "format_failures": format_failures,
        "attempted_actions": len(raw_outputs),
        "completed_weeks": len(actions),
        "local_total_cost": local,
        "system_total_cost": grade.get("costs", {}).get("system_total_cost"),
        "score": score,
        "bullwhip_ratio": grade.get("stability", {}).get("bullwhip_ratio"),
        "normalized_order_volatility": grade.get("stability", {}).get("normalized_order_volatility"),
        "actions": actions,
        "raw_outputs": raw_outputs,
        "demand": episode.outcome.get("research_exogenous"),
    }


def pilot_reference(seed: str, bucket: str, policy: str) -> float:
    from beer_distribution_game.policies import adaptive_policy

    episode = BeerEpisode(research_spec(seed, bucket=bucket), "wholesaler", include_reference=False)
    observation = episode.start()
    policy_obj = adaptive_policy(episode.spec, "wholesaler")
    while not episode.done:
        quantity = 8 if policy == "naive_base_stock" else policy_obj.act(observation)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    return float(episode.outcome["grade"]["primary"]["local_total_cost"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter")
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((ROOT / "experiments/live_y_domain_randomized_grpo_v1/seed_manifest.json").read_text())
    model, tokenizer = load_policy(args.adapter)
    rows = []
    for bucket, seeds in manifest["evaluation"].items():
        for seed in seeds:
            rows.append(evaluate_episode(model, tokenizer, seed, bucket, args.label))
    payload = {"evaluation_kind": "fixed research evaluation", "model": args.label, "adapter": args.adapter, "decoding": {"temperature": 0.7, "top_p": 0.95, "max_completion_tokens": 192}, "summary": aggregate(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
