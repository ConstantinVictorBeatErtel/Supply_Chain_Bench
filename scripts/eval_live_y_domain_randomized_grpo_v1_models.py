#!/usr/bin/env python3
"""Evaluate one Qwen base/adapter against all fixed research buckets."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_colab_grpo_wholesaler as pilot
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import scenario_from_dict
from beer_distribution_rl.agents.llm.prefix_kv_cache import PrefixKVCache
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    ResearchTask,
    research_spec,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
LOCAL_BASE = ROOT / "local_checkpoints" / "qwen35-4b-base"


def default_model_name() -> str:
    if LOCAL_BASE.exists() and (LOCAL_BASE / "config.json").exists():
        return str(LOCAL_BASE)
    return DEFAULT_MODEL


def research_prompt_text(task: Any, observation: dict, tokenizer) -> str:
    messages = [
        {"role": "system", "content": task.data.system_prompt},
        {"role": "user", "content": research_observation_user_message(observation)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


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


def load_policy(adapter: str | None, model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        device_map = "auto"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
        device_map = None
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    if device_map is None:
        model = model.to(device)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        if device_map is None:
            model = model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_episode(model, tokenizer, seed: str, bucket: str, label: str, args: SimpleNamespace) -> dict:
    import torch

    torch.manual_seed(int(seed, 16) & 0x7FFFFFFF)
    task = ResearchTask(
        name=f"live-y-domain-randomized-grpo-v1:eval:{bucket}:{seed}",
        scenario=research_spec(seed, bucket=bucket).to_dict(),
    )
    episode = BeerEpisode(scenario_from_dict(task.scenario), "wholesaler", include_reference=False)
    observation = episode.start()
    raw_outputs: list[str] = []
    actions: list[int] = []
    format_failures = 0
    while not episode.done:
        prompt = research_prompt_text(SimpleNamespace(data=task), observation, tokenizer)
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
    optimal = pilot_reference(seed, bucket, "adaptive_base_stock")
    local = grade.get("primary", {}).get("local_total_cost")
    score = None if local is None else 100.0 * optimal / float(local)
    return {
        "model": label,
        "bucket": bucket,
        "seed": seed,
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "format_failures": format_failures,
        "attempted_actions": len(raw_outputs),
        "completed_weeks": len(actions),
        "local_total_cost": local,
        "optimal_reference_local_cost": optimal,
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
    p.add_argument("--model-name", default=default_model_name())
    p.add_argument("--disable-prefix-kv-cache", action="store_true")
    p.add_argument("--kv-cache-min-prefix-tokens", type=int, default=64)
    p.add_argument(
        "--seeds",
        help="Comma-separated seed hexes to evaluate (default: all evaluation seeds). Resume still skips seeds already in --output.",
    )
    args = p.parse_args()
    manifest = json.loads((ROOT / "experiments/live_y_domain_randomized_grpo_v1/seed_manifest.json").read_text())
    seed_filter = {s.strip() for s in args.seeds.split(",") if s.strip()} if args.seeds else None
    model, tokenizer = load_policy(args.adapter, args.model_name)
    gen_args = SimpleNamespace(
        prompt_max_tokens=2048,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        _prefix_kv_cache=None,
    )
    if not args.disable_prefix_kv_cache:
        gen_args._prefix_kv_cache = PrefixKVCache(
            model, min_prefix_tokens=args.kv_cache_min_prefix_tokens
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done_seeds: set[str] = set()
    if args.output.exists():
        try:
            prior = json.loads(args.output.read_text())
            if prior.get("model") == args.label and isinstance(prior.get("rows"), list):
                rows = list(prior["rows"])
                done_seeds = {str(row.get("seed")) for row in rows if row.get("seed") is not None}
                print(f"[{args.label}] resume {len(rows)} saved rows from {args.output}", flush=True)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[{args.label}] ignore unreadable output ({exc})", flush=True)
            rows = []
            done_seeds = set()

    def write_checkpoint() -> None:
        payload = {
            "evaluation_kind": "fixed research evaluation",
            "model": args.label,
            "adapter": args.adapter,
            "model_name": args.model_name,
            "prompt": "research_observation_user_message",
            "prefix_kv_cache": (
                gen_args._prefix_kv_cache.snapshot()
                if gen_args._prefix_kv_cache is not None
                else {"enabled": False}
            ),
            "decoding": {"temperature": 0.7, "top_p": 0.95, "max_completion_tokens": 64},
            "summary": aggregate(rows),
            "rows": rows,
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    jobs = [(bucket, seed) for bucket, seeds in manifest["evaluation"].items() for seed in seeds]
    if seed_filter is not None:
        jobs = [(bucket, seed) for bucket, seed in jobs if seed in seed_filter]
        missing = seed_filter - {seed for _, seed in jobs}
        if missing:
            raise SystemExit(f"Unknown --seeds not in evaluation manifest: {sorted(missing)}")
    for index, (bucket, seed) in enumerate(jobs, start=1):
        if seed in done_seeds:
            print(f"[{args.label}] {index}/{len(jobs)} {bucket} {seed} skip", flush=True)
            continue
        print(f"[{args.label}] {index}/{len(jobs)} {bucket} {seed}", flush=True)
        rows.append(evaluate_episode(model, tokenizer, seed, bucket, args.label, gen_args))
        done_seeds.add(seed)
        print(
            json.dumps(
                {
                    "bucket": bucket,
                    "seed": seed,
                    "protocol_clean": rows[-1]["protocol_clean"],
                    "local_total_cost": rows[-1]["local_total_cost"],
                    "completed_weeks": rows[-1]["completed_weeks"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        write_checkpoint()
    write_checkpoint()
    print(json.dumps(aggregate(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
