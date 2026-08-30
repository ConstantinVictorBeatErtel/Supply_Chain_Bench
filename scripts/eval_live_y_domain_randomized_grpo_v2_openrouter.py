#!/usr/bin/env python3
"""Evaluate one OpenRouter model on frozen held-out v2 environment seeds."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
    research_system_prompt,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2.environment import training_spec
import scripts.eval_live_y_domain_randomized_grpo_v1_openrouter as v1_openrouter

DEFAULT_MAX_TOKENS = v1_openrouter.DEFAULT_MAX_TOKENS
OpenRouterClient = v1_openrouter.OpenRouterClient
normalize_openrouter_completion = v1_openrouter.normalize_openrouter_completion
REASONING_CONFIG_BY_MODEL = {
    "z-ai/glm-5.3-flash": ("low", 2048),
    "x-ai/grok-4.5": ("low", 768),
    "x-ai/grok-4.6": ("low", 768),
    # The v1 768-token run exhausted its hidden-reasoning budget before
    # producing a visible action. Give Muse the same headroom used for GLM.
    "meta/muse-spark-1.2": ("minimal", 2048),
}


def _parallel_open_url(request: urllib.request.Request, *, timeout: float):
    """Open one request without v1's process-wide IPv4 resolver lock.

    The v1 workaround mutates the process-global resolver and must serialize
    threads. The current host has working direct HTTPS resolution, so v2 can
    retain episode-level request concurrency safely.
    """

    return urllib.request.urlopen(request, timeout=timeout, context=v1_openrouter.ssl_context())


v1_openrouter._open_url = _parallel_open_url

PROTOCOL_ID = "live-y-domain-randomized-grpo-v2"
MANIFEST = ROOT / "experiments/live_y_domain_randomized_grpo_v2/seed_manifest.json"
REFERENCE = ROOT / "artifacts/live_y_domain_randomized_grpo_v2/evaluations/hindsight_reference.json"


def spec_for(seed: str, index: int):
    # The benchmark intentionally calls the training constructor itself.
    return training_spec(seed, index=index)


def adaptive_cost(seed: str, index: int) -> float:
    episode = BeerEpisode(spec_for(seed, index), "wholesaler", include_reference=False)
    observation = episode.start()
    policy = adaptive_policy(episode.spec, "wholesaler")
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    return float(episode.outcome["grade"]["primary"]["local_total_cost"])


def evaluate_episode(
    client: OpenRouterClient,
    model_id: str,
    seed: str,
    index: int,
    reference: dict[str, Any],
) -> dict[str, Any]:
    spec = spec_for(seed, index)
    episode = BeerEpisode(spec, "wholesaler", include_reference=False)
    observation = episode.start()
    system = research_system_prompt(spec, "wholesaler")
    raw_outputs: list[str] = []
    actions: list[int] = []
    prompt_tokens = completion_tokens = cached_tokens = 0
    usage_cost = 0.0
    while not episode.done:
        raw, response = client.complete(
            system=system,
            user=research_observation_user_message(observation),
            session_id=f"{PROTOCOL_ID}|{model_id}|{seed}",
            week=int(observation["week"]),
        )
        raw_outputs.append(raw)
        usage = response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        cached_tokens += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        usage_cost += float(usage.get("cost") or 0.0)
        quantity = parse_completion(normalize_openrouter_completion(raw))
        if quantity is None:
            episode.protocol_failure_outcome(error_count=1, category="invalid_protocol")
            break
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    local = grade.get("primary", {}).get("local_total_cost")
    best = float(reference["best_found_hindsight_local_cost"])
    adaptive = float(reference["adaptive_local_cost"])
    return {
        "model_id": model_id,
        "index": index,
        "seed": seed,
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "format_failures": int(not grade.get("protocol_clean", False)),
        "attempted_actions": len(raw_outputs),
        "completed_weeks": len(actions),
        "local_total_cost": local,
        "best_found_hindsight_local_cost": best,
        "adaptive_local_cost": adaptive,
        "per_episode_hindsight_score": None if local is None else 100.0 * best / float(local),
        "per_episode_adaptive_relative_score": None if local is None else 100.0 * adaptive / float(local),
        "system_total_cost": grade.get("costs", {}).get("system_total_cost"),
        "bullwhip_ratio": grade.get("stability", {}).get("bullwhip_ratio"),
        "normalized_order_volatility": grade.get("stability", {}).get("normalized_order_volatility"),
        "actions": actions,
        "raw_outputs": raw_outputs,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "usage_cost_usd": usage_cost,
        "demand": episode.outcome.get("research_exogenous"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [row for row in rows if row.get("protocol_clean") and row.get("local_total_cost") is not None]
    model_total = sum(float(row["local_total_cost"]) for row in clean)
    best_total = sum(float(row["best_found_hindsight_local_cost"]) for row in clean)
    adaptive_total = sum(float(row["adaptive_local_cost"]) for row in clean)
    episode_scores = [float(row["per_episode_hindsight_score"]) for row in clean]
    costs = [float(row["local_total_cost"]) for row in clean]
    bootstrap_ci = None
    if clean:
        rng = random.Random(20260829)
        bootstrap_scores: list[float] = []
        for _ in range(100_000):
            sample = [clean[rng.randrange(len(clean))] for _ in clean]
            sample_model = sum(float(row["local_total_cost"]) for row in sample)
            sample_best = sum(float(row["best_found_hindsight_local_cost"]) for row in sample)
            bootstrap_scores.append(100.0 * sample_best / sample_model)
        bootstrap_scores.sort()
        bootstrap_ci = [bootstrap_scores[2_500], bootstrap_scores[97_500]]
    return {
        "n_scheduled": len(rows),
        "n_protocol_clean": len(clean),
        "coverage": len(clean) / len(rows) if rows else None,
        "protocol_failure_rate": 1.0 - len(clean) / len(rows) if rows else None,
        "primary_score": 100.0 * best_total / model_total if model_total else None,
        "primary_score_definition": "100 * sum(best_found_hindsight_cost) / sum(model_cost), paired on clean seeds",
        "primary_score_bootstrap_95_ci": bootstrap_ci,
        "bootstrap_resamples": 100_000,
        "bootstrap_seed": 20260829,
        "mean_per_episode_hindsight_score": statistics.mean(episode_scores) if episode_scores else None,
        "model_total_cost": model_total,
        "model_mean_cost": statistics.mean(costs) if costs else None,
        "best_found_hindsight_total_cost": best_total,
        "adaptive_total_cost": adaptive_total,
        "adaptive_score": 100.0 * best_total / adaptive_total if adaptive_total else None,
        "model_vs_adaptive": 100.0 * adaptive_total / model_total if model_total else None,
        "usage_cost_usd": sum(float(row.get("usage_cost_usd") or 0) for row in rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "cached_tokens": sum(int(row.get("cached_tokens") or 0) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="z-ai/glm-5.3-flash")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reasoning_config = REASONING_CONFIG_BY_MODEL.get(args.model)
    if reasoning_config is not None:
        effort, reasoning_max_tokens = reasoning_config
        v1_openrouter.REASONING_EFFORT_BY_MODEL[args.model] = effort
        v1_openrouter.REASONING_MAX_TOKENS = reasoning_max_tokens
    slug = args.model.replace("/", "_").replace(":", "_")
    output = args.output or ROOT / f"artifacts/live_y_domain_randomized_grpo_v2/evaluations/openrouter_{slug}.json"
    manifest = json.loads(MANIFEST.read_text())
    reference_payload = json.loads(REFERENCE.read_text())
    references = {row["seed"]: row for row in reference_payload["rows"]}
    seeds = manifest["evaluation"]["in_distribution"]

    rows: list[dict[str, Any]] = []
    if args.resume and output.exists():
        prior = json.loads(output.read_text())
        if prior.get("model_id") != args.model:
            raise RuntimeError(f"cannot resume output for {prior.get('model_id')}")
        # Preserve successful episodes and genuine model/protocol failures.
        # Retry only infrastructure/API failures carrying an explicit error;
        # otherwise resume would cherry-pick stochastic format failures.
        rows = [
            row
            for row in prior.get("rows", [])
            if row.get("protocol_clean") or not row.get("error")
        ]
    completed = {row["seed"] for row in rows}
    client = OpenRouterClient(args.model, budget_usd=args.budget_usd)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(evaluate_episode, client, args.model, seed, index, references[seed]): (index, seed)
            for index, seed in enumerate(seeds)
            if seed not in completed
        }
        for future in as_completed(futures):
            index, seed = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "model_id": args.model, "index": index, "seed": seed,
                    "protocol_clean": False, "format_failures": 1,
                    "attempted_actions": 0, "completed_weeks": 0,
                    "local_total_cost": None, "error": str(exc),
                }
            rows.append(row)
            print(json.dumps({
                "index": index, "seed": seed,
                "cost": row.get("local_total_cost"),
                "score": row.get("per_episode_hindsight_score"),
                "clean": row.get("protocol_clean"),
                "error": row.get("error"),
            }, sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["index"])
    payload = {
        "evaluation_kind": "fixed held-out v2 in-distribution evaluation",
        "protocol_id": PROTOCOL_ID,
        "model_id": args.model,
        "environment_constructor": "training_spec",
        "seed_manifest": str(MANIFEST.relative_to(ROOT)),
        "reference_artifact": str(REFERENCE.relative_to(ROOT)),
        "reference_caveat": "Best-found feasible hindsight; not a proof of the mathematical optimum.",
        "prompt": "v1 research prompt contract (unchanged role-local inputs), applied to v2 observations",
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_completion_tokens": (
                reasoning_config[1] if reasoning_config is not None else DEFAULT_MAX_TOKENS
            ),
            "reasoning_effort": (
                reasoning_config[0] if reasoning_config is not None else "none"
            ),
            "visible_protocol_limit": 192,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
