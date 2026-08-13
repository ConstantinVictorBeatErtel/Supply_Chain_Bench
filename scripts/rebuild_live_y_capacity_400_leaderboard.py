#!/usr/bin/env python3
"""Rebuild capacity-400 perfect-cost leaderboard from evaluation JSONs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "artifacts" / "live_y_capacity_400" / "evaluations"


def mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.mean(values)
    stderr = statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else 0.0
    return mean, stderr


def score_model(
    name: str,
    model_id: str,
    rows: list[dict],
    perfect_by_seed: dict[str, dict],
) -> dict:
    clean = [row for row in rows if row.get("protocol_clean")]
    per_seed = []
    for row in clean:
        seed = str(row["seed"])
        perfect = perfect_by_seed[seed]
        policy_cost = float(row["local_total_cost"])
        perfect_cost = float(perfect["perfect_local_cost"])
        per_seed.append(
            {
                "bucket": row.get("bucket"),
                "seed": seed,
                "local_total_cost": policy_cost,
                "perfect_local_cost": perfect_cost,
                "score": 100.0 * perfect_cost / policy_cost if policy_cost > 0 else None,
            }
        )
    policy_costs = [item["local_total_cost"] for item in per_seed]
    perfect_costs = [item["perfect_local_cost"] for item in per_seed]
    seed_scores = [item["score"] for item in per_seed if item["score"] is not None]
    policy_mean = statistics.mean(policy_costs) if policy_costs else None
    perfect_mean = statistics.mean(perfect_costs) if perfect_costs else None
    score = 100.0 * perfect_mean / policy_mean if policy_mean and perfect_mean is not None else None
    score_mean, score_stderr = mean_stderr(seed_scores)
    adaptive_scores = []
    for item in per_seed:
        adaptive = float(perfect_by_seed[item["seed"]]["adaptive_local_cost"])
        if item["local_total_cost"] > 0:
            adaptive_scores.append(100.0 * adaptive / item["local_total_cost"])
    return {
        "model_id": model_id,
        "n_episodes": len(rows),
        "n_protocol_clean": len(clean),
        "protocol_failure_rate": (len(rows) - len(clean)) / len(rows) if rows else None,
        "policy_mean_local_cost": policy_mean,
        "hindsight_perfect_mean_on_clean": perfect_mean,
        "score": score,
        "score_mean_of_per_seed": score_mean,
        "score_stderr_per_seed": score_stderr,
        "inline_adaptive_score_mean": statistics.mean(adaptive_scores) if adaptive_scores else None,
        "per_seed": sorted(per_seed, key=lambda item: (item["bucket"] or "", item["seed"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "perfect_cost_leaderboard_capacity_400.json",
    )
    args = parser.parse_args()

    perfect = json.loads((args.eval_dir / "hindsight_perfect_costs.json").read_text())
    perfect_by_seed = {str(row["seed"]): row for row in perfect["rows"]}

    sources = [
        ("GPT-5.6 Luna", "openai_gpt-5.6-luna", "openrouter_openai_gpt-5.6-luna.json"),
        ("Grok 4.5", "x-ai_grok-4.5", "openrouter_x-ai_grok-4.5.json"),
        ("Laguna S 2.1 (free)", "poolside_laguna-s-2.1_free", "openrouter_poolside_laguna-s-2.1_free.json"),
        ("DeepSeek V4 Flash", "deepseek_deepseek-v4-flash-0731", "openrouter_deepseek_deepseek-v4-flash-0731.json"),
        ("Nemotron 3 Ultra (free)", "nvidia_nemotron-3-ultra-550b-a55b_free", "openrouter_nvidia_nemotron-3-ultra-550b-a55b_free.json"),
        ("Qwen3.5-4B (untrained)", "untrained_qwen_capacity_400", "untrained_qwen_capacity_400.json"),
        # The board tracks the current adapter.  The superseded two-update run
        # stays on disk as `trained_qwen_grpo_capacity_400.json` for provenance
        # (it scored 7.48 and was statistically indistinguishable from the
        # untrained base); see docs/LIVE_Y_RL_POSTMORTEM.md.
        (
            "Qwen3.5-4B GRPO",
            "trained_qwen_grpo_v3_capacity_400",
            "trained_qwen_grpo_v3_capacity_400.json",
        ),
    ]

    models: dict[str, dict] = {}
    for name, model_id, filename in sources:
        path = args.eval_dir / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("rows") or []
        if len(rows) < 1:
            continue
        models[name] = score_model(name, model_id, rows, perfect_by_seed)

    board = {
        "capacity": 400,
        "n_seeds": len(perfect_by_seed),
        "hindsight_perfect_mean_local_cost_all_seeds": perfect["summary"]["perfect_mean_local_cost"],
        "adaptive_mean_local_cost_all_seeds": perfect["summary"]["adaptive_mean_local_cost"],
        "score_formula": "100 * mean(C*_seed) / mean(C_policy_seed) over protocol-clean episodes",
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(board, indent=2, sort_keys=True) + "\n")
    ranked = sorted(
        ((name, data["score"], data.get("n_protocol_clean"), data.get("policy_mean_local_cost")) for name, data in models.items() if data.get("score") is not None),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, score, n_clean, mean_cost in ranked:
        print(f"{score:6.2f}  {name}  clean={n_clean}  C={mean_cost:.1f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
