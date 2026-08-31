#!/usr/bin/env python3
"""Build and integrity-check the v2 board from one shared hindsight reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2.environment import training_spec

EVAL_DIR = ROOT / "artifacts/live_y_domain_randomized_grpo_v2/evaluations"
REFERENCE = EVAL_DIR / "hindsight_reference.json"
MANIFEST = ROOT / "experiments/live_y_domain_randomized_grpo_v2/seed_manifest.json"
EXPECTED_N = 16
SOURCES = (
    ("Claude Opus 5", "openrouter_anthropic_claude-opus-5.json"),
    ("DeepSeek V4 Flash", "openrouter_deepseek_deepseek-v4-flash-0731.json"),
    ("GLM-5.3-Flash", "openrouter_z-ai_glm-5.3-flash.json"),
    ("GPT-5.6 Luna", "openrouter_openai_gpt-5.6-luna.json"),
    ("GPT-5.6 Sol", "openrouter_openai_gpt-5.6-sol.json"),
    ("Grok 4.5", "openrouter_x-ai_grok-4.5.json"),
    ("Grok 4.6", "openrouter_x-ai_grok-4.6.json"),
    ("Laguna S 2.1 (free)", "openrouter_poolside_laguna-s-2.1_free.json"),
    ("Muse Spark 1.2", "openrouter_meta_muse-spark-1.2.json"),
    ("Nemotron 3 Ultra (free)", "openrouter_nvidia_nemotron-3-ultra-550b-a55b_free.json"),
    ("Qwen3.5-4B (untrained)", "untrained_qwen_v2.json"),
    ("Qwen3.5-4B GRPO", "trained_qwen_grpo_v2.json"),
)


def replay_clean_row(row: dict[str, Any]) -> None:
    actions = [int(value) for value in row.get("actions") or []]
    if len(actions) != 36:
        raise AssertionError(f"clean row {row.get('seed')} has {len(actions)} actions")
    index = int(row["index"])
    episode = BeerEpisode(training_spec(str(row["seed"]), index=index), "wholesaler", include_reference=False)
    episode.start()
    for quantity in actions:
        episode.place_order(quantity)
    replayed = float(episode.outcome["grade"]["primary"]["local_total_cost"])
    recorded = float(row["local_total_cost"])
    if replayed != recorded:
        raise AssertionError(f"replay mismatch {row.get('seed')}: {replayed} != {recorded}")


def bootstrap_ci(rows: list[dict[str, float]]) -> list[float]:
    rng = random.Random(20260829)
    values: list[float] = []
    for _ in range(100_000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(
            100.0 * sum(row["best"] for row in sample) / sum(row["model"] for row in sample)
        )
    values.sort()
    return [values[2_500], values[97_500]]


def score_source(
    name: str,
    path: Path,
    references: dict[str, dict[str, Any]],
    expected_seeds: list[str],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("protocol_id") != "live-y-domain-randomized-grpo-v2":
        raise AssertionError(f"{path.name}: wrong protocol_id {payload.get('protocol_id')!r}")
    rows = payload.get("rows") or []
    if len(rows) != EXPECTED_N:
        raise AssertionError(f"{path.name}: expected {EXPECTED_N} rows, found {len(rows)}")
    actual_pairs = [(int(row["index"]), str(row["seed"])) for row in rows]
    expected_pairs = list(enumerate(expected_seeds))
    if sorted(actual_pairs) != expected_pairs:
        raise AssertionError(f"{path.name}: rows do not exactly match the frozen seed manifest")
    clean = [row for row in rows if row.get("protocol_clean")]
    for row in clean:
        replay_clean_row(row)
    paired = [
        {
            "model": float(row["local_total_cost"]),
            "best": float(references[str(row["seed"])]["best_found_hindsight_local_cost"]),
            "adaptive": float(references[str(row["seed"])]["adaptive_local_cost"]),
        }
        for row in clean
    ]
    model_total = sum(row["model"] for row in paired)
    best_total = sum(row["best"] for row in paired)
    adaptive_total = sum(row["adaptive"] for row in paired)
    clean_subset_score = 100.0 * best_total / model_total if model_total else None
    clean_subset_ci = bootstrap_ci(paired) if paired else None
    full_coverage = len(clean) == EXPECTED_N
    return {
        "model_id": payload.get("model_id") or payload.get("model"),
        "artifact": str(path.relative_to(ROOT)),
        "n_scheduled": len(rows),
        "n_protocol_clean": len(clean),
        "transport_failures": sum(bool(row.get("error")) for row in rows),
        "protocol_failures": sum(
            not row.get("protocol_clean") and not row.get("error") for row in rows
        ),
        "coverage": len(clean) / len(rows),
        "official_score": clean_subset_score if full_coverage else None,
        "official_score_bootstrap_95_ci": clean_subset_ci if full_coverage else None,
        "clean_subset_score_diagnostic": clean_subset_score,
        "clean_subset_score_bootstrap_95_ci_diagnostic": clean_subset_ci,
        "model_mean_cost_on_clean": statistics.mean(row["model"] for row in paired) if paired else None,
        "adaptive_score_on_clean": 100.0 * best_total / adaptive_total if adaptive_total else None,
        "status": "scored" if full_coverage else "unscored_incomplete_protocol_coverage",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "leaderboard_v2.json")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    expected_seeds = [str(seed) for seed in manifest["evaluation"]["in_distribution"]]
    if len(expected_seeds) != EXPECTED_N or len(set(expected_seeds)) != EXPECTED_N:
        raise AssertionError("frozen seed manifest must contain 16 unique evaluation seeds")
    reference_payload = json.loads(REFERENCE.read_text())
    references = {str(row["seed"]): row for row in reference_payload["rows"]}
    if set(references) != set(expected_seeds):
        raise AssertionError("hindsight reference does not exactly match the frozen seed manifest")
    models = {
        name: score_source(name, EVAL_DIR / filename, references, expected_seeds)
        for name, filename in SOURCES
        if (EVAL_DIR / filename).exists()
    }
    ranked = sorted(
        (
            {"rank": 0, "model": name, "score": row["official_score"]}
            for name, row in models.items()
            if row["official_score"] is not None
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    board = {
        "protocol_id": "live-y-domain-randomized-grpo-v2",
        "score_formula": "100 * sum(best_found_hindsight_cost) / sum(model_cost), paired by held-out seed",
        "coverage_rule": "Official scores require 16/16 protocol-clean episodes; clean-subset scores are diagnostic only.",
        "reference": str(REFERENCE.relative_to(ROOT)),
        "reference_caveat": "Best-found feasible hindsight; not a proof of mathematical optimality.",
        "bootstrap": {"resamples": 100_000, "seed": 20260829, "level": 0.95},
        "ranking": ranked,
        "models": models,
    }
    args.output.write_text(json.dumps(board, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ranking": ranked, "coverage": {
        name: f"{row['n_protocol_clean']}/{row['n_scheduled']}" for name, row in models.items()
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
