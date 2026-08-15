"""Convert committed live-Y capacity-400 evaluations into the v1 schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from supplychainbench.eval import ROOT, _standard_references
from supplychainbench.providers import model_slug
from supplychainbench.results import METRIC_DEFINITIONS, aggregate_episode_rows, validate_result, write_atomic
from supplychainbench.suites import DEFINITIONS, expected_seeds

SOURCE_DIR = ROOT / "artifacts/live_y_capacity_400/evaluations"
MIGRATIONS = (
    ("untrained_qwen_capacity_400.json", "hf:Qwen/Qwen3.5-4B", "Qwen3.5-4B (untrained)", None, "active"),
    ("trained_qwen_grpo_v3_capacity_400.json", "hf:Qwen/Qwen3.5-4B", "Qwen3.5-4B GRPO", "artifacts/live_y_best_adapter", "active"),
    ("trained_qwen_grpo_capacity_400.json", "hf:Qwen/Qwen3.5-4B", "Qwen3.5-4B GRPO (two-update, superseded)", "artifacts/live_y_best_adapter_u15", "superseded"),
    ("openrouter_openai_gpt-5.6-luna.json", "openrouter:openai/gpt-5.6-luna", "GPT-5.6 Luna", None, "active"),
    ("openrouter_x-ai_grok-4.5.json", "openrouter:x-ai/grok-4.5", "Grok 4.5", None, "active"),
    ("openrouter_deepseek_deepseek-v4-flash-0731.json", "openrouter:deepseek/deepseek-v4-flash-0731", "DeepSeek V4 Flash", None, "active"),
    ("openrouter_poolside_laguna-s-2.1_free.json", "openrouter:poolside/laguna-s-2.1:free", "Laguna S 2.1 (free)", None, "active"),
    ("openrouter_nvidia_nemotron-3-ultra-550b-a55b_free.json", "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra (free)", None, "active"),
)


def _commit_time(path: Path) -> tuple[str | None, str | None]:
    try:
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True).strip()
        timestamp = subprocess.check_output(["git", "log", "-1", "--format=%cI", "--", str(path.relative_to(ROOT))], cwd=ROOT, text=True).strip()
        return commit or None, timestamp or None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None, None


def migrate_one(source: Path, model_id: str, label: str, adapter: str | None, publication_status: str, output: Path) -> dict:
    source_payload = json.loads(source.read_text())
    refs = _standard_references()
    rows = []
    for row in source_payload.get("rows", []):
        clean = bool(row.get("protocol_clean"))
        rows.append({
            "seed": str(row["seed"]), "index": None, "bucket": row.get("bucket"),
            "protocol_clean": clean,
            "failure": None if clean else ("format_failure" if row.get("format_failures") else "protocol_failure"),
            "local_total_cost": float(row["local_total_cost"]) if row.get("local_total_cost") is not None else None,
            "reference_cost": refs.get(str(row["seed"])),
            "score": float(row["score"]) if row.get("score") is not None else None,
            "completed_weeks": int(row.get("completed_weeks") or 0),
            "actions": list(row.get("actions") or []), "weekly_local_costs": [], "metrics": {},
        })
    aggregate = aggregate_episode_rows(rows)
    commit, timestamp = _commit_time(source)
    payload = {
        "schema_version": "1.0.0",
        "benchmark": {"id": "supplychainbench", "version": "1.0.0", "environment": "beer-distribution"},
        "model": {"identifier": model_id, "label": label, "provider": model_id.split(":", 1)[0]},
        "suite": {"id": "standard", "version": DEFINITIONS["standard"].version, "expected_seeds": list(expected_seeds("standard"))},
        "episodes": rows, "aggregate": aggregate,
        "protocol_clean_seeds": list(aggregate.get("protocol_clean_seeds", [])),
        "failures": [row for row in rows if not row.get("protocol_clean")],
        "metric_definitions": METRIC_DEFINITIONS,
        "run": {"status": "complete", "timestamp": timestamp or source_payload.get("timestamp") or "unknown", "git_commit": commit, "run_kind": "migrated"},
        "configuration": {"adapter": adapter, "legacy_model": source_payload.get("model"), "legacy_model_name": source_payload.get("model_name"), "reference": "frozen_feasible_hindsight"},
        "provenance": {"legacy": True, "source_path": str(source.relative_to(ROOT)), "source_git_commit": commit, "source_commit_timestamp": timestamp, "publication_status": publication_status},
    }
    validate_result(payload, expected_suite="standard", expected_seeds=set(expected_seeds("standard")))
    write_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate committed capacity-400 evaluations")
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/standard")
    args = parser.parse_args()
    for filename, model_id, label, adapter, publication_status in MIGRATIONS:
        source = args.source_dir / filename
        if not source.exists():
            raise SystemExit(f"missing source artifact: {source}")
        output_name = model_slug(label) + ".json"
        payload = migrate_one(source, model_id, label, adapter, publication_status, args.output_dir / output_name)
        print(f"{label}: clean={payload['aggregate']['protocol_clean_episodes']}/{payload['aggregate']['episodes_attempted']} score={payload['aggregate']['normalized_score']}")


if __name__ == "__main__":
    main()
