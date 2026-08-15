"""One-command SupplyChainBench model evaluation."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Any

from supplychainbench import BENCHMARK_ID, BENCHMARK_VERSION, ENVIRONMENT_ID
from supplychainbench.providers import ProviderError, create_provider, model_slug
from supplychainbench.results import METRIC_DEFINITIONS, aggregate_episode_rows, now_utc, validate_result, write_atomic
from supplychainbench.suites import DEFINITIONS, build_episode, episode_jobs, expected_seeds, prompt_for, reference_cost

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _standard_references() -> dict[str, float]:
    path = ROOT / "artifacts/live_y_capacity_400/evaluations/hindsight_perfect_costs.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {str(row["seed"]): float(row["perfect_local_cost"]) for row in payload.get("rows", [])}


def _episode_metrics(job: Any, weekly: list[float], reference_weekly: list[float] | None) -> dict[str, Any]:
    if not weekly:
        return {}
    if job.event_week is None:
        n = len(weekly)
        split = max(1, n // 3)
        first = sum(weekly[:split]) / split
        last = sum(weekly[-split:]) / split
        ref_first = sum(reference_weekly[:split]) / split if reference_weekly else None
        ref_last = sum(reference_weekly[-split:]) / split if reference_weekly else None
        return {"first_third_mean_cost": first, "final_third_mean_cost": last,
                "adaptation_improvement": first - last,
                "first_third_regret": first - ref_first if ref_first is not None else None,
                "final_third_regret": last - ref_last if ref_last is not None else None}
    k = max(1, min(len(weekly), int(job.event_week))) - 1
    window = 6
    pre = weekly[max(0, k - window):k] or weekly[:1]
    post = weekly[k:min(len(weekly), k + window)] or weekly[-1:]
    regrets = [(a - b) for a, b in zip(weekly, reference_weekly or [0.0] * len(weekly))]
    pre_regret = regrets[max(0, k - window):k] or [0.0]
    post_regret = regrets[k:min(len(regrets), k + window)] or [0.0]
    threshold = sum(pre_regret) / len(pre_regret) + max(1.0, abs(sum(pre_regret) / len(pre_regret)) * 0.1)
    recovery = None
    for start in range(k, max(k, len(regrets) - 2)):
        if len(regrets[start:start + 3]) == 3 and sum(regrets[start:start + 3]) / 3 <= threshold:
            recovery = start - k + 1
            break
    return {"event_week": job.event_week, "pre_event_mean_cost": sum(pre) / len(pre),
            "post_event_mean_cost": sum(post) / len(post),
            "pre_event_mean_regret": sum(pre_regret) / len(pre_regret),
            "post_event_mean_regret": sum(post_regret) / len(post_regret),
            "post_event_regret_auc": sum(max(0.0, value) for value in regrets[k:]),
            "recovery_time": recovery, "recovery_censored": recovery is None}


def evaluate(model: str, suite: str, *, output: Path | None = None, adapter: str | None = None,
             base_url: str | None = None, api_key_env: str | None = None,
             seed_limit: int | None = None, resume: bool = True) -> dict[str, Any]:
    if suite not in DEFINITIONS:
        raise ValueError(f"unknown suite {suite!r}")
    jobs = list(episode_jobs(suite))
    if seed_limit is not None:
        if seed_limit < 1 or seed_limit > len(jobs):
            raise ValueError(f"--seed-limit must be in 1..{len(jobs)}")
        jobs = jobs[:seed_limit]
    output = output or RESULTS_ROOT / suite / f"{model_slug(model)}.json"
    prior_rows: list[dict[str, Any]] = []
    if resume and output.exists():
        try:
            prior = json.loads(output.read_text())
            if prior.get("model", {}).get("identifier") == model and prior.get("suite", {}).get("id") == suite:
                prior_rows = list(prior.get("episodes") or [])
        except json.JSONDecodeError:
            prior_rows = []
    done = {str(row.get("seed")) for row in prior_rows}
    provider = create_provider(model, adapter=adapter, base_url=base_url, api_key_env=api_key_env)
    refs = _standard_references() if suite == "standard" else {}
    rows = list(prior_rows)
    try:
        for job in jobs:
            if job.seed in done:
                continue
            episode = build_episode(job)
            observation = episode.start()
            provider.reset_episode(job, episode, observation)
            actions: list[int] = []
            weekly: list[float] = []
            raw_outputs: list[str] = []
            failure: str | None = None
            while not episode.done:
                system, user = prompt_for(job, observation)
                response = provider.act(system, user, observation)
                raw_outputs.append(response.raw)
                if response.quantity is None:
                    failure = response.error or "invalid_model_action"
                    episode.protocol_failure_outcome(error_count=1, category=failure)
                    break
                result = episode.place_order(int(response.quantity))
                actions.append(int(response.quantity))
                weekly.append(float(episode.operational_transitions[-1]["local_costs"]["wholesaler"]))
                if not result["done"]:
                    observation = result["next_observation"]
            grade = episode.outcome["grade"] if episode.outcome else {}
            clean = bool(grade.get("protocol_clean", False))
            local = grade.get("primary", {}).get("local_total_cost") if clean else None
            reference_row = reference_cost(job) if suite != "standard" else None
            reference = refs.get(job.seed)
            if reference is None:
                reference = float(reference_row["local_total_cost"])
            ref_weekly = reference_row["weekly_local_costs"] if reference_row else None
            row = {"seed": job.seed, "index": job.index, "bucket": job.bucket,
                   "protocol_clean": clean, "failure": failure,
                   "local_total_cost": float(local) if local is not None else None,
                   "reference_cost": float(reference),
                   "score": (100.0 * float(reference) / float(local)) if local and clean else None,
                   "completed_weeks": len(actions), "actions": actions,
                   "weekly_local_costs": weekly,
                   "metrics": _episode_metrics(job, weekly, ref_weekly),
                   "ground_truth_for_evaluator": deepcopy(job.ground_truth) if job.suite != "standard" else None}
            if provider.provider_kind != "agent":
                row["raw_outputs"] = raw_outputs
            rows.append(row)
            done.add(job.seed)
            aggregate = aggregate_episode_rows(rows)
            payload = _payload(model, suite, rows, aggregate, adapter, base_url, api_key_env)
            write_atomic(output, payload)
    finally:
        provider.close()
    aggregate = aggregate_episode_rows(rows)
    payload = _payload(model, suite, rows, aggregate, adapter, base_url, api_key_env)
    payload["run"]["status"] = "complete" if len(rows) == len(jobs) else "incomplete"
    validate_result(payload, expected_suite=suite)
    write_atomic(output, payload)
    return payload


def _payload(model: str, suite: str, rows: list[dict[str, Any]], aggregate: dict[str, Any], adapter: str | None,
             base_url: str | None, api_key_env: str | None) -> dict[str, Any]:
    return {"schema_version": "1.0.0", "benchmark": {"id": BENCHMARK_ID, "version": BENCHMARK_VERSION, "environment": ENVIRONMENT_ID},
            "model": {"identifier": model, "provider": model.split(":", 1)[0]},
            "suite": {"id": suite, "version": DEFINITIONS[suite].version, "expected_seeds": list(expected_seeds(suite))},
            "episodes": rows, "aggregate": aggregate,
            "protocol_clean_seeds": list(aggregate.get("protocol_clean_seeds", [])),
            "failures": [row for row in rows if not row.get("protocol_clean")],
            "metric_definitions": METRIC_DEFINITIONS,
            "run": {"status": "incomplete", "timestamp": now_utc(), "git_commit": _git_commit(), "run_kind": "evaluated"},
            "configuration": {"adapter": adapter, "base_url": base_url, "api_key_env": api_key_env, "reference": DEFINITIONS[suite].reference},
            "provenance": {"legacy": False}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on SupplyChainBench Beer Distribution")
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", choices=tuple(DEFINITIONS), default="standard")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adapter")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env")
    parser.add_argument("--seed-limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    try:
        payload = evaluate(args.model, args.suite, output=args.output, adapter=args.adapter,
                           base_url=args.base_url, api_key_env=args.api_key_env,
                           seed_limit=args.seed_limit, resume=not args.no_resume)
    except ProviderError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
