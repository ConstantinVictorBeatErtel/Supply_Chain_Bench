#!/usr/bin/env python3
"""Cost-controlled rerun of the native Tier-5 frontier benchmark.

The benchmark semantics are unchanged.  This wrapper adds:

* a five-seed full-length screen before the 100-seed run;
* catalog-level rejection of models that cannot disable reasoning;
* HTTP 400/402 circuit breakers;
* one sticky OpenRouter session per seed for provider prompt caching;
* OpenRouter response caching for exact retries/resumptions; and
* a per-model USD budget.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "beer_distribution_game"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from scripts import run_frontier_t5_y_36w as base  # noqa: E402
from beer_distribution_game.episode import BeerEpisode  # noqa: E402
from beer_distribution_game.prompts import observation_user_message, system_prompt  # noqa: E402


MODELS = base.MODELS
TOOLS = base.TOOLS
ROLE = base.ROLE
ORDER_CAP = base.ORDER_CAP


class ModelUnavailable(RuntimeError):
    def __init__(self, reason: str, *, status: int | None = None):
        super().__init__(reason)
        self.reason = reason
        self.status = status


def error_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode()
            parsed = json.loads(body)
        except Exception:
            parsed = {"body": body if "body" in locals() else ""}
        return {"http_status": exc.code, "error": parsed}
    return {"error_type": type(exc).__name__, "error": str(exc)}


class OptimizedClient:
    def __init__(self, model_id: str, raw_path: Path, workers: int, budget_usd: float):
        self.model_id = model_id
        self.raw_path = raw_path
        self.workers = workers
        self.budget_usd = budget_usd
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.lock = threading.Lock()
        self.spent = 0.0
        self.cache_hits = 0
        self.request_errors: Counter[str] = Counter()

    def _write_raw(self, record: dict[str, Any]) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.raw_path.open("a").write(json.dumps(record, sort_keys=True) + "\n")

    def _charge_and_check(self, response: dict[str, Any]) -> bool:
        usage = response.get("usage") or {}
        cost = float(usage.get("cost") or 0.0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        with self.lock:
            self.spent += cost
            self.cache_hits += int(cached > 0)
            return self.spent > self.budget_usd

    def call(
        self,
        observation: dict[str, Any],
        *,
        seed_index: int,
        week: int,
        repair: bool = False,
    ) -> tuple[int | None, str | None, dict[str, Any]]:
        # The system prompt, tools, and dynamic observation remain separate so
        # provider prompt caching can reuse the stable prefix.  One session per
        # seed keeps all 36 weekly calls on the same provider route.
        session_id = f"beer-t5-y-wholesaler-{self.model_id}-{seed_index:03d}"
        request_body: dict[str, Any] = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": 64,
            "reasoning": {"effort": "none"},
            "session_id": session_id,
            "messages": [
                {"role": "system", "content": system_prompt(base.make_spec("0" * 16, 0), ROLE)},
                {"role": "user", "content": observation_user_message(observation, action_format="tool")},
            ],
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "place_order"}},
            "parallel_tool_calls": False,
        }
        if repair:
            request_body["messages"].insert(
                1,
                {
                    "role": "user",
                    "content": "The previous action was invalid. Call place_order exactly once with only an integer quantity from 0 through 128.",
                },
            )
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ConstantinVictorBeatErtel/beer_distribution_RL",
            "X-Title": "Beer Distribution Tier-5 Frontier Evaluation (cached)",
            "X-OpenRouter-Cache": "true",
            "X-OpenRouter-Cache-TTL": "86400",
        }
        raw_response: dict[str, Any]
        last_error: dict[str, Any] | None = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(request_body, separators=(",", ":")).encode(),
                    headers=headers,
                    method="POST",
                )
                import ssl
                try:
                    import certifi
                    context = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    context = ssl.create_default_context()
                with urllib.request.urlopen(request, timeout=120, context=context) as response:
                    raw_response = json.loads(response.read().decode())
                over_budget = self._charge_and_check(raw_response)
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = error_payload(exc)
                status = last_error.get("http_status")
                if status in (400, 402):
                    self.request_errors[str(status)] += 1
                    raw_response = {"_request_error": last_error}
                    record = {"seed_index": seed_index, "week": week, "repair_attempt": repair, "request": request_body, "response": raw_response}
                    self._write_raw(record)
                    raise ModelUnavailable(
                        f"OpenRouter HTTP {status}: {last_error.get('error')}", status=status
                    )
                if status not in (408, 409, 429) and not (isinstance(status, int) and status >= 500):
                    raw_response = {"_request_error": last_error}
                    self.request_errors[str(status or "unknown")] += 1
                    record = {"seed_index": seed_index, "week": week, "repair_attempt": repair, "request": request_body, "response": raw_response}
                    self._write_raw(record)
                    raise ModelUnavailable(f"OpenRouter request failed: {last_error}", status=status)
                time.sleep(2**attempt)
        else:
            self.request_errors["retry_exhausted"] += 1
            raise ModelUnavailable(f"OpenRouter retries exhausted: {last_error}")

        record = {"seed_index": seed_index, "week": week, "repair_attempt": repair, "request": request_body, "response": raw_response}
        self._write_raw(record)
        if over_budget:
            raise ModelUnavailable(
                f"model budget exceeded (${self.spent:.4f} > ${self.budget_usd:.2f})"
            )
        quantity, error = base.parse_tool_action(raw_response)
        return quantity, error, raw_response


def run_episode(client: OptimizedClient, spec) -> dict[str, Any]:
    episode = BeerEpisode(spec, ROLE, include_reference=False)
    observation = episode.start()
    actions: list[int] = []
    format_failures = 0
    protocol_errors: list[str] = []
    while not episode.done:
        week = int(observation["week"])
        quantity, error, _ = client.call(observation, seed_index=spec.seed_index, week=week)
        if error is not None:
            format_failures += 1
            protocol_errors.append(error)
            episode.mark_protocol_error()
            quantity, repair_error, _ = client.call(observation, seed_index=spec.seed_index, week=week, repair=True)
            if repair_error is not None:
                format_failures += 1
                protocol_errors.append(repair_error)
                episode.mark_protocol_error()
                return failure_row(spec, actions, format_failures, protocol_errors, episode.protocol_failure_outcome(error_count=len(protocol_errors), category=repair_error))
            if len(protocol_errors) >= 3:
                return failure_row(spec, actions, format_failures, protocol_errors, episode.protocol_failure_outcome(error_count=len(protocol_errors), category=error))
        assert quantity is not None
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    assert episode.outcome is not None
    grade = episode.outcome["grade"]
    return {
        "seed_index": spec.seed_index,
        "master_seed_hex": spec.master_seed_hex,
        "episode_id": episode.episode_id,
        "actions": actions,
        "format_failures": format_failures,
        "protocol_errors": protocol_errors,
        "status": grade["status"],
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
        "grade": grade,
    }


def failure_row(spec, actions, format_failures, protocol_errors, outcome):
    return {
        "seed_index": spec.seed_index,
        "master_seed_hex": spec.master_seed_hex,
        "episode_id": spec.episode_id(ROLE),
        "actions": actions,
        "format_failures": format_failures,
        "protocol_errors": protocol_errors,
        "status": "protocol_error",
        "protocol_clean": False,
        "outcome": outcome,
    }


def run_stage(key: str, model_id: str, seeds: list[str], *, stage: str, output_root: Path, workers: int, budget_usd: float) -> dict[str, Any]:
    raw_path = output_root / "raw" / stage / f"{key}.jsonl"
    episodes_path = output_root / "episodes" / stage / f"{key}.jsonl"
    client = OptimizedClient(model_id, raw_path, workers, budget_usd)
    specs = [base.make_spec(seed, index) for index, seed in enumerate(seeds)]
    rows: list[dict[str, Any]] = []
    unavailable: ModelUnavailable | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_episode, client, spec): spec.seed_index for spec in specs}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
            except ModelUnavailable as exc:
                unavailable = exc
                for pending in futures:
                    pending.cancel()
                break
            rows.append(row)
            print(f"{stage} {key}: completed {completed}/{len(specs)}", flush=True)
    rows.sort(key=lambda row: row["seed_index"])
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    responses = sum(len(row.get("actions", [])) + int(row.get("format_failures", 0)) for row in rows)
    failures = sum(int(row.get("format_failures", 0)) for row in rows)
    metrics = base.summarize_policy(rows, response_count=responses, format_failures=failures)
    return {
        "status": "unavailable" if unavailable else "complete",
        "model_id": model_id,
        "metrics": metrics,
        "billable_usage_cost": client.spent,
        "cache_hit_responses": client.cache_hits,
        "request_errors": dict(client.request_errors),
        "rows": rows,
        "skip_reason": None if unavailable is None else unavailable.reason,
    }


def catalog_reason(catalog: dict[str, Any], model_id: str) -> str | None:
    metadata = base.model_metadata(catalog, model_id)
    if metadata is None:
        return "model absent from OpenRouter catalog"
    reasoning = metadata.get("reasoning") or {}
    if reasoning.get("mandatory"):
        return "reasoning is mandatory; incompatible with reasoning disabled"
    return None


def write_report(payload: dict[str, Any], report_path: Path) -> None:
    b = payload["baselines"]
    naive = float(b["naive"]["mean_local_wholesaler_cost"])
    oracle = float(b["oracle"]["mean_local_wholesaler_cost"])
    lines = [
        "# Cost-controlled frontier OpenRouter rerun",
        "",
        "Native environment `0.2.0`; Tier-5 Y topology; wholesaler; 36 operational weeks plus settlement; reasoning disabled; one strict `place_order` call per week.",
        "",
        "Cost controls: five-seed full-length screen, per-seed OpenRouter sticky sessions, prompt/response caching, catalog reasoning preflight, HTTP 400/402 circuit breakers, and a per-model USD budget.",
        "",
        "| Model | OpenRouter ID | Status | Cost ± stderr | Format failures | Raw | Score | Billable usage | Cache-hit responses |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        f"| Naive | deterministic | baseline | {naive:.3f} ± {b['naive']['stderr']:.3f} | 0 | 0.000 | 10.000 | — | — |",
        f"| Oracle | adaptive_base_stock_v2 | baseline | {oracle:.3f} ± {b['oracle']['stderr']:.3f} | 0 | 1.000 | 100.000 | — | — |",
    ]
    for key, model_id in MODELS:
        row = payload["models"].get(key)
        if not row:
            continue
        if row["status"] != "complete":
            lines.append(f"| {key} | `{model_id}` | {row['status']}: {row.get('skip_reason','')} | — | — | — | — | {row.get('billable_usage_cost',0):.4f} | {row.get('cache_hit_responses',0)} |")
            continue
        m = row["metrics"]
        if m["mean_local_wholesaler_cost"] is None:
            lines.append(f"| {key} | `{model_id}` | incomplete | — | — | — | — | {row['billable_usage_cost']:.4f} | {row['cache_hit_responses']} |")
            continue
        cost = float(m["mean_local_wholesaler_cost"])
        raw = (naive - cost) / (naive - oracle)
        score = min(100.0, max(0.0, 10.0 + 90.0 * raw))
        lines.append(f"| {key} | `{model_id}` | complete | {cost:.3f} ± {float(m['stderr']):.3f} | {m['format_failures']} / {m['responses']} | {raw:.3f} | {score:.3f} | {row['billable_usage_cost']:.4f} | {row['cache_hit_responses']} |")
    lines += ["", "Screen details and raw request/response JSONL are stored alongside `results.json`.", ""]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "frontier_t5_y_36w_20260802_cached")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--screen-seeds", type=int, default=5)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--only", action="append", choices=[key for key, _ in MODELS])
    args = parser.parse_args()
    if args.workers < 1 or not 1 <= args.screen_seeds <= 100 or args.budget_usd <= 0:
        raise ValueError("invalid workers, screen-seeds, or budget-usd")
    seeds = base.load_split()
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; no model calls made")
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    catalog = base.openrouter_catalog(api_key)
    (output_root / "openrouter_catalog.json").write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    baseline_rows = {policy: [base.run_policy(base.make_spec(seed, index), policy) for index, seed in enumerate(seeds)] for policy in ("naive", "oracle")}
    payload: dict[str, Any] = {
        "protocol": {"environment_version": "0.2.0", "scenario_id": "t5-strategic-y-v2", "topology": "y", "controlled_role": ROLE, "horizon_operational_weeks": 36, "settlement_weeks": 3, "reasoning": "disabled via reasoning.effort=none", "cache_control": {"session_id_per_seed": True, "openrouter_response_cache": True, "openrouter_cache_ttl_seconds": 86400}, "screen_seed_count": args.screen_seeds, "model_budget_usd": args.budget_usd},
        "split": {"path": str(base.SPLIT_PATH.relative_to(ROOT)), "count": len(seeds), "ids": seeds},
        "baselines": {"naive": base.summarize_policy(baseline_rows["naive"]), "oracle": base.summarize_policy(baseline_rows["oracle"])},
        "screen": {},
        "models": {},
    }
    selected = [key for key, _ in MODELS if not args.only or key in args.only]
    for key, model_id in MODELS:
        if key not in selected:
            continue
        reason = catalog_reason(catalog, model_id)
        if reason:
            payload["screen"][key] = {"status": "skipped_preflight", "model_id": model_id, "skip_reason": reason}
            payload["models"][key] = payload["screen"][key]
            print(f"{key}: skipped preflight ({reason})", flush=True)
            continue
        screen = run_stage(key, model_id, seeds[: args.screen_seeds], stage="screen", output_root=output_root, workers=args.workers, budget_usd=args.budget_usd)
        screen_row = {k: v for k, v in screen.items() if k != "rows"}
        payload["screen"][key] = screen_row
        if screen["status"] != "complete" or screen["metrics"]["completed_episodes"] < args.screen_seeds:
            payload["models"][key] = screen_row
            print(f"{key}: not promoted to final ({screen.get('skip_reason')})", flush=True)
            continue
        remaining_budget = max(0.01, args.budget_usd - float(screen["billable_usage_cost"]))
        final = run_stage(key, model_id, seeds, stage="final", output_root=output_root, workers=args.workers, budget_usd=remaining_budget)
        final_row = {k: v for k, v in final.items() if k != "rows"}
        final_row["billable_usage_cost"] = float(screen["billable_usage_cost"]) + float(final["billable_usage_cost"])
        final_row["cache_hit_responses"] = int(screen["cache_hit_responses"]) + int(final["cache_hit_responses"])
        payload["models"][key] = final_row
        (output_root / "episodes" / "final").mkdir(parents=True, exist_ok=True)
        print(f"{key}: final status {final['status']}", flush=True)
    results_path = output_root / "results.json"
    report_path = output_root / "report.md"
    results_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_report(payload, report_path)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
