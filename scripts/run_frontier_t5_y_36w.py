#!/usr/bin/env python3
"""Evaluate version-pinned OpenRouter models on a fresh native Tier-5 split.

This runner uses the framework-neutral Hub ``BeerEpisode`` directly.  It keeps
the model-visible context to the system prompt plus the current observation,
forces exactly one ``place_order`` function call, sends ``reasoning.effort``
``none``, and writes the unmodified JSON response for every request.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "environments" / "beer_distribution_game"
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENV_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_ROOT))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_game.prompts import observation_user_message, system_prompt
from beer_distribution_game.scenario import ScenarioSpec, scenario_for


SPLIT_PATH = ROOT / "eval" / "frontier_t5_y_36w_100_seed_split.json"
OUTPUT_ROOT = ROOT / "artifacts" / "frontier_t5_y_36w_20260802"
CATALOG_PATH = OUTPUT_ROOT / "openrouter_catalog.json"
RESULTS_PATH = OUTPUT_ROOT / "results.json"
REPORT_PATH = OUTPUT_ROOT / "report.md"
ORDER_CAP = 128
ROLE = "wholesaler"
ENVIRONMENT_VERSION = "0.2.0"

# OpenRouter API IDs, not mutable *-latest aliases.  The requested Kimi 3 is
# listed by OpenRouter as Kimi K3.
MODELS: tuple[tuple[str, str], ...] = (
    ("kimi_k3", "moonshotai/kimi-k3"),
    ("claude_opus_5", "anthropic/claude-opus-5"),
    ("muse_spark_1_1", "meta/muse-spark-1.1"),
    ("grok_4_5", "x-ai/grok-4.5"),
    ("gpt_5_6_sol", "openai/gpt-5.6-sol"),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place this week's replenishment order. Call exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": ORDER_CAP,
                    }
                },
                "required": ["quantity"],
                "additionalProperties": False,
            },
        },
    }
]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def load_split() -> list[str]:
    payload = json.loads(SPLIT_PATH.read_text())
    seeds = [str(seed) for seed in payload["seeds"]]
    if payload.get("split_id") != "frontier-t5-y-36w-20260802-v1":
        raise ValueError("unexpected frontier split ID")
    if len(seeds) != 100 or len(set(seeds)) != 100:
        raise ValueError("frontier split must contain 100 unique seeds")
    for index, seed in enumerate(seeds):
        expected = hashlib.sha256(
            f"beer-agent-v1|frontier-t5-y-36w|20260802|{index:05d}".encode()
        ).hexdigest()[:16]
        if seed != expected:
            raise ValueError(f"seed {index} does not match frozen derivation")
    return seeds


def make_spec(seed_hex: str, seed_index: int) -> ScenarioSpec:
    # Reuse the native v0.2.0 Tier-5 scenario parameters and replace only the
    # master seed with the separately frozen 100-seed manifest value.
    template = scenario_for(5, "test", 0, "headline")
    return replace(template, seed_index=seed_index, master_seed_hex=seed_hex)


def run_policy(spec: ScenarioSpec, policy_name: str) -> dict[str, Any]:
    episode = BeerEpisode(spec, ROLE, include_reference=False)
    observation = episode.start()
    actions: list[int] = []
    policy = adaptive_policy(spec, ROLE) if policy_name == "oracle" else None
    while not episode.done:
        if policy_name == "naive":
            quantity = int(observation["state"]["incoming_demand_or_order"])
        elif policy_name == "oracle":
            assert policy is not None
            quantity = int(policy.act(observation))
        else:
            raise ValueError(policy_name)
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
        "status": grade["status"],
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
        "grade": grade,
    }


def mean_stderr(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean_value = statistics.fmean(values)
    stderr = (
        statistics.stdev(values) / math.sqrt(len(values))
        if len(values) > 1
        else 0.0
    )
    return mean_value, stderr


def _request_json(url: str, *, method: str, headers: dict[str, str], body: bytes | None = None) -> Any:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=120, context=_ssl_context()) as response:
        return json.loads(response.read().decode())


def openrouter_catalog(api_key: str) -> dict[str, Any]:
    payload = _request_json(
        "https://openrouter.ai/api/v1/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("OpenRouter catalog response was not a model list")
    return payload


def model_metadata(catalog: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for item in catalog["data"]:
        if isinstance(item, dict) and item.get("id") == model_id:
            return item
    return None


def _error_payload(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode()
            parsed = json.loads(body)
        except Exception:
            parsed = {"body": body if "body" in locals() else ""}
        return {"http_status": exc.code, "error": parsed}
    return {"error_type": type(exc).__name__, "error": str(exc)}


class OpenRouterClient:
    def __init__(self, model_id: str, raw_path: Path, workers: int):
        self.model_id = model_id
        self.raw_path = raw_path
        self.workers = workers
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._raw_lock = threading.Lock()

    def _write_raw(self, record: dict[str, Any]) -> None:
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        with self._raw_lock:
            with self.raw_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def call(self, observation: dict[str, Any], *, seed_index: int, week: int, repair: bool = False) -> tuple[int | None, str | None, dict[str, Any]]:
        request_body = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": 128,
            "reasoning": {"effort": "none"},
            "messages": [
                {"role": "system", "content": system_prompt(make_spec("0" * 16, 0), ROLE)},
                {"role": "user", "content": observation_user_message(observation, action_format="tool")},
            ],
            "tools": TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "place_order"},
            },
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
            "X-Title": "Beer Distribution Tier-5 Frontier Evaluation",
        }
        raw_response: dict[str, Any]
        last_error: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                raw_response = _request_json(
                    "https://openrouter.ai/api/v1/chat/completions",
                    method="POST",
                    headers=headers,
                    body=json.dumps(request_body, separators=(",", ":")).encode(),
                )
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
                last_error = _error_payload(exc)
                status = last_error.get("http_status")
                if status not in (408, 409, 429) and not (isinstance(status, int) and status >= 500):
                    raw_response = {"_request_error": last_error}
                    break
                time.sleep(2**attempt)
        else:
            raw_response = {"_request_error": last_error or {"error": "request_failed"}}

        record = {
            "seed_index": seed_index,
            "week": week,
            "repair_attempt": repair,
            "request": request_body,
            "response": raw_response,
        }
        self._write_raw(record)
        quantity, error = parse_tool_action(raw_response)
        return quantity, error, raw_response


def parse_tool_action(response: dict[str, Any]) -> tuple[int | None, str | None]:
    try:
        message = response["choices"][0]["message"]
        calls = message.get("tool_calls") or []
    except (KeyError, IndexError, TypeError):
        return None, "missing_tool_call"
    if len(calls) != 1:
        return None, "missing_tool_call" if not calls else "multiple_tool_calls"
    call = calls[0]
    if not isinstance(call, dict) or call.get("type") not in (None, "function"):
        return None, "unknown_tool"
    function = call.get("function")
    if not isinstance(function, dict) or function.get("name") != "place_order":
        return None, "unknown_tool"
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(arguments, dict) or set(arguments) != {"quantity"}:
        return None, "invalid_arguments"
    quantity = arguments["quantity"]
    if type(quantity) is not int or not 0 <= quantity <= ORDER_CAP:
        return None, "invalid_quantity"
    return quantity, None


def run_model_episode(client: OpenRouterClient, spec: ScenarioSpec) -> dict[str, Any]:
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
            quantity, repair_error, _ = client.call(
                observation, seed_index=spec.seed_index, week=week, repair=True
            )
            if repair_error is not None:
                format_failures += 1
                protocol_errors.append(repair_error)
                episode.mark_protocol_error()
                outcome = episode.protocol_failure_outcome(
                    error_count=len(protocol_errors), category=repair_error
                )
                return {
                    "seed_index": spec.seed_index,
                    "master_seed_hex": spec.master_seed_hex,
                    "episode_id": episode.episode_id,
                    "actions": actions,
                    "format_failures": format_failures,
                    "protocol_errors": protocol_errors,
                    "status": "protocol_error",
                    "protocol_clean": False,
                    "outcome": outcome,
                }
            if len(protocol_errors) >= 3:
                outcome = episode.protocol_failure_outcome(
                    error_count=len(protocol_errors), category=error
                )
                return {
                    "seed_index": spec.seed_index,
                    "master_seed_hex": spec.master_seed_hex,
                    "episode_id": episode.episode_id,
                    "actions": actions,
                    "format_failures": format_failures,
                    "protocol_errors": protocol_errors,
                    "status": "protocol_error",
                    "protocol_clean": False,
                    "outcome": outcome,
                }
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


def summarize_policy(rows: list[dict[str, Any]], *, response_count: int = 0, format_failures: int = 0) -> dict[str, Any]:
    completed = [row for row in rows if "local_total_cost" in row]
    clean = [row for row in completed if row.get("protocol_clean")]
    costs = [float(row["local_total_cost"]) for row in completed]
    mean_cost, stderr = mean_stderr(costs)
    return {
        "mean_local_wholesaler_cost": mean_cost,
        "stderr": stderr,
        "completed_episodes": len(completed),
        "completed_clean_episodes": len(clean),
        "episodes": len(rows),
        "protocol_failed_episodes": sum(row.get("status") == "protocol_error" for row in rows),
        "format_failures": format_failures,
        "format_failure_rate": format_failures / response_count if response_count else 0.0,
        "responses": response_count,
    }


def write_report(payload: dict[str, Any]) -> None:
    baseline = payload["baselines"]
    naive = float(baseline["naive"]["mean_local_wholesaler_cost"])
    oracle = float(baseline["oracle"]["mean_local_wholesaler_cost"])
    lines = [
        "# Frontier OpenRouter Tier-5 Y wholesaler evaluation",
        "",
        "Native environment `0.2.0`; 36 operational weeks plus native settlement; Tier-5 Y topology; controlled role `wholesaler`; scripted counterparties; temperature 0; reasoning disabled; strict `place_order` tool calls only.",
        "",
        f"Frozen split: `{payload['split']['path']}` ({payload['split']['count']} seeds), created before OpenRouter requests. No training, public live-game, or prior benchmark seeds were used.",
        "",
        "`raw = (naive_cost - model_cost) / (naive_cost - oracle_cost)` and `score = clamp(10 + 90 * raw, 0, 100)`. Naive therefore scores 10 and oracle scores 100.",
        "",
        "Mean and stderr are across completed episodes with a native local-cost grade. Format failures are counted at the response level; a failed repair terminates that episode under the native protocol-error rule and is reported separately.",
        "",
        "| Model | OpenRouter ID | Mean local wholesaler cost ± stderr | Format failures | Raw | Score |",
        "|---|---|---:|---:|---:|---:|",
        f"| Naive | deterministic `incoming_demand_or_order` | {naive:.3f} ± {baseline['naive']['stderr']:.3f} | 0 | 0.000 | 10.000 |",
        f"| Oracle | `adaptive_base_stock_v2` | {oracle:.3f} ± {baseline['oracle']['stderr']:.3f} | 0 | 1.000 | 100.000 |",
    ]
    for key, model_id in MODELS:
        row = payload["models"].get(key)
        if not row:
            continue
        if row.get("status") != "complete":
            reason = row.get("skip_reason", "unavailable")
            lines.append(f"| {key} (skipped: {reason}) | `{model_id}` | — | — | — | — |")
            continue
        metrics = row["metrics"]
        if metrics["mean_local_wholesaler_cost"] is None:
            lines.append(f"| {key} (no completed episodes) | `{model_id}` | — | {metrics['format_failures']} / {metrics['responses']} | — | — |")
            continue
        model_cost = float(metrics["mean_local_wholesaler_cost"])
        raw = (naive - model_cost) / (naive - oracle)
        score = min(100.0, max(0.0, 10.0 + 90.0 * raw))
        status = "" if row.get("status") == "complete" else f" ({row.get('status')})"
        lines.append(
            f"| {key}{status} | `{model_id}` | {model_cost:.3f} ± {float(metrics['stderr']):.3f} | {metrics['format_failures']} / {metrics['responses']} ({100 * float(metrics['format_failure_rate']):.2f}%) | {raw:.3f} | {score:.3f} |"
        )
    lines += [
        "",
        "Raw request/response JSONL is stored under `raw/`; per-episode actions and grades are under `episodes/`; the aggregate JSON is `results.json`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines))


def score_payload(payload: dict[str, Any]) -> None:
    naive = float(payload["baselines"]["naive"]["mean_local_wholesaler_cost"])
    oracle = float(payload["baselines"]["oracle"]["mean_local_wholesaler_cost"])
    denominator = naive - oracle
    for row in payload["models"].values():
        if row.get("status") != "complete":
            row["raw"] = None
            row["score"] = None
            continue
        if row["metrics"]["mean_local_wholesaler_cost"] is None:
            row["raw"] = None
            row["score"] = None
            continue
        cost = float(row["metrics"]["mean_local_wholesaler_cost"])
        raw = (naive - cost) / denominator if denominator else float("nan")
        row["raw"] = raw
        row["score"] = min(100.0, max(0.0, 10.0 + 90.0 * raw))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--only", action="append", choices=[key for key, _ in MODELS])
    parser.add_argument("--catalog-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    seeds = load_split()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; no model calls made")
    catalog = openrouter_catalog(api_key)
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    available = {
        key: model_metadata(catalog, model_id) is not None
        for key, model_id in MODELS
    }
    if args.catalog_only:
        print(json.dumps({"available": available}, indent=2, sort_keys=True))
        return

    baseline_rows = {
        policy: [run_policy(make_spec(seed, index), policy) for index, seed in enumerate(seeds)]
        for policy in ("naive", "oracle")
    }
    fresh_payload: dict[str, Any] = {
        "protocol": {
            "environment_version": ENVIRONMENT_VERSION,
            "scenario_id": "t5-strategic-y-v2",
            "topology": "y",
            "controlled_role": ROLE,
            "horizon_operational_weeks": 36,
            "settlement_weeks": 3,
            "order_contract": "exactly one place_order tool call; integer quantity 0..128",
            "reasoning": "disabled via reasoning.effort=none",
            "temperature": 0,
            "counterparties": "scarcity_aggressive_v1 retailers; adaptive_base_stock_v2 upstream",
            "model_ids_are_versioned_api_ids": True,
        },
        "split": {"path": str(SPLIT_PATH.relative_to(ROOT)), "count": len(seeds), "ids": seeds},
        "baselines": {
            "naive": summarize_policy(baseline_rows["naive"]),
            "oracle": summarize_policy(baseline_rows["oracle"]),
        },
        "models": {},
    }
    if RESULTS_PATH.is_file():
        existing = json.loads(RESULTS_PATH.read_text())
        if existing.get("split", {}).get("ids") == seeds:
            payload = existing
        else:
            payload = fresh_payload
    else:
        payload = fresh_payload
    for policy, rows in baseline_rows.items():
        (OUTPUT_ROOT / "episodes").mkdir(exist_ok=True)
        (OUTPUT_ROOT / "episodes" / f"{policy}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )

    requested = [key for key, _ in MODELS if not args.only or key in args.only]
    for key in requested:
        model_id = dict(MODELS)[key]
        if payload["models"].get(key, {}).get("status") == "complete":
            print(f"{key}: already complete; preserving recorded results", flush=True)
            continue
        if not available[key]:
            payload["models"][key] = {
                "status": "skipped_unavailable",
                "model_id": model_id,
                "metrics": summarize_policy([]),
            }
            continue
        print(f"running {key} ({model_id})", flush=True)
        raw_path = OUTPUT_ROOT / "raw" / f"{key}.jsonl"
        episodes_path = OUTPUT_ROOT / "episodes" / f"{key}.jsonl"
        client = OpenRouterClient(model_id, raw_path, args.workers)
        specs = [make_spec(seed, index) for index, seed in enumerate(seeds)]
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_model_episode, client, spec): spec.seed_index for spec in specs}
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                print(f"{key}: completed {completed}/{len(specs)} episodes", flush=True)
        rows.sort(key=lambda row: row["seed_index"])
        episodes_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        responses = sum(len(row.get("actions", [])) + int(row.get("format_failures", 0)) for row in rows)
        failures = sum(int(row.get("format_failures", 0)) for row in rows)
        payload["models"][key] = {
            "status": "complete",
            "model_id": model_id,
            "metrics": summarize_policy(rows, response_count=responses, format_failures=failures),
        }
        score_payload(payload)
        RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        write_report(payload)

    score_payload(payload)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
