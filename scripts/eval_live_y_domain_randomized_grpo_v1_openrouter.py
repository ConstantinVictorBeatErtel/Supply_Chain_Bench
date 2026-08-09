#!/usr/bin/env python3
"""Evaluate OpenRouter models on the fixed live-y research protocol.

Uses research prompts (no factory capacity), sticky session_id routing, and
explicit system-prefix cache_control so provider prompt caches stay warm across
the 36 weekly turns of each episode. Episodes may run in parallel; weeks within
an episode stay serial.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import ssl
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    research_spec,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
    research_system_prompt,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion

DEFAULT_MODELS = (
    "deepseek/deepseek-v4-flash-0731",
    "openai/gpt-5.6-luna",
    "x-ai/grok-4.5",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "poolside/laguna-s-2.1:free",
)

# Models that reject reasoning.effort=none and spend completion budget on
# hidden reasoning tokens. Leave headroom so the JSON action still fits.
REASONING_REQUIRED_MODELS = {
    "x-ai/grok-4.5",
}
REASONING_MAX_TOKENS = 768
DEFAULT_MAX_TOKENS = 192

FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```\s*$")


def normalize_openrouter_completion(text: str) -> str:
    """Strip optional markdown fences; keep the strict JSON body for parsing."""

    stripped = text.strip()
    match = FENCE_RE.match(stripped)
    return match.group(1).strip() if match else stripped


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_IPV4_LOCK = threading.Lock()


def _open_url(request: urllib.request.Request, *, timeout: float):
    # Prefer IPv4: some local networks leave IPv6 SYN_SENT hanging.
    import socket

    with _IPV4_LOCK:
        original = socket.getaddrinfo

        def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
            return original(host, port, socket.AF_INET, type, proto, flags)

        socket.getaddrinfo = ipv4_only  # type: ignore[assignment]
        try:
            return urllib.request.urlopen(request, timeout=timeout, context=ssl_context())
        finally:
            socket.getaddrinfo = original  # type: ignore[assignment]


def aggregate(rows: list[dict]) -> dict:
    def mean_stderr(key: str):
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        return {
            "mean": statistics.mean(values) if values else None,
            "stderr": (
                statistics.stdev(values) / (len(values) ** 0.5)
                if len(values) > 1
                else 0.0 if values else None
            ),
        }

    out = {
        key: mean_stderr(key)
        for key in (
            "local_total_cost",
            "system_total_cost",
            "score",
            "bullwhip_ratio",
            "normalized_order_volatility",
        )
    }
    out["n"] = len(rows)
    out["protocol_failure_rate"] = (
        sum(not row["protocol_clean"] for row in rows) / len(rows) if rows else None
    )
    out["format_failure_rate"] = sum(row["format_failures"] for row in rows) / max(
        sum(row["attempted_actions"] for row in rows), 1
    )
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in rows)
    cached_tokens = sum(int(row.get("cached_tokens") or 0) for row in rows)
    out["prompt_tokens"] = prompt_tokens
    out["cached_tokens"] = cached_tokens
    out["cache_hit_fraction"] = (cached_tokens / prompt_tokens) if prompt_tokens else None
    out["usage_cost_usd"] = sum(float(row.get("usage_cost_usd") or 0.0) for row in rows)
    return out


def pilot_reference(seed: str, bucket: str) -> float:
    from beer_distribution_game.policies import adaptive_policy

    episode = BeerEpisode(research_spec(seed, bucket=bucket), "wholesaler", include_reference=False)
    observation = episode.start()
    policy_obj = adaptive_policy(episode.spec, "wholesaler")
    while not episode.done:
        result = episode.place_order(policy_obj.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    return float(episode.outcome["grade"]["primary"]["local_total_cost"])


class OpenRouterClient:
    def __init__(self, model_id: str, *, budget_usd: float) -> None:
        self.model_id = model_id
        self.budget_usd = budget_usd
        self.key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.lock = threading.Lock()
        self.spent = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0

    def _charge(self, usage: dict[str, Any]) -> None:
        cost = float(usage.get("cost") or 0.0)
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        with self.lock:
            self.spent += cost
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cached_tokens += cached
            if self.spent > self.budget_usd:
                raise RuntimeError(
                    f"budget exceeded for {self.model_id}: ${self.spent:.4f} > ${self.budget_usd:.2f}"
                )

    def complete(self, *, system: str, user: str, session_id: str, week: int) -> tuple[str, dict[str, Any]]:
        reasoning_required = self.model_id in REASONING_REQUIRED_MODELS
        max_tokens = REASONING_MAX_TOKENS if reasoning_required else DEFAULT_MAX_TOKENS
        body: dict[str, Any] = {
            "model": self.model_id,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "session_id": session_id,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                },
                {"role": "user", "content": user},
            ],
            "usage": {"include": True},
        }
        if reasoning_required:
            # Grok-4.5 rejects effort=none; low keeps hidden reasoning short.
            body["reasoning"] = {"effort": "low"}
        else:
            body["reasoning"] = {"effort": "none"}
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ConstantinVictorBeatErtel/beer_distribution_RL",
            "X-Title": "Beer Distribution Live-Y Research Evaluation",
            "X-OpenRouter-Cache": "true",
            "X-OpenRouter-Cache-TTL": "86400",
        }
        last_error: dict[str, Any] | None = None
        flattened_system = False
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(body, separators=(",", ":")).encode(),
                    headers=headers,
                    method="POST",
                )
                with _open_url(request, timeout=180) as handle:
                    response = json.loads(handle.read().decode())
                usage = response.get("usage") or {}
                self._charge(usage)
                choices = response.get("choices") or []
                message = (choices[0].get("message") if choices else {}) or {}
                content = message.get("content")
                if isinstance(content, list):
                    text = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                else:
                    text = str(content or "")
                return text, response
            except urllib.error.HTTPError as exc:
                try:
                    payload = json.loads(exc.read().decode())
                except Exception:
                    payload = {}
                last_error = {"status": exc.code, "body": payload}
                message = str((payload.get("error") or {}).get("message") or payload).lower()
                # Some models reject structured system content / cache_control / reasoning.
                if exc.code == 400 and (
                    "reasoning is mandatory" in message or "cannot be disabled" in message
                ):
                    body["reasoning"] = {"effort": "low"}
                    body["max_tokens"] = max(int(body.get("max_tokens") or 0), REASONING_MAX_TOKENS)
                elif exc.code == 400 and not flattened_system:
                    body["messages"][0]["content"] = system
                    flattened_system = True
                    if not reasoning_required:
                        body.pop("reasoning", None)
                elif exc.code not in (408, 409, 429) and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = {"type": type(exc).__name__, "message": str(exc)}
            time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenRouter request failed week={week}: {last_error}")


def evaluate_episode(client: OpenRouterClient, seed: str, bucket: str, label: str) -> dict[str, Any]:
    spec = research_spec(seed, bucket=bucket)
    system = research_system_prompt(spec, "wholesaler")
    episode = BeerEpisode(spec, "wholesaler", include_reference=False)
    observation = episode.start()
    session_id = f"live-y-dr-grpo-v1|{client.model_id}|{bucket}|{seed}"
    raw_outputs: list[str] = []
    actions: list[int] = []
    format_failures = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    usage_cost = 0.0
    while not episode.done:
        user = research_observation_user_message(observation)
        raw, response = client.complete(
            system=system, user=user, session_id=session_id, week=int(observation["week"])
        )
        raw_outputs.append(raw)
        usage = response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        cached_tokens += int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        usage_cost += float(usage.get("cost") or 0.0)
        quantity = parse_completion(normalize_openrouter_completion(raw))
        if quantity is None:
            format_failures += 1
            episode.protocol_failure_outcome(error_count=1, category="invalid_protocol")
            break
        actions.append(quantity)
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    optimal = pilot_reference(seed, bucket)
    local = grade.get("primary", {}).get("local_total_cost")
    score = None if local is None else 100.0 * optimal / float(local)
    return {
        "model": label,
        "model_id": client.model_id,
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
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "usage_cost_usd": usage_cost,
        "demand": episode.outcome.get("research_exogenous"),
    }


def run_model(
    model_id: str,
    *,
    output: Path,
    workers: int,
    budget_usd: float,
) -> dict[str, Any]:
    label = model_id.replace("/", "_").replace(":", "_")
    manifest = json.loads(
        (ROOT / "experiments/live_y_domain_randomized_grpo_v1/seed_manifest.json").read_text()
    )
    client = OpenRouterClient(model_id, budget_usd=budget_usd)
    jobs = [
        (seed, bucket)
        for bucket, seeds in manifest["evaluation"].items()
        for seed in seeds
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(evaluate_episode, client, seed, bucket, label): (bucket, seed)
            for seed, bucket in jobs
        }
        for future in as_completed(futures):
            bucket, seed = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "model": model_id,
                            "bucket": bucket,
                            "seed": seed,
                            "score": row.get("score"),
                            "local_total_cost": row.get("local_total_cost"),
                            "protocol_clean": row.get("protocol_clean"),
                            "cached_tokens": row.get("cached_tokens"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": label,
                        "model_id": model_id,
                        "bucket": bucket,
                        "seed": seed,
                        "protocol_clean": False,
                        "format_failures": 1,
                        "attempted_actions": 0,
                        "completed_weeks": 0,
                        "local_total_cost": None,
                        "optimal_reference_local_cost": None,
                        "system_total_cost": None,
                        "score": None,
                        "bullwhip_ratio": None,
                        "normalized_order_volatility": None,
                        "actions": [],
                        "raw_outputs": [],
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cached_tokens": 0,
                        "usage_cost_usd": 0.0,
                        "error": str(exc),
                    }
                )
                print(json.dumps({"model": model_id, "bucket": bucket, "seed": seed, "error": str(exc)}), flush=True)
    rows.sort(key=lambda row: (row["bucket"], row["seed"]))
    payload = {
        "evaluation_kind": "fixed research evaluation",
        "model": label,
        "model_id": model_id,
        "prompt": "research_observation_user_message",
        "cache": {
            "session_id_per_episode": True,
            "system_cache_control": {"type": "ephemeral", "ttl": "1h"},
            "openrouter_cache_headers": True,
        },
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_completion_tokens": (
                REASONING_MAX_TOKENS if model_id in REASONING_REQUIRED_MODELS else DEFAULT_MAX_TOKENS
            ),
            "reasoning_effort": "low" if model_id in REASONING_REQUIRED_MODELS else "none",
        },
        "summary": aggregate(rows),
        "client_totals": {
            "spent_usd": client.spent,
            "prompt_tokens": client.prompt_tokens,
            "completion_tokens": client.completion_tokens,
            "cached_tokens": client.cached_tokens,
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model": model_id, "summary": payload["summary"]}, indent=2, sort_keys=True), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--workers", type=int, default=4, help="Episode parallelism per model")
    parser.add_argument("--budget-usd", type=float, default=25.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/live_y_domain_randomized_grpo_v1/evaluations",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="Run each model in its own process-level thread pool concurrently",
    )
    args = parser.parse_args()

    def _one(model_id: str) -> dict[str, Any]:
        slug = model_id.replace("/", "_").replace(":", "_")
        return run_model(
            model_id,
            output=args.output_dir / f"openrouter_{slug}.json",
            workers=args.workers,
            budget_usd=args.budget_usd,
        )

    if args.parallel_models and len(args.models) > 1:
        with ThreadPoolExecutor(max_workers=len(args.models)) as pool:
            list(pool.map(_one, args.models))
    else:
        for model_id in args.models:
            _one(model_id)


if __name__ == "__main__":
    main()
