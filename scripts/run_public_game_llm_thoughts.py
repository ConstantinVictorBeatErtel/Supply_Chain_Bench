#!/usr/bin/env python3
"""Record the public game's comparison model with a short thought per week.

The published comparison traces (``artifacts/hub_llm/...``) store actions only:
the Hub eval protocol runs with reasoning disabled and a strict
``{"quantity": n}`` reply, so nothing the model *thought* survives. The static
web debrief wants a week-by-week thought tracker, which needs a run where the
rationale and the order come from the same forward pass.

This runner plays the eight playable seeds (development 0-2, validation 0-4) of
the Tier-5 headline Y scenario at the *public* factory capacity of 400 — the
same scenario a human plays in ``static_web`` — with the protocol from
``research/.../protocol.py``: optional bounded ``<think>`` text, then exactly one
JSON object.

Usage:
    OPENROUTER_API_KEY=sk-or-... python scripts/run_public_game_llm_thoughts.py \
        --output artifacts/public_game_llm_thoughts/traces.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import json
import os
from pathlib import Path
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import observation_user_message
from beer_distribution_game.scenario import ScenarioSpec, scenario_for

DEFAULT_MODEL = "x-ai/grok-4.5"
# Models that reject reasoning.effort=none and spend budget on hidden reasoning.
REASONING_REQUIRED_MODELS = {"x-ai/grok-4.5"}
REASONING_MAX_TOKENS = 768
DEFAULT_MAX_TOKENS = 400
PUBLIC_FACTORY_CAPACITY = 400
CONTROLLED_ROLE = "wholesaler"
# static_web/src/sim/scenario.js PLAYABLE_SEEDS
PLAYABLE_SEEDS = [("development", index) for index in range(3)] + [
    ("validation", index) for index in range(5)
]

THINK_RE = re.compile(r"<think>(?P<thought>.*?)</think>", re.DOTALL)
QUANTITY_RE = re.compile(r'"quantity"\s*:\s*(-?\d+)')
THOUGHT_RE = re.compile(r'"thought"\s*:\s*"(?P<thought>(?:[^"\\]|\\.)*)"')
FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*([\s\S]*?)\s*```$")
THOUGHT_WORD_LIMIT = 45

ACTION_INSTRUCTION = (
    "Respond with exactly one JSON object and nothing else, of the form "
    '{"thought": "<one or two short sentences>", "quantity": <integer from 0 '
    'through 128>}. The thought is shown to a person watching the replay back '
    "afterwards: plain language, under "
    f"{THOUGHT_WORD_LIMIT} words, saying what you read in this week's numbers "
    "and why you are ordering that amount. Say it the way you would to a "
    "colleague standing next to you, not as a list of readings. No markdown, no "
    "code fences, no text outside the object."
)


def public_spec(split: str, seed_index: int) -> ScenarioSpec:
    """Tier-5 headline Y scenario at the capacity the public game actually runs."""
    return dataclasses.replace(
        scenario_for(5, split, seed_index), capacity=PUBLIC_FACTORY_CAPACITY
    )


def system_prompt(spec: ScenarioSpec) -> str:
    """Hub system prompt, with the capacity sentence corrected for public play.

    ``prompts.public_demand_blurb`` hard-codes the research capacity of 22, which
    would be a false statement in a capacity-400 episode.
    """
    return (
        f"You control the {CONTROLLED_ROLE} in a {spec.topology} beer-distribution "
        f"supply chain for {spec.horizon} decision weeks. Minimize only your role's "
        f"local holding and backlog cost. Holding costs {spec.holding_cost} per "
        f"unit-week and backlog costs {spec.backlog_cost} per unit-week. Orders take "
        f"{spec.order_delay} week and shipments take {spec.shipment_delay} weeks. "
        "Each retailer has persistent stochastic demand with long-run mean 7.5. "
        f"The factory's capacity is {spec.capacity} units per week and shortages use "
        f"{spec.rationing} allocation. A scripted rival claimant may order "
        "aggressively; its state and policy parameters are private. "
        "Place exactly one order each week. " + ACTION_INSTRUCTION
    )


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _as_order(value: object) -> int | None:
    """Accept 16, 16.0 and "16" — reject anything that is not a whole order."""
    if type(value) is bool:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_reply(text: str) -> tuple[int | None, str]:
    """Return (quantity, thought). Quantity is None when the protocol failed."""
    thought = ""
    body = text.strip()
    think = THINK_RE.search(body)
    if think:
        # Some models narrate in <think> tags instead of the JSON field.
        thought = " ".join(think.group("thought").split())
        body = body[think.end() :].strip()
    fenced = FENCE_RE.match(body)
    if fenced:
        body = fenced.group(1).strip()

    quantity: int | None = None
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        quantity = _as_order(value.get("quantity"))
        if not thought and isinstance(value.get("thought"), str):
            thought = " ".join(value["thought"].split())
    if quantity is None:
        # Salvage a lone order out of near-miss formatting (trailing prose, etc.).
        candidates = QUANTITY_RE.findall(body)
        if len(candidates) == 1:
            quantity = int(candidates[0])
        if not thought:
            loose = THOUGHT_RE.search(body)
            if loose:
                thought = " ".join(
                    json.loads(f'"{loose.group("thought")}"').split()
                )
    if quantity is None or not 0 <= quantity <= 128:
        return None, thought
    return quantity, thought


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
        self.calls = 0

    def _charge(self, usage: dict[str, Any]) -> None:
        with self.lock:
            self.spent += float(usage.get("cost") or 0.0)
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.cached_tokens += int(
                (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            )
            self.calls += 1
            if self.spent > self.budget_usd:
                raise RuntimeError(
                    f"budget exceeded: ${self.spent:.4f} > ${self.budget_usd:.2f}"
                )

    def complete(self, *, system: str, user: str, session_id: str) -> str:
        reasoning_required = self.model_id in REASONING_REQUIRED_MODELS
        body: dict[str, Any] = {
            "model": self.model_id,
            "temperature": 0.7,
            "top_p": 0.95,
            # Hidden reasoning competes with the visible reply for this budget,
            # so models that cannot disable it need the larger allowance.
            "max_tokens": REASONING_MAX_TOKENS if reasoning_required else DEFAULT_MAX_TOKENS,
            "session_id": session_id,
            "reasoning": {"effort": "low" if reasoning_required else "none"},
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
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ConstantinVictorBeatErtel/beer_distribution_RL",
            "X-Title": "Beer Distribution Public Game Thought Traces",
        }
        last_error: Any = None
        flattened_system = False
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=json.dumps(body, separators=(",", ":")).encode(),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(
                    request, timeout=180, context=ssl_context()
                ) as handle:
                    response = json.loads(handle.read().decode())
                self._charge(response.get("usage") or {})
                choices = response.get("choices") or []
                message = (choices[0].get("message") if choices else {}) or {}
                content = message.get("content")
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content or "")
            except urllib.error.HTTPError as exc:
                try:
                    payload = json.loads(exc.read().decode())
                except Exception:
                    payload = {}
                last_error = {"status": exc.code, "body": payload}
                message = str((payload.get("error") or {}).get("message") or payload).lower()
                if exc.code == 400 and (
                    "reasoning is mandatory" in message or "cannot be disabled" in message
                ):
                    body["reasoning"] = {"effort": "low"}
                    body["max_tokens"] = max(
                        int(body.get("max_tokens") or 0), REASONING_MAX_TOKENS
                    )
                elif exc.code == 400 and not flattened_system:
                    body["messages"][0]["content"] = system
                    flattened_system = True
                elif exc.code not in (408, 409, 429) and exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = {"type": type(exc).__name__, "message": str(exc)}
            time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenRouter request failed: {last_error}")


def run_episode(
    client: OpenRouterClient, split: str, seed_index: int, *, retries: int
) -> dict[str, Any]:
    spec = public_spec(split, seed_index)
    system = system_prompt(spec)
    episode = BeerEpisode(spec, CONTROLLED_ROLE)
    observation = episode.start()
    session_id = f"public-game-thoughts|{client.model_id}|{split}|{seed_index}"
    actions: list[int] = []
    weeks: list[dict[str, Any]] = []
    format_failures = 0

    while not episode.done:
        week = int(observation["week"])
        user = observation_user_message(observation)
        quantity: int | None = None
        thought = ""
        raw = ""
        for attempt in range(retries + 1):
            nudge = (
                ""
                if attempt == 0
                else "\n\nYour previous reply did not follow the protocol. "
                + ACTION_INSTRUCTION
            )
            raw = client.complete(
                system=system, user=user + nudge, session_id=session_id
            )
            quantity, thought = parse_reply(raw)
            if quantity is not None:
                break
            format_failures += 1
            print(
                f"[format-failure] {split}-{seed_index} week {week} attempt {attempt}: {raw[:300]!r}",
                file=sys.stderr,
                flush=True,
            )
        if quantity is None:
            raise RuntimeError(
                f"{split}-{seed_index} week {week}: no valid order after "
                f"{retries + 1} attempts; last reply was {raw[:300]!r}"
            )
        actions.append(quantity)
        weeks.append({"week": week, "quantity": quantity, "thought": thought})
        result = episode.place_order(quantity)
        if not result["done"]:
            observation = result["next_observation"]

    assert episode.outcome is not None
    grade = episode.outcome["grade"]
    history = episode.histories[CONTROLLED_ROLE]
    for record, row in zip(weeks, history, strict=True):
        record["demand"] = int(row["incoming_demand_or_order"])
        record["ending_inventory"] = int(row["ending_inventory"])
        record["ending_backlog"] = int(row["ending_backlog"])
        record["local_cost"] = float(row["local_cost"])
    return {
        "id": f"{split}-{seed_index}",
        "split": split,
        "seed_index": seed_index,
        "master_seed_hex": spec.master_seed_hex,
        "episode_id": episode.episode_id,
        "scenario_id": spec.scenario_id,
        "capacity": spec.capacity,
        "actions": actions,
        "weeks": weeks,
        "format_failures": format_failures,
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "local_total_cost": grade["primary"]["local_total_cost"],
        "paired_base_stock_local_total_cost": grade["primary"][
            "paired_base_stock_local_total_cost"
        ],
        "reward": grade["episode_reward"],
        "system_total_cost": grade["costs"]["system_total_cost"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/public_game_llm_thoughts/traces.json",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--budget-usd", type=float, default=2.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="only run the first N seeds (smoke test)"
    )
    args = parser.parse_args()

    jobs = PLAYABLE_SEEDS[: args.limit] if args.limit else PLAYABLE_SEEDS
    client = OpenRouterClient(args.model, budget_usd=args.budget_usd)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_episode, client, split, index, retries=args.retries): (
                split,
                index,
            )
            for split, index in jobs
        }
        for future in as_completed(futures):
            split, index = futures[future]
            row = future.result()
            rows.append(row)
            print(
                json.dumps(
                    {
                        "seed": row["id"],
                        "local_total_cost": row["local_total_cost"],
                        "reward": round(row["reward"], 4),
                        "format_failures": row["format_failures"],
                        "spent_usd": round(client.spent, 5),
                    }
                ),
                flush=True,
            )

    rows.sort(key=lambda row: ({"development": 0, "validation": 1}[row["split"]], row["seed_index"]))
    payload = {
        "schema_version": "1.0.0",
        "environment_version": "0.2.0",
        "scenario_id": "t5-strategic-y-v2",
        "controlled_role": CONTROLLED_ROLE,
        "capacity": PUBLIC_FACTORY_CAPACITY,
        "model": args.model,
        "decoding": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": (
                REASONING_MAX_TOKENS
                if args.model in REASONING_REQUIRED_MODELS
                else DEFAULT_MAX_TOKENS
            ),
        },
        "protocol": 'one JSON object per week: {"thought": ..., "quantity": ...}',
        "usage": {
            "calls": client.calls,
            "prompt_tokens": client.prompt_tokens,
            "completion_tokens": client.completion_tokens,
            "cached_tokens": client.cached_tokens,
            "cost_usd": round(client.spent, 6),
        },
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output} ({len(rows)} episodes, ${client.spent:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
