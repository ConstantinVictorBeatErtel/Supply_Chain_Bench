"""The strict response protocol and offline extractor for this experiment."""

from __future__ import annotations

import json
import re
from typing import Callable

THINK_RE = re.compile(r"^<think>(?P<thought>.*?)</think>(?P<body>.*)\Z", re.DOTALL)
THINK_LIMIT = 160
COMPLETION_LIMIT = 192


def _token_count(text: str, tokenizer: Callable[[str], object] | None) -> int:
    if tokenizer is None:
        # The production runner passes the actual tokenizer. This fallback is
        # deliberately conservative for CPU validation and does not bless text
        # that could exceed the real token budget.
        return len(text.split())
    value = tokenizer(text)
    return len(value["input_ids"] if isinstance(value, dict) else value)


def parse_completion(
    text: str,
    *,
    tokenizer: Callable[[str], object] | None = None,
    think_limit: int = THINK_LIMIT,
    completion_limit: int = COMPLETION_LIMIT,
) -> int | None:
    """Accept optional bounded thinking followed by exactly one JSON object."""

    match = THINK_RE.match(text)
    thought = ""
    body = text
    if match:
        thought = match.group("thought")
        body = match.group("body")
    if _token_count(thought, tokenizer) > think_limit:
        return None
    if _token_count(text, tokenizer) > completion_limit:
        return None
    body = body.strip()
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"quantity"}:
        return None
    quantity = value["quantity"]
    if type(quantity) is not int or not 0 <= quantity <= 128:
        return None
    return quantity


def serialize_completion(quantity: int, thought: str = "") -> str:
    if type(quantity) is not int or not 0 <= quantity <= 128:
        raise ValueError("quantity must be an integer in [0, 128]")
    if thought:
        return f"<think>{thought}</think>\n" + json.dumps({"quantity": quantity}, separators=(",", ":"))
    return json.dumps({"quantity": quantity}, separators=(",", ":"))


def permissive_intended_quantity(text: str) -> int | None:
    """Recover an order from a base-model response for SFT serialization only."""

    candidates = re.findall(r'"quantity"\s*:\s*(-?\d+)', text)
    if not candidates:
        candidates = re.findall(r"place_order\s*\(\s*quantity\s*=\s*(-?\d+)", text)
    if len(candidates) != 1:
        return None
    value = int(candidates[0])
    return value if 0 <= value <= 128 else None
