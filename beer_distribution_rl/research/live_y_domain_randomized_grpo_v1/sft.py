"""Self-generated format-normalization SFT data construction."""

from __future__ import annotations

from typing import Any, Callable

from beer_distribution_game.prompts import sft_chat_example

from .protocol import permissive_intended_quantity, serialize_completion


def build_self_generated_records(
    episodes: list[Any],
    raw_outputs: list[list[str]],
    *,
    target: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Re-serialize the base model's own selected action; never substitute a teacher."""

    rows: list[dict[str, Any]] = []
    dropped = 0
    for episode, outputs in zip(episodes, raw_outputs):
        observations = getattr(episode, "research_observations", None)
        if observations is None:
            raise ValueError("episode must provide research_observations captured during base rollout")
        for observation, raw in zip(observations, outputs):
            quantity = permissive_intended_quantity(raw)
            if quantity is None:
                dropped += 1
                continue
            row = sft_chat_example(
                episode.spec,
                episode.controlled_role,
                observation,
                quantity,
                action_format="json",
            )
            row["messages"][-1]["content"] = serialize_completion(quantity)
            row["meta"]["source"] = "base-model-self-generated-format-normalization"
            rows.append(row)
            if len(rows) >= target:
                return rows, {"records": len(rows), "dropped_no_integer": dropped}
    return rows, {"records": len(rows), "dropped_no_integer": dropped}


def protocol_rate(outputs: list[str], tokenizer: Callable[[str], object] | None = None) -> float:
    from .protocol import parse_completion

    if not outputs:
        return 0.0
    return sum(parse_completion(text, tokenizer=tokenizer) is not None for text in outputs) / len(outputs)
