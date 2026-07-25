"""Verifiers-free prompt helpers shared by Hub eval, human export, and Colab SFT/GRPO."""

from __future__ import annotations

from typing import Any, Literal

from .scenario import Role, ScenarioSpec, canonical_json

ActionFormat = Literal["tool", "json"]

JSON_ACTION_INSTRUCTION = (
    "Respond with exactly one JSON object of the form "
    '{"quantity": <integer from 0 through 128>}. Do not include any other text.'
)


def public_demand_blurb(spec: ScenarioSpec) -> str:
    if spec.tier == 1:
        return "Customer demand is constant at 8 units per week."
    if spec.tier == 2:
        return "Demand is stationary, persistent, and stochastic with long-run mean 7.5."
    if spec.tier in (3, 4):
        return (
            "Demand begins near mean 7.5 and may undergo one persistent change. "
            "Its time, direction, and new mean are not disclosed."
        )
    return (
        "Each retailer has persistent stochastic demand with long-run mean 7.5. "
        "The factory's public capacity is 22 and shortages use the stated rationing rule."
    )


def system_prompt(spec: ScenarioSpec, role: Role) -> str:
    """Exact Hub system prompt text used by BeerTaskset."""
    rival = (
        " A scripted rival claimant may order aggressively; its state and policy parameters are private."
        if spec.tier == 5
        else ""
    )
    mechanism = (
        f" Factory capacity is {spec.capacity}; allocation under shortage is {spec.rationing}."
        if spec.tier == 5
        else ""
    )
    return (
        f"You control the {role} in a {spec.topology} beer-distribution supply chain for "
        f"{spec.horizon} decision weeks. Minimize only your role's local holding and backlog "
        f"cost. Holding costs {spec.holding_cost} per unit-week and backlog costs "
        f"{spec.backlog_cost} per unit-week. Orders take {spec.order_delay} week and shipments "
        f"take {spec.shipment_delay} weeks. {public_demand_blurb(spec)}{mechanism}{rival} "
        f"Place exactly one order each week by calling place_order with one integer quantity "
        f"from 0 through {spec.order_cap}. Do not answer with a plain-text order or call any "
        "other tool. There is no separate final answer."
    )


def observation_user_message(
    observation: dict[str, Any],
    *,
    action_format: ActionFormat = "tool",
) -> str:
    body = "Current observation:\n" + canonical_json(observation)
    if action_format == "json":
        return body + "\n\n" + JSON_ACTION_INSTRUCTION
    return body


def assistant_action_message(
    quantity: int,
    *,
    action_format: ActionFormat = "json",
) -> str:
    if action_format == "json":
        return canonical_json({"quantity": int(quantity)})
    return f"place_order(quantity={int(quantity)})"


def sft_chat_example(
    spec: ScenarioSpec,
    role: Role,
    observation: dict[str, Any],
    quantity: int,
    *,
    action_format: ActionFormat = "json",
) -> dict[str, Any]:
    """One supervised week: system + observation → labeled action."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt(spec, role)},
            {
                "role": "user",
                "content": observation_user_message(
                    observation, action_format=action_format
                ),
            },
            {
                "role": "assistant",
                "content": assistant_action_message(
                    quantity, action_format=action_format
                ),
            },
        ],
        "meta": {
            "tier": spec.tier,
            "role": role,
            "split": spec.split,
            "seed_index": spec.seed_index,
            "seed": spec.master_seed_hex,
            "scenario_id": spec.scenario_id,
            "variant": spec.variant,
            "week": observation.get("week"),
            "quantity": int(quantity),
            "action_format": action_format,
            "environment_version": spec.environment_version,
        },
    }
