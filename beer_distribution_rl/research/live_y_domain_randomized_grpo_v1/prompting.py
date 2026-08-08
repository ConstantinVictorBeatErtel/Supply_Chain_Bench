"""Human-information-parity prompt helpers for the research protocol.

The frozen Hub prompt remains unchanged. This research prompt deliberately
does not disclose the demand law/parameters or factory capacity. The model
still receives its local state, retailer orders visible to the wholesaler, and
the bounded action interface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from beer_distribution_game.scenario import Role, ScenarioSpec, canonical_json

from .protocol import COMPLETION_LIMIT, THINK_LIMIT


def research_system_prompt(spec: ScenarioSpec, role: Role) -> str:
    rival = (
        " A scripted rival claimant may order aggressively; its state and policy parameters are private."
        if spec.tier == 5
        else ""
    )
    return (
        f"You control the {role} in a {spec.topology} beer-distribution supply chain for "
        f"{spec.horizon} decision weeks. Minimize only your role's local holding and backlog "
        f"cost. Holding costs {spec.holding_cost} per unit-week and backlog costs "
        f"{spec.backlog_cost} per unit-week. Orders take {spec.order_delay} week and shipments "
        f"take {spec.shipment_delay} weeks. Demand-generating parameters and other roles' "
        f"private information are not disclosed; use only the role-local observation and "
        f"the history supplied each week.{rival} "
        f"Place exactly one order each week. Return an integer quantity from 0 through "
        f"{spec.order_cap} using exactly one JSON object {{\"quantity\": INTEGER}}. "
        f"An optional <think> block is allowed up to {THINK_LIMIT} tokens, with a total "
        f"completion budget of {COMPLETION_LIMIT} tokens. Do not include prose after the JSON."
    )


def research_observation_user_message(observation: dict[str, Any]) -> str:
    """Serialize the same local state while omitting undisclosed global facts."""

    value = deepcopy(observation)
    constraints = value.get("constraints")
    if isinstance(constraints, dict):
        constraints.pop("factory_capacity", None)
    return "Current observation:\n" + canonical_json(value) + (
        "\n\nRespond with exactly one JSON object of the form "
        '{"quantity": <integer from 0 through 128>}. Do not include any other text.'
    )
