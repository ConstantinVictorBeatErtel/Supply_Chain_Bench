"""Render Hub observation dicts for humans without leaking extra state."""

from __future__ import annotations

from typing import Any

# Exact top-level keys from BeerGameCore.observation (Hub FOW contract).
OBSERVATION_KEYS = frozenset(
    {
        "episode_id",
        "scenario_id",
        "week",
        "horizon",
        "weeks_remaining",
        "role",
        "topology",
        "observation_mode",
        "state",
        "costs",
        "constraints",
        "recent_history",
    }
)

STATE_KEYS_AGGREGATE = frozenset(
    {
        "inventory_on_hand",
        "backlog",
        "inventory_position",
        "on_order",
        "shipment_received",
        "incoming_demand_or_order",
        "units_filled",
        "last_order_placed",
    }
)

STATE_KEYS_SHIPMENT_NOTICES = STATE_KEYS_AGGREGATE | {"inbound_shipment_pipeline"}

COST_KEYS = frozenset(
    {
        "holding_per_unit",
        "backlog_per_unit",
        "current_inventory_backlog_cost",
        "cumulative_local_cost_through_previous_week",
    }
)

CONSTRAINT_KEYS = frozenset(
    {
        "minimum_order",
        "maximum_order",
        "factory_capacity",
    }
)


def observation_field_keys(observation: dict[str, Any]) -> set[str]:
    return set(observation)


def assert_observation_parity(observation: dict[str, Any]) -> None:
    """Raise if the observation contains keys outside the Hub FOW contract."""
    extra = set(observation) - OBSERVATION_KEYS
    if extra:
        raise ValueError(f"observation leaked unexpected keys: {sorted(extra)}")
    mode = observation.get("observation_mode")
    allowed_state = (
        STATE_KEYS_SHIPMENT_NOTICES
        if mode == "shipment_notices"
        else STATE_KEYS_AGGREGATE
    )
    state = observation.get("state", {})
    state_extra = set(state) - allowed_state
    if state_extra:
        raise ValueError(f"state leaked unexpected keys: {sorted(state_extra)}")
    cost_extra = set(observation.get("costs", {})) - COST_KEYS
    if cost_extra:
        raise ValueError(f"costs leaked unexpected keys: {sorted(cost_extra)}")
    constraint_extra = set(observation.get("constraints", {})) - CONSTRAINT_KEYS
    if constraint_extra:
        raise ValueError(
            f"constraints leaked unexpected keys: {sorted(constraint_extra)}"
        )


def format_observation_markdown(observation: dict[str, Any]) -> str:
    """Human-readable view of the exact Hub observation (no added fields)."""
    assert_observation_parity(observation)
    state = observation["state"]
    costs = observation["costs"]
    constraints = observation["constraints"]
    lines = [
        f"**Week {observation['week']} / {observation['horizon']}** "
        f"({observation['weeks_remaining']} remaining)",
        f"Role: `{observation['role']}` · Topology: `{observation['topology']}` · "
        f"Mode: `{observation['observation_mode']}`",
        "",
        "### State",
        f"- Inventory on hand: **{state['inventory_on_hand']}**",
        f"- Backlog: **{state['backlog']}**",
        f"- Inventory position: **{state['inventory_position']}**",
        f"- On order: **{state['on_order']}**",
        f"- Shipment received: **{state['shipment_received']}**",
        f"- Incoming demand/order: **{state['incoming_demand_or_order']}**",
        f"- Units filled: **{state['units_filled']}**",
        f"- Last order placed: **{state['last_order_placed']}**",
    ]
    if "inbound_shipment_pipeline" in state:
        lines.append(
            f"- Inbound shipment pipeline: `{state['inbound_shipment_pipeline']}`"
        )
    lines.extend(
        [
            "",
            "### Costs",
            f"- Holding per unit: {costs['holding_per_unit']}",
            f"- Backlog per unit: {costs['backlog_per_unit']}",
            f"- Current inventory/backlog cost: **{costs['current_inventory_backlog_cost']}**",
            f"- Cumulative local cost (through previous week): "
            f"**{costs['cumulative_local_cost_through_previous_week']}**",
            "",
            "### Constraints",
            f"- Order bounds: [{constraints['minimum_order']}, "
            f"{constraints['maximum_order']}]",
            f"- Factory capacity: {constraints['factory_capacity']}",
            "",
            "### Recent history (own role, last "
            f"{len(observation['recent_history'])} weeks)",
        ]
    )
    history = observation["recent_history"]
    if not history:
        lines.append("_No prior weeks yet._")
    else:
        for row in history:
            lines.append(
                f"- Week {row['week']}: demand/order={row['incoming_demand_or_order']}, "
                f"received={row['shipment_received']}, ordered={row['order_placed']}, "
                f"inv={row['ending_inventory']}, backlog={row['ending_backlog']}, "
                f"cost={row['local_cost']}"
            )
    lines.extend(
        [
            "",
            "<details><summary>Raw observation JSON (same fields the LLM receives)</summary>",
            "",
            "```json",
            _canonical_pretty(observation),
            "```",
            "",
            "</details>",
        ]
    )
    return "\n".join(lines)


def _canonical_pretty(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)
