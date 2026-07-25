"""Render Hub observation dicts for humans without leaking extra state."""

from __future__ import annotations

from html import escape
import json
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


def _role_label(role: str) -> str:
    return role.replace("_", " ").title()


def format_meters_html(observation: dict[str, Any]) -> str:
    assert_observation_parity(observation)
    costs = observation["costs"]
    week_cost = costs["current_inventory_backlog_cost"]
    total = costs["cumulative_local_cost_through_previous_week"]
    return f"""
<div class="beer-meters" aria-live="polite">
  <div class="beer-meter">
    <span class="label">Week</span>
    <span class="value">{observation["week"]}</span>
    <span class="sub">/ {observation["horizon"]}</span>
  </div>
  <div class="beer-meter">
    <span class="label">Your week cost</span>
    <span class="value">{week_cost:g}</span>
  </div>
  <div class="beer-meter">
    <span class="label">Your total</span>
    <span class="value">{total:g}</span>
  </div>
</div>
""".strip()


def format_observation_html(observation: dict[str, Any]) -> str:
    """Station panel HTML using only Hub FOW observation fields."""
    assert_observation_parity(observation)
    state = observation["state"]
    costs = observation["costs"]
    constraints = observation["constraints"]
    role = _role_label(str(observation["role"]))
    demand_label = (
        "Customer demand"
        if str(observation["role"]).startswith("retailer")
        else "Incoming order"
    )

    pipeline_row = ""
    if "inbound_shipment_pipeline" in state:
        pipe = escape(str(state["inbound_shipment_pipeline"]))
        pipeline_row = f"""
        <div>
          <dt>Ship pipeline</dt>
          <dd style="font-size:0.95rem">{pipe}</dd>
        </div>
        """

    history_items = []
    for row in observation["recent_history"]:
        history_items.append(
            "<li>"
            f"W{row['week']}: dem={row['incoming_demand_or_order']} · "
            f"recv={row['shipment_received']} · ord={row['order_placed']} · "
            f"inv={row['ending_inventory']} · bl={row['ending_backlog']} · "
            f"cost={row['local_cost']}"
            "</li>"
        )
    history_html = (
        "".join(history_items) if history_items else "<li>No prior weeks yet.</li>"
    )
    raw = escape(
        json.dumps(observation, indent=2, sort_keys=True, ensure_ascii=True)
    )

    return f"""
{format_meters_html(observation)}
<section class="beer-panel" aria-label="Your station">
  <h2>Your station · {escape(role)}</h2>
  <dl class="beer-stats">
    <div><dt>Inventory</dt><dd>{state["inventory_on_hand"]}</dd></div>
    <div><dt>Backlog</dt><dd>{state["backlog"]}</dd></div>
    <div><dt>{escape(demand_label)}</dt><dd>{state["incoming_demand_or_order"]}</dd></div>
    <div><dt>Received</dt><dd>{state["shipment_received"]}</dd></div>
    <div><dt>On order</dt><dd>{state["on_order"]}</dd></div>
    <div><dt>Last order</dt><dd>{state["last_order_placed"]}</dd></div>
    <div><dt>Inventory position</dt><dd>{state["inventory_position"]}</dd></div>
    <div><dt>Units filled</dt><dd>{state["units_filled"]}</dd></div>
    {pipeline_row}
  </dl>
  <div class="beer-cost-box">
    <h3>How your cost is calculated</h3>
    <p>Week cost = ({costs["holding_per_unit"]} × inventory) + ({costs["backlog_per_unit"]} × backlog)</p>
    <p>Orders must be integers in [{constraints["minimum_order"]}, {constraints["maximum_order"]}]. Factory capacity: {constraints["factory_capacity"]}.</p>
  </div>
  <div class="beer-history">
    <h3>Recent history (own role only)</h3>
    <ul>{history_html}</ul>
  </div>
  <details style="margin-top:0.9rem;color:var(--muted);font-size:0.8rem">
    <summary>Raw observation JSON (same fields the LLM receives)</summary>
    <pre style="white-space:pre-wrap;font-family:var(--mono);font-size:0.72rem;color:var(--muted)">{raw}</pre>
  </details>
</section>
""".strip()


def format_observation_markdown(observation: dict[str, Any]) -> str:
    """Plain-text fallback used by tests; same FOW fields only."""
    assert_observation_parity(observation)
    state = observation["state"]
    return (
        f"Week {observation['week']}/{observation['horizon']} · "
        f"Inventory on hand: {state['inventory_on_hand']}"
    )


def format_summary_html(summary: dict[str, Any]) -> str:
    if not summary or summary.get("status") == "in_progress":
        return ""
    status = str(summary["status"])
    title = "Session complete" if status == "completed" else "Session abandoned"
    rows = [
        ("Weeks played", summary.get("weeks_played")),
        ("Seed", f"{summary.get('seed')} ({summary.get('split')} #{summary.get('seed_index')})"),
    ]
    if status == "completed":
        rows.extend(
            [
                ("Final total cost", summary.get("final_total_cost")),
                ("Same-seed base-stock cost", summary.get("base_stock_cost")),
                ("Episode reward", summary.get("episode_reward")),
            ]
        )
    else:
        rows.append(
            (
                "Note",
                "Abandoned before week 36; partial actions were logged anonymously.",
            )
        )
    body = "".join(
        f'<div class="row"><span>{escape(str(k))}</span>'
        f"<span>{escape(str(v))}</span></div>"
        for k, v in rows
    )
    return f'<section class="beer-summary"><h2>{escape(title)}</h2>{body}</section>'
