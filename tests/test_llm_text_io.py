"""Unit tests for agents/llm text I/O (no paid spend; Ollama optional)."""

from __future__ import annotations

import pytest

from beer_distribution_rl.agents.llm.grammar import (
    map_delta_to_order,
    order_delta_gbnf,
    delta_json_schema,
)
from beer_distribution_rl.agents.llm.memory import AgentMemory, WeekRecord
from beer_distribution_rl.agents.llm.parser import (
    parse_delta_json,
    parse_order_from_delta_text,
    parse_order_legacy,
)
from beer_distribution_rl.agents.llm.serializer import (
    FORBIDDEN_SUBSTRINGS,
    OWN_HISTORY_FIELDS,
    observe_local,
    prompt_leak_report,
    serialize_prompt,
)
from beer_distribution_rl.env.core import BeerGameCore, Role, y_topology_env_config


def _y_core(seed: int = 0) -> BeerGameCore:
    core = BeerGameCore(y_topology_env_config(horizon=52, seed=seed))
    core.reset(seed)
    return core


def test_delta_schema_bounds():
    schema = delta_json_schema(8)
    assert schema["properties"]["delta"]["minimum"] == -8
    assert schema["properties"]["delta"]["maximum"] == 8
    assert "delta" in schema["required"]


def test_gbnf_covers_signed_range():
    g = order_delta_gbnf(8)
    assert "-8" in g and "8" in g
    assert "root" in g


def test_map_delta_to_order_clip():
    assert map_delta_to_order(0, 7, order_cap=128) == 7
    assert map_delta_to_order(8, 120, order_cap=128) == 128
    assert map_delta_to_order(-8, 3, order_cap=128) == 0
    with pytest.raises(ValueError):
        map_delta_to_order(9, 7)


def test_parse_delta_json_strict_and_range():
    assert parse_delta_json('{"delta": 3}') == 3
    assert parse_delta_json('{"delta": -8}') == -8
    assert parse_delta_json('{"delta": 9}') is None
    assert parse_delta_json("twelve") is None
    assert parse_delta_json('noise {"delta": 2} trailing') == 2
    assert parse_order_from_delta_text('{"delta": 2}', 10) == 12


def test_parse_order_legacy_check5():
    assert parse_order_legacy("ORDER: 12") == 12
    assert parse_order_legacy("I think we should ORDER: 5\n") == 5
    assert parse_order_legacy("twelve cases please") is None
    assert parse_order_legacy("ORDER: 133") is None


def test_serializer_includes_info_set_and_no_leak():
    core = _y_core(0)
    role = core.roles[0]
    mem = AgentMemory(role=role, role_name=core.role_names[role])
    obs = observe_local(core, role)
    for k in OWN_HISTORY_FIELDS:
        assert k in obs
    prompt = serialize_prompt(
        mem, obs, order_cap=core.config.order_cap, holding=0.5, backlog_cost=1.0
    )
    for k in (
        "inventory=",
        "backlog=",
        "on_order=",
        "last_demand_or_order=",
        "last_shipment_received=",
        "last_order_placed=",
        "ship_pipeline=",
        "order_pipeline=",
    ):
        assert k in prompt
    assert "signals" not in prompt.lower() or "Do not broadcast signals" in prompt
    assert prompt_leak_report(prompt, role, core) == []
    for s in FORBIDDEN_SUBSTRINGS:
        assert s not in prompt.lower()


def test_round_trip_logged_observations_three_weeks():
    """Round-trip on real logged observations — three example weeks."""
    core = _y_core(0)
    role = Role.RETAILER if Role.RETAILER in core.roles else core.roles[0]
    mem = AgentMemory(role=role, role_name=core.role_names[role])
    examples = []
    for week in range(3):
        obs = observe_local(core, role)
        prompt = serialize_prompt(
            mem, obs, order_cap=core.config.order_cap, holding=0.5, backlog_cost=1.0
        )
        assert prompt_leak_report(prompt, role, core) == []
        # Simulate constrained model output
        raw = '{"delta": 0}'
        order = parse_order_from_delta_text(
            raw, int(obs["last_demand_or_order"]), order_cap=core.config.order_cap
        )
        assert order is not None
        examples.append(
            {
                "week": week,
                "role": core.role_names[role],
                "obs": {k: obs[k] for k in OWN_HISTORY_FIELDS},
                "prompt_excerpt": "\n".join(prompt.splitlines()[-12:]),
                "raw": raw,
                "parsed_order": order,
            }
        )
        orders = {
            r: max(0, min(core.config.order_cap, int(core._states[r].last_demand_or_order)))
            for r in core.roles
        }
        orders[role] = order
        ship_before = int(core._states[role].last_shipment_received)
        _, _, _, info = core.step(orders)
        st = core._states[role]
        mem.append(
            WeekRecord(
                week=week,
                demand_or_incoming=int(obs["last_demand_or_order"]),
                ship_in=ship_before,
                ordered=int(order),
                alloc_recv=int(st.last_shipment_received),
                inventory=int(st.inventory),
                backlog=int(st.backlog),
                on_order=int(st.on_order),
                ship_pipeline=list(obs["ship_pipeline"]),
                order_pipeline=list(obs["order_pipeline"]),
                local_cost=float(info.local_costs[role]),
            )
        )
    assert len(examples) == 3
    # Persistence: week-1+ prompt must contain prior ordered=
    assert len(mem.history) == 3
    obs = observe_local(core, role)
    later = serialize_prompt(
        mem, obs, order_cap=core.config.order_cap, holding=0.5, backlog_cost=1.0
    )
    assert f"ordered={mem.history[0].ordered}" in later


def test_upstream_prompt_has_no_customer_demand_field():
    core = _y_core(1)
    factory = Role.FACTORY
    assert factory in core.roles
    mem = AgentMemory(role=factory, role_name=core.role_names[factory])
    obs = observe_local(core, factory)
    prompt = serialize_prompt(
        mem, obs, order_cap=128, holding=0.5, backlog_cost=1.0
    )
    assert "customer_demand=" not in prompt
    assert "true_demand=" not in prompt
    assert prompt_leak_report(prompt, factory, core) == []
