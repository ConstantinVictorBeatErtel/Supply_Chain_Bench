"""Rolling-window memory: persistence, E1, W knob, token budget ($0, no LLM)."""

from __future__ import annotations

import pytest

from beer_distribution_rl.agents.baselines import base_stock_order
from beer_distribution_rl.agents.llm import (
    DEFAULT_ROLLING_WINDOW,
    AgentMemory,
    WeekRecord,
    estimate_prompt_tokens,
    observe_local,
    prompt_leak_report,
    serialize_prompt,
)
from beer_distribution_rl.env.core import BeerGameCore, Role, y_topology_env_config


def _y_core(seed: int = 0, horizon: int = 52) -> BeerGameCore:
    core = BeerGameCore(y_topology_env_config(horizon=horizon, seed=seed))
    core.reset(seed)
    return core


def _append_week(
    mem: AgentMemory,
    core: BeerGameCore,
    role: Role,
    obs: dict,
    order: int,
    info,
    ship_before: int,
) -> None:
    st = core._states[role]
    mem.append(
        WeekRecord(
            week=core.t - 1,
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


def test_default_window_is_eight():
    assert DEFAULT_ROLLING_WINDOW == 8
    mem = AgentMemory(role=Role.RETAILER, role_name="retailer")
    assert mem.window == 8


def test_window_knob_is_one_line_change():
    """Ablation-ready: constructing with window=W is the sensitivity knob."""
    mem = AgentMemory(role=Role.RETAILER, role_name="retailer", window=4)
    assert mem.window == 4
    for i in range(6):
        mem.append(
            WeekRecord(
                week=i,
                demand_or_incoming=i,
                ship_in=0,
                ordered=i + 10,
                alloc_recv=0,
                inventory=0,
                backlog=0,
                on_order=0,
                ship_pipeline=[0, 0],
                order_pipeline=[0],
                local_cost=0.0,
            )
        )
    assert [r.week for r in mem.windowed_history()] == [2, 3, 4, 5]
    assert [r.week for r in mem.windowed_history(window=2)] == [4, 5]


def test_rolling_window_drops_old_weeks_from_prompt():
    core = _y_core(0)
    role = Role.RETAILER if Role.RETAILER in core.roles else core.roles[0]
    mem = AgentMemory(role=role, role_name=core.role_names[role], window=8)
    levels = {r: 30 for r in core.roles}
    done = False
    while not done:
        obs = observe_local(core, role)
        ship_before = int(core._states[role].last_shipment_received)
        orders = {
            r: base_stock_order(core._states[r], levels[r], order_cap=core.config.order_cap)
            for r in core.roles
        }
        _, _, done, info = core.step(orders)
        _append_week(mem, core, role, obs, orders[role], info, ship_before)

    assert len(mem.history) == 52
    assert len(mem.windowed_history()) == 8
    assert [r.week for r in mem.windowed_history()] == list(range(44, 52))

    obs = observe_local(core, role)
    prompt = serialize_prompt(
        mem, obs, order_cap=core.config.order_cap, holding=0.5, backlog_cost=1.0
    )
    assert "W=8" in prompt
    assert "week=51:" in prompt
    assert "week=44:" in prompt
    assert "week=0:" not in prompt
    assert "week=43:" not in prompt
    assert f"ordered={mem.history[51].ordered}" in prompt


def test_persistence_across_full_t52_episode_w8():
    """Week t+1's prompt contains week t's outcome for every t, W=8, T=52."""
    core = _y_core(0, horizon=52)
    role = Role.RETAILER if Role.RETAILER in core.roles else core.roles[0]
    mem = AgentMemory(role=role, role_name=core.role_names[role], window=8)
    levels = {r: 30 for r in core.roles}
    misses: list[int] = []
    leak_hits = 0
    token_counts: list[int] = []

    for week in range(52):
        obs = observe_local(core, role)
        prompt = serialize_prompt(
            mem, obs, order_cap=core.config.order_cap, holding=0.5, backlog_cost=1.0
        )
        token_counts.append(estimate_prompt_tokens(prompt))
        if prompt_leak_report(prompt, role, core):
            leak_hits += 1
        if week >= 1:
            prev = mem.history[week - 1]
            needle = f"week={prev.week}: demand_or_incoming={prev.demand_or_incoming}, "
            if needle not in prompt or f"ordered={prev.ordered}" not in prompt:
                misses.append(week)
            # Window bound: weeks older than W must be absent once history > W
            if week > 8:
                oldest_kept = week - 8
                assert f"week={oldest_kept - 1}:" not in prompt

        ship_before = int(core._states[role].last_shipment_received)
        orders = {
            r: base_stock_order(core._states[r], levels[r], order_cap=core.config.order_cap)
            for r in core.roles
        }
        _, _, done, info = core.step(orders)
        _append_week(mem, core, role, obs, orders[role], info, ship_before)
        if done:
            break

    assert misses == [], f"persistence misses at weeks {misses}"
    assert leak_hits == 0
    # Steady-state (full W=8) prompts must fit 32k with margin
    steady = token_counts[8:]
    assert steady, "expected prompts after window fills"
    assert max(steady) < 4000  # ≪ 32k
    assert max(token_counts) < 32000


def test_e1_no_leak_on_upstream_with_full_window():
    core = _y_core(1)
    factory = Role.FACTORY
    mem = AgentMemory(role=factory, role_name=core.role_names[factory], window=8)
    levels = {r: 30 for r in core.roles}
    for _ in range(10):
        obs = observe_local(core, factory)
        prompt = serialize_prompt(
            mem, obs, order_cap=128, holding=0.5, backlog_cost=1.0
        )
        assert "customer_demand=" not in prompt
        assert "true_demand=" not in prompt
        assert prompt_leak_report(prompt, factory, core) == []
        ship_before = int(core._states[factory].last_shipment_received)
        orders = {
            r: base_stock_order(core._states[r], levels[r], order_cap=core.config.order_cap)
            for r in core.roles
        }
        _, _, done, info = core.step(orders)
        _append_week(mem, core, factory, obs, orders[factory], info, ship_before)
        if done:
            break


@pytest.mark.parametrize("w", [1, 4, 8, 16])
def test_window_override_on_serialize(w: int):
    mem = AgentMemory(role=Role.RETAILER, role_name="retailer_a", window=8)
    for i in range(20):
        mem.append(
            WeekRecord(
                week=i,
                demand_or_incoming=1,
                ship_in=0,
                ordered=i,
                alloc_recv=0,
                inventory=1,
                backlog=0,
                on_order=0,
                ship_pipeline=[0, 0],
                order_pipeline=[0],
                local_cost=0.5,
            )
        )
    core = _y_core(0)
    obs = observe_local(core, Role.RETAILER)
    prompt = serialize_prompt(
        mem, obs, order_cap=128, holding=0.5, backlog_cost=1.0, window=w
    )
    assert f"W={w}" in prompt
    assert len(mem.windowed_history(w)) == w
    assert f"week={20 - w}:" in prompt
    if 20 - w - 1 >= 0:
        assert f"week={20 - w - 1}:" not in prompt
