"""Serialize BeerGameCore state into spectator and fog-of-war player frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from beer_distribution_rl.env.core import BeerGameCore, RoleState, StepInfo
from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Y_ROLES, Role


def _role_map(
    values: Mapping[Role, int] | Mapping[Role, float],
    roles: tuple[Role, ...] = Y_ROLES,
    names: Mapping[Role, str] = Y_ROLE_NAMES,
) -> dict[str, int | float]:
    return {names[r]: values[r] for r in roles}


@dataclass(frozen=True)
class SpectatorFrame:
    """God-view snapshot for one week (or t=0 after reset)."""

    t: int
    inventories: dict[str, int]
    backlogs: dict[str, int]
    orders: dict[str, int]
    shipments: dict[str, int]
    shipments_received: dict[str, int]
    incoming: dict[str, int]
    customer_demand: int | None
    customer_demands: dict[str, int]
    allocations: dict[str, dict[str, int]]
    local_costs: dict[str, float]
    cumulative_local_costs: dict[str, float]
    system_cost: float
    cumulative_cost: float
    terminated: bool
    horizon: int

    def to_dict(self) -> dict:
        return asdict(self)


def initial_frame(
    states: Mapping[Role, RoleState],
    *,
    horizon: int,
    roles: tuple[Role, ...] = Y_ROLES,
    names: Mapping[Role, str] = Y_ROLE_NAMES,
) -> SpectatorFrame:
    """Snapshot right after reset (no orders yet)."""
    return SpectatorFrame(
        t=0,
        inventories={names[r]: int(states[r].inventory) for r in roles},
        backlogs={names[r]: int(states[r].backlog) for r in roles},
        orders={names[r]: 0 for r in roles},
        shipments={names[r]: 0 for r in roles},
        shipments_received={names[r]: 0 for r in roles},
        incoming={names[r]: 0 for r in roles},
        customer_demand=None,
        customer_demands={names[r]: 0 for r in roles if r in (Role.RETAILER, Role.RETAILER_B)},
        allocations={},
        local_costs={names[r]: 0.0 for r in roles},
        cumulative_local_costs={names[r]: 0.0 for r in roles},
        system_cost=0.0,
        cumulative_cost=0.0,
        terminated=False,
        horizon=horizon,
    )


def frame_from_step(
    states: Mapping[Role, RoleState],
    info: StepInfo,
    *,
    t: int,
    cumulative_cost: float,
    cumulative_local_costs: Mapping[Role, float],
    terminated: bool,
    horizon: int,
    roles: tuple[Role, ...] = Y_ROLES,
    names: Mapping[Role, str] = Y_ROLE_NAMES,
) -> SpectatorFrame:
    """Build a spectator frame from post-step states and StepInfo."""
    customer_demands = {
        names[r]: int(info.customer_demands[r])
        for r in roles
        if r in info.customer_demands
    }
    allocations = {
        names[src]: {names[dst]: int(qty) for dst, qty in dst_map.items()}
        for src, dst_map in info.allocations.items()
    }
    return SpectatorFrame(
        t=t,
        inventories={names[r]: int(states[r].inventory) for r in roles},
        backlogs={names[r]: int(states[r].backlog) for r in roles},
        orders=_role_map(info.orders_placed, roles, names),
        shipments=_role_map(info.shipments, roles, names),
        shipments_received=_role_map(info.shipments_received, roles, names),
        incoming=_role_map(info.incoming_orders, roles, names),
        customer_demand=info.customer_demand,
        customer_demands=customer_demands,
        allocations=allocations,
        local_costs={names[r]: float(info.local_costs[r]) for r in roles},
        cumulative_local_costs={
            names[r]: float(cumulative_local_costs[r]) for r in roles
        },
        system_cost=float(info.system_cost),
        cumulative_cost=float(cumulative_cost),
        terminated=terminated,
        horizon=horizon,
    )


def player_frame_from_core(
    core: BeerGameCore,
    *,
    human_role: Role,
    week_cost: float,
    cumulative_own_cost: float,
    last_order: int | None = None,
    terminated: bool = False,
) -> dict[str, Any]:
    """Fog-of-war view: only the human role's local observation + own costs.

    Topology silhouette roles are listed so the UI can render muted nodes, but
    no rival inventories, orders, or costs are included.
    """
    name = Y_ROLE_NAMES[human_role]
    obs = core.observe(human_role)
    # At t=0 after reset, last_demand_or_order may still be the init sentinel.
    demand_or_incoming = int(obs["last_demand_or_order"])
    is_retailer = human_role in (Role.RETAILER, Role.RETAILER_B)
    return {
        "t": int(core.t),
        "horizon": int(core.config.horizon),
        "human_role": name,
        "roles": [Y_ROLE_NAMES[r] for r in Y_ROLES],
        "inventory": int(obs["inventory"]),
        "backlog": int(obs["backlog"]),
        "on_order": int(obs["on_order"]),
        "ship_pipeline": list(obs["ship_pipeline"]),
        "order_pipeline": list(obs["order_pipeline"]),
        "last_demand_or_order": demand_or_incoming,
        "last_shipment_received": int(obs["last_shipment_received"]),
        "last_order_placed": int(obs["last_order_placed"]),
        "inventory_position": int(obs["inventory_position"]),
        "order_cap": int(obs["order_cap"]),
        "week_cost": float(week_cost),
        "cumulative_own_cost": float(cumulative_own_cost),
        "own_order": int(last_order) if last_order is not None else int(obs["last_order_placed"]),
        "is_retailer": is_retailer,
        "customer_demand": demand_or_incoming if is_retailer else None,
        "terminated": bool(terminated),
        "awaiting_order": not terminated,
    }


def end_reveal(
    *,
    human_role: Role,
    cumulative_own_cost: float,
    cumulative_system_cost: float,
    horizon: int,
    ai_mode: str,
    seed: int,
) -> dict[str, Any]:
    """End-of-episode summary without dumping every rival's private history."""
    return {
        "type": "reveal",
        "human_role": Y_ROLE_NAMES[human_role],
        "cumulative_own_cost": float(cumulative_own_cost),
        "cumulative_system_cost": float(cumulative_system_cost),
        "horizon": int(horizon),
        "ai_mode": ai_mode,
        "seed": int(seed),
    }
