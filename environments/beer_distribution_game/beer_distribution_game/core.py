"""Framework-neutral, deterministic two-phase Beer Game simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any

from .scenario import Role, ScenarioSpec, canonical_json, derive_seed

Claimant = str


def _topology(spec: ScenarioSpec) -> tuple[
    dict[Role, Role | None], dict[Role, tuple[Role, ...]], tuple[Role, ...]
]:
    if spec.topology == "serial":
        upstream: dict[Role, Role | None] = {
            "retailer": "wholesaler",
            "wholesaler": "distributor",
            "distributor": "factory",
            "factory": None,
        }
        downstream: dict[Role, tuple[Role, ...]] = {
            "retailer": (),
            "wholesaler": ("retailer",),
            "distributor": ("wholesaler",),
            "factory": ("distributor",),
        }
        return upstream, downstream, ("retailer",)
    upstream = {
        "retailer_a": "wholesaler",
        "retailer_b": "wholesaler",
        "wholesaler": "distributor",
        "distributor": "factory",
        "factory": None,
    }
    downstream = {
        "retailer_a": (),
        "retailer_b": (),
        "wholesaler": ("retailer_a", "retailer_b"),
        "distributor": ("wholesaler",),
        "factory": ("distributor",),
    }
    return upstream, downstream, ("retailer_a", "retailer_b")


@dataclass
class RoleState:
    inventory: int
    backlog: int
    shipment_pipeline: list[int]
    order_pipelines: dict[Claimant, list[int]]
    claimant_backlog: dict[Claimant, int]
    last_order_placed: int = 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "inventory": self.inventory,
            "backlog": self.backlog,
            "shipment_pipeline": list(self.shipment_pipeline),
            "order_pipelines": {
                key: list(value) for key, value in sorted(self.order_pipelines.items())
            },
            "claimant_backlog": dict(sorted(self.claimant_backlog.items())),
            "last_order_placed": self.last_order_placed,
        }


@dataclass
class PreparedWeek:
    week: int
    operational: bool
    received: dict[Role, int]
    incoming_by_claimant: dict[Role, dict[Claimant, int]]
    backlog_before_by_claimant: dict[Role, dict[Claimant, int]]
    allocations: dict[Role, dict[Claimant, int]]
    local_costs: dict[Role, float]


class DemandGenerator:
    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.research_demand = None
        if spec.demand_process in {
            "episode_randomized_y_poisson_v1",
            "canonical_y_step_4_to_8_v1",
            "overdispersed_y_nb_mean10_var20_v1",
            "burst_collapse_y_poisson_v1",
        }:
            from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.demand import (
                EpisodeRandomizedPoissonDemand,
                EvaluationDemand,
            )

            if spec.demand_process == "episode_randomized_y_poisson_v1":
                self.research_demand = EpisodeRandomizedPoissonDemand(
                    spec.master_seed_hex,
                    float(spec.demand_parameters.get("lambda_low", 2.0)),
                    float(spec.demand_parameters.get("lambda_high", 8.0)),
                    str(spec.demand_parameters.get("demand_seed") or spec.master_seed_hex),
                )
            else:
                self.research_demand = EvaluationDemand(spec.demand_process, spec.master_seed_hex)
        self.rng = random.Random(derive_seed(spec.master_seed_hex, "demand"))
        params = spec.demand_parameters
        self.x = float(params.get("x0") or params.get("mu") or params.get("mu_before") or 0)
        self.common = float(params.get("common0") or 0.0)

    def sample(self, week: int, customers: tuple[Role, ...]) -> dict[Role, int]:
        process = self.spec.demand_process
        p = self.spec.demand_parameters
        if self.research_demand is not None:
            return self.research_demand.sample(week, customers)
        if process == "constant_v1":
            return {customers[0]: int(p["value"])}
        if process in ("ar1_v1", "shifted_ar1_v1"):
            mu = float(p.get("mu") or p.get("mu_before") or 0.0)
            shift_week = p.get("shift_week")
            if shift_week is not None and week >= int(shift_week):
                mu = float(p["mu_after"])
            phi = float(p["phi"])
            self.x = mu + phi * (self.x - mu) + self.rng.gauss(0.0, float(p["sigma"]))
            return {customers[0]: max(0, int(round(self.x)))}
        if process == "correlated_y_ar1_v1":
            self.common = float(p["phi"]) * self.common + self.rng.gauss(
                0.0, float(p["sigma_common"])
            )
            return {
                role: max(
                    0,
                    int(
                        round(
                            float(p["mu"])
                            + self.common
                            + self.rng.gauss(0.0, float(p["sigma_idiosyncratic"]))
                        )
                    ),
                )
                for role in customers
            }
        raise ValueError(f"unsupported demand process {process!r}")

    def research_manifest(self) -> dict[str, object] | None:
        return None if self.research_demand is None else self.research_demand.manifest()


def _allocate(
    requested: dict[Claimant, int], available: int, rationing: str
) -> dict[Claimant, int]:
    total = sum(requested.values())
    if available <= 0 or total <= 0:
        return {key: 0 for key in requested}
    if available >= total:
        return dict(requested)
    if rationing == "uniform":
        out = {key: 0 for key in requested}
        remaining = available
        while remaining:
            progressed = False
            for key in sorted(requested):
                if remaining == 0:
                    break
                if out[key] < requested[key]:
                    out[key] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return out

    raw = {key: available * requested[key] / total for key in requested}
    out = {key: min(requested[key], int(raw[key])) for key in requested}
    remaining = available - sum(out.values())
    order = sorted(
        requested,
        key=lambda key: (raw[key] - int(raw[key]), key),
        reverse=True,
    )
    while remaining:
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if out[key] < requested[key]:
                out[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return out


class BeerGameCore:
    """Two-phase simulator: prepare observations, then atomically commit all orders."""

    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.upstream, self.downstream, self.customers = _topology(spec)
        self.demand = DemandGenerator(spec)
        self.week = 0
        self.prepared: PreparedWeek | None = None
        self.states: dict[Role, RoleState] = {}
        for role in spec.roles:
            claimants: tuple[Claimant, ...]
            if role in self.customers:
                claimants = ()
            else:
                claimants = self.downstream[role]
            self.states[role] = RoleState(
                inventory=spec.initial_inventory,
                backlog=0,
                shipment_pipeline=[
                    spec.initial_shipment_pipeline
                    for _ in range(spec.shipment_delay)
                ],
                order_pipelines={
                    claimant: [
                        spec.initial_order_pipeline for _ in range(spec.order_delay)
                    ]
                    for claimant in claimants
                },
                claimant_backlog={claimant: 0 for claimant in claimants},
            )

    def _virtual_customer(self, role: Role) -> Claimant:
        return f"customer:{role}"

    def on_order(self, role: Role) -> int:
        state = self.states[role]
        total = sum(state.shipment_pipeline)
        upstream = self.upstream[role]
        if upstream is None:
            return total
        upstream_state = self.states[upstream]
        return (
            total
            + sum(upstream_state.order_pipelines.get(role, ()))
            + upstream_state.claimant_backlog.get(role, 0)
        )

    def inventory_position(self, role: Role) -> int:
        state = self.states[role]
        return state.inventory - state.backlog + self.on_order(role)

    def prepare_week(self, *, operational: bool = True) -> PreparedWeek:
        if self.prepared is not None:
            raise RuntimeError("current week is already prepared")
        week = self.week + 1
        received: dict[Role, int] = {}
        for role in self.spec.roles:
            pipeline = self.states[role].shipment_pipeline
            received[role] = pipeline.pop(0) if pipeline else 0

        customer_demand = (
            self.demand.sample(week, self.customers)
            if operational
            else {role: 0 for role in self.customers}
        )
        incoming: dict[Role, dict[Claimant, int]] = {}
        backlog_before: dict[Role, dict[Claimant, int]] = {}
        for role in self.spec.roles:
            state = self.states[role]
            if role in self.customers:
                claimant = self._virtual_customer(role)
                incoming[role] = {claimant: customer_demand[role]}
                backlog_before[role] = {claimant: state.backlog}
            else:
                incoming[role] = {
                    claimant: (pipeline.pop(0) if pipeline else 0)
                    for claimant, pipeline in sorted(state.order_pipelines.items())
                }
                backlog_before[role] = dict(state.claimant_backlog)

        allocations: dict[Role, dict[Claimant, int]] = {}
        for role in self.spec.roles:
            state = self.states[role]
            requested = {
                claimant: backlog_before[role].get(claimant, 0) + quantity
                for claimant, quantity in incoming[role].items()
            }
            available = state.inventory + received[role]
            allocation = _allocate(requested, available, self.spec.rationing)
            allocations[role] = allocation
            state.inventory = available - sum(allocation.values())
            remaining = {
                claimant: requested[claimant] - allocation.get(claimant, 0)
                for claimant in requested
            }
            state.backlog = sum(remaining.values())
            state.claimant_backlog = (
                {} if role in self.customers else remaining
            )
            if state.inventory < 0 or state.backlog < 0:
                raise RuntimeError("negative physical state")
            if state.inventory and state.backlog:
                raise RuntimeError("inventory and backlog cannot both be positive")

        for supplier in self.spec.roles:
            for claimant, quantity in allocations[supplier].items():
                if claimant in self.states:
                    self.states[claimant].shipment_pipeline.append(quantity)

        local_costs = {
            role: (
                self.spec.holding_cost * self.states[role].inventory
                + self.spec.backlog_cost * self.states[role].backlog
            )
            for role in self.spec.roles
        }
        self.prepared = PreparedWeek(
            week=week,
            operational=operational,
            received=received,
            incoming_by_claimant=incoming,
            backlog_before_by_claimant=backlog_before,
            allocations=allocations,
            local_costs=local_costs,
        )
        return self.prepared

    def observation(
        self,
        role: Role,
        *,
        episode_id: str,
        recent_history: list[dict[str, Any]],
        cumulative_local_cost: float,
    ) -> dict[str, Any]:
        prepared = self.prepared
        if prepared is None:
            raise RuntimeError("prepare_week() must be called before observation")
        state = self.states[role]
        incoming = sum(prepared.incoming_by_claimant[role].values())
        filled = sum(prepared.allocations[role].values())
        state_view: dict[str, Any] = {
            "inventory_on_hand": state.inventory,
            "backlog": state.backlog,
            "inventory_position": self.inventory_position(role),
            "on_order": self.on_order(role),
            "shipment_received": prepared.received[role],
            "incoming_demand_or_order": incoming,
            "units_filled": filled,
            "last_order_placed": state.last_order_placed,
        }
        if self.spec.observation_mode == "shipment_notices":
            state_view["inbound_shipment_pipeline"] = list(state.shipment_pipeline)
        return {
            "episode_id": episode_id,
            "scenario_id": self.spec.scenario_id,
            "week": prepared.week,
            "horizon": self.spec.horizon,
            "weeks_remaining": self.spec.horizon - prepared.week + 1,
            "role": role,
            "topology": self.spec.topology,
            "observation_mode": self.spec.observation_mode,
            "state": state_view,
            "costs": {
                "holding_per_unit": self.spec.holding_cost,
                "backlog_per_unit": self.spec.backlog_cost,
                "current_inventory_backlog_cost": prepared.local_costs[role],
                "cumulative_local_cost_through_previous_week": cumulative_local_cost,
            },
            "constraints": {
                "minimum_order": 0,
                "maximum_order": self.spec.order_cap,
                "factory_capacity": self.spec.capacity,
            },
            "recent_history": recent_history[-self.spec.history_window :],
        }

    def commit_orders(self, orders: dict[Role, int]) -> dict[str, Any]:
        prepared = self.prepared
        if prepared is None:
            raise RuntimeError("prepare_week() must be called before commit_orders()")
        if set(orders) != set(self.spec.roles):
            missing = sorted(set(self.spec.roles) - set(orders))
            extra = sorted(set(orders) - set(self.spec.roles))
            raise ValueError(f"orders must cover every role; missing={missing}, extra={extra}")
        for role, quantity in orders.items():
            if type(quantity) is not int or not 0 <= quantity <= self.spec.order_cap:
                raise ValueError(f"invalid order for {role}: {quantity!r}")

        states_before_orders = {
            role: self.states[role].snapshot() for role in self.spec.roles
        }
        capacity_bound = False
        production = 0
        for role in self.spec.roles:
            quantity = orders[role]
            state = self.states[role]
            state.last_order_placed = quantity
            upstream = self.upstream[role]
            if upstream is None:
                production = quantity
                if self.spec.capacity is not None and production > self.spec.capacity:
                    production = self.spec.capacity
                    capacity_bound = True
                state.shipment_pipeline.append(production)
            else:
                self.states[upstream].order_pipelines[role].append(quantity)

        transition = {
            "week": prepared.week,
            "operational": prepared.operational,
            "received": dict(prepared.received),
            "incoming_by_claimant": prepared.incoming_by_claimant,
            "backlog_before_by_claimant": prepared.backlog_before_by_claimant,
            "allocations": prepared.allocations,
            "states_after_fulfillment": states_before_orders,
            "orders": dict(orders),
            "factory_production": production,
            "capacity_bound": capacity_bound,
            "local_costs": dict(prepared.local_costs),
            "system_cost": sum(prepared.local_costs.values()),
        }
        self.week += 1
        self.prepared = None
        return transition

    def snapshot(self) -> dict[str, Any]:
        return {
            "week": self.week,
            "scenario": self.spec.to_dict(),
            "states": {role: self.states[role].snapshot() for role in self.spec.roles},
        }

    def canonical_snapshot(self) -> str:
        return canonical_json(self.snapshot())

    def research_exogenous_manifest(self) -> dict[str, object] | None:
        """Expose only the additive research trace; frozen scenarios are unchanged."""

        return self.demand.research_manifest()
