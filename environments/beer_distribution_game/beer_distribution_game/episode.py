"""Single-controlled-role episode orchestration over the deterministic core."""

from __future__ import annotations

from typing import Any

from .core import BeerGameCore
from .grading import grade_episode
from .policies import Policy, adaptive_policy, counterparty_policies
from .scenario import Role, ScenarioSpec


class BeerEpisode:
    def __init__(
        self,
        spec: ScenarioSpec,
        controlled_role: Role,
        *,
        include_reference: bool = True,
    ):
        if controlled_role not in spec.roles:
            raise ValueError(f"role {controlled_role!r} is not in {spec.topology}")
        self.spec = spec
        self.controlled_role = controlled_role
        self.episode_id = spec.episode_id(controlled_role)
        self.include_reference = include_reference
        self.core = BeerGameCore(spec)
        self.counterparties = counterparty_policies(spec, controlled_role)
        self.histories: dict[Role, list[dict[str, Any]]] = {
            role: [] for role in spec.roles
        }
        self.cumulative_costs: dict[Role, float] = {
            role: 0.0 for role in spec.roles
        }
        self.operational_transitions: list[dict[str, Any]] = []
        self.settlement_transitions: list[dict[str, Any]] = []
        self.protocol_clean = True
        self.started = False
        self.done = False
        self.outcome: dict[str, Any] | None = None

    def mark_protocol_error(self) -> None:
        self.protocol_clean = False

    def _observations(self) -> dict[Role, dict[str, Any]]:
        return {
            role: self.core.observation(
                role,
                episode_id=self.episode_id,
                recent_history=self.histories[role],
                cumulative_local_cost=self.cumulative_costs[role],
            )
            for role in self.spec.roles
        }

    def start(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("episode already started")
        self.started = True
        self.core.prepare_week(operational=True)
        return self._observations()[self.controlled_role]

    def place_order(self, quantity: int) -> dict[str, Any]:
        if not self.started or self.done:
            raise RuntimeError("order is out of turn")
        if type(quantity) is not int or not 0 <= quantity <= self.spec.order_cap:
            raise ValueError(f"quantity must be an integer in [0, {self.spec.order_cap}]")

        observations = self._observations()
        orders: dict[Role, int] = {self.controlled_role: quantity}
        for role, policy in self.counterparties.items():
            orders[role] = policy.act(observations[role])
        transition = self.core.commit_orders(orders)
        self.operational_transitions.append(transition)

        for role in self.spec.roles:
            local_cost = float(transition["local_costs"][role])
            state = transition["states_after_fulfillment"][role]
            self.histories[role].append(
                {
                    "week": transition["week"],
                    "incoming_demand_or_order": sum(
                        transition["incoming_by_claimant"][role].values()
                    ),
                    "shipment_received": transition["received"][role],
                    "order_placed": transition["orders"][role],
                    "ending_inventory": state["inventory"],
                    "ending_backlog": state["backlog"],
                    "local_cost": local_cost,
                }
            )
            self.cumulative_costs[role] += local_cost

        if len(self.operational_transitions) >= self.spec.horizon:
            self._finish()
            assert self.outcome is not None
            return {
                "status": "accepted",
                "completed_week": transition["week"],
                "order_placed": quantity,
                "done": True,
                "termination_reason": "horizon_completed",
                "summary": {
                    "episode_reward": self.outcome["grade"]["episode_reward"],
                    "local_total_cost": self.outcome["grade"]["primary"][
                        "local_total_cost"
                    ],
                    "system_total_cost": self.outcome["grade"]["costs"][
                        "system_total_cost"
                    ],
                },
            }

        self.core.prepare_week(operational=True)
        return {
            "status": "accepted",
            "completed_week": transition["week"],
            "order_placed": quantity,
            "done": False,
            "next_observation": self._observations()[self.controlled_role],
        }

    def _finish(self) -> None:
        for _ in range(self.spec.settlement_weeks):
            self.core.prepare_week(operational=False)
            self.settlement_transitions.append(
                self.core.commit_orders({role: 0 for role in self.spec.roles})
            )
        terminal_positions = {
            role: self.core.inventory_position(role) for role in self.spec.roles
        }
        base_reference = None
        if self.include_reference:
            base_reference = _run_reference(self.spec, self.controlled_role)
        grade = grade_episode(
            spec=self.spec,
            controlled_role=self.controlled_role,
            operational=self.operational_transitions,
            settlement=self.settlement_transitions,
            terminal_inventory_positions=terminal_positions,
            protocol_clean=self.protocol_clean,
            base_reference=base_reference,
        )
        self.done = True
        self.outcome = {
            "episode_id": self.episode_id,
            "scenario": self.spec.to_dict(),
            "controlled_role": self.controlled_role,
            "counterparties": {
                role: {
                    "policy_id": policy.policy_id,
                    "policy_version": policy.policy_version,
                }
                for role, policy in self.counterparties.items()
            },
            "operational_transitions": self.operational_transitions,
            "settlement_transitions": self.settlement_transitions,
            "terminal_inventory_positions": terminal_positions,
            "final_state": self.core.snapshot(),
            "grade": grade,
        }
        research_manifest = self.core.research_exogenous_manifest()
        if research_manifest is not None:
            self.outcome["research_exogenous"] = research_manifest

    def protocol_failure_outcome(
        self, *, error_count: int, category: str
    ) -> dict[str, Any]:
        self.protocol_clean = False
        self.done = True
        self.outcome = {
            "episode_id": self.episode_id,
            "scenario": self.spec.to_dict(),
            "controlled_role": self.controlled_role,
            "operational_transitions": self.operational_transitions,
            "settlement_transitions": [],
            "final_state": self.core.snapshot(),
            "grade": {
                "grader_version": "1.0.0",
                "status": "protocol_error",
                "episode_reward": 0.0,
                "protocol_clean": False,
                "termination_reason": "protocol_error",
                "protocol_error_count": error_count,
                "last_protocol_error": category,
                "completed_operational_weeks": len(self.operational_transitions),
            },
        }
        research_manifest = self.core.research_exogenous_manifest()
        if research_manifest is not None:
            self.outcome["research_exogenous"] = research_manifest
        return self.outcome


def _run_reference(spec: ScenarioSpec, controlled_role: Role) -> dict[str, float]:
    episode = BeerEpisode(spec, controlled_role, include_reference=False)
    policy: Policy = adaptive_policy(spec, controlled_role)
    observation = episode.start()
    while not episode.done:
        result = episode.place_order(policy.act(observation))
        if not result["done"]:
            observation = result["next_observation"]
    assert episode.outcome is not None
    grade = episode.outcome["grade"]
    return {
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "system_total_cost": float(grade["costs"]["system_total_cost"]),
    }
