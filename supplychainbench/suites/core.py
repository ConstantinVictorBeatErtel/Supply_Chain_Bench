"""Suite definitions and deterministic hidden-dynamics runtimes.

The default path delegates to the frozen live-Y research protocol.  New
variants use the optional runtime hook in ``BeerGameCore``; the hook is absent
from standard/Hub episodes, preserving their serialized state and trajectories.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable

import supplychainbench._compat  # noqa: F401  (installs the bundled environment path)

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.policies import adaptive_policy
from beer_distribution_game.scenario import ScenarioSpec, scenario_for
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.demand import poisson
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    research_spec,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
    research_system_prompt,
)

STANDARD_BUCKETS: dict[str, tuple[str, ...]] = {
    "in_distribution": (
        "1e594a5f4f13c914", "1ef97b9065013134", "60014380abcd0c62", "ba0eef18cf04ad40",
    ),
    "canonical_held_out_step": (
        "722ad6103d9490ad", "2b5eae5b998e4f1d", "6ad00ac691773fce", "0fa94b60e8eb54e2",
    ),
    "shifted_mean_doubled_variance": (
        "d9e921eb4abf1311", "98155973e5223d05", "a01e69edae32256e", "ba32f821074ddb20",
    ),
    "burst_and_collapse": (
        "b7c3c42c06605a25", "3385fc1772a83c0f", "6004dde3fcee7f67", "f0cd8b5418cd1e55",
    ),
}

SUITE_IDS = (
    "standard", "demand_shift", "unknown_lead_time", "capacity_shock",
    "supply_disruption", "held_out_dynamics",
)


def derive_seed(suite: str, index: int, *, phase: str = "evaluation") -> str:
    """Stable seed derivation; changing it requires a new benchmark version."""

    value = f"supplychainbench|beer-distribution|1.0.0|{suite}|{phase}|{index:05d}"
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def expected_seeds(suite: str) -> tuple[str, ...]:
    if suite not in SUITE_IDS:
        raise ValueError(f"unknown suite {suite!r}; choose from {SUITE_IDS}")
    if suite == "standard":
        return tuple(seed for bucket in STANDARD_BUCKETS.values() for seed in bucket)
    return tuple(derive_seed(suite, index) for index in range(16))


@dataclass(frozen=True)
class SuiteDefinition:
    suite_id: str
    version: str
    episode_count: int
    reference: str
    hidden_information: tuple[str, ...]
    event_metric: str | None = None


DEFINITIONS: dict[str, SuiteDefinition] = {
    "standard": SuiteDefinition(
        "standard", "1.0.0", 16, "frozen_feasible_hindsight", (
            "demand law and parameters", "factory capacity", "counterparty private state",
        )
    ),
    "demand_shift": SuiteDefinition(
        "demand_shift", "1.0.0", 16, "aware_adaptive_heuristic", (
            "demand rates", "change week", "post-change regime",
        ), "demand_shift"
    ),
    "unknown_lead_time": SuiteDefinition(
        "unknown_lead_time", "1.0.0", 16, "aware_adaptive_heuristic", (
            "order delay", "shipment delay", "demand parameters",
        )
    ),
    "capacity_shock": SuiteDefinition(
        "capacity_shock", "1.0.0", 16, "aware_adaptive_heuristic", (
            "capacity schedule", "shock week", "demand parameters",
        ), "capacity_shock"
    ),
    "supply_disruption": SuiteDefinition(
        "supply_disruption", "1.0.0", 16, "aware_adaptive_heuristic", (
            "disruption interval", "factory production state", "demand parameters",
        ), "supply_disruption"
    ),
    "held_out_dynamics": SuiteDefinition(
        "held_out_dynamics", "1.0.0", 16, "aware_adaptive_heuristic", (
            "demand parameters", "order delay", "shipment delay", "capacity",
        )
    ),
}


class _PoissonDemand:
    def __init__(self, seed: str, rate: float | None = None, before: float | None = None,
                 after: float | None = None, shift_week: int | None = None):
        self.seed = seed
        self.rate = rate
        self.before = before
        self.after = after
        self.shift_week = shift_week
        self.trace: list[dict[str, int | float]] = []

    def sample(self, week: int, customers: tuple[str, ...]) -> dict[str, int]:
        if self.shift_week is not None:
            rate = self.before if week < self.shift_week else self.after
        else:
            rate = self.rate
        assert rate is not None
        values = {
            role: poisson(self.seed, f"scb/demand/week-{week}/{role}", float(rate), index)
            for index, role in enumerate(customers)
        }
        self.trace.append({"week": week, "rate": float(rate), **values})
        return values

    def manifest(self) -> dict[str, Any]:
        return {"process_id": "supplychainbench_poisson_v1", "trace": list(self.trace)}


@dataclass(frozen=True)
class SuiteEpisode:
    suite: str
    seed: str
    index: int
    bucket: str | None
    spec: ScenarioSpec
    ground_truth: dict[str, Any]
    runtime: object | None
    event_week: int | None = None


class _Runtime:
    def __init__(self, demand: _PoissonDemand, capacity_fn: Callable[[int, int | None], int | None], truth: dict[str, Any]):
        self.demand = demand
        self._capacity_fn = capacity_fn
        self.ground_truth = truth

    def capacity_for_week(self, week: int, base: int | None) -> int | None:
        return self._capacity_fn(week, base)


class _AwareOrderUpToPolicy:
    """Reference policy with the suite's hidden schedule injected evaluator-side."""

    def __init__(self, job: SuiteEpisode):
        self.job = job
        self.delay = job.spec.order_delay + job.spec.shipment_delay
        truth = job.ground_truth
        self.base_rate = float(truth.get("rate", truth.get("rate_before", 6.0)))

    def _rate(self, week: int) -> float:
        truth = self.job.ground_truth
        if self.job.suite == "demand_shift" and week >= int(truth["change_week"]):
            return float(truth["rate_after"])
        return self.base_rate

    def act(self, observation: dict[str, Any]) -> int:
        week = int(observation["week"])
        target = math.ceil(self.delay * self._rate(week))
        position = int(observation["state"]["inventory_position"])
        quantity = max(0, target - position)
        if self.job.runtime is not None and hasattr(self.job.runtime, "capacity_for_week"):
            cap = self.job.runtime.capacity_for_week(week, self.job.spec.capacity)
            if cap is not None:
                quantity = min(quantity, max(0, int(cap)))
        return min(self.job.spec.order_cap, quantity)


def _unit_capacity(week: int, base: int | None) -> int | None:
    del week
    return base


def _new_episode(suite: str, seed: str, index: int) -> SuiteEpisode:
    prototype = scenario_for(5, "development", 0, "headline")
    values = prototype.to_dict()
    values.update({
        "environment_version": "supplychainbench-beer-v1",
        "scenario_id": f"scb-beer-v1:{suite}",
        "split": f"scb_{suite}",
        "seed_index": index,
        "master_seed_hex": seed,
        "capacity": 400,
        "demand_process": "supplychainbench_poisson_v1",
        "demand_parameters": {"hidden": True},
    })
    truth: dict[str, Any] = {"suite": suite}
    event_week: int | None = None
    demand = _PoissonDemand(seed, rate=6.0)
    capacity_fn: Callable[[int, int | None], int | None] = _unit_capacity

    if suite == "demand_shift":
        event_week = (15, 19, 23)[int(seed[:2], 16) % 3]
        high_first = int(seed[2:4], 16) % 2 == 0
        before, after = ((4.0, 12.0) if high_first else (12.0, 4.0))
        demand = _PoissonDemand(seed, before=before, after=after, shift_week=event_week)
        truth.update({"change_week": event_week, "rate_before": before, "rate_after": after})
    elif suite == "unknown_lead_time":
        values["order_delay"] = 1 + int(seed[0], 16) % 2
        values["shipment_delay"] = 1 + int(seed[1], 16) % 3
        values["initial_shipment_pipeline"] = 4
        values["initial_order_pipeline"] = 4
        truth.update({"order_delay": values["order_delay"], "shipment_delay": values["shipment_delay"]})
    elif suite == "capacity_shock":
        event_week = (15, 19, 23)[int(seed[:2], 16) % 3]
        shock_capacity = (24, 32)[int(seed[2:4], 16) % 2]
        capacity_fn = lambda week, base, ew=event_week, cap=shock_capacity: cap if week >= ew else base
        truth.update({"change_week": event_week, "capacity_before": 400, "capacity_after": shock_capacity})
    elif suite == "supply_disruption":
        event_week = 13 + int(seed[:2], 16) % 9
        duration = 3 + int(seed[2:4], 16) % 3
        capacity_fn = lambda week, base, ew=event_week, d=duration: 0 if ew <= week < ew + d else base
        truth.update({"disruption_start": event_week, "disruption_duration": duration})
    elif suite == "held_out_dynamics":
        rate = float(9 + int(seed[:2], 16) % 6)
        values["order_delay"] = 2 + int(seed[2], 16) % 2
        values["shipment_delay"] = 3 + int(seed[3], 16) % 2
        capacity = (24, 32, 40)[int(seed[4:6], 16) % 3]
        values["capacity"] = capacity
        demand = _PoissonDemand(seed, rate=rate)
        truth.update({"rate": rate, "order_delay": values["order_delay"],
                      "shipment_delay": values["shipment_delay"], "capacity": capacity})

    spec = ScenarioSpec(**{**values, "roles": tuple(values["roles"])})
    runtime = _Runtime(demand, capacity_fn, truth)
    return SuiteEpisode(suite, seed, index, None, spec, truth, runtime, event_week)


def episode_jobs(suite: str) -> tuple[SuiteEpisode, ...]:
    if suite == "standard":
        jobs: list[SuiteEpisode] = []
        for bucket, seeds in STANDARD_BUCKETS.items():
            for index, seed in enumerate(seeds):
                spec = research_spec(seed, bucket=bucket, index=index)
                jobs.append(SuiteEpisode(suite, seed, index, bucket, spec, {}, None, None))
        return tuple(jobs)
    return tuple(_new_episode(suite, seed, index) for index, seed in enumerate(expected_seeds(suite)))


def continual_episode_jobs(suite: str, phase: str, count: int, *, world_seed: str | None = None) -> tuple[SuiteEpisode, ...]:
    """Create fresh demand streams under one fixed hidden world."""

    if suite not in {"demand_shift", "unknown_lead_time", "capacity_shock", "supply_disruption", "held_out_dynamics"}:
        raise ValueError("continual experiments require a hidden-dynamics suite")
    world = _new_episode(suite, world_seed or derive_seed(suite, 0, phase="world"), 0)
    jobs: list[SuiteEpisode] = []
    for index in range(count):
        seed = derive_seed(suite, index, phase=phase)
        truth = deepcopy(world.ground_truth)
        if suite == "demand_shift":
            demand = _PoissonDemand(seed, before=float(truth["rate_before"]), after=float(truth["rate_after"]), shift_week=int(truth["change_week"]))
        elif suite == "held_out_dynamics":
            demand = _PoissonDemand(seed, rate=float(truth["rate"]))
        else:
            demand = _PoissonDemand(seed, rate=6.0)
        runtime = _Runtime(demand, world.runtime._capacity_fn, truth)  # type: ignore[union-attr]
        values = world.spec.to_dict()
        values["master_seed_hex"] = seed
        values["split"] = f"continual_{phase}"
        values["seed_index"] = index
        spec = ScenarioSpec(**{**values, "roles": tuple(values["roles"])})
        jobs.append(SuiteEpisode(suite, seed, index, None, spec, truth, runtime, world.event_week))
    return tuple(jobs)


def build_episode(job: SuiteEpisode, *, include_reference: bool = False) -> BeerEpisode:
    return BeerEpisode(job.spec, "wholesaler", include_reference=include_reference, runtime=job.runtime)


def project_observation(job: SuiteEpisode, observation: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(observation)
    if job.suite != "standard":
        value.pop("episode_id", None)
        value.pop("scenario_id", None)
    constraints = value.get("constraints")
    if isinstance(constraints, dict):
        constraints.pop("factory_capacity", None)
    return value


def prompt_for(job: SuiteEpisode, observation: dict[str, Any]) -> tuple[str, str]:
    projected = project_observation(job, observation)
    if job.suite == "standard":
        return research_system_prompt(job.spec, "wholesaler"), research_observation_user_message(projected)
    system = (
        "You control the wholesaler in a Y beer-distribution supply chain for 36 decision weeks. "
        "Minimize your local holding and backlog cost. Demand, supply capacity, and in some tracks "
        "the order and shipping delays may be hidden or change during the episode. Infer them only "
        "from your role-local observations and history. Place exactly one integer order from 0 through "
        "128 as JSON: {\"quantity\": INTEGER}. Do not include prose."
    )
    return system, "Current observation:\n" + _canonical(projected) + (
        "\n\nRespond with exactly one JSON object of the form "
        '{"quantity": <integer from 0 through 128>}. Do not include any other text.'
    )


def _canonical(value: object) -> str:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def reference_cost(job: SuiteEpisode) -> dict[str, Any]:
    """Run the deterministic environment-aware reference policy."""

    episode = build_episode(job)
    observation = episode.start()
    policy = adaptive_policy(job.spec, "wholesaler") if job.suite == "standard" else _AwareOrderUpToPolicy(job)
    costs: list[float] = []
    while not episode.done:
        quantity = int(policy.act(observation))
        result = episode.place_order(quantity)
        costs.append(float(episode.operational_transitions[-1]["local_costs"]["wholesaler"]))
        if not result["done"]:
            observation = result["next_observation"]
    grade = episode.outcome["grade"]
    return {
        "local_total_cost": float(grade["primary"]["local_total_cost"]),
        "weekly_local_costs": costs,
        "event_week": job.event_week,
        "ground_truth": deepcopy(job.ground_truth),
    }
