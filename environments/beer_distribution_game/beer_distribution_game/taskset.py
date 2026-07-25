"""Native Verifiers v1 taskset, state, toolset, rewards, and metrics."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field
import verifiers.v1 as vf

from .episode import BeerEpisode
from .policies import adaptive_policy
from .prompts import system_prompt as _system_prompt
from .scenario import (
    SPLIT_SIZES,
    Role,
    ScenarioSpec,
    Split,
    Variant,
    canonical_json,
    roles_for,
    scenario_for,
    scenario_from_dict,
)


class BeerRolloutState(vf.State):
    protocol_error_count: int = 0
    invalid_attempts_this_week: int = 0
    completed_weeks: int = 0
    done: bool = False
    outcome: dict[str, Any] | None = None


class BeerToolset(vf.Toolset[vf.ToolsetConfig, BeerRolloutState]):
    TOOL_PREFIX = "beer_game"

    async def setup_task(self, task) -> None:
        spec = scenario_from_dict(task.scenario)
        self.episode = BeerEpisode(spec, task.controlled_role)
        initial = self.episode.start()
        if canonical_json(initial) != canonical_json(task.initial_observation):
            raise RuntimeError("task prompt and tool episode disagree on initial observation")

    def _json(self, value: dict[str, Any]) -> str:
        return canonical_json(value)

    @vf.tool
    async def place_order(
        self,
        quantity: Annotated[int, Field(strict=True, ge=0, le=128)],
    ) -> str:
        """Place this week's replenishment order.

        Args:
            quantity: Absolute integer order quantity from 0 through 128.
        """
        if self.state.done:
            return self._json(
                {"status": "error", "category": "out_of_turn", "done": True}
            )
        if type(quantity) is not int or not 0 <= quantity <= 128:
            return await self.record_protocol_error("invalid_quantity")
        result = self.episode.place_order(quantity)
        self.state.completed_weeks = len(self.episode.operational_transitions)
        self.state.invalid_attempts_this_week = 0
        if self.episode.done:
            self.state.done = True
            self.state.outcome = self.episode.outcome
        return self._json(result)

    @vf.tool
    async def record_protocol_error(self, category: str) -> str:
        """Internal harness hook; never exposed to the evaluated model."""
        if self.state.done:
            return self._json(
                {"status": "error", "category": "out_of_turn", "done": True}
            )
        self.episode.mark_protocol_error()
        self.state.protocol_error_count += 1
        self.state.invalid_attempts_this_week += 1
        terminate = (
            self.state.invalid_attempts_this_week >= 2
            or self.state.protocol_error_count >= 3
        )
        week = self.episode.core.prepared.week if self.episode.core.prepared else None
        if terminate:
            self.state.done = True
            self.state.outcome = self.episode.protocol_failure_outcome(
                error_count=self.state.protocol_error_count,
                category=category,
            )
        return self._json(
            {
                "status": "error",
                "category": category,
                "week": week,
                "protocol_error_count": self.state.protocol_error_count,
                "retry_allowed": not terminate,
                "done": terminate,
                "message": (
                    "Call place_order exactly once with {\"quantity\": <integer 0..128>}."
                    if not terminate
                    else "Episode terminated because the protocol-error limit was reached."
                ),
            }
        )


class BeerTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig()


class BeerTaskData(vf.TaskData):
    scenario: dict[str, Any]
    controlled_role: Role
    episode_id: str
    initial_observation: dict[str, Any]


class BeerTask(vf.Task[BeerTaskData, BeerRolloutState, BeerTaskConfig]):
    tools = (BeerToolset,)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        del runtime
        outcome = trace.state.outcome
        if outcome is None:
            outcome = {
                "episode_id": self.data.episode_id,
                "scenario": self.data.scenario,
                "controlled_role": self.data.controlled_role,
                "grade": {
                    "grader_version": "1.0.0",
                    "status": "protocol_error",
                    "episode_reward": 0.0,
                    "protocol_clean": False,
                    "termination_reason": "agent_or_harness_stopped_early",
                    "completed_operational_weeks": trace.state.completed_weeks,
                },
            }
        trace.info["beer_game"] = outcome

    @vf.stop
    async def episode_done(self, trace: vf.Trace) -> bool:
        return trace.state.done

    @vf.reward(weight=1.0)
    async def supply_chain_reward(self, trace: vf.Trace) -> float:
        return float(trace.info["beer_game"]["grade"]["episode_reward"])

    @vf.metric
    async def beer_game_metrics(self, trace: vf.Trace) -> dict[str, float]:
        grade = trace.info["beer_game"]["grade"]
        metrics = {
            "protocol_clean": float(grade.get("protocol_clean", False)),
            "completed_operational_weeks": float(
                len(trace.info["beer_game"].get("operational_transitions", []))
            ),
        }
        if grade.get("status") != "scored":
            return metrics
        metrics.update(
            {
                "local_total_cost": float(grade["primary"]["local_total_cost"]),
                "cost_score": float(grade["primary"]["cost_score"]),
                "system_total_cost": float(grade["costs"]["system_total_cost"]),
                "immediate_fill_rate": float(
                    grade["service"]["immediate_fill_rate"]
                ),
                "horizon_fulfillment": float(
                    grade["service"]["horizon_fulfillment"]
                ),
            }
        )
        for name in ("bullwhip_ratio", "normalized_order_volatility"):
            value = grade["stability"][name]
            if value is not None:
                metrics[name] = float(value)
        return metrics

    async def validate(self, runtime: vf.Runtime) -> bool:
        del runtime
        spec = scenario_from_dict(self.data.scenario)
        episode = BeerEpisode(spec, self.data.controlled_role)
        observation = episode.start()
        policy = adaptive_policy(spec, self.data.controlled_role)
        while not episode.done:
            result = episode.place_order(policy.act(observation))
            if not result["done"]:
                observation = result["next_observation"]
        return bool(
            episode.outcome
            and episode.outcome["grade"]["episode_reward"] == 0.5
        )


class BeerTasksetConfig(vf.TasksetConfig):
    split: Split = "development"
    tiers: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    role_mode: Literal["core", "all"] = "core"
    controlled_roles: list[Role] | None = None
    seed_limit: int | None = None
    tier5_controls: bool = False
    task: BeerTaskConfig = BeerTaskConfig()


class BeerTaskset(vf.Taskset[BeerTask, BeerTasksetConfig]):
    def load(self) -> list[BeerTask]:
        invalid_tiers = sorted(set(self.config.tiers) - set(range(1, 6)))
        if invalid_tiers:
            raise ValueError(f"invalid tiers: {invalid_tiers}")
        if self.config.controlled_roles is not None:
            if not self.config.controlled_roles:
                raise ValueError("controlled_roles must contain at least one role")
            if len(set(self.config.controlled_roles)) != len(
                self.config.controlled_roles
            ):
                raise ValueError("controlled_roles must not contain duplicates")
        available = SPLIT_SIZES[self.config.split]
        count = available if self.config.seed_limit is None else self.config.seed_limit
        if count < 1 or count > available:
            raise ValueError(f"seed_limit must be in 1..{available} for {self.config.split}")

        tasks: list[BeerTask] = []
        for tier in self.config.tiers:
            variants: tuple[Variant, ...] = ("headline",)
            if tier == 5 and self.config.tier5_controls:
                variants += ("t5_control_base_rival", "t5_control_uniform")
            for variant in variants:
                for seed_index in range(count):
                    spec = scenario_for(tier, self.config.split, seed_index, variant)
                    selected_roles = (
                        tuple(self.config.controlled_roles)
                        if self.config.controlled_roles is not None
                        else roles_for(spec, self.config.role_mode)
                    )
                    unavailable = [role for role in selected_roles if role not in spec.roles]
                    if unavailable:
                        raise ValueError(
                            f"controlled roles {unavailable} are unavailable in "
                            f"tier {tier} ({spec.topology} topology); available roles: "
                            f"{list(spec.roles)}"
                        )
                    for role in selected_roles:
                        episode = BeerEpisode(spec, role, include_reference=False)
                        initial_observation = episode.start()
                        data = BeerTaskData(
                            idx=len(tasks),
                            name=f"{spec.scenario_id}:{role}:{self.config.split}:{seed_index}",
                            description="Delayed supply-chain control with programmatic grading.",
                            system_prompt=_system_prompt(spec, role),
                            prompt="Current observation:\n" + canonical_json(initial_observation),
                            scenario=spec.to_dict(),
                            controlled_role=role,
                            episode_id=spec.episode_id(role),
                            initial_observation=initial_observation,
                        )
                        tasks.append(BeerTask(data, self.config.task))
        return tasks


if __name__ == "__main__":
    BeerToolset.run()
