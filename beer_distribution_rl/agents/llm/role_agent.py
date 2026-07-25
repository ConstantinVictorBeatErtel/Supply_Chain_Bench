"""Per-role OpenRouter LLM agent with rolling own-history memory."""

from __future__ import annotations

from typing import Any

from beer_distribution_rl.agents.llm.memory import AgentMemory, WeekRecord
from beer_distribution_rl.agents.llm.openrouter import OpenRouterOrderDecoder
from beer_distribution_rl.agents.llm.serializer import observe_local, serialize_prompt
from beer_distribution_rl.env.core import BeerGameCore
from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Role


class LLMRoleAgent:
    """Orders for one supply-chain seat via a named OpenRouter model."""

    def __init__(
        self,
        role: Role,
        *,
        model: str,
        label: str | None = None,
        api_key: str | None = None,
        order_cap: int = 128,
    ) -> None:
        self.role = role
        self.role_name = Y_ROLE_NAMES.get(role, role.name.lower())
        self.model = model
        self.label = label or model
        self.memory = AgentMemory(role=role, role_name=self.role_name)
        self.decoder = OpenRouterOrderDecoder(
            model=model,
            api_key=api_key,
            order_cap=order_cap,
        )
        self._pending_obs: dict[str, Any] | None = None
        self._pending_order: int | None = None

    def reset(self) -> None:
        self.memory.reset()
        self._pending_obs = None
        self._pending_order = None

    def order(self, core: BeerGameCore) -> int:
        obs = observe_local(core, self.role)
        costs_h = float(obs["holding_cost"])
        costs_b = float(obs["backlog_cost"])
        prompt = serialize_prompt(
            self.memory,
            obs,
            order_cap=int(obs["order_cap"]),
            holding=costs_h,
            backlog_cost=costs_b,
        )
        result = self.decoder.sample_order(
            prompt,
            int(obs["last_demand_or_order"]),
            stats_key=self.role_name,
        )
        self._pending_obs = obs
        self._pending_order = int(result.order)
        return int(result.order)

    def commit_week(self, core: BeerGameCore, local_cost: float) -> None:
        """Append own-history after the env step completes."""
        if self._pending_obs is None or self._pending_order is None:
            return
        obs = self._pending_obs
        post = core.observe(self.role)
        self.memory.append(
            WeekRecord(
                week=int(obs["t"]),
                demand_or_incoming=int(obs["last_demand_or_order"]),
                ship_in=int(obs["last_shipment_received"]),
                ordered=int(self._pending_order),
                alloc_recv=int(post["last_shipment_received"]),
                inventory=int(post["inventory"]),
                backlog=int(post["backlog"]),
                on_order=int(post["on_order"]),
                ship_pipeline=list(post["ship_pipeline"]),
                order_pipeline=list(post["order_pipeline"]),
                local_cost=float(local_cost),
            )
        )
        self._pending_obs = None
        self._pending_order = None

    def info(self) -> dict[str, str]:
        return {
            "role": self.role_name,
            "model": self.model,
            "label": self.label,
        }
