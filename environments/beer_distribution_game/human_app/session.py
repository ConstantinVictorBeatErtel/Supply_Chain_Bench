"""Human play sessions on the Hub BeerEpisode stack (Tier 5 wholesaler)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
from typing import Any, Literal
from uuid import uuid4

from beer_distribution_game import __version__ as ENV_VERSION
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import (
    SPLIT_SIZES,
    Split,
    master_seed_hex,
    scenario_for,
)

FIXED_TIER = 5
FIXED_ROLE = "wholesaler"
FIXED_VARIANT = "headline"
HUMAN_SPLITS: tuple[Split, ...] = ("development", "validation")
PriorExperience = Literal["yes", "no", "unsure"]
SessionStatus = Literal["in_progress", "completed", "abandoned"]

EVAL_SEED_POOL: tuple[tuple[Split, int], ...] = tuple(
    (split, index)
    for split in HUMAN_SPLITS
    for index in range(SPLIT_SIZES[split])
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class HumanSession:
    """One anonymous human episode with Hub-eval condition parity."""

    prior_beer_game_experience: PriorExperience
    rng: random.Random = field(default_factory=random.Random)
    session_uuid: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=_utc_now_iso)
    status: SessionStatus = "in_progress"
    actions: list[int] = field(default_factory=list)
    weekly: list[dict[str, Any]] = field(default_factory=list)
    observation: dict[str, Any] | None = None
    _logged: bool = field(default=False, init=False, repr=False)

    split: Split = field(init=False)
    seed_index: int = field(init=False)
    seed: str = field(init=False)
    scenario_id: str = field(init=False)
    episode: BeerEpisode = field(init=False)

    def __post_init__(self) -> None:
        self.split, self.seed_index = self.rng.choice(EVAL_SEED_POOL)
        self.seed = master_seed_hex(self.split, self.seed_index)
        spec = scenario_for(
            FIXED_TIER,
            self.split,
            self.seed_index,
            variant=FIXED_VARIANT,
        )
        self.scenario_id = spec.scenario_id
        self.episode = BeerEpisode(spec, FIXED_ROLE)
        self.observation = self.episode.start()

    @property
    def done(self) -> bool:
        return self.status != "in_progress"

    @property
    def weeks_played(self) -> int:
        return len(self.actions)

    def place_order(self, quantity: int) -> dict[str, Any]:
        if self.status != "in_progress":
            raise RuntimeError("session is already finished")
        if type(quantity) is not int or isinstance(quantity, bool):
            raise ValueError("quantity must be an integer in [0, 128]")
        if not 0 <= quantity <= self.episode.spec.order_cap:
            raise ValueError(
                f"quantity must be an integer in [0, {self.episode.spec.order_cap}]"
            )

        result = self.episode.place_order(quantity)
        self.actions.append(quantity)
        transition = self.episode.operational_transitions[-1]
        role = FIXED_ROLE
        state = transition["states_after_fulfillment"][role]
        self.weekly.append(
            {
                "week": transition["week"],
                "inventory": int(state["inventory"]),
                "backlog": int(state["backlog"]),
                "local_cost": float(transition["local_costs"][role]),
            }
        )
        if result["done"]:
            self.status = "completed"
            self.observation = None
        else:
            self.observation = result["next_observation"]
        return result

    def abandon(self) -> None:
        if self.status != "in_progress":
            return
        self.status = "abandoned"
        self.observation = None

    def to_log_record(self) -> dict[str, Any]:
        final_total_cost = None
        base_stock_cost = None
        episode_reward = None
        if self.status == "completed" and self.episode.outcome is not None:
            grade = self.episode.outcome["grade"]
            final_total_cost = float(grade["primary"]["local_total_cost"])
            paired = grade["primary"]["paired_base_stock_local_total_cost"]
            base_stock_cost = None if paired is None else float(paired)
            reward = grade["episode_reward"]
            episode_reward = None if reward is None else float(reward)

        return {
            "session_uuid": self.session_uuid,
            "timestamp": self.timestamp,
            "env_version": ENV_VERSION,
            "tier": FIXED_TIER,
            "role": FIXED_ROLE,
            "split": self.split,
            "seed_index": self.seed_index,
            "seed": self.seed,
            "scenario_id": self.scenario_id,
            "variant": FIXED_VARIANT,
            "prior_beer_game_experience": self.prior_beer_game_experience,
            "status": self.status,
            "actions": list(self.actions),
            "weekly": list(self.weekly),
            "final_total_cost": final_total_cost,
            "base_stock_cost": base_stock_cost,
            "episode_reward": episode_reward,
        }

    def end_summary(self) -> dict[str, Any]:
        record = self.to_log_record()
        return {
            "status": record["status"],
            "weeks_played": len(record["actions"]),
            "actions": list(record["actions"]),
            "weekly": list(record["weekly"]),
            "final_total_cost": record["final_total_cost"],
            "base_stock_cost": record["base_stock_cost"],
            "episode_reward": record["episode_reward"],
            "seed": record["seed"],
            "split": record["split"],
            "seed_index": record["seed_index"],
        }
