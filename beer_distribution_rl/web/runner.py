"""Turn-based human-vs-AI episode runner for the playable Y-topology game."""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from beer_distribution_rl.agents.baselines import StermanAgent
from beer_distribution_rl.agents.ippo.policy_agent import (
    IPPOTeam,
    PolicyLoadError,
    default_ippo_checkpoint_dir,
    load_ippo_team,
)
from beer_distribution_rl.env.core import BeerGameCore, y_topology_env_config
from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Y_ROLES, Role
from beer_distribution_rl.web.frames import end_reveal, player_frame_from_core

Listener = Callable[[dict[str, Any]], None]

ROLE_ORDER: tuple[str, ...] = tuple(Y_ROLE_NAMES[r] for r in Y_ROLES)
AiMode = Literal["sterman", "ippo"]

_NAME_TO_ROLE: dict[str, Role] = {Y_ROLE_NAMES[r]: r for r in Y_ROLES}


class GameError(RuntimeError):
    """User-facing game control error."""


class GameRunner:
    """Y-topology Beer Game: one human role, AI counterparties, fog-of-war.

    Thread-safe for FastAPI / WebSocket callers. Turn-based: each week waits
    for ``submit_order`` before advancing.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        ippo_checkpoint_dir: Path | str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._seed = int(seed)
        self._ippo_checkpoint_dir = (
            Path(ippo_checkpoint_dir)
            if ippo_checkpoint_dir is not None
            else default_ippo_checkpoint_dir()
        )

        self._core = BeerGameCore(y_topology_env_config())
        self._human_role: Role | None = None
        self._ai_mode: AiMode | None = None
        self._phase: Literal["setup", "playing", "finished"] = "setup"
        self._awaiting_order = False

        self._sterman: dict[Role, StermanAgent] = {r: StermanAgent() for r in Y_ROLES}
        self._ippo_team: IPPOTeam | None = None

        self._cumulative_system_cost = 0.0
        self._cumulative_own_cost = 0.0
        self._last_week_cost = 0.0
        self._last_own_order: int | None = None
        self._last_frame: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._reveal: dict[str, Any] | None = None

    # --- listeners ---------------------------------------------------------

    def add_listener(self, listener: Listener) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, message: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(message)
            except Exception:
                pass

    # --- status / snapshots ------------------------------------------------

    def _status_payload(self) -> dict[str, Any]:
        demand = self._core.config.demand
        return {
            "type": "status",
            "phase": self._phase,
            "awaiting_order": self._awaiting_order,
            "seed": self._seed,
            "roles": list(ROLE_ORDER),
            "horizon": self._core.config.horizon,
            "order_cap": self._core.config.order_cap,
            "topology": self._core.topology.name,
            "demand_model": getattr(demand, "name", type(demand).__name__),
            "human_role": (
                Y_ROLE_NAMES[self._human_role] if self._human_role is not None else None
            ),
            "ai_mode": self._ai_mode,
        }

    def _emit_status(self) -> None:
        self._emit(self._status_payload())

    def _build_player_frame(self, *, terminated: bool) -> dict[str, Any]:
        assert self._human_role is not None
        frame = player_frame_from_core(
            self._core,
            human_role=self._human_role,
            week_cost=self._last_week_cost,
            cumulative_own_cost=self._cumulative_own_cost,
            last_order=self._last_own_order,
            terminated=terminated,
        )
        frame["awaiting_order"] = bool(self._awaiting_order and not terminated)
        return frame

    def snapshot(self) -> dict[str, Any]:
        """Filtered sync payload for a newly connected client."""
        with self._lock:
            status = self._status_payload()
            out: dict[str, Any] = {
                **status,
                "type": "snapshot",
                "history": list(self._history),
            }
            if self._last_frame is not None:
                out["frame"] = dict(self._last_frame)
            if self._reveal is not None:
                out["reveal"] = dict(self._reveal)
            return out

    # --- game control ------------------------------------------------------

    def start(
        self,
        role: str | Role,
        ai_mode: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Begin an episode as ``role`` against Sterman or IPPO opponents."""
        with self._lock:
            human = self._parse_role(role)
            mode = self._parse_ai_mode(ai_mode)
            if seed is not None:
                self._seed = int(seed)

            self._human_role = human
            self._ai_mode = mode
            self._phase = "playing"
            self._awaiting_order = True
            self._cumulative_system_cost = 0.0
            self._cumulative_own_cost = 0.0
            self._last_week_cost = 0.0
            self._last_own_order = None
            self._history = []
            self._reveal = None

            if mode == "sterman":
                self._ippo_team = None
                for agent in self._sterman.values():
                    agent.reset()
            else:
                try:
                    self._ippo_team = load_ippo_team(
                        self._core.config,
                        checkpoint_dir=self._ippo_checkpoint_dir,
                    )
                    self._ippo_team.reset()
                except PolicyLoadError:
                    self._human_role = None
                    self._ai_mode = None
                    self._phase = "setup"
                    self._awaiting_order = False
                    raise

            self._core.reset(seed=self._seed)
            frame = self._build_player_frame(terminated=False)
            self._last_frame = frame
            self._history.append(dict(frame))

        self._emit_status()
        self._emit({"type": "frame", **frame})
        return self.snapshot()

    def submit_order(self, quantity: int | float) -> dict[str, Any]:
        """Accept the human order, fill AI orders, and advance one week."""
        with self._lock:
            if self._phase != "playing" or self._human_role is None or self._ai_mode is None:
                raise GameError("No active game. Call start first.")
            if not self._awaiting_order:
                raise GameError("Not awaiting an order.")
            if self._core._terminated:
                raise GameError("Episode already finished.")

            order_cap = int(self._core.config.order_cap)
            try:
                qty = int(quantity)
            except (TypeError, ValueError) as exc:
                raise GameError("Order quantity must be an integer.") from exc
            if qty < 0 or qty > order_cap:
                raise GameError(f"Order must be between 0 and {order_cap}.")

            human = self._human_role
            orders: dict[Role, int] = {human: qty}
            for role in Y_ROLES:
                if role == human:
                    continue
                orders[role] = self._ai_order(role)

            _states, _rewards, terminated, info = self._core.step(orders)
            own_cost = float(info.local_costs[human])
            self._last_week_cost = own_cost
            self._cumulative_own_cost += own_cost
            self._cumulative_system_cost += float(info.system_cost)
            self._last_own_order = qty

            if terminated:
                self._phase = "finished"
                self._awaiting_order = False
                frame = self._build_player_frame(terminated=True)
                self._last_frame = frame
                self._history.append(dict(frame))
                reveal = end_reveal(
                    human_role=human,
                    cumulative_own_cost=self._cumulative_own_cost,
                    cumulative_system_cost=self._cumulative_system_cost,
                    horizon=self._core.config.horizon,
                    ai_mode=self._ai_mode,
                    seed=self._seed,
                )
                self._reveal = reveal
            else:
                self._awaiting_order = True
                frame = self._build_player_frame(terminated=False)
                self._last_frame = frame
                self._history.append(dict(frame))
                reveal = None

        self._emit({"type": "frame", **frame})
        if reveal is not None:
            self._emit(reveal)
            self._emit_status()
        return self.snapshot()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Return to setup; clears the active episode."""
        with self._lock:
            if seed is not None:
                self._seed = int(seed)
            self._human_role = None
            self._ai_mode = None
            self._phase = "setup"
            self._awaiting_order = False
            self._cumulative_system_cost = 0.0
            self._cumulative_own_cost = 0.0
            self._last_week_cost = 0.0
            self._last_own_order = None
            self._last_frame = None
            self._history = []
            self._reveal = None
            self._ippo_team = None
            for agent in self._sterman.values():
                agent.reset()
        self._emit_status()
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            self._phase = "setup"
            self._awaiting_order = False
            self._ippo_team = None

    # --- internals ---------------------------------------------------------

    def _ai_order(self, role: Role) -> int:
        state = self._core.states[role]
        if self._ai_mode == "ippo":
            assert self._ippo_team is not None
            return int(self._ippo_team.order(role, state, self._core))
        return int(self._sterman[role].order(state))

    @staticmethod
    def _parse_role(role: str | Role) -> Role:
        if isinstance(role, Role):
            if role not in Y_ROLES:
                raise GameError(f"Unsupported role: {role}")
            return role
        key = str(role).strip().lower().replace(" ", "_")
        if key == "retailer":
            key = "retailer_a"
        if key not in _NAME_TO_ROLE:
            raise GameError(
                f"Unknown role '{role}'. Choose one of: {', '.join(ROLE_ORDER)}"
            )
        return _NAME_TO_ROLE[key]

    @staticmethod
    def _parse_ai_mode(ai_mode: str) -> AiMode:
        mode = str(ai_mode).strip().lower()
        if mode in ("sterman", "heuristic"):
            return "sterman"
        if mode in ("ippo", "trained", "rl", "ai"):
            return "ippo"
        raise GameError("ai_mode must be 'sterman' or 'ippo'")


# Backward-compatible alias for older imports / serve scripts.
EpisodeRunner = GameRunner
