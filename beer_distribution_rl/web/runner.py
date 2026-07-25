"""Turn-based human-vs-AI episode runner for the playable Y-topology game."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from beer_distribution_rl.agents.baselines import StermanAgent
from beer_distribution_rl.agents.ippo.policy_agent import (
    IPPOTeam,
    PolicyLoadError,
    default_ippo_checkpoint_dir,
    load_ippo_team,
)
from beer_distribution_rl.agents.llm.openrouter import (
    OpenRouterError,
    model_catalog,
    require_openrouter_api_key,
)
from beer_distribution_rl.agents.llm.role_agent import LLMRoleAgent
from beer_distribution_rl.env.core import BeerGameCore, y_topology_env_config
from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Y_ROLES, Role
from beer_distribution_rl.web.frames import end_reveal, player_frame_from_core

Listener = Callable[[dict[str, Any]], None]

ROLE_ORDER: tuple[str, ...] = tuple(Y_ROLE_NAMES[r] for r in Y_ROLES)
AiMode = Literal["sterman", "ippo", "llm"]
LlmScope = Literal["retailers", "all"]
CompareMode = Literal["sterman", "llm"]
LLM_HORIZON = 16
_RETAILER_ROLES = (Role.RETAILER, Role.RETAILER_B)

_NAME_TO_ROLE: dict[str, Role] = {Y_ROLE_NAMES[r]: r for r in Y_ROLES}
_LABELS = {m["id"]: m["label"] for m in model_catalog()}


class GameError(RuntimeError):
    """User-facing game control error."""


class GameRunner:
    """Y-topology Beer Game: one human role, AI/LLM counterparties, fog-of-war."""

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
        self._llm_agents: dict[Role, LLMRoleAgent] = {}
        self._model_map: dict[str, dict[str, str]] = {}
        self._default_model: str | None = None
        self._llm_scope: LlmScope = "retailers"
        self._compare_mode: CompareMode = "sterman"

        self._cumulative_system_cost = 0.0
        self._cumulative_own_cost = 0.0
        self._last_week_cost = 0.0
        self._last_own_order: int | None = None
        self._last_frame: dict[str, Any] | None = None
        self._history: list[dict[str, Any]] = []
        self._reveal: dict[str, Any] | None = None
        self._human_series: list[dict[str, Any]] = []

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

    def _ai_role_names(self) -> list[str]:
        assert self._human_role is not None
        return [Y_ROLE_NAMES[r] for r in Y_ROLES if r != self._human_role]

    def _status_payload(self) -> dict[str, Any]:
        demand = self._core.config.demand
        return {
            "type": "status",
            "phase": self._phase,
            "awaiting_order": self._awaiting_order,
            "seed": self._seed,
            "roles": list(ROLE_ORDER),
            "ai_roles": self._ai_role_names() if self._human_role is not None else [],
            "horizon": self._core.config.horizon,
            "order_cap": self._core.config.order_cap,
            "topology": self._core.topology.name,
            "demand_model": getattr(demand, "name", type(demand).__name__),
            "human_role": (
                Y_ROLE_NAMES[self._human_role] if self._human_role is not None else None
            ),
            "ai_mode": self._ai_mode,
            "models": self._model_map,
            "model_catalog": model_catalog(),
            "default_model": self._default_model,
            "llm_scope": self._llm_scope,
            "compare_mode": self._compare_mode,
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
            ai_roles=self._ai_role_names(),
        )
        frame["awaiting_order"] = bool(self._awaiting_order and not terminated)
        frame["models"] = dict(self._model_map)
        return frame

    def snapshot(self) -> dict[str, Any]:
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
        *,
        model: str | None = None,
        retailer_a_model: str | None = None,
        retailer_b_model: str | None = None,
        llm_scope: str = "retailers",
        compare_mode: str = "sterman",
    ) -> dict[str, Any]:
        """Begin an episode. LLM mode uses OpenRouter; dual retailer models when
        the human is not a retailer.

        ``llm_scope="retailers"`` (default) only calls LLMs for retailer seats;
        other AI seats use Sterman. ``compare_mode="sterman"`` (default) uses a
        fast Sterman shadow for the end-screen You-vs-AI charts.
        """
        with self._lock:
            human = self._parse_role(role)
            mode = self._parse_ai_mode(ai_mode)
            scope = self._parse_llm_scope(llm_scope)
            compare = self._parse_compare_mode(compare_mode)
            if seed is not None:
                self._seed = int(seed)

            self._human_role = human
            self._ai_mode = mode
            self._llm_scope = scope
            self._compare_mode = compare
            self._phase = "playing"
            self._awaiting_order = True
            self._cumulative_system_cost = 0.0
            self._cumulative_own_cost = 0.0
            self._last_week_cost = 0.0
            self._last_own_order = None
            self._history = []
            self._human_series = []
            self._reveal = None
            self._llm_agents = {}
            self._model_map = {}
            self._default_model = None
            self._ippo_team = None

            # LLM games use a shorter horizon so OpenRouter latency stays playable.
            # Preserve a preconfigured horizon for Sterman/IPPO (tests / custom setups).
            horizon = LLM_HORIZON if mode == "llm" else int(self._core.config.horizon)
            self._core = BeerGameCore(y_topology_env_config(horizon=horizon))

            try:
                if mode == "sterman":
                    for agent in self._sterman.values():
                        agent.reset()
                elif mode == "ippo":
                    self._ippo_team = load_ippo_team(
                        self._core.config,
                        checkpoint_dir=self._ippo_checkpoint_dir,
                    )
                    self._ippo_team.reset()
                else:
                    for agent in self._sterman.values():
                        agent.reset()
                    self._setup_llm_agents(
                        human,
                        model=model,
                        retailer_a_model=retailer_a_model,
                        retailer_b_model=retailer_b_model,
                        llm_scope=scope,
                    )
            except (PolicyLoadError, OpenRouterError, GameError):
                self._human_role = None
                self._ai_mode = None
                self._phase = "setup"
                self._awaiting_order = False
                self._llm_agents = {}
                raise

            self._core.reset(seed=self._seed)
            frame = self._build_player_frame(terminated=False)
            self._last_frame = frame
            self._history.append(dict(frame))

        self._emit_status()
        self._emit({"type": "frame", **frame})
        return self.snapshot()

    def submit_order(self, quantity: int | float) -> dict[str, Any]:
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
            try:
                orders, ai_orders, ai_models = self._collect_ai_orders(human, qty)
            except OpenRouterError as exc:
                raise GameError(str(exc)) from exc

            assert len(ai_orders) == len(Y_ROLES) - 1
            assert set(orders) == set(Y_ROLES)

            _states, _rewards, terminated, info = self._core.step(orders)
            own_cost = float(info.local_costs[human])
            self._last_week_cost = own_cost
            self._cumulative_own_cost += own_cost
            self._cumulative_system_cost += float(info.system_cost)
            self._last_own_order = qty

            for role, agent in self._llm_agents.items():
                agent.commit_week(self._core, float(info.local_costs[role]))

            self._human_series.append(
                {
                    "t": int(self._core.t),
                    "order": qty,
                    "week_cost": own_cost,
                    "cumulative_own_cost": float(self._cumulative_own_cost),
                    "demand_or_incoming": int(
                        self._core.states[human].last_demand_or_order
                    ),
                    "inventory": int(self._core.states[human].inventory),
                    "backlog": int(self._core.states[human].backlog),
                    "ai_orders": ai_orders,
                    "ai_models": ai_models,
                }
            )

            if terminated:
                self._phase = "finished"
                self._awaiting_order = False
                frame = self._build_player_frame(terminated=True)
                self._last_frame = frame
                self._history.append(dict(frame))
                try:
                    shadow = self._run_shadow_human_seat()
                except OpenRouterError as exc:
                    raise GameError(str(exc)) from exc
                reveal = end_reveal(
                    human_role=human,
                    cumulative_own_cost=self._cumulative_own_cost,
                    cumulative_system_cost=self._cumulative_system_cost,
                    horizon=self._core.config.horizon,
                    ai_mode=self._ai_mode,
                    seed=self._seed,
                    ai_roles=self._ai_role_names(),
                    human_series=list(self._human_series),
                    ai_series=shadow["series"],
                    ai_own_cost=shadow["own_cost"],
                    ai_system_cost=shadow["system_cost"],
                )
                reveal["models"] = dict(self._model_map)
                reveal["shadow_model"] = shadow.get("shadow_model")
                reveal["llm_scope"] = self._llm_scope
                reveal["compare_mode"] = self._compare_mode
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
            self._human_series = []
            self._reveal = None
            self._ippo_team = None
            self._llm_agents = {}
            self._model_map = {}
            self._default_model = None
            self._llm_scope = "retailers"
            self._compare_mode = "sterman"
            self._core = BeerGameCore(y_topology_env_config())
            for agent in self._sterman.values():
                agent.reset()
        self._emit_status()
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            self._phase = "setup"
            self._awaiting_order = False
            self._ippo_team = None
            self._llm_agents = {}

    # --- internals ---------------------------------------------------------

    def _collect_ai_orders(
        self, human: Role, human_qty: int
    ) -> tuple[dict[Role, int], dict[str, int], dict[str, dict[str, str]]]:
        """Fill non-human orders; LLM seats are queried in parallel."""
        roles = [r for r in Y_ROLES if r != human]
        orders: dict[Role, int] = {human: human_qty}
        ai_orders: dict[str, int] = {}
        ai_models: dict[str, dict[str, str]] = {}

        def _one(role: Role) -> tuple[Role, int]:
            return role, self._ai_order(role)

        # Parallelize OpenRouter latency across AI seats (Sterman seats are instant).
        with ThreadPoolExecutor(max_workers=max(1, len(roles))) as pool:
            futures = [pool.submit(_one, role) for role in roles]
            for fut in as_completed(futures):
                role, ai_qty = fut.result()
                orders[role] = ai_qty
                name = Y_ROLE_NAMES[role]
                ai_orders[name] = ai_qty
                if name in self._model_map:
                    ai_models[name] = dict(self._model_map[name])
        return orders, ai_orders, ai_models

    def _setup_llm_agents(
        self,
        human: Role,
        *,
        model: str | None,
        retailer_a_model: str | None,
        retailer_b_model: str | None,
        llm_scope: LlmScope,
    ) -> None:
        catalog_ids = {m["id"] for m in model_catalog()}
        default = (model or "deepseek/deepseek-v4-flash").strip()
        if default not in catalog_ids and "/" not in default:
            raise GameError(f"Unknown model: {default}")
        self._default_model = default

        human_is_retailer = human in _RETAILER_ROLES
        ra = (retailer_a_model or default).strip()
        rb = (retailer_b_model or default).strip()
        if not human_is_retailer:
            if not retailer_a_model or not retailer_b_model:
                raise GameError(
                    "When you are not a retailer, pick two different models "
                    "for Retailer A and Retailer B."
                )
            if ra == rb:
                raise GameError(
                    "Retailer A and Retailer B must use different LLM models."
                )

        api_key = require_openrouter_api_key()

        llm_roles: list[Role] = []
        for role in Y_ROLES:
            if role == human:
                continue
            if llm_scope == "retailers" and role not in _RETAILER_ROLES:
                self._model_map[Y_ROLE_NAMES[role]] = {
                    "model": "sterman",
                    "label": "Sterman",
                }
                continue
            llm_roles.append(role)

        for role in llm_roles:
            if role == Role.RETAILER:
                mid = ra if not human_is_retailer else default
            elif role == Role.RETAILER_B:
                mid = rb if not human_is_retailer else default
            else:
                mid = default
            label = _LABELS.get(mid, mid)
            agent = LLMRoleAgent(
                role,
                model=mid,
                label=label,
                api_key=api_key,
                order_cap=int(self._core.config.order_cap),
            )
            self._llm_agents[role] = agent
            self._model_map[Y_ROLE_NAMES[role]] = {
                "model": mid,
                "label": label,
            }

    def _shadow_env(self) -> BeerGameCore:
        cfg = self._core.config
        return BeerGameCore(
            y_topology_env_config(
                horizon=cfg.horizon,
                order_cap=cfg.order_cap,
                ship_delay=cfg.ship_delay,
                order_delay=cfg.order_delay,
                capacity=cfg.capacity,
                regime=cfg.regime,
                signaling_enabled=cfg.signaling_enabled,
            )
        )

    def _run_shadow_human_seat(self) -> dict[str, Any]:
        """Same seed; replay recorded AI counterparties; AI/LLM plays human seat."""
        assert self._human_role is not None
        assert self._ai_mode is not None
        human = self._human_role
        core = self._shadow_env()
        core.reset(seed=self._seed)

        sterman = {r: StermanAgent() for r in Y_ROLES}
        ippo: IPPOTeam | None = None
        shadow_llm: LLMRoleAgent | None = None
        shadow_model_info: dict[str, str] | None = None

        if self._ai_mode == "ippo":
            ippo = load_ippo_team(
                core.config,
                checkpoint_dir=self._ippo_checkpoint_dir,
            )
            ippo.reset()
        elif self._ai_mode == "llm" and self._compare_mode == "llm":
            # Optional slow path: LLM in the human seat for end-screen comparison.
            mid = self._default_model or "deepseek/deepseek-v4-flash"
            label = _LABELS.get(mid, mid)
            shadow_llm = LLMRoleAgent(
                human,
                model=mid,
                label=label,
                api_key=require_openrouter_api_key(),
                order_cap=int(core.config.order_cap),
            )
            shadow_model_info = {"model": mid, "label": label}
        elif self._ai_mode == "llm":
            # Default fast path: Sterman baseline in your seat (no extra API calls).
            shadow_model_info = {"model": "sterman", "label": "Sterman"}

        series: list[dict[str, Any]] = []
        own_cost = 0.0
        system_cost = 0.0

        for row in self._human_series:
            recorded = row.get("ai_orders") or {}
            orders: dict[Role, int] = {}
            for role in Y_ROLES:
                if role == human:
                    if shadow_llm is not None:
                        orders[role] = shadow_llm.order(core)
                    elif self._ai_mode == "ippo":
                        assert ippo is not None
                        orders[role] = int(ippo.order(role, core.states[role], core))
                    else:
                        orders[role] = int(sterman[role].order(core.states[role]))
                else:
                    name = Y_ROLE_NAMES[role]
                    if name in recorded:
                        orders[role] = int(recorded[name])
                    elif self._ai_mode == "ippo":
                        assert ippo is not None
                        orders[role] = int(ippo.order(role, core.states[role], core))
                    else:
                        orders[role] = int(sterman[role].order(core.states[role]))

            _states, _rewards, _term, info = core.step(orders)
            week = float(info.local_costs[human])
            own_cost += week
            system_cost += float(info.system_cost)
            if shadow_llm is not None:
                shadow_llm.commit_week(core, week)
            series.append(
                {
                    "t": int(core.t),
                    "order": int(info.orders_placed[human]),
                    "week_cost": week,
                    "cumulative_own_cost": own_cost,
                    "demand_or_incoming": int(core.states[human].last_demand_or_order),
                    "inventory": int(core.states[human].inventory),
                    "backlog": int(core.states[human].backlog),
                }
            )
        return {
            "series": series,
            "own_cost": own_cost,
            "system_cost": system_cost,
            "shadow_model": shadow_model_info,
        }

    def _ai_order(self, role: Role) -> int:
        if self._ai_mode == "llm":
            agent = self._llm_agents.get(role)
            if agent is not None:
                return int(agent.order(self._core))
            # Hybrid fast mode: non-LLM seats use Sterman.
            return int(self._sterman[role].order(self._core.states[role]))
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
        if mode in ("ippo", "trained", "rl"):
            return "ippo"
        if mode in ("llm", "openrouter", "ai"):
            return "llm"
        raise GameError("ai_mode must be 'llm', 'sterman', or 'ippo'")

    @staticmethod
    def _parse_llm_scope(llm_scope: str) -> LlmScope:
        scope = str(llm_scope).strip().lower()
        if scope in ("retailers", "fast", "hybrid"):
            return "retailers"
        if scope in ("all", "full"):
            return "all"
        raise GameError("llm_scope must be 'retailers' or 'all'")

    @staticmethod
    def _parse_compare_mode(compare_mode: str) -> CompareMode:
        mode = str(compare_mode).strip().lower()
        if mode in ("sterman", "heuristic", "fast"):
            return "sterman"
        if mode in ("llm", "openrouter", "model"):
            return "llm"
        raise GameError("compare_mode must be 'sterman' or 'llm'")


EpisodeRunner = GameRunner
