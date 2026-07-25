"""Lightweight IPPO inference agents for the playable web game.

Loads frozen per-role checkpoints without the training loop. Supports
Markovian MLP and recurrent GRU policies used by Regime A Y-topology runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from beer_distribution_rl.agents.ippo.obs import obs_dim, state_to_obs
from beer_distribution_rl.env.core import BeerGameCore, EnvConfig, RoleState
from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Y_ROLES, Role

# Prefer a trained Y recurrent run when present locally; otherwise use the
# packaged playable policies shipped with the web UI.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACTS_IPPO_DIR = (
    _REPO_ROOT
    / "artifacts"
    / "runs"
    / "ippo"
    / "recurrent_baseline"
    / "regimeA_topoy_capinf_ratproportional_demar1_seed0"
    / "checkpoints"
)
_PACKAGED_IPPO_DIR = (
    Path(__file__).resolve().parents[2] / "web" / "assets" / "ippo_y_default" / "checkpoints"
)


def default_ippo_checkpoint_dir() -> Path:
    if _ARTIFACTS_IPPO_DIR.is_dir():
        return _ARTIFACTS_IPPO_DIR
    return _PACKAGED_IPPO_DIR


DEFAULT_IPPO_CHECKPOINT_DIR = _PACKAGED_IPPO_DIR


class PolicyLoadError(RuntimeError):
    """Raised when torch or checkpoint files are unavailable."""


def _require_torch():
    try:
        import torch
        from beer_distribution_rl.agents.ippo.networks import (
            ActorCritic,
            RecurrentActorCritic,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional deps
        raise PolicyLoadError(
            "IPPO opponents require torch. Install with: pip install -e \".[marl]\""
        ) from exc
    return torch, ActorCritic, RecurrentActorCritic


class IPPOPolicyAgent:
    """Single-role frozen policy used as a non-human counterparty."""

    def __init__(
        self,
        role: Role,
        policy,
        *,
        action_mode: str = "relative",
        action_delta_max: int = 8,
        order_cap: int = 128,
        recurrent: bool = False,
        device: str = "cpu",
    ) -> None:
        self.role = role
        self.policy = policy
        self.action_mode = action_mode
        self.action_delta_max = int(action_delta_max)
        self.order_cap = int(order_cap)
        self.recurrent = bool(recurrent)
        self.device = device
        self._h = None
        self.reset()

    def reset(self) -> None:
        torch, _, RecurrentActorCritic = _require_torch()
        if self.recurrent:
            assert isinstance(self.policy, RecurrentActorCritic)
            self._h = self.policy.initial_hidden(1, torch.device(self.device))
        else:
            self._h = None

    def _decode_order(self, action_idx: int, state: RoleState) -> int:
        if self.action_mode == "absolute":
            return int(action_idx)
        delta = int(action_idx) - self.action_delta_max
        raw = int(state.last_demand_or_order) + delta
        return max(0, min(self.order_cap, raw))

    def order(self, state: RoleState, core: BeerGameCore) -> int:
        torch, _, RecurrentActorCritic = _require_torch()
        obs = state_to_obs(state, self.role, core)
        ot = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.recurrent:
                assert isinstance(self.policy, RecurrentActorCritic)
                assert self._h is not None
                dist, _value, h_new = self.policy.forward(ot, self._h)
                self._h = h_new.detach()
                action_idx = int(dist.probs.argmax(dim=-1).item())
            else:
                dist, _value = self.policy.forward(ot)
                action_idx = int(dist.probs.argmax(dim=-1).item())
        return self._decode_order(action_idx, state)


class IPPOTeam:
    """All non-human Y-topology roles driven by one checkpoint directory."""

    def __init__(
        self,
        agents: dict[Role, IPPOPolicyAgent],
        *,
        checkpoint_dir: Path,
        config: dict[str, Any],
    ) -> None:
        self.agents = agents
        self.checkpoint_dir = checkpoint_dir
        self.config = config

    def reset(self) -> None:
        for agent in self.agents.values():
            agent.reset()

    def order(self, role: Role, state: RoleState, core: BeerGameCore) -> int:
        return self.agents[role].order(state, core)


def load_ippo_team(
    env_config: EnvConfig,
    *,
    checkpoint_dir: Path | str | None = None,
    device: str = "cpu",
    roles: tuple[Role, ...] = Y_ROLES,
) -> IPPOTeam:
    """Load per-role policies for Y-topology inference."""
    torch, ActorCritic, RecurrentActorCritic = _require_torch()

    ckpt = Path(checkpoint_dir) if checkpoint_dir else default_ippo_checkpoint_dir()
    if not ckpt.is_dir():
        raise PolicyLoadError(f"IPPO checkpoint directory not found: {ckpt}")

    # Match the default recurrent Y baseline used in artifacts.
    action_mode = "relative"
    action_delta_max = 8
    hidden = 256
    gru_hidden = 128
    recurrent = True
    order_cap = int(env_config.order_cap)

    # Prefer sibling config.yaml when present.
    cfg_path = ckpt.parent / "config.yaml"
    meta: dict[str, Any] = {
        "action_mode": action_mode,
        "action_delta_max": action_delta_max,
        "hidden": hidden,
        "gru_hidden": gru_hidden,
        "recurrent": recurrent,
        "order_cap": order_cap,
        "checkpoint_dir": str(ckpt),
    }
    if cfg_path.is_file():
        try:
            import yaml

            raw = yaml.safe_load(cfg_path.read_text()) or {}
            action_mode = str(raw.get("action_mode", action_mode))
            action_delta_max = int(raw.get("action_delta_max", action_delta_max))
            hidden = int(raw.get("hidden", hidden))
            gru_hidden = int(raw.get("gru_hidden", gru_hidden))
            recurrent = bool(raw.get("recurrent", recurrent))
            order_cap = int(raw.get("order_cap", order_cap))
            meta.update(
                {
                    "action_mode": action_mode,
                    "action_delta_max": action_delta_max,
                    "hidden": hidden,
                    "gru_hidden": gru_hidden,
                    "recurrent": recurrent,
                    "order_cap": order_cap,
                }
            )
        except Exception:
            pass

    if action_mode == "relative":
        n_order = 2 * action_delta_max + 1
    elif action_mode == "absolute":
        n_order = order_cap + 1
    else:
        raise PolicyLoadError(f"unknown action_mode: {action_mode}")

    odim = obs_dim(env_config)
    agents: dict[Role, IPPOPolicyAgent] = {}
    for role in roles:
        name = Y_ROLE_NAMES.get(role, role.name.lower())
        pt = ckpt / f"policy_{name}.pt"
        if not pt.exists() and role == Role.RETAILER:
            pt = ckpt / "policy_retailer.pt"
        if not pt.exists():
            raise PolicyLoadError(f"Missing policy checkpoint: {pt}")

        if recurrent:
            policy = RecurrentActorCritic(odim, n_order, hidden, gru_hidden)
        else:
            policy = ActorCritic(odim, n_order, hidden)
        state = torch.load(pt, map_location=device, weights_only=True)
        policy.load_state_dict(state)
        policy.to(device)
        policy.eval()
        agents[role] = IPPOPolicyAgent(
            role,
            policy,
            action_mode=action_mode,
            action_delta_max=action_delta_max,
            order_cap=order_cap,
            recurrent=recurrent,
            device=device,
        )

    return IPPOTeam(agents, checkpoint_dir=ckpt, config=meta)
