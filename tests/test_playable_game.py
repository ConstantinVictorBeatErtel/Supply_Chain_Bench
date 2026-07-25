"""Tests for the playable Y-topology human-vs-AI web game."""

from __future__ import annotations

import pytest

from beer_distribution_rl.env.core_types import Y_ROLE_NAMES, Y_ROLES, Role
from beer_distribution_rl.web.frames import player_frame_from_core
from beer_distribution_rl.web.runner import GameError, GameRunner


@pytest.fixture
def runner() -> GameRunner:
    return GameRunner(seed=0)


def test_start_defaults_to_setup(runner: GameRunner) -> None:
    snap = runner.snapshot()
    assert snap["phase"] == "setup"
    assert snap["human_role"] is None
    assert snap["awaiting_order"] is False
    assert "frame" not in snap or snap.get("frame") is None


@pytest.mark.parametrize("role_name", list(Y_ROLE_NAMES[r] for r in Y_ROLES))
def test_sterman_start_and_submit_advances(runner: GameRunner, role_name: str) -> None:
    snap = runner.start(role_name, "sterman", seed=1)
    assert snap["phase"] == "playing"
    assert snap["human_role"] == role_name
    assert snap["ai_mode"] == "sterman"
    assert snap["awaiting_order"] is True
    frame = snap["frame"]
    assert frame["t"] == 0
    assert frame["human_role"] == role_name
    assert frame["awaiting_order"] is True

    snap2 = runner.submit_order(5)
    assert snap2["frame"]["t"] == 1
    assert snap2["frame"]["own_order"] == 5
    assert snap2["frame"]["last_order_placed"] == 5
    assert snap2["awaiting_order"] is True


def test_double_submit_without_order_state_blocked(runner: GameRunner) -> None:
    runner.start("wholesaler", "sterman", seed=2)
    runner.submit_order(3)
    # After a successful submit we are awaiting again; force-clear to simulate race.
    runner._awaiting_order = False  # noqa: SLF001
    with pytest.raises(GameError, match="awaiting"):
        runner.submit_order(4)


def test_order_clamp_validation(runner: GameRunner) -> None:
    runner.start("factory", "sterman", seed=3)
    with pytest.raises(GameError, match="between 0 and"):
        runner.submit_order(-1)
    with pytest.raises(GameError, match="between 0 and"):
        runner.submit_order(10_000)


def test_fog_frame_hides_other_roles(runner: GameRunner) -> None:
    runner.start("distributor", "sterman", seed=4)
    frame = runner.snapshot()["frame"]
    # Local fields only — no rival inventories / orders maps.
    assert "inventories" not in frame
    assert "backlogs" not in frame
    assert "orders" not in frame
    assert "allocations" not in frame
    assert "local_costs" not in frame
    assert "cumulative_cost" not in frame  # system cost hidden mid-game
    assert "cumulative_own_cost" in frame
    assert frame["human_role"] == "distributor"
    assert set(frame["roles"]) == set(Y_ROLE_NAMES[r] for r in Y_ROLES)


def test_player_frame_from_core_uses_observe(runner: GameRunner) -> None:
    runner.start("retailer_a", "sterman", seed=5)
    core = runner._core  # noqa: SLF001
    framed = player_frame_from_core(
        core,
        human_role=Role.RETAILER,
        week_cost=0.0,
        cumulative_own_cost=0.0,
        terminated=False,
    )
    obs = core.observe(Role.RETAILER)
    assert framed["inventory"] == obs["inventory"]
    assert framed["backlog"] == obs["backlog"]
    assert framed["last_demand_or_order"] == obs["last_demand_or_order"]
    assert framed["is_retailer"] is True
    assert framed["customer_demand"] == obs["last_demand_or_order"]


def test_reset_returns_to_setup(runner: GameRunner) -> None:
    runner.start("wholesaler", "sterman", seed=6)
    runner.submit_order(4)
    snap = runner.reset()
    assert snap["phase"] == "setup"
    assert snap["human_role"] is None
    assert snap.get("history") == []
    assert "reveal" not in snap


def test_full_episode_ends_with_reveal(runner: GameRunner) -> None:
    # Short horizon for a fast end-to-end.
    from beer_distribution_rl.env.core import BeerGameCore, y_topology_env_config

    runner._core = BeerGameCore(y_topology_env_config(horizon=3))  # noqa: SLF001
    runner.start("retailer_b", "sterman", seed=7)
    snap = None
    for qty in (4, 5, 6):
        snap = runner.submit_order(qty)
    assert snap is not None
    assert snap["phase"] == "finished"
    assert snap["awaiting_order"] is False
    assert snap["frame"]["terminated"] is True
    reveal = snap["reveal"]
    assert reveal["human_role"] == "retailer_b"
    assert reveal["ai_mode"] == "sterman"
    assert reveal["horizon"] == 3
    assert reveal["cumulative_own_cost"] >= 0
    assert reveal["cumulative_system_cost"] >= reveal["cumulative_own_cost"]
    assert set(reveal["ai_roles"]) == {
        "retailer_a",
        "wholesaler",
        "distributor",
        "factory",
    }
    assert len(reveal["human_series"]) == 3
    assert len(reveal["ai_series"]) == 3
    assert "ai_own_cost" in reveal
    # AI placed orders for every non-human role each week.
    for row in reveal["human_series"]:
        assert set(row["ai_orders"]) == set(reveal["ai_roles"])
        assert all(isinstance(v, int) and v >= 0 for v in row["ai_orders"].values())


def test_submit_before_start_errors(runner: GameRunner) -> None:
    with pytest.raises(GameError, match="No active game"):
        runner.submit_order(1)


def test_unknown_role_errors(runner: GameRunner) -> None:
    with pytest.raises(GameError, match="Unknown role"):
        runner.start("bartender", "sterman")


def test_ippo_path_or_clear_error(runner: GameRunner) -> None:
    """IPPO either loads and plays, or fails with a clear PolicyLoadError/GameError path."""
    try:
        snap = runner.start("wholesaler", "ippo", seed=8)
    except Exception as exc:
        # Missing torch or missing checkpoints should surface clearly.
        assert "IPPO" in str(exc) or "torch" in str(exc).lower() or "checkpoint" in str(exc).lower()
        return

    assert snap["ai_mode"] == "ippo"
    snap2 = runner.submit_order(4)
    assert snap2["frame"]["t"] == 1
    order_cap = snap2["frame"]["order_cap"]
    assert 0 <= snap2["frame"]["own_order"] <= order_cap


def test_llm_requires_dual_retailer_models(runner: GameRunner) -> None:
    with pytest.raises(GameError, match="different LLM models"):
        runner.start(
            "wholesaler",
            "llm",
            seed=1,
            model="deepseek/deepseek-v4-flash",
            retailer_a_model="deepseek/deepseek-v4-flash",
            retailer_b_model="deepseek/deepseek-v4-flash",
        )


def test_llm_requires_api_key(monkeypatch: pytest.MonkeyPatch, runner: GameRunner) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(Exception, match="OPENROUTER_API_KEY"):
        runner.start(
            "retailer_a",
            "llm",
            seed=1,
            model="deepseek/deepseek-v4-flash",
        )
