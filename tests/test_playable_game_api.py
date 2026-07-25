"""API smoke tests for the playable game server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from beer_distribution_rl.web.runner import GameRunner
from beer_distribution_rl.web.server import create_app


def test_api_start_order_reset_fog() -> None:
    runner = GameRunner(seed=0)
    # Short episode for reveal coverage.
    from beer_distribution_rl.env.core import BeerGameCore, y_topology_env_config

    runner._core = BeerGameCore(y_topology_env_config(horizon=2))  # noqa: SLF001
    app = create_app(runner)
    client = TestClient(app)

    state = client.get("/api/state").json()
    assert state["phase"] == "setup"

    started = client.post(
        "/api/start",
        json={"role": "wholesaler", "ai_mode": "sterman", "seed": 11},
    )
    assert started.status_code == 200
    body = started.json()
    assert body["phase"] == "playing"
    frame = body["frame"]
    assert "inventories" not in frame
    assert frame["human_role"] == "wholesaler"

    step1 = client.post("/api/order", json={"quantity": 4})
    assert step1.status_code == 200
    assert step1.json()["frame"]["t"] == 1

    step2 = client.post("/api/order", json={"quantity": 5})
    assert step2.status_code == 200
    finished = step2.json()
    assert finished["phase"] == "finished"
    assert "reveal" in finished
    assert finished["reveal"]["cumulative_system_cost"] >= 0

    bad = client.post("/api/order", json={"quantity": 1})
    assert bad.status_code == 400

    reset = client.post("/api/control", json={"action": "reset"})
    assert reset.status_code == 200
    assert reset.json()["phase"] == "setup"
