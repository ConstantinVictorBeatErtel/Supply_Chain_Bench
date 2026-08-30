from __future__ import annotations

from beer_distribution_game.episode import BeerEpisode

from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2 import (
    COUNTERPARTY_POLICY_ID,
    PROTOCOL_ID,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v2.environment import (
    derive_training_seed,
    training_spec,
)


def test_training_seed_derivation_is_unique_and_replayable() -> None:
    seeds = [derive_training_seed(index) for index in range(128)]
    assert len(set(seeds)) == len(seeds)
    assert all(len(seed) == 16 for seed in seeds)
    assert seeds == [derive_training_seed(index) for index in range(128)]


def test_training_and_play_spec_is_explicitly_versioned() -> None:
    spec = training_spec("0123456789abcdef", index=7)
    assert spec.environment_version == PROTOCOL_ID
    assert spec.scenario_id == f"{PROTOCOL_ID}:in_distribution"
    assert spec.split == "training"
    assert spec.seed_index == 7
    assert spec.capacity == 400
    assert spec.counterparty_policy == COUNTERPARTY_POLICY_ID


def test_wholesaler_receives_weekly_demand_responsive_retailer_orders() -> None:
    spec = training_spec("0123456789abcdef")
    episode = BeerEpisode(spec, "wholesaler", include_reference=False)
    observation = episode.start()
    observed_orders: list[int] = []

    while not episode.done:
        observed_orders.append(int(observation["state"]["incoming_demand_or_order"]))
        result = episode.place_order(8)
        transition = episode.operational_transitions[-1]

        for role in ("retailer_a", "retailer_b"):
            customer_demand = next(iter(transition["incoming_by_claimant"][role].values()))
            assert transition["orders"][role] == customer_demand + 8

        if len(episode.operational_transitions) >= 2:
            previous = episode.operational_transitions[-2]
            expected = previous["orders"]["retailer_a"] + previous["orders"]["retailer_b"]
            assert transition["incoming_by_claimant"]["wholesaler"] == {
                "retailer_a": previous["orders"]["retailer_a"],
                "retailer_b": previous["orders"]["retailer_b"],
            }
            assert observed_orders[-1] == expected

        if not result["done"]:
            observation = result["next_observation"]

    # A stochastic weekly signal, rather than the archived constant-16 floor.
    assert len(set(observed_orders[1:])) >= 6
    assert sum(left != right for left, right in zip(observed_orders[1:], observed_orders[2:])) >= 20
    assert sum(value == 16 for value in observed_orders[1:]) < 5
