from __future__ import annotations

from types import SimpleNamespace

import pytest

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.advantages import (
    FAILURE_PENALTY,
    assign_group_advantages,
    return_to_go,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.demand import (
    negative_binomial_r10_p05,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    compare_paired_rollouts,
    research_spec,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
    research_system_prompt,
)
from scripts.train_colab_grpo_wholesaler import ActionRecord


def test_episode_lambda_and_common_random_numbers_are_paired():
    spec = research_spec("0123456789abcdef", bucket="in_distribution")
    left, right = compare_paired_rollouts(spec, [0, 128, 4, 8], [128, 0, 96, 2])
    assert left["scenario"] == right["scenario"]
    assert left["demand"]["demand_seed"] == right["demand"]["demand_seed"]
    assert left["demand"]["lambda"] == right["demand"]["lambda"]
    assert left["demand"]["trace"] == right["demand"]["trace"]
    assert left["counterparty_rng_streams"] == right["counterparty_rng_streams"]


def test_research_prompt_hides_demand_parameters_and_factory_capacity():
    spec = research_spec("0123456789abcdef", bucket="in_distribution")
    prompt = research_system_prompt(spec, "wholesaler")
    assert "long-run mean" not in prompt
    assert "Poisson" not in prompt
    assert "Factory capacity" not in prompt
    assert "22" not in prompt

    observation = {
        "constraints": {"minimum_order": 0, "maximum_order": 128, "factory_capacity": 22},
        "state": {"incoming_demand_or_order": 8},
    }
    rendered = research_observation_user_message(observation)
    assert "factory_capacity" not in rendered
    assert "22" not in rendered
    assert '"maximum_order":128' in rendered


def test_episode_randomized_poisson_is_fixed_for_the_episode():
    spec = research_spec("fedcba9876543210", bucket="in_distribution")
    episode = BeerEpisode(spec, "wholesaler", include_reference=False)
    episode.start()
    for _ in range(4):
        if not episode.done:
            episode.place_order(8)
    demand = episode.core.demand.research_demand
    assert 2 <= demand.lambda_value < 8
    assert len({row["lambda"] for row in demand.trace}) == 1
    assert all(set(row) >= {"week", "retailer_a", "retailer_b"} for row in demand.trace)


def test_overdispersed_count_moments_are_calibrated():
    values = [negative_binomial_r10_p05(f"{index:016x}", "calibration") for index in range(5000)]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    assert mean == pytest.approx(10.0, abs=0.35)
    assert variance == pytest.approx(20.0, abs=1.2)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"quantity": 12}', 12),
        ('<think>brief</think>\n{"quantity":12}', 12),
        ('{"quantity":12}\ntrailing', None),
        ('<think>' + ' x' * 161 + '</think>{"quantity":12}', None),
        ('{"quantity":12,"reason":"x"}', None),
        ('{"quantity":12.0}', None),
        ('{"quantity":129}', None),
    ],
)
def test_research_protocol_parser(text, expected):
    assert parse_completion(text) == expected


def _run(costs, *, group=1, clean=True, valid=None):
    if valid is None:
        valid = [True] * len(costs)
    records = [
        ActionRecord([], [1], group, "", 1 if ok else None, ok)
        for ok in valid
    ]
    transitions = [{"local_costs": {"wholesaler": cost}} for cost in costs]
    grade = {
        "protocol_clean": clean,
        "costs": {"settlement_local_cost": 4.0, "terminal_exposure_cost": 7.0},
    }
    episode = SimpleNamespace(
        controlled_role="wholesaler",
        operational_transitions=transitions,
        outcome={"grade": grade},
    )
    return SimpleNamespace(group_id=group, records=records, episode=episode)


def test_return_to_go_includes_weekly_settlement_and_terminal_cost():
    run = _run([1.0, 2.0, 3.0])
    assert return_to_go(run) == pytest.approx([-17.0, -16.0, -14.0])


def test_same_timestep_baseline_and_different_turn_advantages():
    first = _run([1.0, 2.0, 3.0], group=4)
    second = _run([4.0, 2.0, 1.0], group=4)
    diagnostics = assign_group_advantages([first, second])
    assert first.records[0].group_baseline == pytest.approx(-17.5)
    assert first.records[0].advantage == pytest.approx(0.5)
    assert first.records[1].advantage == pytest.approx(-1.0)
    assert first.records[2].advantage == pytest.approx(-1.0)
    assert diagnostics["total_group_timesteps"] == 3
    assert all(record.advantage == record.advantage for record in first.records)


def test_zero_variance_group_is_finite_zero():
    first = _run([1.0, 2.0], group=8)
    second = _run([1.0, 2.0], group=8)
    diagnostics = assign_group_advantages([first, second])
    assert [record.advantage for record in first.records] == [0.0, 0.0]
    assert diagnostics["zero_variance_timesteps"] == 2


def test_protocol_failure_penalty_propagates_to_preceding_turns():
    run = _run([1.0, 2.0], clean=False, valid=[True, False])
    values = return_to_go(run)
    assert values == pytest.approx([-100001.0, FAILURE_PENALTY])


def test_unequal_group_lengths_are_masked_per_timestep():
    short = _run([1.0], group=2)
    long = _run([1.0, 10.0], group=2)
    diagnostics = assign_group_advantages([short, long])
    assert short.records[0].group_baseline == long.records[0].group_baseline
    assert long.records[1].group_baseline == pytest.approx(long.records[1].return_to_go)
    assert diagnostics["total_group_timesteps"] == 2
