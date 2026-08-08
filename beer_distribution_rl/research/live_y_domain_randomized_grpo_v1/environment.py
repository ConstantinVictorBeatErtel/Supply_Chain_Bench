"""Research scenario construction and common-random-number checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.scenario import ScenarioSpec, scenario_for

from . import DEMAND_PROCESS_ID, PROTOCOL_ID
from .prompting import research_system_prompt

EVAL_PROCESSES = {
    "in_distribution": DEMAND_PROCESS_ID,
    "canonical_held_out_step": "canonical_y_step_4_to_8_v1",
    "shifted_mean_doubled_variance": "overdispersed_y_nb_mean10_var20_v1",
    "burst_and_collapse": "burst_collapse_y_poisson_v1",
}


def derive_research_seed(bucket: str, index: int) -> str:
    return hashlib.sha256(f"{PROTOCOL_ID}|evaluation|{bucket}|{index:05d}".encode()).hexdigest()[:16]


def research_spec(seed_hex: str, *, bucket: str, index: int = 0) -> ScenarioSpec:
    process = EVAL_PROCESSES.get(bucket, bucket)
    prototype = scenario_for(5, "development", 0, "headline")
    values = prototype.to_dict()
    values.update(
        {
            "environment_version": PROTOCOL_ID,
            "scenario_id": f"{PROTOCOL_ID}:{bucket}",
            "split": f"research_{bucket}",
            "seed_index": index,
            "master_seed_hex": seed_hex,
            "demand_process": process,
            "demand_parameters": (
                {"lambda_low": 2.0, "lambda_high": 8.0, "demand_seed": seed_hex}
                if process == DEMAND_PROCESS_ID
                else {"process_id": process}
            ),
        }
    )
    return ScenarioSpec(**{**values, "roles": tuple(values["roles"])})


@dataclass
class ResearchTask:
    name: str
    scenario: dict[str, Any]
    controlled_role: str = "wholesaler"

    @property
    def system_prompt(self) -> str:
        return research_system_prompt(
            ScenarioSpec(**{**self.scenario, "roles": tuple(self.scenario["roles"])}),
            self.controlled_role,
        )


def tasks_for(bucket: str, count: int = 4) -> list[ResearchTask]:
    return [
        ResearchTask(
            name=f"{PROTOCOL_ID}:{bucket}:{index}",
            scenario=research_spec(derive_research_seed(bucket, index), bucket=bucket, index=index).to_dict(),
        )
        for index in range(count)
    ]


def start_research_episode(task: ResearchTask, *, include_reference: bool = False) -> BeerEpisode:
    spec = ScenarioSpec(**{**task.scenario, "roles": tuple(task.scenario["roles"])})
    return BeerEpisode(spec, task.controlled_role, include_reference=include_reference)


def exogenous_manifest(episode: BeerEpisode) -> dict[str, Any]:
    manifest = episode.core.research_exogenous_manifest()
    if manifest is None:
        raise AssertionError("research episode did not expose an exogenous manifest")
    return {
        "scenario": episode.spec.to_dict(),
        "demand": manifest,
        "counterparty_rng_streams": {
            role: hashlib.sha256(
                f"{PROTOCOL_ID}|{episode.spec.master_seed_hex}|counterparty|{role}".encode()
            ).hexdigest()
            for role in episode.counterparties
        },
    }


def compare_paired_rollouts(spec: ScenarioSpec, actions_a: list[int], actions_b: list[int]) -> tuple[dict, dict]:
    """Run paired policies and compare exogenous draws before policy effects."""

    first = BeerEpisode(spec, "wholesaler", include_reference=False)
    second = BeerEpisode(spec, "wholesaler", include_reference=False)
    first.start()
    second.start()
    for left, right in zip(actions_a, actions_b):
        first.place_order(left)
        second.place_order(right)
        left_trace = first.core.research_exogenous_manifest()["trace"]
        right_trace = second.core.research_exogenous_manifest()["trace"]
        if left_trace != right_trace:
            raise AssertionError("paired rollouts diverged in exogenous demand")
        if first.done or second.done:
            break
    return exogenous_manifest(first), exogenous_manifest(second)
