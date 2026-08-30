"""Scenario construction shared by corrected model training and human play."""

from __future__ import annotations

import hashlib

from beer_distribution_game.scenario import ScenarioSpec, scenario_for

from . import COUNTERPARTY_POLICY_ID, DEMAND_PROCESS_ID, PROTOCOL_ID


def derive_training_seed(index: int) -> str:
    """Derive a unique replayable seed for one training episode."""

    if index < 0:
        raise ValueError("training seed index must be non-negative")
    payload = f"{PROTOCOL_ID}|training|{index:08d}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def training_spec(seed_hex: str, *, index: int = 0) -> ScenarioSpec:
    """Return the exact corrected scenario used by training and browser play.

    Customer demand is sampled independently for every retailer/week from an
    episode-randomized Poisson process.  Retailer replenishment equals that
    weekly customer demand plus the declared scarcity increment, so the
    wholesaler's role-local incoming orders retain weekly variation.
    """

    if len(seed_hex) != 16 or any(ch not in "0123456789abcdef" for ch in seed_hex):
        raise ValueError("seed must be 16 lowercase hexadecimal characters")
    if index < 0:
        raise ValueError("training seed index must be non-negative")

    prototype = scenario_for(5, "development", 0, "headline")
    values = prototype.to_dict()
    values.update(
        {
            "environment_version": PROTOCOL_ID,
            "scenario_id": f"{PROTOCOL_ID}:in_distribution",
            "split": "training",
            "seed_index": index,
            "master_seed_hex": seed_hex,
            "demand_process": DEMAND_PROCESS_ID,
            "demand_parameters": {
                "lambda_low": 2.0,
                "lambda_high": 8.0,
                "demand_seed": seed_hex,
            },
            "capacity": 400,
            "counterparty_policy": COUNTERPARTY_POLICY_ID,
            "aggressive_retailers": True,
        }
    )
    return ScenarioSpec(**{**values, "roles": tuple(values["roles"])})
