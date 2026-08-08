#!/usr/bin/env python3
"""Frozen split and policy-contract guards for the live Tier-5 Y comparison.

This module is deliberately independent of Torch and Verifiers so seed checks
can run before a training or evaluation process allocates a GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "experiments" / "live_y_qwen_rl" / "splits.json"
PROTOCOL_PATH = ROOT / "experiments" / "live_y_qwen_rl" / "evaluation_protocol.json"

SPLIT_ID = "beer-live-y-rl-v1"
PROTOCOL_ID = "tier5-y-wholesaler-36w-policy-v1"

SplitName = Literal["train", "research_development", "held_out_test"]
SeedPurpose = Literal[
    "training_rollout",
    "training_data_generation",
    "calibration",
    "development_evaluation",
    "prompt_tuning",
    "hyperparameter_selection",
    "early_stopping",
    "held_out_test",
]

_EXPECTED_SPLITS: dict[SplitName, tuple[str, int]] = {
    "train": ("train", 16),
    # The ten historical values were derived with the label ``eval``.  Their
    # clearer role name does not change that frozen derivation.
    "research_development": ("eval", 10),
    "held_out_test": ("held_out_test", 20),
}
_PURPOSE_SPLIT: dict[SeedPurpose, SplitName] = {
    "training_rollout": "train",
    "training_data_generation": "train",
    "calibration": "research_development",
    "development_evaluation": "research_development",
    "prompt_tuning": "research_development",
    "hyperparameter_selection": "research_development",
    "early_stopping": "research_development",
    "held_out_test": "held_out_test",
}


def _derived_seed(label: str, index: int) -> str:
    value = f"{SPLIT_ID}|{label}|{index:05d}".encode()
    return hashlib.sha256(value).hexdigest()[:16]


def load_seed_splits(path: Path = SPLIT_PATH) -> dict[SplitName, tuple[str, ...]]:
    """Load and fully verify the immutable seed values and their derivations."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "2.0" or payload.get("split_id") != SPLIT_ID:
        raise ValueError(f"unexpected live-Y split schema or ID in {path}")
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or set(raw_splits) != set(_EXPECTED_SPLITS):
        raise ValueError(f"{path} must define exactly {sorted(_EXPECTED_SPLITS)}")

    result: dict[SplitName, tuple[str, ...]] = {}
    seen: dict[str, SplitName] = {}
    for split_name, (derivation_label, expected_count) in _EXPECTED_SPLITS.items():
        row = raw_splits[split_name]
        values = tuple(str(value) for value in row.get("seeds", ()))
        if row.get("count") != expected_count or len(values) != expected_count:
            raise ValueError(f"{split_name} must contain exactly {expected_count} seeds")
        for index, value in enumerate(values):
            expected = _derived_seed(derivation_label, index)
            if value != expected:
                raise AssertionError(
                    f"{split_name} seed {index} no longer matches its frozen derivation"
                )
            if value in seen:
                raise AssertionError(
                    f"seed {value} appears in both {seen[value]} and {split_name}"
                )
            seen[value] = split_name
        result[split_name] = values
    return result


def seeds_for(
    purpose: SeedPurpose,
    *,
    allow_held_out_test: bool = False,
    path: Path = SPLIT_PATH,
) -> tuple[str, ...]:
    """Return only the split authorized for ``purpose``.

    Test access is intentionally noisy and opt-in.  Calling code should expose
    an equally explicit command-line switch and record it in the run config.
    """

    if purpose not in _PURPOSE_SPLIT:
        raise ValueError(f"unknown seed purpose: {purpose!r}")
    split_name = _PURPOSE_SPLIT[purpose]
    if split_name == "held_out_test" and not allow_held_out_test:
        raise PermissionError(
            "held-out test seeds require explicit allow_held_out_test=True after model selection"
        )
    return load_seed_splits(path)[split_name]


def assert_seed_use(
    seed_hexes: Iterable[str],
    purpose: SeedPurpose,
    *,
    allow_held_out_test: bool = False,
    path: Path = SPLIT_PATH,
) -> tuple[str, ...]:
    """Assert that every seed belongs to the one split authorized for a purpose."""

    values = tuple(str(value) for value in seed_hexes)
    if not values or len(values) != len(set(values)):
        raise ValueError("seed use must contain non-empty, unique seed IDs")
    allowed = set(
        seeds_for(
            purpose,
            allow_held_out_test=allow_held_out_test,
            path=path,
        )
    )
    forbidden = sorted(set(values) - allowed)
    if forbidden:
        raise AssertionError(
            f"seed(s) forbidden for {purpose}: {forbidden}; expected split {_PURPOSE_SPLIT[purpose]}"
        )
    return values


def assert_training_data_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_live_y_split: bool = False,
    path: Path = SPLIT_PATH,
) -> None:
    """Prevent development/test seed leakage through an SFT-style dataset.

    Unrelated legacy datasets remain usable.  Once a row identifies any seed
    from this experiment (or ``require_live_y_split`` is set), every row must
    carry a recognized live-Y training seed.
    """

    splits = load_seed_splits(path)
    recognized = set().union(*map(set, splits.values()))
    seed_rows: list[str] = []
    for index, row in enumerate(rows):
        meta = row.get("meta") or {}
        seed = meta.get("seed")
        if seed is not None:
            seed_rows.append(str(seed))
        elif require_live_y_split:
            raise ValueError(f"training-data row {index} has no meta.seed provenance")

    uses_live_y = any(seed in recognized for seed in seed_rows)
    if require_live_y_split or uses_live_y:
        if len(seed_rows) != len(rows):
            raise ValueError("every live-Y training-data row must carry meta.seed provenance")
        assert_seed_use(set(seed_rows), "training_data_generation", path=path)


def parse_policy_action(text: str) -> int | None:
    """Parse the frozen strict-JSON policy response, returning None if invalid."""

    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"quantity"}:
        return None
    quantity = value["quantity"]
    if type(quantity) is not int or not 0 <= quantity <= 128:
        return None
    return quantity


def load_policy_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0" or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"unexpected live-Y policy protocol schema or ID in {path}")
    environment = payload.get("environment", {})
    expected = {
        "environment_version": "0.2.0",
        "scenario_id": "t5-strategic-y-v2",
        "tier": 5,
        "variant": "headline",
        "topology": "y",
        "controlled_role": "wholesaler",
        "operational_weeks": 36,
        "settlement_weeks": 3,
        "order_cap": 128,
        "capacity": 22,
        "rationing": "proportional",
        "counterparty_policy": "adaptive_base_stock_v2",
        "aggressive_retailers": True,
    }
    mismatches = {
        key: (environment.get(key), value)
        for key, value in expected.items()
        if environment.get(key) != value
    }
    if mismatches:
        raise AssertionError(f"live-Y policy protocol changed: {mismatches}")
    return payload


def assert_policy_scenario(spec: Any, controlled_role: str = "wholesaler") -> None:
    """Assert that a ScenarioSpec matches the single frozen leaderboard cell."""

    load_policy_protocol()
    expected = {
        "environment_version": "0.2.0",
        "scenario_id": "t5-strategic-y-v2",
        "tier": 5,
        "variant": "headline",
        "topology": "y",
        "horizon": 36,
        "order_cap": 128,
        "capacity": 22,
        "rationing": "proportional",
        "counterparty_policy": "adaptive_base_stock_v2",
        "aggressive_retailers": True,
    }
    if controlled_role != "wholesaler":
        raise AssertionError("the live-Y leaderboard controls the wholesaler only")
    mismatches = {
        key: (getattr(spec, key, None), value)
        for key, value in expected.items()
        if getattr(spec, key, None) != value
    }
    if getattr(spec, "settlement_weeks", None) != 3:
        mismatches["settlement_weeks"] = (getattr(spec, "settlement_weeks", None), 3)
    if mismatches:
        raise AssertionError(f"scenario is outside the live-Y policy contract: {mismatches}")
