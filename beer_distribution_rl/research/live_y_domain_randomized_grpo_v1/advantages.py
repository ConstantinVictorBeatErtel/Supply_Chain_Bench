"""Per-decision return-to-go advantages for critic-free group-relative updates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

FAILURE_PENALTY = -100_000.0


@dataclass
class TurnDiagnostic:
    turn: int
    return_to_go: float
    baseline: float
    advantage: float


def _terminal_reward(run: Any) -> float:
    grade = (run.episode.outcome or {}).get("grade", {})
    costs = grade.get("costs", {})
    terminal = float(costs.get("settlement_local_cost", 0.0)) + float(
        costs.get("terminal_exposure_cost", 0.0)
    )
    return -terminal


def return_to_go(run: Any) -> list[float]:
    """Compute G[i,t] from local costs, native settlement, terminal exposure."""

    records = list(run.records)
    transitions = list(run.episode.operational_transitions)
    clean = bool((run.episode.outcome or {}).get("grade", {}).get("protocol_clean"))
    values: list[float] = [0.0] * len(records)
    running = _terminal_reward(run) if clean else 0.0
    for turn in range(len(records) - 1, -1, -1):
        record = records[turn]
        if not record.valid:
            running += FAILURE_PENALTY
        elif turn < len(transitions):
            running -= float(transitions[turn]["local_costs"][run.episode.controlled_role])
        values[turn] = running
    return values


def assign_group_advantages(runs: list[Any]) -> dict[str, Any]:
    """Attach same-timestep group-relative advantages; unequal lengths are masked."""

    by_group: dict[int, list[Any]] = {}
    for run in runs:
        by_group.setdefault(run.group_id, []).append(run)
    diagnostics: list[dict[str, float | int]] = []
    zero_variance = 0
    total_timesteps = 0
    for group_id, members in by_group.items():
        returns = {id(run): return_to_go(run) for run in members}
        max_turns = max((len(values) for values in returns.values()), default=0)
        for turn in range(max_turns):
            available = [values[turn] for values in returns.values() if turn < len(values)]
            if not available:
                continue
            baseline = mean(available)
            variance = mean((value - baseline) ** 2 for value in available)
            if variance <= 1e-24:
                zero_variance += 1
            for run in members:
                if turn >= len(returns[id(run)]):
                    continue
                advantage = returns[id(run)][turn] - baseline
                if not math.isfinite(advantage):
                    raise FloatingPointError("non-finite per-turn advantage")
                record = run.records[turn]
                record.return_to_go = returns[id(run)][turn]
                record.group_baseline = baseline
                record.advantage = advantage
                diagnostics.append({
                    "group_id": group_id,
                    "turn": turn,
                    "return_to_go": returns[id(run)][turn],
                    "baseline": baseline,
                    "advantage": advantage,
                })
            total_timesteps += 1
    return {
        "diagnostics": diagnostics,
        "zero_variance_timesteps": zero_variance,
        "total_group_timesteps": total_timesteps,
        "zero_variance_rate": zero_variance / total_timesteps if total_timesteps else 0.0,
    }
