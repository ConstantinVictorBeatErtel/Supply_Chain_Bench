"""Per-decision return-to-go advantages for critic-free group-relative updates.

The credit window matters more than it looks.  With an unbounded return-to-go,
``G[i,t]`` is dominated by the inventory position the trajectory was already
carrying at ``t``, not by the order placed at ``t``: once two group members
diverge by a few hundred units they keep paying that difference every remaining
week, so ``G[i,t] - baseline`` stays roughly constant for the rest of the
episode.  Measured on the ``runpod_final`` rollouts, only ~10% of the advantage
variance was per-turn -- the update degenerated into episode-level REINFORCE
with a four-sample baseline.

Bounding the window to the causal horizon of one order (order delay 1 +
shipment delay 2, so effects land within ~3 weeks and wash out by ~6) restores
per-turn credit to ~73% of the variance.  See
``docs/LIVE_Y_RL_POSTMORTEM.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any

FAILURE_PENALTY = -100_000.0

#: Causal credit horizon in weeks.  ``None`` restores the unbounded
#: return-to-go used by the archived two-update run.
DEFAULT_CREDIT_WINDOW = 6

#: Per-week discount applied inside the window.  Kept at 1.0 by default; the
#: window is doing the localization, and stacking both is redundant.
DEFAULT_DISCOUNT = 1.0

#: Safety clamp applied after group normalization.  Group-relative
#: normalization already bounds a group of ``n`` members to ``sqrt(n - 1)``, so
#: this only ever fires on a degenerate group; it exists so no single record can
#: run away with a minibatch.
ADVANTAGE_CLIP = 10.0

#: Advantage assigned to any decision whose credit window contains a protocol
#: failure, in normalized units.  Group normalization would otherwise rescale
#: the raw ``-1e5`` penalty down to roughly ``-1.7`` -- indistinguishable from
#: an ordinary bad week -- while leaving it unnormalized would let one record
#: own the entire update.  A fixed value keeps failure decisively worst than any
#: clean outcome (which cannot exceed ``sqrt(n - 1)``) and bounded.
FAILURE_ADVANTAGE = -5.0


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


def return_to_go(
    run: Any,
    window: int | None = None,
    discount: float = DEFAULT_DISCOUNT,
) -> list[float]:
    """Compute G[i,t] from local costs, native settlement, terminal exposure.

    ``window`` bounds how many weeks of downstream cost a single decision is
    charged for.  ``None`` sums to the end of the episode (the original
    unbounded behaviour).  Settlement and terminal exposure are only attributed
    to turns whose window actually reaches the horizon, which is the same
    causality rule applied to the weekly costs.
    """

    if window is not None and window < 1:
        raise ValueError("credit window must be >= 1 week")
    records = list(run.records)
    transitions = list(run.episode.operational_transitions)
    clean = bool((run.episode.outcome or {}).get("grade", {}).get("protocol_clean"))
    role = run.episode.controlled_role
    horizon = len(records)
    terminal = _terminal_reward(run) if clean else 0.0
    values: list[float] = []
    for turn in range(horizon):
        end = horizon if window is None else min(horizon, turn + window)
        total = 0.0
        for step in range(turn, end):
            weight = discount ** (step - turn)
            record = records[step]
            if not record.valid:
                total += weight * FAILURE_PENALTY
            elif step < len(transitions):
                total -= weight * float(transitions[step]["local_costs"][role])
        if end >= horizon:
            total += (discount ** (horizon - turn)) * terminal
        values.append(total)
    return values


def assign_group_advantages(
    runs: list[Any],
    window: int | None = DEFAULT_CREDIT_WINDOW,
    discount: float = DEFAULT_DISCOUNT,
    normalize: bool = True,
) -> dict[str, Any]:
    """Attach same-timestep group-relative advantages; unequal lengths are masked.

    ``normalize`` divides each timestep's advantages by the group standard
    deviation so the surrogate sees a unit-scale signal instead of raw cost
    units (which ranged over ~500 and left the effective step size at the mercy
    of gradient-norm clipping).  Decisions whose credit window contains a
    protocol failure bypass normalization and take ``FAILURE_ADVANTAGE``.
    """

    by_group: dict[int, list[Any]] = {}
    for run in runs:
        by_group.setdefault(run.group_id, []).append(run)
    diagnostics: list[dict[str, float | int]] = []
    zero_variance = 0
    total_timesteps = 0
    for group_id, members in by_group.items():
        returns = {
            id(run): return_to_go(run, window=window, discount=discount) for run in members
        }
        max_turns = max((len(values) for values in returns.values()), default=0)
        for turn in range(max_turns):
            available = [values[turn] for values in returns.values() if turn < len(values)]
            if not available:
                continue
            baseline = mean(available)
            variance = mean((value - baseline) ** 2 for value in available)
            if variance <= 1e-24:
                zero_variance += 1
            scale = math.sqrt(variance) if normalize and variance > 1e-24 else 1.0
            for run in members:
                if turn >= len(returns[id(run)]):
                    continue
                advantage = (returns[id(run)][turn] - baseline) / scale
                if normalize:
                    if returns[id(run)][turn] <= FAILURE_PENALTY / 2:
                        advantage = FAILURE_ADVANTAGE
                    else:
                        advantage = max(-ADVANTAGE_CLIP, min(ADVANTAGE_CLIP, advantage))
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
        "credit_window": window,
        "discount": discount,
        "normalized": normalize,
        "diagnostics": diagnostics,
        "zero_variance_timesteps": zero_variance,
        "total_group_timesteps": total_timesteps,
        "zero_variance_rate": zero_variance / total_timesteps if total_timesteps else 0.0,
    }
