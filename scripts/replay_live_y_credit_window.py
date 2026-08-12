#!/usr/bin/env python3
"""Replay logged rollouts under alternative credit-assignment windows.

This is the free checkpoint before buying GPU time.  The archived
``live-y-domain-randomized-grpo-v1`` run logged, per trajectory, the weekly
local wholesaler cost and the per-turn return-to-go that was actually used to
build the advantages.  That is enough to recompute the advantages under any
other window/discount offline and ask a single question:

    how much of the advantage signal is genuinely about *this* week's decision?

With the unbounded return-to-go the answer was ~10%: two group members that had
diverged in inventory kept paying that difference every remaining week, so
``G[i,t] - baseline`` stayed roughly constant and the update degenerated into
episode-level REINFORCE with a four-sample baseline.  Bounding the window to the
causal horizon of one order restores per-turn credit.

Usage::

    python scripts/replay_live_y_credit_window.py \
        --rollouts artifacts/live_y_domain_randomized_grpo_v1/runpod_final/rollouts.jsonl \
        --out artifacts/live_y_domain_randomized_grpo_v1/credit_window_replay.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.advantages import (  # noqa: E402
    assign_group_advantages,
    return_to_go,
)

ROLE = "wholesaler"

SCHEMES: list[dict[str, Any]] = [
    {"name": "to-end, undiscounted (archived)", "window": None, "discount": 1.0},
    {"name": "to-end, discount 0.9", "window": None, "discount": 0.9},
    {"name": "window 12 weeks", "window": 12, "discount": 1.0},
    {"name": "window 8 weeks", "window": 8, "discount": 1.0},
    {"name": "window 6 weeks", "window": 6, "discount": 1.0},
    {"name": "window 4 weeks", "window": 4, "discount": 1.0},
    {"name": "window 6 weeks, discount 0.9", "window": 6, "discount": 0.9},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rollouts",
        default="artifacts/live_y_domain_randomized_grpo_v1/runpod_final/rollouts.jsonl",
    )
    p.add_argument("--out", default=None, help="Write the full report as JSON here.")
    return p.parse_args()


def build_run(row: dict[str, Any], group_id: int) -> Any:
    """Rebuild the minimal run object ``advantages`` needs from a logged row.

    The logged ``return_to_go`` is the unbounded series, so its last entry is
    ``-cost[-1] + terminal``; that inverts exactly to the settlement plus
    terminal-exposure charge, which the row does not store on its own.
    """
    costs = [float(value) for value in row["weekly_local_costs"]]
    logged = [float(value) for value in row["return_to_go"]]
    terminal = logged[-1] + costs[-1] if costs else 0.0
    clean = bool(row.get("protocol_clean", True))
    records = [
        SimpleNamespace(
            valid=True,
            advantage=0.0,
            return_to_go=None,
            group_baseline=None,
        )
        for _ in costs
    ]
    episode = SimpleNamespace(
        controlled_role=ROLE,
        operational_transitions=[{"local_costs": {ROLE: cost}} for cost in costs],
        outcome={
            "grade": {
                "protocol_clean": clean,
                "costs": {
                    "settlement_local_cost": -terminal if clean else 0.0,
                    "terminal_exposure_cost": 0.0,
                },
            }
        },
    )
    return SimpleNamespace(
        group_id=group_id,
        records=records,
        episode=episode,
        _logged_return_to_go=logged,
        _task=row["task"],
        _update=row["update"],
        _cost=row.get("local_total_cost"),
    )


def verify_reconstruction(runs: list[Any]) -> float:
    """Largest absolute disagreement with the logged return-to-go series."""
    worst = 0.0
    for run in runs:
        rebuilt = return_to_go(run, window=None, discount=1.0)
        for actual, expected in zip(rebuilt, run._logged_return_to_go):
            worst = max(worst, abs(actual - expected))
    return worst


def variance_split(matrix: list[list[float]]) -> dict[str, float]:
    """How much of the advantage signal varies week to week vs per trajectory."""
    within = statistics.mean(statistics.pstdev(row) for row in matrix)
    between = statistics.pstdev([statistics.mean(row) for row in matrix])
    total = within**2 + between**2
    return {
        "within_trajectory_sd": within,
        "between_trajectory_sd": between,
        "per_turn_variance_share": (within**2 / total) if total else 0.0,
    }


def sign_agreement(matrix: list[list[float]]) -> float:
    """Fraction of weeks whose advantage sign just echoes the episode verdict."""
    shares = []
    for row in matrix:
        episode_sign = math.copysign(1.0, statistics.mean(row))
        shares.append(
            statistics.mean(1.0 if math.copysign(1.0, v) == episode_sign else 0.0 for v in row)
        )
    return statistics.mean(shares)


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return float("nan")
    left_sd = statistics.pstdev(left)
    right_sd = statistics.pstdev(right)
    if left_sd <= 1e-12 or right_sd <= 1e-12:
        return float("nan")
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    covariance = statistics.mean(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    return covariance / (left_sd * right_sd)


def evaluate(runs: list[Any], window: int | None, discount: float) -> dict[str, Any]:
    # Normalization is a per-timestep affine rescale, so it cannot change how
    # the signal splits between weeks and trajectories.  Leave it off here so
    # the comparison isolates the credit window.
    assign_group_advantages(runs, window=window, discount=discount, normalize=False)
    matrix = [[record.advantage for record in run.records] for run in runs]
    horizon = min(len(row) for row in matrix)
    report = variance_split(matrix)
    report["sign_agreement_with_episode_verdict"] = sign_agreement(matrix)
    first = [row[0] for row in matrix]
    report["corr_week1_week12"] = (
        correlation(first, [row[11] for row in matrix]) if horizon > 11 else float("nan")
    )
    report["corr_week1_week24"] = (
        correlation(first, [row[23] for row in matrix]) if horizon > 23 else float("nan")
    )
    report["mean_abs_advantage"] = statistics.mean(
        abs(value) for row in matrix for value in row
    )
    return report


def main() -> None:
    args = parse_args()
    path = Path(args.rollouts)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    groups: dict[tuple[int, str], int] = {}
    runs: list[Any] = []
    for row in rows:
        # The same task recurs across updates; those are separate groups.
        key = (row["update"], row["task"])
        group_id = groups.setdefault(key, len(groups))
        runs.append(build_run(row, group_id))

    drift = verify_reconstruction(runs)
    print(f"rollouts: {len(runs)} trajectories in {len(groups)} groups")
    print(f"reconstruction check vs logged return-to-go: max abs error {drift:.2e}")
    if drift > 1e-6:
        raise SystemExit(
            "reconstructed return-to-go does not match the log; refusing to report"
        )
    print()

    header = f"{'scheme':32} {'per-turn share':>14} {'sign echo':>10} {'r(w1,w12)':>10}"
    print(header)
    print("-" * len(header))
    results = []
    for scheme in SCHEMES:
        report = evaluate(runs, scheme["window"], scheme["discount"])
        results.append({**scheme, **report})
        print(
            f"{scheme['name']:32} "
            f"{report['per_turn_variance_share']:>14.3f} "
            f"{report['sign_agreement_with_episode_verdict']:>10.3f} "
            f"{report['corr_week1_week12']:>10.2f}"
        )

    payload = {
        "source": str(path),
        "trajectories": len(runs),
        "groups": len(groups),
        "reconstruction_max_abs_error": drift,
        "role": ROLE,
        "metric_notes": {
            "per_turn_variance_share": (
                "share of advantage variance that varies week to week within a "
                "trajectory rather than being a per-trajectory constant; 1.0 "
                "would be pure per-turn credit, 0.0 pure episode-level verdict"
            ),
            "sign_agreement_with_episode_verdict": (
                "fraction of weeks whose advantage sign merely repeats the "
                "trajectory's overall verdict"
            ),
        },
        "schemes": results,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
