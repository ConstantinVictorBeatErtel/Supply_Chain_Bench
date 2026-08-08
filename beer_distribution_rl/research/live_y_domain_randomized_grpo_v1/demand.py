"""Episode-randomized Y demand for the research extension only."""

from __future__ import annotations

from dataclasses import dataclass, field

from .randomness import digest, poisson, uniform

PROCESS_ID = "episode_randomized_y_poisson_v1"
CANONICAL_STEP_ID = "canonical_y_step_4_to_8_v1"
OVERDISPERSED_ID = "overdispersed_y_nb_mean10_var20_v1"
BURST_COLLAPSE_ID = "burst_collapse_y_poisson_v1"


@dataclass
class EpisodeRandomizedPoissonDemand:
    master_seed_hex: str
    lambda_low: float = 2.0
    lambda_high: float = 8.0
    demand_seed: str = ""
    lambda_value: float = field(init=False)
    trace: list[dict[str, int | float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lambda_low < 0 or self.lambda_high < self.lambda_low:
            raise ValueError("invalid lambda range")
        # The seed is recorded separately so the CRN contract can compare it.
        self.demand_seed = self.demand_seed or self.master_seed_hex
        self.lambda_value = self.lambda_low + (self.lambda_high - self.lambda_low) * uniform(
            self.demand_seed, "episode/lambda"
        )

    def sample(self, week: int, customers: tuple[str, ...]) -> dict[str, int]:
        values = {
            role: poisson(
                self.demand_seed,
                f"customer-demand/week-{week}/{role}",
                self.lambda_value,
            )
            for role in customers
        }
        self.trace.append({"week": week, "lambda": self.lambda_value, **values})
        return values

    def manifest(self) -> dict[str, object]:
        return {
            "process_id": PROCESS_ID,
            "demand_seed": self.demand_seed,
            "lambda_low": self.lambda_low,
            "lambda_high": self.lambda_high,
            "lambda": self.lambda_value,
            "trace": list(self.trace),
        }


def negative_binomial_r10_p05(seed_hex: str, namespace: str, index: int = 0) -> int:
    """NB(r=10,p=.5) as ten seeded geometric counts (mean 10, variance 20)."""

    total = 0
    successes = 0
    while successes < 10:
        draw = uniform(seed_hex, f"{namespace}/geometric", index + total + successes * 4096)
        if draw >= 0.5:
            successes += 1
        else:
            total += 1
    return total


@dataclass
class EvaluationDemand:
    process_id: str
    master_seed_hex: str
    trace: list[dict[str, int | float]] = field(default_factory=list)
    burst_start: int | None = None
    burst_length: int = 4

    def __post_init__(self) -> None:
        if self.process_id == BURST_COLLAPSE_ID:
            self.burst_start = 8 + int.from_bytes(digest(self.master_seed_hex, "burst-window")[:2], "big") % 9

    def sample(self, week: int, customers: tuple[str, ...]) -> dict[str, int]:
        values: dict[str, int] = {}
        for role_index, role in enumerate(customers):
            if self.process_id == CANONICAL_STEP_ID:
                value = 4 if week < 5 else 8
            elif self.process_id == OVERDISPERSED_ID:
                value = negative_binomial_r10_p05(
                    self.master_seed_hex, f"{self.process_id}/week-{week}/{role}", role_index
                )
            elif self.process_id == BURST_COLLAPSE_ID:
                assert self.burst_start is not None
                burst = self.burst_start <= week < self.burst_start + self.burst_length
                collapse = self.burst_start + self.burst_length <= week < self.burst_start + 2 * self.burst_length
                rate = 12.0 if burst else 1.0 if collapse else 4.0
                value = poisson(self.master_seed_hex, f"{self.process_id}/week-{week}/{role}", rate, role_index)
            else:
                raise ValueError(f"unsupported evaluation process {self.process_id!r}")
            values[role] = value
        self.trace.append({"week": week, **values})
        return values

    def manifest(self) -> dict[str, object]:
        return {
            "process_id": self.process_id,
            "burst_start": self.burst_start,
            "burst_length": self.burst_length if self.process_id == BURST_COLLAPSE_ID else None,
            "trace": list(self.trace),
        }
