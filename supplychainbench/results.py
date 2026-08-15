"""Standard result schema, validation, aggregation, and atomic persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

RESULT_SCHEMA_VERSION = "1.0.0"

METRIC_DEFINITIONS = {
    "standard_score": "100 * mean(reference_cost) / mean(policy_cost) across clean episodes",
    "episode_cost": "controlled wholesaler holding + backlog cost including settlement and terminal exposure",
    "weekly_regret": "controlled weekly local cost minus aware-reference weekly local cost",
    "post_event_regret_auc": "sum(max(0, weekly_regret)) from the hidden event week onward",
}


class ResultValidationError(ValueError):
    pass


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _stderr(values: list[float]) -> float | None:
    return statistics.stdev(values) / (len(values) ** 0.5) if len(values) > 1 else (0.0 if values else None)


def aggregate_episode_rows(rows: list[dict[str, Any]], *, reference_costs: dict[str, float] | None = None) -> dict[str, Any]:
    clean = [row for row in rows if row.get("protocol_clean") and row.get("local_total_cost") is not None]
    costs = [float(row["local_total_cost"]) for row in clean]
    refs = [float(reference_costs[str(row["seed"])]) for row in clean] if reference_costs else [float(row["reference_cost"]) for row in clean if row.get("reference_cost") is not None]
    scores = [100.0 * ref / cost for ref, cost in zip(refs, costs) if cost > 0]
    failures = [row for row in rows if not row.get("protocol_clean")]
    out = {
        "episodes_attempted": len(rows),
        "protocol_clean_episodes": len(clean),
        "protocol_failure_episodes": len(failures),
        "protocol_failure_rate": (len(failures) / len(rows)) if rows else None,
        "protocol_coverage": (len(clean) / len(rows)) if rows else None,
        "mean_local_cost": _mean(costs),
        "stderr_local_cost": _stderr(costs),
        "mean_reference_cost_on_clean": _mean(refs),
        "normalized_score": (100.0 * _mean(refs) / _mean(costs)) if costs and refs and _mean(costs) else None,
        "mean_per_episode_score": _mean(scores),
        "stderr_per_episode_score": _stderr(scores),
        "protocol_clean_seeds": sorted(str(row["seed"]) for row in clean),
        "failed_seeds": sorted(str(row.get("seed")) for row in failures),
    }
    recovery_rows = [row for row in clean if "recovery_censored" in row.get("metrics", {})]
    if recovery_rows:
        out["recovery_censor_rate"] = sum(bool(row["metrics"].get("recovery_censored")) for row in recovery_rows) / len(recovery_rows)
    for key in ("recovery_time", "post_event_regret_auc", "adaptation_improvement"):
        values = [float(row["metrics"][key]) for row in clean if row.get("metrics", {}).get(key) is not None]
        if values:
            out[f"mean_{key}"] = _mean(values)
            out[f"stderr_{key}"] = _stderr(values)
    return out


def validate_result(payload: dict[str, Any], *, expected_suite: str | None = None, expected_seeds: set[str] | None = None) -> None:
    if not isinstance(payload, dict):
        raise ResultValidationError("result must be a JSON object")
    required = {"schema_version", "benchmark", "model", "suite", "episodes", "aggregate", "run", "configuration"}
    missing = sorted(required - set(payload))
    if missing:
        raise ResultValidationError(f"missing required fields: {missing}")
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultValidationError(f"unsupported result schema {payload['schema_version']!r}")
    suite = payload["suite"]
    if not isinstance(suite, dict) or not isinstance(suite.get("id"), str):
        raise ResultValidationError("suite.id is required")
    if expected_suite and suite["id"] != expected_suite:
        raise ResultValidationError(f"suite mismatch: {suite['id']!r} != {expected_suite!r}")
    episodes = payload["episodes"]
    if not isinstance(episodes, list):
        raise ResultValidationError("episodes must be a list")
    seeds = [str(row.get("seed")) for row in episodes if isinstance(row, dict)]
    if len(seeds) != len(set(seeds)):
        raise ResultValidationError("duplicate episode seeds")
    if expected_seeds is not None and not set(seeds).issubset(expected_seeds):
        raise ResultValidationError("result contains seeds outside the frozen manifest")
    for index, row in enumerate(episodes):
        if not isinstance(row, dict) or not isinstance(row.get("seed"), str):
            raise ResultValidationError(f"episode {index} is missing seed")
        if not isinstance(row.get("protocol_clean"), bool):
            raise ResultValidationError(f"episode {index} protocol_clean must be bool")
        if row["protocol_clean"] and row.get("local_total_cost") is None:
            raise ResultValidationError(f"clean episode {index} has no local_total_cost")
        if not row["protocol_clean"] and not row.get("failure"):
            raise ResultValidationError(f"failed episode {index} must include failure")
    aggregate = payload["aggregate"]
    if not isinstance(aggregate, dict):
        raise ResultValidationError("aggregate must be an object")
    recomputed = aggregate_episode_rows(episodes)
    for key in ("episodes_attempted", "protocol_clean_episodes", "protocol_failure_episodes"):
        if aggregate.get(key) != recomputed[key]:
            raise ResultValidationError(f"aggregate.{key} does not match episode rows")
    if aggregate.get("normalized_score") is not None and recomputed.get("normalized_score") is not None:
        if abs(float(aggregate["normalized_score"]) - float(recomputed["normalized_score"])) > 1e-7:
            raise ResultValidationError("aggregate.normalized_score does not match episode rows")
    if "protocol_clean_seeds" in aggregate and aggregate["protocol_clean_seeds"] != recomputed["protocol_clean_seeds"]:
        raise ResultValidationError("aggregate.protocol_clean_seeds does not match episode rows")
    if "failed_seeds" in aggregate and aggregate["failed_seeds"] != recomputed["failed_seeds"]:
        raise ResultValidationError("aggregate.failed_seeds does not match episode rows")
    if "protocol_clean_seeds" in payload and payload["protocol_clean_seeds"] != recomputed["protocol_clean_seeds"]:
        raise ResultValidationError("protocol_clean_seeds does not match episode rows")
    if "failures" in payload:
        failure_seeds = sorted(str(row.get("seed")) for row in payload["failures"] if isinstance(row, dict))
        if failure_seeds != recomputed["failed_seeds"]:
            raise ResultValidationError("failures does not match episode rows")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
