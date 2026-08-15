"""Generate deterministic browser-native benchmark replay data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from supplychainbench.suites import build_episode, episode_jobs

ROOT = Path(__file__).resolve().parents[1]


def _load_actions(path: Path, seed: str) -> list[int]:
    payload = json.loads(path.read_text())
    for row in payload.get("episodes", []):
        if str(row.get("seed")) == seed:
            return [int(value) for value in row.get("actions", [])]
    raise ValueError(f"seed {seed} not found in {path}")


def generate(output: Path, *, seed: str | None = None) -> dict[str, Any]:
    jobs = episode_jobs("standard")
    sources = {
        "adaptive baseline": ROOT / "results/standard/agent-adaptive.json",
        "untrained Qwen": ROOT / "results/standard/qwen3.5-4b-untrained.json",
        "trained Qwen": ROOT / "results/standard/qwen3.5-4b-grpo.json",
    }
    if seed is None:
        available = []
        for path in sources.values():
            payload = json.loads(path.read_text())
            available.append({str(row["seed"]) for row in payload.get("episodes", []) if len(row.get("actions", [])) == 36})
        common = sorted(set.intersection(*available))
        if not common:
            raise ValueError("no shared replay seed has actions for all three policies")
        seed = common[0]
    job = next((item for item in jobs if item.seed == seed), None)
    if job is None:
        raise ValueError(f"unknown standard seed {seed!r}")
    models: list[dict[str, Any]] = []
    for label, path in sources.items():
        actions = _load_actions(path, job.seed)
        episode = build_episode(job)
        observation = episode.start()
        frames = [{"week": 0, "inventory": observation["state"]["inventory_on_hand"], "backlog": observation["state"]["backlog"], "order": None, "cost": 0.0}]
        cumulative = 0.0
        for action in actions:
            result = episode.place_order(action)
            transition = episode.operational_transitions[-1]
            state = transition["states_after_fulfillment"]["wholesaler"]
            cumulative += float(transition["local_costs"]["wholesaler"])
            frames.append({"week": int(transition["week"]), "inventory": state["inventory"], "backlog": state["backlog"], "order": int(action), "cost": cumulative})
            if result.get("done"):
                break
            observation = result["next_observation"]
        models.append({"label": label, "actions": actions, "frames": frames,
                       "local_total_cost": episode.outcome["grade"]["primary"]["local_total_cost"]})
    payload = {"schema_version": "1.0.0", "suite": "standard", "seed": job.seed,
               "models": models}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "static_web/public/data/benchmark-replay.json")
    parser.add_argument("--seed")
    args = parser.parse_args()
    payload = generate(args.output, seed=args.seed)
    print(f"wrote {args.output} ({payload['seed']})")


if __name__ == "__main__":
    main()
