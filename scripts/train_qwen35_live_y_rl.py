#!/usr/bin/env python3
"""Teacher-free RL continuation for the exact public Tier-5 Y game.

The implementation reuses the native trajectory PPO/GRPO pilot, but replaces
its public development/validation tasks with the separately frozen experiment
split in ``experiments/live_y_qwen_rl/splits.json``.  Training advantages use
only terminal *local* wholesaler cost; no expert actions or base-stock reward
are used.  Evaluation enables the native paired base-stock reference solely for
reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_colab_grpo_wholesaler as pilot
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import system_prompt
from beer_distribution_game.scenario import scenario_for

SPLITS = ROOT / "experiments" / "live_y_qwen_rl" / "splits.json"
ADAPTER = "/workspace/outputs/beer-wholesaler-qwen35-4b-lora/adapter"
INTEGER_RE = re.compile(r"\s*(\d{1,3})\s*")


@dataclass
class LocalTask:
    data: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--adapter", default=ADAPTER)
    parser.add_argument("--output-dir", default="/workspace/outputs/beer-wholesaler-qwen35-4b-live-y-rl")
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_seed_split() -> dict[str, list[str]]:
    payload = json.loads(SPLITS.read_text())
    train = [str(value) for value in payload["train"]]
    evaluation = [str(value) for value in payload["eval"]]
    if len(train) != 16 or len(evaluation) != 10 or set(train).intersection(evaluation):
        raise ValueError("live-Y RL split must contain disjoint 16 train and 10 eval seeds")
    for name, values in (("train", train), ("eval", evaluation)):
        for index, value in enumerate(values):
            expected = hashlib.sha256(f"beer-live-y-rl-v1|{name}|{index:05d}".encode()).hexdigest()[:16]
            if value != expected:
                raise AssertionError(f"{name} seed {index} no longer matches its frozen derivation")
    return {"train": train, "eval": evaluation}


def tasks_for(seed_hexes: list[str], split_name: str) -> list[LocalTask]:
    prototype = scenario_for(5, "development", 0, "headline")
    tasks: list[LocalTask] = []
    for index, seed_hex in enumerate(seed_hexes):
        scenario = prototype.to_dict()
        scenario.update({"split": f"research_{split_name}", "seed_index": index, "master_seed_hex": seed_hex})
        spec = type(prototype)(**{**scenario, "roles": tuple(scenario["roles"])})
        tasks.append(
            LocalTask(
                SimpleNamespace(
                    name=f"t5-strategic-y-v2:wholesaler:research_{split_name}:{index}",
                    scenario=spec.to_dict(),
                    controlled_role="wholesaler",
                    system_prompt=system_prompt(spec, "wholesaler"),
                )
            )
        )
    return tasks


def configure_pilot(args: argparse.Namespace) -> None:
    splits = load_seed_split()

    def load_tasks(_split: str, _seeds: list[int], tier5_controls: bool = False) -> list[LocalTask]:
        if tier5_controls:
            raise ValueError("Tier-5 control variants are not part of live-Y RL training")
        return tasks_for(splits["eval"] if args.mode == "eval" else splits["train"], args.mode)

    def start_episode(task: LocalTask, group_id: int) -> Any:
        spec = pilot.scenario_from_dict(task.data.scenario)
        episode = BeerEpisode(spec, task.data.controlled_role, include_reference=args.mode == "eval")
        return pilot.EpisodeRun(
            group_id=group_id,
            task_name=task.data.name,
            episode=episode,
            observation=episode.start(),
        )

    def extract_quantity(text: str) -> int | None:
        parsed = pilot.QUANTITY_RE.search(text)
        if parsed is not None:
            quantity = int(parsed.group(1))
            return quantity if 0 <= quantity <= 128 else None
        bare = INTEGER_RE.fullmatch(text)
        return int(bare.group(1)) if bare is not None and int(bare.group(1)) <= 128 else None

    def assign_advantages(runs: list[Any]) -> None:
        grouped: dict[int, list[tuple[Any, float]]] = {}
        for run in runs:
            grade = run.episode.outcome["grade"] if run.episode.outcome else {}
            local_cost = grade.get("primary", {}).get("local_total_cost")
            reward = -float(local_cost) if local_cost is not None and grade.get("protocol_clean") else -100_000.0
            grouped.setdefault(run.group_id, []).append((run, reward))
        for members in grouped.values():
            mean_reward = sum(value for _, value in members) / len(members)
            for run, reward in members:
                for record in run.records:
                    record.advantage = reward - mean_reward

    original_summary = pilot.episode_summary

    def episode_summary(run: Any) -> dict[str, Any]:
        row = original_summary(run)
        row["reward"] = -float(row["local_total_cost"]) if row["local_total_cost"] is not None and row["protocol_clean"] else -100_000.0
        return row

    def load_policy(pilot_args: Any) -> tuple[Any, Any]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("live-Y RL requires CUDA")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-4B", trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=args.mode == "train")
        if args.mode == "train":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.print_trainable_parameters()
        model.config.use_cache = True
        return model, tokenizer

    def pilot_args() -> Any:
        return SimpleNamespace(
            model_name="Qwen/Qwen3.5-4B",
            output_dir=args.output_dir,
            updates=args.updates,
            group_size=args.group_size,
            train_seeds=list(range(len(splits["train"]))),
            eval_seeds=list(range(len(splits["eval"]))),
            eval_split="validation",
            tier5_controls=False,
            eval_only=args.mode == "eval",
            adapter=args.adapter,
            dry_run=args.dry_run,
            seed=args.seed,
            max_new_tokens=8,
            prompt_max_tokens=2048,
            train_minibatch=2,
            inference_minibatch=4,
            learning_rate=args.learning_rate,
            temperature=0.7,
            top_p=0.95,
            no_4bit=True,
        )

    pilot.load_tasks = load_tasks
    pilot.start_episode = start_episode
    pilot.extract_quantity = extract_quantity
    pilot.assign_advantages = assign_advantages
    pilot.episode_summary = episode_summary
    pilot.load_policy = load_policy
    pilot.parse_args = pilot_args


def main() -> None:
    args = parse_args()
    load_seed_split()
    configure_pilot(args)
    pilot.main()


if __name__ == "__main__":
    main()
