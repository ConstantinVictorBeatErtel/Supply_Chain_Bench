#!/usr/bin/env python3
"""Teacher-free Qwen LoRA RL for the exact public Tier-5 Y game.

The implementation reuses the native trajectory PPO/GRPO pilot, but replaces
its public development/validation tasks with the separately frozen experiment
split in ``experiments/live_y_qwen_rl/splits.json``.  Training advantages use
only terminal *local* wholesaler cost; no expert actions or base-stock reward
are used.  Evaluation enables the native paired base-stock reference solely for
reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.train_colab_grpo_wholesaler as pilot
from scripts.live_y_integrity import (
    PROTOCOL_ID,
    SPLIT_ID,
    SPLIT_PATH,
    assert_policy_scenario,
    assert_seed_use,
    load_seed_splits,
    parse_policy_action,
    seeds_for,
)
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import system_prompt
from beer_distribution_game.scenario import scenario_for

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class LocalTask:
    data: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("train", "eval", "held-out-test"),
        default="train",
        help="eval is research-development; held-out-test is one-shot final reporting",
    )
    parser.add_argument(
        "--adapter",
        help=(
            "Optional adapter to evaluate; omit for the base model. "
            "Training always creates a fresh, teacher-free live-Y adapter."
        ),
    )
    parser.add_argument("--output-dir", default="/workspace/outputs/beer-wholesaler-qwen35-4b-live-y-rl")
    parser.add_argument("--updates", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--train-minibatch", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument(
        "--allow-held-out-test",
        action="store_true",
        help="Required with --mode held-out-test after model selection is frozen.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tasks_for(seed_hexes: list[str] | tuple[str, ...], split_name: str) -> list[LocalTask]:
    prototype = scenario_for(5, "development", 0, "headline")
    tasks: list[LocalTask] = []
    for index, seed_hex in enumerate(seed_hexes):
        scenario = prototype.to_dict()
        scenario.update({"split": f"research_{split_name}", "seed_index": index, "master_seed_hex": seed_hex})
        spec = type(prototype)(**{**scenario, "roles": tuple(scenario["roles"])})
        assert_policy_scenario(spec)
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
    splits = load_seed_splits()

    if args.allow_held_out_test and args.mode != "held-out-test":
        raise ValueError("--allow-held-out-test is only valid with --mode held-out-test")
    if args.mode == "held-out-test" and not args.allow_held_out_test:
        # Access the registry here, before the model or CUDA is loaded.
        seeds_for("held_out_test", allow_held_out_test=False)

    evaluation_purpose = (
        "held_out_test" if args.mode == "held-out-test" else "development_evaluation"
    )
    evaluation_seeds = seeds_for(
        evaluation_purpose,
        allow_held_out_test=args.allow_held_out_test,
    )

    def load_tasks(_split: str, _seeds: list[int], tier5_controls: bool = False) -> list[LocalTask]:
        if tier5_controls:
            raise ValueError("Tier-5 control variants are not part of live-Y RL training")
        if args.mode == "train":
            seeds = assert_seed_use(splits["train"], "training_rollout")
            return tasks_for(seeds, "train")
        seeds = assert_seed_use(
            evaluation_seeds,
            evaluation_purpose,
            allow_held_out_test=args.allow_held_out_test,
        )
        return tasks_for(seeds, "held_out_test" if args.mode == "held-out-test" else "development")

    def development_tasks_for_training(_pilot_args: Any) -> list[LocalTask]:
        seeds = assert_seed_use(splits["research_development"], "development_evaluation")
        return tasks_for(seeds, "development")

    def start_episode(task: LocalTask, group_id: int) -> Any:
        spec = pilot.scenario_from_dict(task.data.scenario)
        assert_policy_scenario(spec, task.data.controlled_role)
        include_reference = spec.split != "research_train"
        episode = BeerEpisode(spec, task.data.controlled_role, include_reference=include_reference)
        return pilot.EpisodeRun(
            group_id=group_id,
            task_name=task.data.name,
            episode=episode,
            observation=episode.start(),
        )

    def extract_quantity(text: str) -> int | None:
        return parse_policy_action(text)

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
        # The native grade does not expose this convenience field; every
        # accepted action advances exactly one 36-week decision boundary.
        row["completed_weeks"] = len(run.actions)
        return row

    def load_policy(pilot_args: Any) -> tuple[Any, Any]:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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
        if args.mode == "train":
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    target_modules=list(LORA_TARGETS),
                    bias="none",
                ),
            )
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.print_trainable_parameters()
            model.config.use_cache = False
        else:
            if args.adapter:
                model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
            model.config.use_cache = True
        return model, tokenizer

    def pilot_args() -> Any:
        return SimpleNamespace(
            model_name="Qwen/Qwen3.5-4B",
            output_dir=args.output_dir,
            updates=args.updates,
            group_size=args.group_size,
            train_seeds=list(range(len(splits["train"]))),
            eval_seeds=list(range(len(evaluation_seeds))),
            eval_split=("held_out_test" if args.mode == "held-out-test" else "research_development"),
            tier5_controls=False,
            eval_only=args.mode != "train",
            allow_base_eval=True,
            adapter=args.adapter,
            dry_run=args.dry_run,
            seed=args.seed,
            max_new_tokens=8,
            prompt_max_tokens=2048,
            train_minibatch=args.train_minibatch,
            inference_minibatch=4,
            learning_rate=args.learning_rate,
            temperature=0.7,
            top_p=0.95,
            no_4bit=True,
        )

    pilot.load_tasks = load_tasks
    pilot.development_tasks_for_training = development_tasks_for_training
    pilot.start_episode = start_episode
    pilot.extract_quantity = extract_quantity
    pilot.assign_advantages = assign_advantages
    pilot.episode_summary = episode_summary
    pilot.load_policy = load_policy
    pilot.parse_args = pilot_args


def main() -> None:
    args = parse_args()
    load_seed_splits()
    configure_pilot(args)
    pilot.main()
    (Path(args.output_dir) / "live_y_rl_config.json").write_text(
        json.dumps(
            {
                "environment": "v0.2.0 Tier-5 Y headline, wholesaler, 36 weeks",
                "model": "Qwen/Qwen3.5-4B",
                "precision": "bf16",
                "quantization": "none",
                "lora": {"rank": 16, "alpha": 16, "targets": list(LORA_TARGETS)},
                "training_reward": "group-relative negative terminal local wholesaler cost; protocol failures = -100000",
                "teacher": "none; fresh LoRA initialized from the base model",
                "split_id": SPLIT_ID,
                "split_file": str(SPLIT_PATH.relative_to(ROOT)),
                "seed_purpose": (
                    "training_rollout"
                    if args.mode == "train"
                    else "held_out_test"
                    if args.mode == "held-out-test"
                    else "development_evaluation"
                ),
                "policy_protocol_id": PROTOCOL_ID,
                "policy_response": "strict JSON object with only integer quantity in [0, 128]",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
