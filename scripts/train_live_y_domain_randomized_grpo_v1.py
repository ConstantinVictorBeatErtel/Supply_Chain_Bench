#!/usr/bin/env python3
"""Live-Y domain-randomized critic-free GRPO research runner.

Additive wrapper around the single-GPU multi-turn group-relative trainer.
Uses the research prompt (no demand-law / factory-capacity leak), research
``capacity=400`` feasible-supply override, and a longer wraparound seed
schedule (defaults: 12 updates × 8 seeds × group 6).

Do not launch a paid GPU job from this script unless the user explicitly
approves; see ``experiments/live_y_feasible_supply_grpo_v2/DESIGN.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "environments" / "beer_distribution_game"
for path in (ROOT, ENV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.train_colab_grpo_wholesaler as pilot
from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import system_prompt
from beer_distribution_game.scenario import scenario_from_dict
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1 import PROTOCOL_ID
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.advantages import (
    DEFAULT_CREDIT_WINDOW,
    DEFAULT_DISCOUNT,
    assign_group_advantages,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import (
    ResearchTask,
    derive_research_seed,
    research_spec,
)
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import parse_completion
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import (
    research_observation_user_message,
)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/workspace/outputs/live-y-domain-randomized-grpo-v1")
    p.add_argument(
        "--updates",
        type=int,
        default=12,
        help="Optimizer updates. Each update rolls out seeds_per_update training seeds × group_size.",
    )
    p.add_argument("--group-size", type=int, default=6)
    p.add_argument(
        "--seeds-per-update",
        type=int,
        default=8,
        help="How many of the 16 training seeds to roll out each update (cycled).",
    )
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=20260808)
    p.add_argument("--adapter", help="format-scaffold adapter; omit for base cold start")
    p.add_argument(
        "--credit-window",
        type=int,
        default=DEFAULT_CREDIT_WINDOW,
        help=(
            "Weeks of downstream cost charged to one decision. 0 restores the "
            "unbounded return-to-go used by the archived two-update run."
        ),
    )
    p.add_argument("--credit-discount", type=float, default=DEFAULT_DISCOUNT)
    p.add_argument(
        "--raw-advantages",
        action="store_true",
        help="Skip group-standard-deviation normalization of the advantages.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save the adapter every N updates; 0 saves only at the end.",
    )
    p.add_argument(
        "--no-keep-all-checkpoints",
        dest="keep_all_checkpoints",
        action="store_false",
        help="Keep only the latest and best adapters instead of one per update.",
    )
    p.add_argument("--clip-epsilon", type=float, default=pilot.DEFAULT_CLIP_EPSILON)
    p.add_argument("--dual-clip", type=float, default=pilot.DEFAULT_DUAL_CLIP)
    p.add_argument(
        "--all-completion-tokens",
        action="store_true",
        help="Score every completion token instead of only the digits of the order.",
    )
    p.add_argument("--generation-token-cap", type=int, default=32)
    p.add_argument("--kv-cache-min-prefix-tokens", type=int, default=64)
    p.add_argument("--disable-prefix-kv-cache", action="store_true")
    p.add_argument("--smoke", action="store_true", help="one training seed, one group, one update")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def make_tasks(indices: list[int], bucket: str = "in_distribution") -> list[ResearchTask]:
    rows: list[ResearchTask] = []
    for index in indices:
        seed = derive_research_seed("training", index)
        spec = research_spec(seed, bucket=bucket, index=index)
        rows.append(
            ResearchTask(
                name=f"{PROTOCOL_ID}:training:{index}",
                scenario=spec.to_dict(),
            )
        )
    return rows


def configure(requested_args: argparse.Namespace) -> None:
    count = 1 if requested_args.smoke else 16
    all_tasks = make_tasks(list(range(count)))
    original_rollout = pilot.rollout_batch
    update_number = [0]

    def load_tasks(_split: str, _seeds: list[int], tier5_controls: bool = False) -> list[Any]:
        if tier5_controls:
            raise ValueError("research protocol has no Tier-5 control variants")
        return [SimpleNamespace(data=task) for task in all_tasks]

    def development_tasks_for_training(_args: Any) -> list[Any]:
        # Development seeds are distinct from training and are never passed to
        # rollout_batch when it is collecting training data.
        return [
            SimpleNamespace(
                data=ResearchTask(
                    name=f"{PROTOCOL_ID}:development:{index}",
                    scenario=research_spec(
                        derive_research_seed("development", index),
                        bucket="in_distribution",
                        index=index,
                    ).to_dict(),
                )
            )
            for index in range(4)
        ]

    def start_episode(task: Any, group_id: int) -> Any:
        spec = scenario_from_dict(task.data.scenario)
        include_reference = "development" in task.data.name
        episode = BeerEpisode(spec, task.data.controlled_role, include_reference=include_reference)
        return pilot.EpisodeRun(group_id, task.data.name, episode, episode.start())

    def extract_quantity(text: str) -> int | None:
        return parse_completion(text, tokenizer=getattr(pilot, "_research_tokenizer", None))

    def prompt_text(task: Any, observation: dict[str, Any], tokenizer: Any) -> str:
        messages = [
            {"role": "system", "content": task.data.system_prompt},
            {"role": "user", "content": research_observation_user_message(observation)},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def assign_advantages(runs: list[Any]) -> None:
        diagnostics = assign_group_advantages(
            runs,
            window=requested_args.credit_window or None,
            discount=requested_args.credit_discount,
            normalize=not requested_args.raw_advantages,
        )
        if not hasattr(pilot, "_research_advantage_diagnostics"):
            pilot._research_advantage_diagnostics = []
        pilot._research_advantage_diagnostics.append(diagnostics)

    original_summary = pilot.episode_summary

    def episode_summary(run: Any) -> dict[str, Any]:
        row = original_summary(run)
        row["protocol_id"] = PROTOCOL_ID
        row["weekly_local_costs"] = [
            float(t["local_costs"][run.episode.controlled_role])
            for t in run.episode.operational_transitions
        ]
        row["return_to_go"] = [record.return_to_go for record in run.records]
        row["per_turn_advantages"] = [record.advantage for record in run.records]
        row["per_turn_baselines"] = [record.group_baseline for record in run.records]
        return row

    def rollout_batch(model: Any, tokenizer: Any, tasks: list[Any], args: Any, group_size: int, sample: bool) -> list[Any]:
        if sample and not requested_args.smoke:
            # Cycle through the 16 train seeds so longer runs keep seeing every seed.
            width = max(1, min(int(requested_args.seeds_per_update), len(tasks)))
            start = (update_number[0] * width) % len(tasks)
            selected = [tasks[(start + offset) % len(tasks)] for offset in range(width)]
            update_number[0] += 1
        else:
            selected = tasks
        return original_rollout(model, tokenizer, selected, args, group_size, sample)

    def load_policy(_args: Any) -> tuple[Any, Any]:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("research GRPO requires CUDA; no GPU is available on this host")
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        pilot._research_tokenizer = tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-4B", trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
        )
        if requested_args.adapter:
            model = PeftModel.from_pretrained(model, requested_args.adapter, is_trainable=True)
        else:
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
        model.config.use_cache = False
        return model, tokenizer

    def make_pilot_args() -> Any:
        return SimpleNamespace(
            model_name="Qwen/Qwen3.5-4B",
            output_dir=requested_args.output_dir,
            updates=1 if requested_args.smoke else requested_args.updates,
            group_size=requested_args.group_size,
            train_seeds=list(range(count)),
            eval_seeds=list(range(4)),
            eval_split="research_development",
            tier5_controls=False,
            eval_only=False,
            adapter=requested_args.adapter,
            dry_run=requested_args.dry_run,
            seed=requested_args.seed,
            max_new_tokens=requested_args.generation_token_cap,
            prompt_max_tokens=2048,
            train_minibatch=4,
            inference_minibatch=4,
            learning_rate=requested_args.learning_rate,
            temperature=0.7,
            top_p=0.95,
            no_4bit=True,
            checkpoint_every=requested_args.checkpoint_every,
            keep_all_checkpoints=requested_args.keep_all_checkpoints,
            clip_epsilon=requested_args.clip_epsilon,
            dual_clip=requested_args.dual_clip,
            all_completion_tokens=requested_args.all_completion_tokens,
            kv_cache_min_prefix_tokens=requested_args.kv_cache_min_prefix_tokens,
            disable_prefix_kv_cache=requested_args.disable_prefix_kv_cache,
        )

    pilot.load_tasks = load_tasks
    pilot.development_tasks_for_training = development_tasks_for_training
    pilot.start_episode = start_episode
    pilot.extract_quantity = extract_quantity
    pilot.prompt_text = prompt_text
    pilot.assign_advantages = assign_advantages
    pilot.episode_summary = episode_summary
    pilot.rollout_batch = rollout_batch
    pilot.load_policy = load_policy
    pilot.parse_args = make_pilot_args


def main() -> None:
    requested = args()
    configure(requested)
    pilot.main()
    out = Path(requested.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "research_protocol.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "method": "critic-free multi-turn group-relative PPO/GRPO-style update",
                "sft": "optional self-generated format-normalization scaffolding only",
                "demand_process": "episode_randomized_y_poisson_v1",
                "updates": 1 if requested.smoke else requested.updates,
                "group_size": requested.group_size,
                "scheduled_training_indices": [0] if requested.smoke else list(range(16)),
                "decoding": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "generation_token_cap": requested.generation_token_cap,
                    "protocol_max_completion_tokens": 192,
                },
                "prompt": {
                    "system": "research_system_prompt",
                    "observation": "research_observation_user_message",
                    "hides_factory_capacity": True,
                    "hides_demand_parameters": True,
                },
                "environment": {
                    "amendment_id": "live-y-feasible-supply-grpo-v2",
                    "factory_capacity": 400,
                    "initial_inventory": 12,
                    "initial_shipment_pipeline": 4,
                    "initial_order_pipeline": 4,
                    "aggressive_retailers": True,
                    "design_note": "experiments/live_y_feasible_supply_grpo_v2/DESIGN.md",
                },
                "schedule": {
                    "updates": 1 if requested.smoke else requested.updates,
                    "group_size": requested.group_size,
                    "seeds_per_update": requested.seeds_per_update,
                    "learning_rate": requested.learning_rate,
                    "seed_cycling": "wraparound",
                    "max_trajectories": (
                        requested.group_size
                        if requested.smoke
                        else requested.updates * requested.seeds_per_update * requested.group_size
                    ),
                },
                "efficiency": {
                    "prefix_kv_cache": not requested.disable_prefix_kv_cache,
                    "kv_cache_min_prefix_tokens": requested.kv_cache_min_prefix_tokens,
                    "rolling_history_window": 8,
                    "bf16": True,
                    "lora": True,
                    "gradient_checkpointing": True,
                    "training_forward_use_cache": False,
                },
                "reward": "negative local wholesaler return-to-go; no system/team cost",
                "credit_assignment": {
                    "window_weeks": requested.credit_window or None,
                    "discount": requested.credit_discount,
                    "rationale": (
                        "order delay 1 + shipment delay 2, so one order's cost "
                        "footprint lands within ~3 weeks and washes out by ~6"
                    ),
                },
                "surrogate": {
                    "ratio": "per-token",
                    "clip_epsilon": requested.clip_epsilon,
                    "dual_clip": requested.dual_clip,
                    "scored_tokens": (
                        "all completion tokens"
                        if requested.all_completion_tokens
                        else "digits of the order only"
                    ),
                },
                "per_turn_normalization": (
                    "none; same-timestep mean baseline only"
                    if requested.raw_advantages
                    else "same-timestep group mean and standard deviation"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
