#!/usr/bin/env python3
"""Small one-GPU GRPO/LoRA pilot for the Tier-5 Y wholesaler.

This is intentionally separate from PRIME-RL's two-process launcher. A hosted
Colab runtime normally gives one GPU, so rollouts and the LoRA update share the
same Transformers model in this pilot. The environment transition and grading
are still the native BeerEpisode implementation; only the local action
serializer emits JSON which is converted to place_order(quantity).

The script is development-only by default. It never constructs validation or
test tasks during training. Use --eval-only for a held-out evaluation after a
checkpoint has been selected.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Colab T4 runs can retain large unused CUDA allocator segments after batched
# generation.  Expandable segments reduce fragmentation; setting this before
# importing torch is required for it to take effect.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

try:
    import torch
    from torch.nn.utils.rnn import pad_sequence
except ModuleNotFoundError:  # Allows --dry-run with only the environment installed.
    torch = None  # type: ignore[assignment]
    pad_sequence = None  # type: ignore[assignment]

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_game.prompts import observation_user_message
from beer_distribution_game.scenario import scenario_from_dict
QUANTITY_RE = re.compile(r'"quantity"\s*:\s*(-?\d+)')


@dataclass
class ActionRecord:
    prompt_ids: list[int]
    completion_ids: list[int]
    group_id: int
    raw_text: str
    quantity: int | None
    valid: bool
    advantage: float = 0.0
    old_logprob: float = 0.0
    return_to_go: float | None = None
    group_baseline: float | None = None


@dataclass
class EpisodeRun:
    group_id: int
    task_name: str
    episode: BeerEpisode
    observation: dict[str, Any] | None
    actions: list[int] = field(default_factory=list)
    raw_outputs: list[str] = field(default_factory=list)
    records: list[ActionRecord] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="outputs/beer-wholesaler-qwen3-0p6b-colab")
    p.add_argument("--updates", type=int, default=10)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--train-seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--eval-split", choices=["development", "validation"], default="validation")
    p.add_argument("--tier5-controls", action="store_true")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--adapter", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--seed", type=int, default=20260718)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prompt-max-tokens", type=int, default=4096)
    p.add_argument(
        "--train-minibatch",
        type=int,
        default=2,
        help="Per-forward training batch; 2 is conservative for a Colab T4.",
    )
    p.add_argument(
        "--inference-minibatch",
        type=int,
        default=8,
        help="No-grad batch for old-policy log probabilities.",
    )
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--no-4bit", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tasks(split: str, seeds: list[int], tier5_controls: bool = False) -> list[Any]:
    # Taskset is the only import in this script that requires Verifiers.  Keep
    # it lazy so split/protocol dry-runs and integrity tests remain CPU-only.
    from beer_distribution_game.taskset import BeerTaskset, BeerTasksetConfig

    if not seeds or min(seeds) < 0:
        raise ValueError("seeds must be non-empty non-negative integers")
    max_seed = max(seeds)
    cfg = BeerTasksetConfig(
        id="beer-distribution-game",
        split=split,
        tiers=[5],
        controlled_roles=["wholesaler"],
        seed_limit=max_seed + 1,
        tier5_controls=tier5_controls,
    )
    tasks = BeerTaskset(cfg).load()
    selected = []
    for task in tasks:
        seed = int(task.data.name.rsplit(":", 1)[-1])
        if seed in set(seeds) and (tier5_controls or task.data.scenario.get("variant") == "headline"):
            selected.append(task)
    expected = len(seeds) * (3 if tier5_controls else 1)
    if len(selected) != expected:
        raise RuntimeError(f"expected {expected} task rows, found {len(selected)}")
    return selected


def development_tasks_for_training(args: argparse.Namespace) -> list[Any]:
    """Tasks used for pre/post training diagnostics.

    Experiment wrappers with dedicated seed registries override this hook.  The
    default retains the original Colab pilot behavior for its normative
    development-only smoke configuration.
    """

    return load_tasks("development", args.train_seeds)


def prompt_text(task: Any, observation: dict[str, Any], tokenizer: Any) -> str:
    messages = [
        {"role": "system", "content": task.data.system_prompt},
        {
            "role": "user",
            "content": observation_user_message(observation, action_format="json"),
        },
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


def extract_quantity(text: str) -> int | None:
    match = QUANTITY_RE.search(text)
    if match is None:
        return None
    quantity = int(match.group(1))
    return quantity if 0 <= quantity <= 128 else None


def model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def load_policy(args: argparse.Namespace) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("Install torch, transformers, and peft in the Colab runtime first.")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires a CUDA GPU. Select a Colab GPU runtime.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": dtype,
    }
    if not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=not args.eval_only)
    elif not args.eval_only:
        if not args.no_4bit:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    if not args.eval_only:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    model.config.use_cache = True
    return model, tokenizer


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    args: argparse.Namespace,
    sample: bool,
) -> list[tuple[list[int], list[int], str]]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.prompt_max_tokens,
        add_special_tokens=False,
    )
    device = model_device(model)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            do_sample=sample,
            temperature=args.temperature if sample else 1.0,
            top_p=args.top_p if sample else 1.0,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    width = encoded["input_ids"].shape[1]
    outputs: list[tuple[list[int], list[int], str]] = []
    for row, prompt_len in zip(generated, encoded["attention_mask"].sum(dim=1).tolist()):
        prompt_ids = encoded["input_ids"][len(outputs), -int(prompt_len) :].detach().cpu().tolist()
        completion = row[width:].detach().cpu().tolist()
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in completion:
            completion = completion[: completion.index(tokenizer.eos_token_id) + 1]
        while completion and completion[-1] == tokenizer.pad_token_id:
            completion.pop()
        text = tokenizer.decode(completion, skip_special_tokens=True)
        outputs.append((prompt_ids, completion, text))
    return outputs


def start_episode(task: Any, group_id: int) -> EpisodeRun:
    spec = scenario_from_dict(task.data.scenario)
    episode = BeerEpisode(spec, task.data.controlled_role, include_reference=True)
    return EpisodeRun(
        group_id=group_id,
        task_name=task.data.name,
        episode=episode,
        observation=episode.start(),
    )


def finish_invalid(run: EpisodeRun, category: str) -> None:
    run.episode.protocol_failure_outcome(error_count=1, category=category)
    run.observation = None


def rollout_batch(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    args: argparse.Namespace,
    group_size: int,
    sample: bool,
) -> list[EpisodeRun]:
    task_map = task_lookup(tasks)
    runs: list[EpisodeRun] = []
    for task_index, task in enumerate(tasks):
        for replicate in range(group_size):
            runs.append(start_episode(task, task_index))

    active = list(runs)
    while active:
        prompts = [prompt_text(task_map[run.task_name], run.observation, tokenizer) for run in active]
        generated = generate_batch(model, tokenizer, prompts, args, sample=sample)
        next_active: list[EpisodeRun] = []
        for run, (prompt_ids, completion_ids, raw_text) in zip(active, generated):
            quantity = extract_quantity(raw_text)
            valid = quantity is not None
            run.raw_outputs.append(raw_text)
            if valid:
                result = run.episode.place_order(quantity)
                run.actions.append(quantity)
                run.records.append(
                    ActionRecord(
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        group_id=run.group_id,
                        raw_text=raw_text,
                        quantity=quantity,
                        valid=True,
                    )
                )
                if not result["done"]:
                    run.observation = result["next_observation"]
                    next_active.append(run)
                else:
                    run.observation = None
            else:
                run.records.append(
                    ActionRecord(
                        prompt_ids=prompt_ids,
                        completion_ids=completion_ids,
                        group_id=run.group_id,
                        raw_text=raw_text,
                        quantity=None,
                        valid=False,
                    )
                )
                finish_invalid(run, "invalid_json_action")
        active = next_active
    return runs


def episode_summary(run: EpisodeRun) -> dict[str, Any]:
    grade = run.episode.outcome["grade"] if run.episode.outcome else {}
    return {
        "task": run.task_name,
        "variant": run.episode.spec.variant,
        "reward": float(grade.get("episode_reward") or 0.0),
        "protocol_clean": bool(grade.get("protocol_clean", False)),
        "completed_weeks": int(grade.get("completed_operational_weeks", 0)),
        "local_total_cost": grade.get("primary", {}).get("local_total_cost"),
        "cost_score": grade.get("primary", {}).get("cost_score"),
        "system_total_cost": grade.get("costs", {}).get("system_total_cost"),
        "immediate_fill_rate": grade.get("service", {}).get("immediate_fill_rate"),
        "bullwhip_ratio": grade.get("stability", {}).get("bullwhip_ratio"),
        "order_cap_hit_rate": grade.get("stability", {}).get("order_cap_hit_rate"),
        "actions": run.actions,
        "raw_outputs": run.raw_outputs,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_sd(key: str) -> tuple[float | None, float | None]:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            return None, None
        return statistics.mean(values), statistics.pstdev(values)

    out: dict[str, Any] = {"n": len(rows)}
    for key in (
        "reward",
        "local_total_cost",
        "cost_score",
        "system_total_cost",
        "immediate_fill_rate",
        "bullwhip_ratio",
        "order_cap_hit_rate",
    ):
        out[f"{key}_mean"], out[f"{key}_sd"] = mean_sd(key)
    out["protocol_clean_rate"] = statistics.mean(float(row["protocol_clean"]) for row in rows)
    out["completed_weeks_mean"] = statistics.mean(row["completed_weeks"] for row in rows)
    return out


def task_lookup(tasks: list[Any]) -> dict[str, Any]:
    return {task.data.name: task for task in tasks}


def completion_mean_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    prompt_len: int,
    completion_len: int,
) -> torch.Tensor:
    """Mean log probability of completion tokens without prompt-sized softmax."""
    if completion_len == 0:
        return logits.new_zeros(())
    # Position t predicts input_ids[t + 1].  Slice before softmax so the
    # temporary [sequence, vocabulary] log-probability tensor is only as long
    # as the generated completion, not the full prompt.
    start = prompt_len - 1
    completion_logits = logits[start : start + completion_len].float()
    completion_labels = input_ids[prompt_len : prompt_len + completion_len]
    token_logp = torch.log_softmax(completion_logits, dim=-1).gather(
        -1, completion_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logp.mean()


def sequence_logprobs(model: Any, records: list[ActionRecord], minibatch: int) -> torch.Tensor:
    if torch is None or pad_sequence is None:
        raise RuntimeError("Install torch in the Colab runtime first.")
    device = model_device(model)
    values: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(records), minibatch):
        batch = records[start : start + minibatch]
        sequences = [torch.tensor(r.prompt_ids + r.completion_ids, dtype=torch.long) for r in batch]
        lengths = [len(r.prompt_ids) for r in batch]
        completion_lengths = [len(r.completion_ids) for r in batch]
        ids = pad_sequence(sequences, batch_first=True, padding_value=0).to(device)
        mask = torch.zeros_like(ids, dtype=torch.long)
        for row, seq in enumerate(sequences):
            mask[row, : len(seq)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        for row, (prompt_len, completion_len) in enumerate(zip(lengths, completion_lengths)):
            values.append(
                completion_mean_logprob(logits[row], ids[row], prompt_len, completion_len)
                .cpu()
            )
    return torch.stack(values)


def train_update(model: Any, optimizer: Any, records: list[ActionRecord], args: argparse.Namespace) -> dict[str, float]:
    if torch is None or pad_sequence is None:
        raise RuntimeError("Install torch in the Colab runtime first.")
    if not records:
        return {"loss": 0.0, "trainable_actions": 0.0, "mean_advantage": 0.0}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    old = sequence_logprobs(model, records, args.inference_minibatch)
    # Batched generation and the no-grad old-policy pass can leave large
    # reclaimable segments in the CUDA caching allocator.  Return them before
    # constructing the backward graph, which is the peak-memory phase.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for record, value in zip(records, old.tolist()):
        record.old_logprob = float(value)
    trainable = [record for record in records if abs(record.advantage) > 1e-12 and record.completion_ids]
    if not trainable:
        return {"loss": 0.0, "trainable_actions": 0.0, "mean_advantage": 0.0}

    model.train()
    losses: list[float] = []
    for start in range(0, len(trainable), args.train_minibatch):
        batch = trainable[start : start + args.train_minibatch]
        sequences = [torch.tensor(r.prompt_ids + r.completion_ids, dtype=torch.long) for r in batch]
        lengths = [len(r.prompt_ids) for r in batch]
        completion_lengths = [len(r.completion_ids) for r in batch]
        ids = pad_sequence(sequences, batch_first=True, padding_value=0).to(model_device(model))
        mask = torch.zeros_like(ids, dtype=torch.long)
        for row, seq in enumerate(sequences):
            mask[row, : len(seq)] = 1
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        current: list[torch.Tensor] = []
        for row, (prompt_len, completion_len) in enumerate(zip(lengths, completion_lengths)):
            current.append(completion_mean_logprob(logits[row], ids[row], prompt_len, completion_len))
        current_logp = torch.stack(current)
        old_logp = torch.tensor([r.old_logprob for r in batch], device=current_logp.device)
        advantages = torch.tensor([r.advantage for r in batch], device=current_logp.device)
        ratio = torch.exp((current_logp - old_logp).clamp(-5.0, 5.0))
        clipped = torch.clamp(ratio, 0.8, 1.2)
        loss = -(torch.minimum(ratio * advantages, clipped * advantages)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "loss": statistics.mean(losses) if losses else 0.0,
        "trainable_actions": float(len(trainable)),
        "mean_advantage": statistics.mean(record.advantage for record in trainable),
    }


def assign_advantages(runs: list[EpisodeRun]) -> None:
    rewards: dict[int, list[float]] = {}
    for run in runs:
        reward = float(run.episode.outcome["grade"].get("episode_reward") or 0.0)
        rewards.setdefault(run.group_id, []).append(reward)
    baselines = {group: statistics.mean(values) for group, values in rewards.items()}
    for run in runs:
        reward = float(run.episode.outcome["grade"].get("episode_reward") or 0.0)
        advantage = reward - baselines[run.group_id]
        for record in run.records:
            record.advantage = advantage


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def evaluate(model: Any, tokenizer: Any, tasks: list[Any], args: argparse.Namespace) -> dict[str, Any]:
    runs = rollout_batch(model, tokenizer, tasks, args, group_size=1, sample=False)
    rows = [episode_summary(run) for run in runs]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    return {
        "summary": aggregate(rows),
        "summary_by_variant": {variant: aggregate(group) for variant, group in by_variant.items()},
        "episodes": rows,
    }


def main() -> None:
    args = parse_args()
    if args.tier5_controls and not args.eval_only:
        raise ValueError("Tier-5 controls are evaluation-only and cannot enter training.")
    if args.eval_only and not args.adapter and not getattr(args, "allow_base_eval", False):
        raise ValueError("--eval-only requires --adapter.")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        train_tasks: list[Any] = []
        development_tasks: list[Any] = []
        eval_tasks = load_tasks(
            args.eval_split,
            args.eval_seeds,
            tier5_controls=args.tier5_controls,
        )
    else:
        train_tasks = load_tasks("development", args.train_seeds)
        development_tasks = development_tasks_for_training(args)
        eval_tasks = []
    save_json(
        output_dir / "run_config.json",
        {
            "model_name": args.model_name,
            "train_seeds": args.train_seeds,
            "eval_split": args.eval_split,
            "eval_seeds": args.eval_seeds,
            "group_size": args.group_size,
            "updates": args.updates,
            "reward": "native BeerEpisode grade.episode_reward",
            "action_serializer": "strict JSON quantity converted to place_order",
        },
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "train_tasks": [task.data.name for task in train_tasks],
                    "development_tasks": [task.data.name for task in development_tasks],
                    "eval_tasks_constructed": bool(eval_tasks),
                    "eval_tasks": [task.data.name for task in eval_tasks],
                    "output_dir": str(output_dir),
                },
                indent=2,
            )
        )
        return

    print("Loading policy model and LoRA adapter...", flush=True)
    model, tokenizer = load_policy(args)
    print("Policy loaded.", flush=True)
    if args.eval_only:
        print(f"Running {args.eval_split} evaluation...", flush=True)
        result = evaluate(model, tokenizer, eval_tasks, args)
        save_json(output_dir / f"eval_{args.eval_split}.json", result)
        print(json.dumps(result["summary"], indent=2))
        return

    print("Running pre-training development evaluation...", flush=True)
    pre = evaluate(model, tokenizer, development_tasks, args)
    save_json(output_dir / "eval_pre_development.json", pre)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    update_rows: list[dict[str, Any]] = []
    for update in range(1, args.updates + 1):
        print(f"Update {update}/{args.updates}: collecting rollouts...", flush=True)
        runs = rollout_batch(model, tokenizer, train_tasks, args, args.group_size, sample=True)
        assign_advantages(runs)
        records = [record for run in runs for record in run.records]
        print(f"Update {update}/{args.updates}: optimizing {len(records)} action records...", flush=True)
        train_stats = train_update(model, optimizer, records, args)
        episode_rows = [episode_summary(run) for run in runs]
        row = {
            "update": update,
            **train_stats,
            **aggregate(episode_rows),
            "valid_actions": sum(int(record.valid) for record in records),
            "total_actions": len(records),
        }
        update_rows.append(row)
        print(json.dumps(row, sort_keys=True))
        save_json(output_dir / "training_metrics.json", update_rows)
        with (output_dir / "rollouts.jsonl").open("a") as handle:
            for episode in episode_rows:
                handle.write(json.dumps({"update": update, **episode}) + "\n")

    print("Saving LoRA adapter...", flush=True)
    model.config.use_cache = True
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print("Running post-training development evaluation...", flush=True)
    post = evaluate(model, tokenizer, development_tasks, args)
    save_json(output_dir / "eval_post_development.json", post)
    print(json.dumps({"pre": pre["summary"], "post": post["summary"]}, indent=2))


if __name__ == "__main__":
    main()
