#!/usr/bin/env python3
"""Generate 256 self-selected format examples from training seeds only.

This command intentionally never calls a reference policy: an integer is
recovered from the base model output and then re-serialized unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "environments" / "beer_distribution_game"):
    sys.path.insert(0, str(path))

from beer_distribution_game.episode import BeerEpisode
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.environment import make_tasks
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.protocol import permissive_intended_quantity, serialize_completion
from beer_distribution_rl.research.live_y_domain_randomized_grpo_v1.prompting import research_observation_user_message
from beer_distribution_game.prompts import sft_chat_example


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=256)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"GPU generation dependencies are required: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("base-model SFT generation requires CUDA; no GPU is available")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    rows = []
    dropped = 0
    for task in make_tasks(count=16, bucket="in_distribution"):
        from beer_distribution_game.scenario import scenario_from_dict
        scenario = scenario_from_dict(task.scenario)
        episode = BeerEpisode(scenario, "wholesaler", include_reference=False)
        observation = episode.start()
        episode.research_observations = []
        while not episode.done and len(rows) < args.target:
            episode.research_observations.append(observation)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": task.system_prompt}, {"role": "user", "content": research_observation_user_message(observation)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=192)
            raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            quantity = permissive_intended_quantity(raw)
            if quantity is None:
                dropped += 1
                episode.protocol_failure_outcome(error_count=1, category="no_recoverable_integer")
                break
            rows.append(sft_chat_example(scenario, "wholesaler", observation, quantity))
            rows[-1]["messages"][-1]["content"] = serialize_completion(quantity)
            rows[-1]["meta"].update({"source": "base-model-self-generated-format-normalization", "protocol_id": "live-y-domain-randomized-grpo-v1"})
            result = episode.place_order(quantity)
            if not result["done"]:
                observation = result["next_observation"]
    if len(rows) < args.target:
        raise SystemExit(f"only recovered {len(rows)} records; dropped {dropped}; continue training seeds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows[: args.target]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps({"records": args.target, "dropped_no_recoverable_integer": dropped, "source": "base-model-self-generated-format-normalization", "training_seeds_only": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
