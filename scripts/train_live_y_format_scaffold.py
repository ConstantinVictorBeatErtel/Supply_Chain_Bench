#!/usr/bin/env python3
"""Train the optional 256-example format-normalization LoRA scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not 200 <= len(rows) <= 500:
        raise ValueError(f"format scaffold must contain 200-500 records, got {len(rows)}")
    for row in rows:
        meta = row.get("meta", {})
        if meta.get("source") != "base-model-self-generated-format-normalization":
            raise ValueError("format scaffold contains non-self-generated data")
        if meta.get("protocol_id") not in (None, "live-y-domain-randomized-grpo-v1"):
            raise ValueError("format scaffold contains another protocol")
    report = {
        "method": "self-generated format-normalization SFT; not policy supervision",
        "model": "Qwen/Qwen3.5-4B",
        "precision": "bf16",
        "quantization": "none",
        "lora": {"rank": 16, "alpha": 16, "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "learning_rate": 1e-5,
        "epochs": 1,
        "max_example_presentations": 256,
        "examples": len(rows),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError as exc:
        raise SystemExit(f"SFT dependencies are required on the GPU runtime: {exc}")
    if not torch.cuda.is_available():
        raise SystemExit("format scaffold training requires CUDA")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    dataset = Dataset.from_list(rows)
    dataset = dataset.map(lambda row: {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)}, remove_columns=dataset.column_names)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        peft_config=LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none", task_type="CAUSAL_LM"),
        args=TrainingArguments(output_dir=str(args.output_dir), learning_rate=1e-5, num_train_epochs=1, max_steps=256, per_device_train_batch_size=1, bf16=True, report_to=[], save_strategy="no"),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(args.output_dir / "adapter")
    (args.output_dir / "scaffold_config.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
