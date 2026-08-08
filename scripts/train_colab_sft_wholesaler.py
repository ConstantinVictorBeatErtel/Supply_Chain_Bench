#!/usr/bin/env python3
"""Supervised fine-tune a wholesaler LoRA on exported human demo JSONL.

Warm-start step before GRPO (`scripts/train_colab_grpo_wholesaler.py --adapter ...`).
Requires GPU + transformers/peft/trl in the training runtime. Use --dry-run to
validate dataset shape without loading a model.

Example:

    PYTHONPATH=environments/beer_distribution_game \\
      python3 scripts/train_colab_sft_wholesaler.py \\
        --data data/human_demos/wholesaler_sft.jsonl \\
        --output-dir outputs/beer-wholesaler-sft \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parents[1]
_ENV = _ROOT / "environments" / "beer_distribution_game"
for path in (_ENV, _ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.live_y_integrity import assert_training_data_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/beer-wholesaler-sft"))
    p.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        help="Base model (default Qwen3-0.6B for Colab T4; use Qwen2.5-7B-Instruct when VRAM allows)",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument(
        "--require-live-y-train-seeds",
        action="store_true",
        help=(
            "Require every example to carry a seed from the frozen live-Y train split. "
            "Development and held-out-test seeds are always rejected if detected."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-4bit", action="store_true")
    return p.parse_args()


def load_examples(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" not in row:
                raise ValueError(f"{path}:{line_no}: missing messages")
            roles = [m.get("role") for m in row["messages"]]
            if roles != ["system", "user", "assistant"]:
                raise ValueError(f"{path}:{line_no}: expected system/user/assistant")
            meta = row.get("meta") or {}
            if meta.get("role") not in (None, "wholesaler"):
                raise ValueError(f"{path}:{line_no}: non-wholesaler demo")
            rows.append(row)
    if not rows:
        raise ValueError(f"no SFT examples in {path}")
    return rows


def dry_run_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = sorted({(r.get("meta") or {}).get("seed") for r in rows})
    sessions = sorted({(r.get("meta") or {}).get("session_uuid") for r in rows})
    return {
        "examples": len(rows),
        "unique_seeds": len([s for s in seeds if s is not None]),
        "unique_sessions": len([s for s in sessions if s is not None]),
        "sample_assistant": rows[0]["messages"][2]["content"],
        "sample_week": (rows[0].get("meta") or {}).get("week"),
    }


def train(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("SFT training requires a CUDA GPU runtime.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(example: dict[str, Any]) -> dict[str, str]:
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    dataset = Dataset.from_list(rows).map(to_text)

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if not args.no_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        max_length=args.max_seq_len,
        dataset_text_field="text",
        report_to=[],
        seed=args.seed,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "adapter"))
    meta_path = args.output_dir / "sft_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "examples": len(rows),
                "data": str(args.data),
                "adapter": str(args.output_dir / "adapter"),
                "next_step": (
                    "python3 scripts/train_colab_grpo_wholesaler.py "
                    f"--adapter {args.output_dir / 'adapter'}"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    rows = load_examples(args.data)
    assert_training_data_rows(
        rows,
        require_live_y_split=args.require_live_y_train_seeds,
    )
    report = dry_run_report(rows)
    print(json.dumps({"dry_run": args.dry_run, **report}, indent=2))
    if args.dry_run:
        return 0
    train(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
