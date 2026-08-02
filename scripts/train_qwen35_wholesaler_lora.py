from unsloth import FastLanguageModel

import json
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


MAX_SEQ_LENGTH = 2048
ROOT = Path("/workspace/beer_distribution_RL")
OUT = Path("/workspace/outputs/beer-wholesaler-qwen35-4b-lora")


def render_text(row: dict) -> dict:
    """Render the fixed, no-thinking prompt without depending on chat-template sentinels."""
    user, assistant = row["messages"]
    return {
        "text": (
            f"<|im_start|>user\n{user['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant['content']}<|im_end|>"
        )
    }


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=MAX_SEQ_LENGTH,
)
model.print_trainable_parameters()

dataset = load_dataset(
    "json",
    data_files=str(ROOT / "data/wholesaler_base_stock_train.jsonl"),
    split="train",
)
dataset = dataset.map(render_text, remove_columns=dataset.column_names)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=SFTConfig(
        output_dir=str(OUT),
        dataset_text_field="text",
        max_length=MAX_SEQ_LENGTH,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=3407,
        dataset_num_proc=1,
    ),
)
trainer.train()
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(OUT / "adapter"))
tokenizer.save_pretrained(str(OUT / "adapter"))
(OUT / "training_meta.json").write_text(json.dumps({
    "model": "Qwen/Qwen3.5-4B",
    "load_in_4bit": False,
    "bf16": True,
    "r": 16,
    "alpha": 16,
    "examples": len(dataset),
}, indent=2))
print("TRAINING_OK")
