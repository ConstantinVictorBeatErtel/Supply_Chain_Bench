# Human → model learning (Tier 5 Y wholesaler)

## Topology choice (why wholesaler, Y unchanged)

Keep the **Tier 5 Y network** exactly as in Hub eval:

```text
Customers A → Retailer A ─┐
                          ├→ Wholesaler → Distributor → Factory
Customers B → Retailer B ─┘
```

**Do not** put the human and the LLM in the same live episode. Both play the
**wholesaler** seat in *separate* episodes against the same scripted
counterparties (aggressive retailers + adaptive base-stock upstream in the
headline variant). That preserves:

- identical FOW observations and seeds
- attributable local-cost reward
- exact replay from `(seed, action sequence)`

Learning signal: if humans beat base-stock (or simply provide diverse competent
orders), export those trajectories as supervised demos, then continue with GRPO
on the programmatic wholesaler reward.

## Where / which model

| Step | Where | Default base model |
|---|---|---|
| Human play + logging | Your laptop (Gradio) | — |
| Export / `--dry-run` | Laptop or Colab (CPU OK) | — |
| SFT + GRPO | **Google Colab GPU** (T4+) | **`Qwen/Qwen3-0.6B`** |

Use `--model-name Qwen/Qwen2.5-7B-Instruct` when you have more VRAM (training-gate target).

## One command (preferred)

After you have `human_sessions.jsonl` from the Gradio app:

```bash
# CPU check
PYTHONPATH=environments/beer_distribution_game \
  python3 scripts/run_human_to_model.py \
    --sessions human_sessions.jsonl \
    --dry-run --beat-base-stock

# Full train on Colab GPU
PYTHONPATH=environments/beer_distribution_game \
  python3 scripts/run_human_to_model.py \
    --sessions human_sessions.jsonl \
    --beat-base-stock
```

That runs **export → SFT → GRPO** (skips GRPO under `--dry-run`).  
Colab notebook: [`notebooks/colab_human_to_model.ipynb`](../notebooks/colab_human_to_model.ipynb).

## Pipeline (what the one command does)

```text
Human Gradio app (wholesaler)
    → sessions.jsonl
    → export (BeerEpisode replay → chat JSONL)
    → SFT LoRA warm-start
    → GRPO on programmatic wholesaler reward
```

### Collect (laptop)

```bash
cd environments/beer_distribution_game
PYTHONPATH=. python3 human_app/app.py
```

Optional: set `HF_TOKEN` / `HF_DATASET_REPO` so `CommitScheduler` publishes
`sessions.jsonl`.

Underlying scripts remain available: `export_human_sft.py`,
`train_colab_sft_wholesaler.py`, `train_colab_grpo_wholesaler.py`.  
See also [`LLM_RL_RUNBOOK.md`](LLM_RL_RUNBOOK.md) for the Prime-RL two-GPU path.

## What is intentionally out of scope here

- Live human↔LLM shared control of one wholesaler seat
- Changing Y edges or letting humans play retailers for this learning cell
- Training on abandoned / incomplete sessions
