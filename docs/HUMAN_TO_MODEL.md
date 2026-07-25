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

## Pipeline

```text
Human Gradio app (wholesaler)
    → sessions.jsonl  (seed + actions + costs; anonymous UUID)
    → scripts/export_human_sft.py  (BeerEpisode replay → chat JSONL)
    → scripts/train_colab_sft_wholesaler.py  (LoRA SFT warm-start)
    → scripts/train_colab_grpo_wholesaler.py --adapter <sft>  (RL)
    → Hub / Colab eval vs base-stock and human baseline costs
```

### 1. Collect

```bash
cd environments/beer_distribution_game
PYTHONPATH=. python3 human_app/app.py
```

Optional: set `HF_TOKEN` / `HF_DATASET_REPO` so `CommitScheduler` publishes
`sessions.jsonl`.

### 2. Export SFT rows

```bash
PYTHONPATH=environments/beer_distribution_game \
  python3 scripts/export_human_sft.py \
    --input human_sessions.jsonl \
    --output data/human_demos/wholesaler_sft.jsonl \
    --beat-base-stock
```

Each completed session yields 36 chat examples (system + current observation →
`{"quantity": N}`), using [`beer_distribution_game/prompts.py`](../environments/beer_distribution_game/beer_distribution_game/prompts.py)
so labels match the Colab GRPO JSON action format.

### 3. SFT warm-start

```bash
PYTHONPATH=environments/beer_distribution_game \
  python3 scripts/train_colab_sft_wholesaler.py \
    --data data/human_demos/wholesaler_sft.jsonl \
    --output-dir outputs/beer-wholesaler-sft \
    --dry-run   # validate dataset; drop --dry-run on GPU
```

### 4. GRPO continue

```bash
PYTHONPATH=environments/beer_distribution_game \
  python3 scripts/train_colab_grpo_wholesaler.py \
    --adapter outputs/beer-wholesaler-sft/adapter \
    --updates 10
```

See also [`LLM_RL_RUNBOOK.md`](LLM_RL_RUNBOOK.md) for the Prime-RL two-GPU path.

## What is intentionally out of scope here

- Live human↔LLM shared control of one wholesaler seat
- Changing Y edges or letting humans play retailers for this learning cell
- Training on abandoned / incomplete sessions
