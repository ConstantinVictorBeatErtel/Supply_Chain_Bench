# LLM RL runbook

This is the first actual RL experiment for the Verifiers environment. The
Akash DeepSeek and Qwen runs were zero-shot baselines; they did not update model
weights. This run trains one Qwen3-0.6B LoRA adapter with GRPO from the
programmatic wholesaler reward.

Human demos for the same wholesaler seat can warm-start the adapter via
[`HUMAN_TO_MODEL.md`](HUMAN_TO_MODEL.md) (`export_human_sft.py` →
`train_colab_sft_wholesaler.py` → this GRPO script with `--adapter`).

## Frozen experiment definition

- Model: `Qwen/Qwen3-0.6B`
- Trainable parameters: one LoRA adapter for the controlled wholesaler only
- Algorithm: Prime-RL GRPO
- Environment: v0.2.0, Tier 5 Y topology, headline proportional-rationing variant
- Split: development seeds 0–2 only
- Reward: controlled wholesaler cost score, with protocol failure scored as zero
- Group size: 8
- Batch size: 16 rollouts
- Maximum: 10 updates for the smoke; extend to 50 only after reviewing the smoke
- Learning rate: `5e-6`
- Context: `seq_len=8192`, environment horizon 36 weeks, `max_turns=38`
- LoRA: rank 8, alpha 16, dropout 0

The authoritative launcher config is
`configs/prime_rl/beer_wholesaler_qwen3_0p6b_smoke.toml`.

The Colab pilot uses the same taskset scenario generation, observation JSON,
integer action bounds, deterministic counterparties, settlement, and grader,
but replaces the OpenAI-compatible tool-call transport with a local strict JSON
action serializer so rollouts and training can share one GPU. Therefore its
first result is an RL-signal pilot, not yet a claim of native Hub tool-call
protocol performance. A selected checkpoint must receive a separate native
Verifiers harness evaluation before publication.

## Hardware constraint

The current Prime-RL launcher partitions GPUs between the inference server and
trainer. This configuration therefore requests two NVIDIA GPUs. The local Mac
has no CUDA device, so the RL run cannot be honestly launched from this
workspace. A one-GPU Colab runtime is insufficient for this Prime-RL setup;
use a two-GPU runtime or a Prime/Akash compute allocation. The Akash model API
key is an inference credential, not a GPU-training credential.

Prime-RL's current checkout should be pinned before installation. The checked
current commit during this audit was `256809b2ecb92fb9776c25a3bddfbe3d6c934861`.
The environment repository commit must also be recorded after the current
working tree is committed; do not train from an unrecorded dirty tree.

## Two-GPU launch sketch

From a Linux Python 3.12 GPU machine:

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
git checkout 256809b2ecb92fb9776c25a3bddfbe3d6c934861
git submodule update --init --recursive
uv sync --all-extras --all-packages
uv pip install --no-deps -e /path/to/beer_distribution_rl/environments/beer_distribution_game
```

First validate the environment import and task count:

```bash
uv run python - <<'PY'
from verifiers.v1.loaders import taskset_config_type
from beer_distribution_game import BeerTaskset
cfg = taskset_config_type("beer-distribution-game")(
    id="beer-distribution-game", split="development", tiers=[5],
    controlled_roles=["wholesaler"], seed_limit=3,
)
print(BeerTaskset(cfg).load().__len__())
PY
```

Then run a config dry-run before allocating GPU time:

```bash
uv run rl @ /path/to/beer_distribution_rl/configs/prime_rl/beer_wholesaler_qwen3_0p6b_smoke.toml \
  --output-dir outputs/beer-wholesaler-y-qwen3-0p6b-smoke \
  --dry-run
```

After the dry-run passes, remove `--dry-run` to launch the 10-update
development smoke. Keep the output directory unique and preserve the resolved
`rl.toml`, `trainer.toml`, `orchestrator.toml`, `inference.toml`, logs, and
adapter checkpoint.

## Evaluation gates after training

1. Evaluate the base Qwen3-0.6B and the trained adapter on development seeds
   0–2, with the same tool protocol and deterministic scripted counterparties.
2. Check protocol-clean rate, completed weeks, local cost, cost score,
   immediate fill, bullwhip, and order-cap hits.
3. Stop if protocol-clean rate is below 95%, ordering collapses to zero/cap,
   or no cost/order signal appears by update 10.
4. Only if the smoke is non-degenerate, extend to 50 updates using development
   seeds only.
5. Freeze the selected checkpoint, then evaluate once on the five validation
   seeds and both Tier-5 controls. Do not tune on validation.

The publication comparison is trained-small-model-before/after RL versus the
frozen zero-shot DeepSeek/Qwen baselines, adaptive base-stock, and random.
