# How the Qwen3.5-4B wholesaler policy was trained

The board entry **Qwen3.5-4B GRPO (20.64)** is a LoRA adapter over
`Qwen/Qwen3.5-4B`, trained with a critic-free group-relative policy update on
the live-Y research protocol. This document is the recipe. For *why* the first
attempt failed, see [`LIVE_Y_RL_POSTMORTEM.md`](LIVE_Y_RL_POSTMORTEM.md); for
the compute-efficiency machinery, [`LIVE_Y_EFFICIENCY.md`](LIVE_Y_EFFICIENCY.md).

Weights: `artifacts/live_y_best_adapter/` and the Hugging Face Hub
(`beer-game-wholesaler-qwen3.5-4b-grpo-lora`).

## Result

| | untrained | trained |
|---|---:|---:|
| capacity-400 board score (16 held-out seeds) | 6.73 | **20.64** |
| board mean local cost | 4264.7 | **1391.8** |
| development-seed local cost (4 seeds) | 2360.6 | **1430.9** |
| protocol-clean episodes on the board | 16/16 | **16/16** |

Improvement on every bucket, including the three never trained on:

| bucket | untrained-era adapter | this adapter |
|---|---:|---:|
| `in_distribution` | 3524 | 1305 |
| `canonical_held_out_step` | 3648 | 1337 |
| `burst_and_collapse` | 3506 | 1378 |
| `shifted_mean_doubled_variance` | 4677 | 1548 |

## What the policy learns

One habit dominates the cost: **over-ordering**. The base model orders far more
than it must ship, inventory accumulates, and the 0.5/unit-week holding charge
swamps everything. Mean weekly order fell from ~22.9 at update 1 to ~17.5 by
the end, against an obligation near 16/week — essentially onto the adaptive
base-stock reference of 16.7.

`inventory_position` and `on_order` are both in the observation, so this was
never an observability problem. The base model can see the supply line and does
not use it.

## The loop

One update is: roll out, grade each decision, reweight the tokens that carried
it.

1. **Roll out.** Take `seeds_per_update` training seeds; play each `group_size`
   times. Group members share scenario, demand seed, and counterparty streams
   (common random numbers), so they differ only through sampling. Each week the
   policy sees a role-local JSON observation and emits `{"quantity": N}`.
2. **Grade.** Each decision gets a windowed return-to-go, normalized against its
   groupmates at the same timestep.
3. **Update.** A token-level clipped importance-ratio step on the LoRA
   parameters only. The prefix KV cache is cleared afterwards, since the weights
   have moved and cached keys/values are stale.

### Credit window — the change that mattered

An order clears order delay (1 week) and shipment delay (2 weeks) and washes out
within about 6. So a decision is charged for `W = 6` weeks of downstream local
cost, not the whole remaining episode:

```
G[i,t] = -sum_{u=t}^{min(t+W, H)-1} gamma^(u-t) * c[i,u]
         + [t+W >= H] * gamma^(H-t) * terminal
```

Settlement and terminal exposure are attributed only to decisions whose window
reaches the horizon — the same causality rule as the weekly costs.

Unbounded return-to-go (the original) made `G[i,t]` a per-trajectory constant:
once two group members diverged in inventory they paid that difference every
remaining week, and a same-timestep group mean cannot remove it. Measured on the
archived rollouts, **only 9.9% of the advantage variance was per-turn** and 91%
of weeks merely echoed the episode verdict. A 6-week window restores that to
**73.6%**.

Reproduce the comparison offline, no GPU:

```bash
python scripts/replay_live_y_credit_window.py
```

### Advantage

Same-timestep group mean and standard deviation:

```
A[i,t] = (G[i,t] - mean_g) / sd_g
```

Normalization matters because raw advantages were in cost units (sd ~148, max
~483) sharing a path with a -1e5 protocol-failure penalty, leaving the effective
step size at the mercy of gradient-norm clipping. Decisions whose window
contains a protocol failure bypass normalization and take a fixed -5.0: group
normalization would otherwise rescale -1e5 down to about -1.7, indistinguishable
from an ordinary bad week.

### Objective — token-level, double-sided

For each scored token `k`, with `r = exp(logp_new - logp_old)`:

```
s   = min(r * A, clip(r, 1-eps, 1+eps) * A)
loss = -mean_k( A < 0 ? max(s, c * A) : s )
```

`eps = 0.2`, dual-clip `c = 3`. The dual clip bounds the surrogate below for
negative advantages, where the single-sided minimum is unbounded and one badly
off-policy token can dominate a minibatch.

**Only the digits of the order are scored.** `{"quantity": 48}` is ~8 tokens and
one carries the decision; averaging log-probabilities over the completion let
boilerplate outvote it, so the ratio barely responded and the clip never
engaged. Scoring the digits alone took `clip_fraction` from ~0 to 15–30%, and
cut scored tokens from ~8 to ~1.5 per decision.

## Configuration

| | |
|---|---|
| base | `Qwen/Qwen3.5-4B`, bf16, frozen |
| adapter | LoRA r=16, alpha=16, dropout 0, on q/k/v/o/gate/up/down |
| updates | 16 |
| seeds per update | 8 (cycling the 16 training seeds) |
| group size | 6 → 48 trajectories/update, 768 total |
| learning rate | 1e-5, AdamW, grad-norm clip 1.0 |
| train minibatch | 4 |
| credit window | 6 weeks, discount 1.0 |
| decoding | temperature 0.7, top-p 0.95, 32-token cap |
| training bucket | `in_distribution` only |
| seed | 20260808 |
| hardware | 1× A40, 15.4 h, ~$7.50 |

## Reproducing

```bash
python scripts/train_live_y_domain_randomized_grpo_v1.py \
  --updates 16 --group-size 6 --seeds-per-update 8 \
  --credit-window 6 --checkpoint-every 1 \
  --output-dir /workspace/outputs/live-y-run
```

Requires CUDA. `--credit-window 0` restores the unbounded return-to-go of the
archived run; `--all-completion-tokens` restores whole-completion scoring.

Evaluate against the board:

```bash
python scripts/eval_live_y_domain_randomized_grpo_v1_models.py \
  --adapter artifacts/live_y_best_adapter \
  --label my_run --output artifacts/live_y_capacity_400/evaluations/my_run.json
python scripts/rebuild_live_y_capacity_400_leaderboard.py
python scripts/render_capacity_400_scoreboard.py
```

Checkpointing writes `adapter/` (latest), `adapter_best/` (lowest training cost
among protocol-clean updates) and `checkpoints/update_NNN/` every update, so the
stopping point can be chosen after seeing the whole curve. Two helper scripts
support unattended runs: `sync_runpod_artifacts.sh` mirrors outputs with byte
verification, and `finalize_runpod_run.sh` tears the pod down only after a
verified copy.

## Stopping point, and why it matters

Training does **not** converge monotonically — it over-optimizes.

An earlier 20-update run at the same hyperparameters reached a *lower* training
cost (732 at update 14) and then fell apart: it overshot into under-ordering
(13.0 orders/week at update 16, cost back up to 3016) and then into malformed
JSON, ending with a post-training development evaluation of
`protocol_clean_rate 0.0` — a completely broken policy. Its rolling checkpoint
had overwritten the good adapter, so nothing was recoverable.

The 16-update run kept every update and stopped before the collapse. Its final
update was both the cheapest and the one the development evaluation measured.

Practical consequences, both now enforced in the trainer:

- **Keep every checkpoint.** The right stopping point is only visible after the
  curve is.
- **Watch `clip_fraction`.** It climbed 17% → 44% ahead of the collapse; the
  policy was moving further off-policy each update.
- **Watch `protocol_clean_rate` per update.** It degrades before the output
  becomes unusable.

## Honest limits

- **Single seed.** Both runs used `20260808`, so run-to-run variance is
  unmeasured. Two runs at one seed are not two samples.
- **The margin over a blind baseline is narrow.** The best constant-order policy
  that never reads the observation scores **19.80**; this adapter scores 20.64.
  It beats its own base model 3×, but "it reasons about the supply chain" is not
  established by 0.84 points.
- **The training distribution is partly degenerate.** Scripted retailers carry a
  +8/week ordering floor exceeding per-retailer demand across the whole training
  λ range, so the wholesaler's incoming orders sit near a constant ~16/week and
  the demand signal is largely masked. The policy was never required to react to
  demand during training. Only the `shifted_mean_doubled_variance` bucket breaks
  this.
- **Development evaluation is 4 seeds**, in-distribution only.
