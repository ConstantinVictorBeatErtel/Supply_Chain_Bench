---
base_model: Qwen/Qwen3.5-4B
library_name: peft
license: apache-2.0
tags:
  - lora
  - reinforcement-learning
  - grpo
  - supply-chain
  - beer-distribution-game
---

# Beer Distribution Game — wholesaler policy (Qwen3.5-4B LoRA, GRPO)

A LoRA adapter that plays the **wholesaler** seat in a stochastic Y-topology
beer distribution game under order/shipment delay and partial observability.
Trained with a critic-free, group-relative (GRPO-style) multi-turn policy
update.

Each week the policy sees a role-local JSON observation (inventory on hand,
backlog, on-order pipeline, inventory position, incoming retailer orders, recent
history) and emits exactly one order: `{"quantity": <int 0..128>}`.

## Results

Evaluated on 4 held-out **development** seeds not used in training, greedy
decoding:

| | untrained `Qwen3.5-4B` | this adapter |
|---|---:|---:|
| mean local cost (36 weeks + settlement) | 2360.6 | **1430.9** |
| protocol-clean rate | 1.00 | **1.00** |
| immediate fill rate | 0.740 | 0.763 |

**39.4% reduction in local holding + backlog cost**, with output-format
compliance fully intact.

Training-rollout cost fell from 3443 to 1511 over 16 updates. Protocol-clean
rate was 1.00 at every update except update 4 (0.979); the policy was still
improving when training stopped.

The learned behaviour is a correction to chronic **over-ordering**: the base
model orders far more than it must ship, so inventory accumulates and holding
cost dominates.

## What this adapter is *not*

- **Not evaluated on the project's 16-seed held-out benchmark board.** The
  number above is 4 development seeds from the in-distribution bucket only. Do
  not quote it as a benchmark score.
- **Single training seed (20260808).** Run-to-run variance has not been
  measured. An earlier run with the same hyperparameters reached a lower
  training cost and then collapsed into malformed output; that failure mode is
  real and this adapter is one sample, not a characterised method.
- **Trained only on the in-distribution demand bucket.** The step-change,
  overdispersed, and burst/collapse buckets are held out by design and were
  never seen.
- **The training environment has a known degeneracy.** Scripted retailers carry
  a +8/week ordering floor that exceeds per-retailer demand across the whole
  training range, so the wholesaler's incoming order stream sits near a constant
  ~16/week and the demand signal is largely masked. The policy was therefore
  never required to react to demand during training.

## Training

| | |
|---|---|
| base | `Qwen/Qwen3.5-4B`, bf16, frozen |
| adapter | LoRA r=16, alpha=16, dropout=0, on q/k/v/o/gate/up/down |
| algorithm | critic-free group-relative policy update (GRPO-style) |
| updates | 16 × 8 seeds × group 6 = 768 trajectories |
| credit window | 6 weeks of downstream local cost per decision |
| advantage | same-timestep group mean and standard deviation |
| objective | per-token clipped ratio, ε=0.2, dual-clip c=3 |
| scored tokens | digits of the order only |
| decoding | temperature 0.7, top-p 0.95, 32-token cap |
| hardware | 1× A40, ~15 h |

Two design choices matter more than the rest:

**Bounded credit window.** An order clears order delay (1 week) plus shipment
delay (2 weeks) and washes out within ~6. Charging a decision for cost through
the end of the episode makes the advantage a per-trajectory constant — measured
at only ~10% per-turn variance, i.e. episode-level REINFORCE wearing a per-turn
costume. A 6-week window restores that to ~73%.

**Token-level surrogate.** `{"quantity": 48}` is ~8 tokens and only one carries
the decision. Averaging log-probabilities over the completion lets boilerplate
outvote the order, so the ratio barely responds and the clip never engages.
Scoring the digits alone fixed this: the clip engages on ~15–30% of scored
tokens.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "<this-repo>")
```

The policy expects the research system prompt and observation format from the
source repository; it is not a general-purpose chat adapter.

## Provenance

`MANIFEST.json` records the selected update, SHA-256 of the weights, full
hyperparameters, and the development-eval numbers above.
