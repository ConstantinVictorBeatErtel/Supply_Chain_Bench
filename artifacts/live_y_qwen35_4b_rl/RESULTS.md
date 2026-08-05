# Teacher-free Qwen3.5-4B LoRA — Tier-5 live Y game

This is a development-only research result, not a replacement for either the
100-seed frontier Tier-5 evaluation or the 100-seed serial LoRA benchmark.

The policy controlled the **wholesaler** for 36 operational weeks plus native
settlement in the Tier-5 Y-headline scenario. It was trained on 16 dedicated
research seeds and evaluated on 10 distinct dedicated research seeds from
[`experiments/live_y_qwen_rl/splits.json`](../../experiments/live_y_qwen_rl/splits.json).
Those seed sets are SHA-256-derived and disjoint from the public live game,
normative development/validation/test splits, and prior benchmarks.

| Policy | Mean local cost ± stderr | Score (0–100) | Protocol-clean |
|---|---:|---:|---:|
| Naive (last observed incoming order) | 1,284.80 ± 72.67 | 50.00 | 100% |
| Adaptive base-stock | 601.60 ± 72.54 | 68.11 | 100% |
| Qwen3.5-4B, base | 1,579.69 ± 232.46 | 44.85 | 100% |
| Qwen3.5-4B, teacher-free LoRA | **1,096.31 ± 134.66** | **53.96** | **100%** |

The adapter reduced mean local wholesaler cost by **30.60%** against its
untuned base model on this evaluation condition. The score is computed from
the policy mean cost, with the same non-clipping formula used by the serial
LoRA benchmark:

`score = 100 × naive_mean_cost / (naive_mean_cost + policy_mean_cost)`

Thus the naive policy is 50; lower cost gives a higher score. The compact,
machine-readable record is [`summary.json`](summary.json). The Runpod output
directory retains the adapter, full per-run evaluation JSON, rollout trace, and
training log at the paths recorded in that summary.
