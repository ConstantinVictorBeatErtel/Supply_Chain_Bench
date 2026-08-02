# Wholesaler LoRA benchmark

All policies run the wholesaler seat for 20 weeks on the same 100 frozen held-out demand seeds in `eval/held_out_seeds.json`. Retailer, distributor, and factory use scripted base-stock policies. The total-cost value is the supply-chain holding plus backlog cost, reported as mean ± standard error.

| Policy | Mean total cost | Benchmark score | Bullwhip | Format failures |
| --- | ---: | ---: | ---: | ---: |
| Naive (last observed order) | 1730.82 ± 58.12 | 50.00 | 1.821 | 0.0% |
| Tuned base stock | 1624.97 ± 43.39 | 51.58 | 1.821 | 0.0% |
| Qwen3.5-4B, untuned | 20452.36 ± 13.18 | 7.80 | 1.821 | 0.0% |
| GPT-5.6 Terra, zero-shot | 1952.84 ± 78.46 | 46.99 | 1.821 | 0.0% |
| Qwen3.5-4B, LoRA | 1632.25 ± 40.99 | 51.47 | 1.821 | 0.0% |

The 0–100 benchmark score is `100 * naive_cost / (naive_cost + policy_cost)`: the naive policy anchors at 50, a zero-cost policy approaches 100, and worse policies remain positive rather than being clipped to zero.

The LoRA adapter uses bf16 (not 4-bit), rank 16, alpha 16, and targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. It is evaluated locally with batched `generate()`; GPT-5.6 Terra is the only API policy.

## Independent robustness replication

`eval/robustness_seeds.json` is a separately frozen set of 100 sequences, derived from `random.Random(20260805)` and created after training. It is never used for data generation, training, or base-stock tuning. The same policies, 20-week horizon, and score formula are used; the resulting replication is recorded in `results/robustness.json`.

| Policy | Mean total cost | Benchmark score | Bullwhip | Format failures |
| --- | ---: | ---: | ---: | ---: |
| Naive (last observed order) | 1717.55 ± 58.60 | 50.00 | 1.889 | 0.0% |
| Tuned base stock | 1612.23 ± 43.58 | 51.58 | 1.889 | 0.0% |
| Qwen3.5-4B, untuned | 20468.56 ± 6.61 | 7.74 | 1.889 | 0.0% |
| GPT-5.6 Terra, zero-shot | 1913.81 ± 78.61 | 47.30 | 1.889 | 0.0% |
| Qwen3.5-4B, LoRA | 1619.47 ± 41.38 | 51.47 | 1.889 | 0.0% |
