# Wholesaler benchmark

This directory has one public leaderboard: 20-week serial Beer Game episodes
on the 100 frozen demand sequences in `eval/held_out_seeds.json`. The policy
controls the wholesaler; retailer, distributor, and factory are fixed
base-stock counterparties. Total cost is supply-chain holding plus backlog cost
and is reported as mean ± standard error across the same 100 episodes.

| Model | Mean total cost | Score / 100 | Bullwhip | Format failures |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-4B (untuned) | 20,452.36 ± 13.18 | 7.80 | 1.821 | 0.0% |
| Qwen3.5-4B + LoRA | 1,632.25 ± 40.99 | 51.47 | 1.821 | 0.0% |
| Claude Opus 5 (zero-shot) | 1,754.48 ± 61.14 | 49.66 | 1.821 | 0.0% |

The score is:

`100 × naive_mean_cost / (naive_mean_cost + policy_mean_cost)`

The naive wholesaler orders its last observed downstream order. It has mean
cost 1,730.82 and therefore scores 50. A zero-cost policy approaches 100; a
policy worse than naive remains positive rather than being clipped. The
base-stock level is tuned on train seeds only. The frozen evaluation seeds are
never used for training, data generation, or tuning.

The LoRA adapter is bf16, rank 16, alpha 16, and targets `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Opus is evaluated
zero-shot through OpenRouter with a compact state tuple, strict one-integer JSON
response, reasoning disabled, a 16-token completion cap, same-week request
deduplication, and response caching. This keeps cost low and makes an
interrupted run restart-safe without changing the model's decision condition.

`baseline.json` is the machine-readable source for the serial leaderboard.
Historical robustness and live-Y research records are retained as supplementary
artifacts and are not additional public leaderboards.
