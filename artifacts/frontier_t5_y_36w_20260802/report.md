# Frontier OpenRouter Tier-5 Y wholesaler evaluation

Native environment `0.2.0`; 36 operational weeks plus native settlement; Tier-5 Y topology; controlled role `wholesaler`; scripted counterparties; temperature 0; reasoning disabled; strict `place_order` tool calls only.

Frozen split: `eval/frontier_t5_y_36w_100_seed_split.json` (100 seeds), created before OpenRouter requests. No training, public live-game, or prior benchmark seeds were used.

`raw = (naive_cost - model_cost) / (naive_cost - oracle_cost)` and `score = clamp(10 + 90 * raw, 0, 100)`. Naive therefore scores 10 and oracle scores 100.

Mean and stderr are across completed episodes with a native local-cost grade. Format failures are counted at the response level; a failed repair terminates that episode under the native protocol-error rule and is reported separately.

| Model | OpenRouter ID | Mean local wholesaler cost ± stderr | Format failures | Raw | Score |
|---|---|---:|---:|---:|---:|
| Naive | deterministic `incoming_demand_or_order` | 1453.050 ± 56.976 | 0 | 0.000 | 10.000 |
| Oracle | `adaptive_base_stock_v2` | 776.020 ± 57.754 | 0 | 1.000 | 100.000 |
| kimi_k3 | `moonshotai/kimi-k3` | 1346.410 ± 57.561 | 113 / 3450 (3.28%) | 0.158 | 24.176 |
| claude_opus_5 (skipped: OpenRouter HTTP 402 insufficient credits before 100-episode completion) | `anthropic/claude-opus-5` | — | — | — | — |
| muse_spark_1_1 (skipped: reasoning mandatory (HTTP 400); subsequent requests also hit HTTP 402 insufficient credits) | `meta/muse-spark-1.1` | — | — | — | — |
| grok_4_5 (skipped: OpenRouter HTTP 402 insufficient credits) | `x-ai/grok-4.5` | — | — | — | — |
| gpt_5_6_sol (skipped: OpenRouter HTTP 402 insufficient credits) | `openai/gpt-5.6-sol` | — | — | — | — |

Skipped models were not assigned cost, raw, score, or format-failure metrics. Their raw request errors and any partial valid calls remain in `results.json` and the per-model JSONL logs.

Raw request/response JSONL is stored under `raw/`; per-episode actions and grades are under `episodes/`; the aggregate JSON is `results.json`.
