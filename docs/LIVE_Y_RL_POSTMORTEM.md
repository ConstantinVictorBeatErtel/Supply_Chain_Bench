# Live-Y GRPO postmortem: why the Qwen fine-tune did not move

Status: diagnosis complete, fixes landed, the corrected training run has **not**
been executed yet. Everything below the "What changed" heading is measured on
the archived artifacts; nothing here claims a new benchmark result.

## The result that started this

Capacity-400 board, sixteen fixed live-Y seeds, protocol-clean episodes only:

| Model | Mean local cost | Score (`100·C*/C̄`) | Mean per-seed score |
|---|---:|---:|---:|
| `Qwen3.5-4B` untrained | 4264.7 ± 545 | 6.73 | 9.99 ± 1.63 |
| `Qwen3.5-4B` GRPO (archived adapter) | 3838.9 ± 301 | 7.48 | 10.08 ± 1.67 |

Both estimators appear on the board and they disagree in sign, which is itself
the finding: the mean-of-per-seed-scores moved by **0.09 against a standard
error of 1.6**. Training produced no measurable improvement. The 10% drop in
mean cost sits inside the noise of either estimator.

For scale, the adaptive base-stock baseline on the same seeds averages 388.84
(score **73.9**) and hindsight-perfect averages 287.22. Every LLM on the board,
including Grok 4.5 at 29.78, is far worse than a short heuristic.

## The behavioural failure mode

One habit accounts for most of the leaderboard ordering: **chronic
over-ordering**. Mean weekly order divided by mean weekly retailer demand, and
the cumulative unit surplus a policy accumulates over 36 weeks:

| Policy | order ÷ demand | cumulative surplus units | Score |
|---|---:|---:|---:|
| Qwen untrained | 2.79 | 602 | 6.7 |
| Qwen GRPO | 2.48 | 492 | 7.5 |
| GPT-5.6 Luna | 1.78 | 242 | 14.8 |
| Grok 4.5 | 1.63 | 167 | 29.8 |

(Untrained ratios are computed on the four seeds whose per-week actions survived
in the eval JSON; the remaining twelve rows were recovered from logs without
their action traces.)

On seed `ba0eef18` (λ = 2.19) the trained model orders 30.7 units/week against
4.50 units/week of demand — 6.8×. Inventory climbs into the hundreds and the
0.5/unit-week holding charge dominates the episode.

This is not an observability problem. `inventory_position` and `on_order` are
both in the research observation; see `prompting.py`. The policy can see the
supply line and does not use it. That is precisely the kind of single-parameter
habit RL should correct quickly — given a signal.

## Root cause 1: the per-turn advantage was not per-turn

`advantages.py` computed `G[i,t]` as the **undiscounted sum of local cost from
week `t` to the end of the episode**, then subtracted the same-timestep mean over
the group. The intent was per-decision credit assignment. Measured on the 64
archived trajectories, it was not:

| | |
|---|---:|
| mean within-trajectory sd of advantage across 36 weeks | 46.0 |
| sd of the trajectory-mean advantage | 138.8 |
| **share of advantage variance that is genuinely per-turn** | **0.099** |
| weeks whose advantage sign merely echoes the episode verdict | 0.911 |
| corr(advantage at week 1, advantage at week 12) | 0.96 |

**90% of the advantage was a per-trajectory constant broadcast to all 36
actions.** The update degenerated into episode-level REINFORCE with a
four-sample baseline. Every action inside a good episode was reinforced,
including the bad ones; every action inside a bad episode was punished,
including the good ones.

The mechanism is state persistence. Group members share a CRN seed but diverge
in inventory. Once member *i* is carrying 200 units more than member *j*, every
remaining week costs *i* roughly 100 more, so `G[i,t] − G[j,t]` stays nearly
constant for the rest of the episode. Subtracting a same-timestep group mean
does not remove that, because the baseline averages over four members standing
in four different states. The signal answers "how did this trajectory get here,"
not "was this week's order good."

Effective learning signal: **64 episode verdicts**, not 2,304 decisions.

## Root cause 2: the decision token was outvoted

`completion_mean_logprob` returned the **mean** per-token log probability of the
completion, and the importance ratio was built from it:

```
r = exp(mean_logp_new − mean_logp_old)
```

`{"quantity": 48}` is roughly eight tokens and exactly one carries the order;
the braces, key, and colon are boilerplate the policy already emits with
probability ≈1. Averaging first means a large change in the order distribution
barely moves `r`, so the surrogate is insensitive to the only thing being
trained and the `[0.8, 1.2]` clip effectively never engages.

## Root cause 3: data starvation

`runpod_final/run_config.json` records `updates: 2`, `group_size: 4`, 16 seeds —
**64 trajectories, one A40, $0.5382**. The launcher's own defaults are 12
updates × 8 seeds × group 6. The optimizer did take ~576 AdamW steps
(`train_minibatch=4` over 1,151 trainable records × 2 updates), so this is not a
step-count problem; the model simply never saw more than 64 episodes of
experience, and 576 steps on a degenerate 64-sample signal is not training.

## Root cause 4: train/eval bucket mismatch

`make_tasks` defaults to `bucket="in_distribution"`, so all 16 training seeds are
in-distribution while 12 of the 16 evaluation seeds are `canonical_held_out_step`,
`shifted_mean_doubled_variance`, or `burst_and_collapse`. The trained model's
worst bucket (4677.2 mean cost) is the one it never trained on.

## Root cause 5 (cosmetic but misleading)

`training_metrics.json` logs `reward_mean: 0.0, reward_sd: 0.0`. That is the
unused episode-level `episode_reward` field from the pilot summary, not the
per-turn path that actually drove the update, and it reads as a dead reward
signal when the signal was in fact present (advantage sd 148, 0/16 groups with
identical action sequences, 0.1% exactly-zero advantages).

## What changed

### Bounded credit window (`advantages.py`)

`return_to_go` now takes `window` and `discount`. An order placed in week `t`
clears the order delay (1 week) and shipment delay (2 weeks) within ~3 weeks and
its effect washes out by ~6, so charging a decision for cost beyond that horizon
is attributing to week `t` the consequences of week `t+10`. Settlement and
terminal exposure are attributed only to turns whose window actually reaches the
horizon — the same causality rule applied to the weekly costs.

`DEFAULT_CREDIT_WINDOW = 6`. `window=None` restores the archived behaviour
exactly, and `scripts/train_live_y_domain_randomized_grpo_v1.py --credit-window 0`
exposes that for reproduction.

Advantages are additionally normalized by the same-timestep group standard
deviation. Previously they were raw cost units (sd ≈ 148, max ≈ 483) sharing a
path with a `-1e5` protocol-failure penalty, which left the effective step size
entirely at the mercy of the 1.0 gradient-norm clip. Because group-relative
normalization bounds a group of `n` to `sqrt(n-1)`, it would also have quietly
rescaled a protocol failure down to ≈ −1.7 — indistinguishable from an ordinary
bad week — so failures bypass normalization and take a fixed `FAILURE_ADVANTAGE
= -5.0`.

### Token-level surrogate (`train_colab_grpo_wholesaler.py`)

The ratio is now per-token, the clip is applied per-token, and the loss is
restricted to the tokens that encode the integer order (found by decoding each
completion token on its own and keeping the ones containing a digit — robust to
tokenizer merges, and it falls back to the whole span for a completion with no
digits, which is a protocol failure that still has to carry its penalty).

A double-sided bound is applied: for negative advantages the single-sided
`min(rA, clip(r)A)` is unbounded below, so one badly off-policy token can
dominate a minibatch. `surrogate = max(surrogate, dual_clip · A)` when `A < 0`,
with `dual_clip = 3.0`.

`train_update` now also reports `scored_tokens` and `clip_fraction`, so the next
run can be checked for whether the clip ever engages instead of assuming it.

### Not changed

LoRA. Rank 16, alpha 16, dropout 0, over q/k/v/o/gate/up/down on
`Qwen/Qwen3.5-4B` in bf16. It was never the bottleneck: the target behaviour is
"order roughly what you sold, and count the pipeline," which is a small
behavioural adjustment rather than new knowledge, and rank 16 across all seven
projections has ample capacity for it.

## Offline validation

`scripts/replay_live_y_credit_window.py` recomputes the advantages from the 64
archived trajectories under alternative windows. The logged rows carry the
weekly local cost and the unbounded return-to-go series, which inverts exactly to
the settlement plus terminal charge; the script asserts that its reconstruction
reproduces the logged series before reporting anything (**max abs error 0.0**).

| Scheme | per-turn variance share | sign echo | corr(w1, w12) |
|---|---:|---:|---:|
| to-end, undiscounted (archived) | 0.099 | 0.911 | 0.96 |
| to-end, discount 0.9 | 0.432 | 0.778 | 0.84 |
| window 12 weeks | 0.601 | 0.697 | 0.47 |
| window 8 weeks | 0.692 | 0.673 | 0.76 |
| **window 6 weeks (new default)** | **0.736** | 0.664 | 0.75 |
| window 4 weeks | 0.778 | 0.656 | 0.65 |
| window 6 weeks, discount 0.9 | 0.737 | 0.659 | 0.74 |

Bit-identical on macOS/CPU and on an A40 pod. Shrinking the window past 6 keeps
buying per-turn share, but 4 weeks starts truncating the shipment-delay tail of
a decision's real consequences, so 6 is the causal-horizon choice rather than
the argmax of this table.

Reproduce:

```bash
python scripts/replay_live_y_credit_window.py --out artifacts/live_y_domain_randomized_grpo_v1/credit_window_replay.json
```

`test_unbounded_unnormalized_path_reproduces_the_archived_advantages` pins that
`--credit-window 0` still reproduces the advantages the archived run actually
optimized, to within 1e-9, so the refactor did not quietly redefine the baseline.

## GPU verification

One update on an A40 (pod `003n55c0oesf3q`, ~$0.09, artifacts in
`artifacts/live_y_domain_randomized_grpo_v1/credit_window_smoke/`), one training
seed, group 4:

| Metric | Value | Reading |
|---|---:|---|
| `valid_actions` / `total_actions` | 144 / 144 | format unaffected by the change |
| `protocol_clean_rate` | 1.0 | — |
| `trainable_actions` | 144 | windowed, normalized advantages do not collapse to zero |
| `scored_tokens` | 214 | 1.49 tokens per decision, i.e. the order's digits — was ~8 |
| `clip_fraction` | 0.121 | the clip engages; under the mean-logprob ratio it effectively never did |
| `loss` | 0.0276 | finite |

This is a plumbing check, not a result: one update on one seed says nothing
about whether the corrected objective learns.

Two efficiency findings surfaced in the same log and are **not** fixed:

- `PrefixKVCache` fell back on 31 of 72 lookups with
  `AttributeError: 'LinearAttentionLayer' object has no attribute 'batch_repeat_interleave'`
  on Qwen3.5's hybrid linear-attention layers (transformers 5.14.1). Correct but
  slow; it will inflate the cost of the real run.
- A `use_cache=True is incompatible with gradient checkpointing` warning appears
  around the evaluation rollouts. `load_policy` here enables gradient
  checkpointing and sets `config.use_cache = False` and never restores it for
  generation, unlike `train_qwen35_live_y_rl.py`. Worth confirming generation is
  actually using a KV cache before paying for 20 updates of rollouts.

## What is still open

1. **The corrected training run has not been done.** 12–20 updates, group 6–8,
   mixed buckets. Until it runs, nothing here is a claim about GRPO's ceiling on
   this task — only about why the archived run could not have worked.
2. **`make_tasks` still hardcodes `in_distribution`.** Root cause 4 is diagnosed,
   not fixed; mixing the buckets belongs with the paid run.
3. **Evaluation samples at temperature 0.7.** Greedy decoding would remove free
   variance from a 16-seed benchmark.
4. **No value model.** The group baseline is the weakest remaining component: it
   compares a member against groupmates in different states. A learned `V(s_t)`
   is the principled fix and is what SAO (arXiv 2607.07508) uses; its
   asynchronous-infrastructure half solves a throughput problem this project does
   not have at one-GPU scale, and adopting it wholesale is not warranted, but its
   value model and token-level clipping both target real defects listed above.
   Token-level clipping has landed; the value head has not.
