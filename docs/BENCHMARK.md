# SupplyChainBench Beer Distribution benchmark

## Research question

Can a language-model agent control a delayed, partially observed supply chain
when its actions have consequences several weeks later and the demand/supply
dynamics are not fully disclosed?

Beer Distribution is Environment #1 of SupplyChainBench. The benchmark controls
one wholesaler seat; retailers, distributor, and factory are deterministic
counterparties. A result is attributable to the controlled model rather than to
sampling from several independently failing model agents.

## Current stochastic v2 protocol

`live-y-domain-randomized-grpo-v2` is the current playable and model-evaluation
protocol:

- Y topology, wholesaler control, 36 decision weeks, and three settlement weeks.
- Order delay 1, shipment delay 2, order range 0–128, and factory capacity 400.
- Customer demand is newly sampled for every retailer and operational week.
  Retailer orders carry that variation to the wholesaler rather than collapsing
  its incoming signal to a constant.
- Each game receives a fresh seed. The model benchmark uses 16 fixed, held-out,
  SHA-derived v2 seeds. Training and evaluation both construct the game with
  `training_spec`; browser play is parity-tested against the same v2 dynamics.
- The model receives only its local state, delayed incoming order, supply-line
  summary, and bounded own history. It never receives the demand law, capacity,
  evaluator traces, hindsight actions, or another role's private state.

The action is exactly one JSON object, `{"quantity": INTEGER}`, with an integer
from 0 through 128. A malformed action terminates the episode as a protocol
failure; it is never clamped or silently repaired. Ranked results require all
16 expected seeds and 16 protocol-clean episodes.

The v2 normalized score is paired by seed:

`100 × sum_s(C_best-found,s) / sum_s(C_policy,s)`.

The saved reference is the lowest feasible cost found through policy grids,
model warm starts, and coordinate descent. It is not a claim of exact global
optimality. Uncertainty is a 100,000-resample paired bootstrap over the 16 seeds
with a fixed bootstrap seed.

| Rank | Model | Score | 95% CI | Clean episodes |
| ---: | --- | ---: | ---: | ---: |
| 1 | Muse Spark 1.2 | 51.37 | 46.39–55.93 | 16/16 |
| 2 | Grok 4.6 | 46.78 | 44.56–48.95 | 16/16 |
| 3 | GLM-5.3-Flash | 44.17 | 39.18–49.47 | 16/16 |
| 4 | GPT-5.6 Sol | 43.81 | 37.61–50.62 | 16/16 |
| 5 | Qwen3.5-4B GRPO | 38.42 | 35.66–41.69 | 16/16 |
| 6 | GPT-5.6 Luna | 18.51 | 15.31–23.21 | 16/16 |
| 7 | Qwen3.5-4B (untrained) | 11.78 | 9.41–14.72 | 16/16 |
| — | Claude Opus 5 | 54.81* | 52.39–57.07* | 15/16 |

\* Diagnostic clean-subset result. Opus had one genuine protocol failure and
is not ranked.

The committed leaderboard, coverage failures, action traces, and hindsight
reference are in
[`artifacts/live_y_domain_randomized_grpo_v2/evaluations/`](../artifacts/live_y_domain_randomized_grpo_v2/evaluations/).

## Hidden-dynamics suites

Each suite has 16 deterministic evaluation seeds and an explicit ground-truth
schedule retained by the evaluator only.

| Suite | Hidden mechanism | Primary adaptation measure |
| --- | --- | --- |
| `standard` | Existing hidden demand/capacity contract | Standard normalized score |
| `demand_shift` | Seeded low/high Poisson regime change | Pre/post regret and recovery |
| `unknown_lead_time` | Seeded order/shipping delays | First-third vs final-third regret |
| `capacity_shock` | Permanent seeded capacity reduction | Post-shock regret AUC/recovery |
| `supply_disruption` | Temporary zero-production interval | Post-disruption regret AUC/recovery |
| `held_out_dynamics` | Demand, delays, and capacity outside training ranges | Adaptation improvement |

For event week `k`, six-week pre/post means are calculated from the weekly
controlled-role costs. Weekly regret is `r_t = c_t - c_t^aware`, where the
aware reference knows the law and schedule but not future random draws. The
post-event regret AUC is

`Σ[t=k..H] max(0, r_t)`.

Recovery time is the first week beginning at or after `k` whose next three-week
mean regret is no greater than the pre-event mean plus
`max(1, 0.1 × |pre_event_mean_regret|)`. If no such window exists, recovery is
`null` and `recovery_censored` is true. Non-event suites compare the first and
final thirds of the episode. Stress-suite normalized scores are suite-local and
must not be compared numerically with the standard score.

## Seeds and reproducibility

The 16 held-out benchmark seeds are frozen in
[`experiments/live_y_domain_randomized_grpo_v2/seed_manifest.json`](../experiments/live_y_domain_randomized_grpo_v2/seed_manifest.json)
and derive from

`SHA256("live-y-domain-randomized-grpo-v2|evaluation|index:08d")[:16]`.

The evaluator records suite/version, exact seeds, git commit, provider
configuration (never credentials), timestamp, action trace, protocol failures,
and reference metadata. Re-running a deterministic baseline with the same
configuration produces the same action trace and cost. API model outputs remain
provider-dependent and are reported as such.

Use `python -m supplychainbench.eval --help` for the supported provider URI and
result options. Use `python -m supplychainbench.leaderboard` to validate all
result files and regenerate the JSON, Markdown, and chart outputs.
