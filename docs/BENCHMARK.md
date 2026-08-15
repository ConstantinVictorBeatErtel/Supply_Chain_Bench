# SupplyChainBench Beer Distribution benchmark

## Research question

Can a language-model agent control a delayed, partially observed supply chain
when its actions have consequences several weeks later and the demand/supply
dynamics are not fully disclosed?

Beer Distribution is Environment #1 of SupplyChainBench. The benchmark controls
one wholesaler seat; retailers, distributor, and factory are deterministic
counterparties. A result is attributable to the controlled model rather than to
sampling from several independently failing model agents.

## Frozen standard protocol

`standard` is the existing live-Y capacity-400 research protocol:

- Y topology, wholesaler control, 36 decision weeks, and three settlement weeks.
- Order delay 1, shipment delay 2, order range 0–128.
- Holding cost 0.5 per unit-week; backlog cost 1.0 per unit-week.
- Factory capacity 400 is hidden from the model; the four existing demand
  buckets contain four fixed SHA-derived seeds each.
- The model receives only its local state, delayed incoming order, supply-line
  summary, and bounded own history. It never receives evaluator traces,
  hindsight actions, or another role's private state.

The committed standard seeds and source artifacts are unchanged. The Verifiers
Hub Tier-5 capacity-22 protocol and the older 100-seed serial benchmark are
legacy tracks and must not be combined with this board.

## Interface and score

The action is exactly one JSON object, `{"quantity": INTEGER}`, with an integer
from 0 through 128. A malformed action terminates the episode as a protocol
failure; it is never clamped or silently repaired.

For weekly local inventory `I_t` and backlog `B_t`,

`c_t = 0.5 I_t + 1.0 B_t`.

Episode cost includes decision weeks, settlement weeks, and terminal inventory
position exposure. Standard normalized score is

`100 × mean_s(C*_s) / mean_s(C_policy,s)`.

`C*` is the committed feasible hindsight-search reference for that seed. It is
an upper bound on the true optimum, not a claim of exact global optimality.
Runs with protocol failures retain a diagnostic clean-subset score but are
unranked. Ranked results require all 16 expected seeds and 16 clean episodes.

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

Standard uses the historical manifest. New seeds derive from

`SHA256("supplychainbench|beer-distribution|1.0.0|suite|phase|index")[:16]`.

The evaluator records suite/version, exact seeds, git commit, provider
configuration (never credentials), timestamp, action trace, protocol failures,
and reference metadata. Re-running a deterministic baseline with the same
configuration produces the same action trace and cost. API model outputs remain
provider-dependent and are reported as such.

Use `python -m supplychainbench.eval --help` for the supported provider URI and
result options. Use `python -m supplychainbench.leaderboard` to validate all
result files and regenerate the JSON, Markdown, and chart outputs.
