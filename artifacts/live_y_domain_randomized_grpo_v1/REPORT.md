# `live-y-domain-randomized-grpo-v1`

## Status

**Capacity-22 archive.** This report documents the original live-Y research run
under factory capacity **22** (hindsight-perfect mean ≈ **720.88**). The
**current** public/live-Y board uses capacity **400** (perfect ≈ **287.22**)
and lives in [`../live_y_capacity_400/`](../live_y_capacity_400/) with the
feasible-supply design note in
[`../../experiments/live_y_feasible_supply_grpo_v2/DESIGN.md`](../../experiments/live_y_feasible_supply_grpo_v2/DESIGN.md).

This is a research extension, not the frozen v0.2.0 Tier-5 Y Hub leaderboard.
RL is the intended headline method: the existing critic-free multi-turn
group-relative PPO/GRPO-style trainer is extended with per-turn return-to-go
advantages. SFT is optional, self-generated format-normalization scaffolding
only; it is not policy supervision and is not the headline method.

The research smoke and fixed two-update GPU run completed on one A40. The
historical adapter was not recovered: the old pod had no network volume
attached, and the volume was in a placement with no available A40. It remains
explicitly unavailable and is not conflated with the new endpoint.

The base model produced 100% protocol-clean development output in the smoke,
so the optional SFT scaffold was skipped as unnecessary. The A40 smoke had
144 valid actions, finite nonzero loss, and a saved/reloaded adapter. The full
run had 2,304 valid actions across 64 trajectories.
The full job wrote the adapter and both update metrics before its shell ended;
the optional final development JSON was not emitted. The verified adapter was
then evaluated separately on all 16 fixed research seeds below.

## Frozen research design

- Training demand is `episode_randomized_y_poisson_v1`: one SHA-256-derived
  latent λ sampled Uniform(2, 8) at episode construction, fixed for all 36
  operational weeks; retailer A and B draw independent Poisson counts from
  that λ.
- Every rollout in a group receives the same ScenarioSpec, λ, demand seed and
  draws, parameter draws, and counterparty streams. Policy-dependent states and
  actions may diverge.
- The canonical 4→8 step is held out, as are the shifted mean/variance and
  burst/collapse buckets. Fixed seeds are in `seed_manifest.json` and are
  disjoint from the repository's existing manifests.
- Topology, rationing, costs, initial conditions, delays, horizon, and native
  settlement matched Tier-5 Y. **This archived board used capacity 22**; the
  current research override is capacity 400. The environment extension is
  labeled with this protocol ID rather than v0.2.0.

## Training contract

The fixed endpoint is two updates, group size four, seeds 0–7 then 8–15, 64
trajectories maximum and 2,304 decision turns maximum. It uses bf16,
unquantized `Qwen/Qwen3.5-4B`, LoRA rank/alpha 16/16 over q/k/v/o/gate/up/down,
learning rate `5e-6`, temperature `0.7`, top-p `0.95`, and maximum completion
budget 192 tokens. A one-seed smoke must pass before expansion.

For clean rollouts, `G[i,t]` is the negative local wholesaler cost from turn
`t` through operational costs, native settlement, and terminal exposure. The
same-timestep group mean is subtracted; no variance normalization is used.
Protocol failure adds `-100000` at the failure turn, propagating to preceding
return-to-go values. System/team cost is reporting-only.

## Evaluation and reporting (capacity-22 archive)

Evaluations store per-episode demand manifests (`research_exogenous` / `demand`
traces), actions, and local costs. Hindsight search then computes a feasible
perfect local cost for each of the 16 fixed seeds under the same CRN demand and
scripted counterparties. The published score is:

`100 * hindsight_perfect_mean_local_cost / policy_mean_local_cost`

Adaptive base-stock is retained as a reporting heuristic only. On this
**capacity-22** seed set its mean local cost is **783.59**, while hindsight
perfect averages **720.88** (~1.09×). A score of 100% means matching the
hindsight-perfect reference; lower percentages mean higher cost.

Dashboard models (research prompt, rescored vs perfect, **capacity 22**):

| Model | Local cost ± stderr | Perfect-cost percentage | Clean |
|---|---:|---:|---:|
| Hindsight perfect reference | 720.88 | 100.00% | 16/16 |
| GPT-5.6 Luna | 1577.56 ± 412.22 | 45.70% | 16/16 |
| Final per-turn GRPO | 1706.81 ± 277.24 | 42.24% | 16/16 |
| Untrained base | 1934.84 ± 299.83 | 37.26% | 16/16 |
| DeepSeek V4 Flash | 1831.42 ± 407.82 | 33.06% | 12/16 |
| Laguna S 2.1 | 1132.88 ± 179.72 | 16.68% | 4/16 |

Artifacts (archive):
`evaluations/hindsight_perfect_costs.json`,
`evaluations/perfect_cost_leaderboard.json`, and
`docs/assets/live-y-domain-randomized-benchmark.svg`.

Current capacity-400 board:
`artifacts/live_y_capacity_400/evaluations/` and
`docs/assets/live-y-capacity-400-benchmark.svg`.

Runpod billing was $0.5996907949 total: $0.5381918224 for the A40 run and
$0.0614989726 for the bounded recovery-pod inventory. The A40 was briefly
resumed for final artifact verification and stopped again; no pods remain
running.

The generated CPU reference-only rows (`cpu_reference_results.json`) are:

| Bucket | Naive base-stock local cost ± stderr | Adaptive base-stock local cost ± stderr |
|---|---:|---:|
| In-distribution | 6275.00 ± 73.34 | 228.25 ± 20.09 |
| Canonical held-out step | 6152.00 ± 0.00 | 196.00 ± 0.00 |
| Shifted mean / doubled variance | 10932.25 ± 694.19 | 2442.50 ± 567.01 |
| Burst-and-collapse | 6461.25 ± 139.39 | 267.63 ± 32.54 |

These are not Qwen results and are not used for selection.
