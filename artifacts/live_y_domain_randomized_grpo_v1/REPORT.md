# `live-y-domain-randomized-grpo-v1`

## Status

This is a research extension, not the frozen v0.2.0 Tier-5 Y leaderboard.
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
- Topology, capacity, rationing, costs, initial conditions, delays, horizon,
  and native settlement are unchanged. The environment extension is labeled
  with this protocol ID rather than v0.2.0.

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

## Evaluation and reporting

The evaluation runner evaluated the untrained base, final per-turn GRPO, naive
base-stock, and adaptive base-stock under identical prompts, parsers, decoding,
scenarios, and seeds. Recovered historical RL is unavailable, and format SFT
was skipped by the predeclared 95% protocol gate.
It records paired per-seed local/team costs, stderr, the exact score formula,
protocol/format failure rates, bullwhip ratio, normalized order volatility,
weekly costs, return-to-go, advantages, hashes, GPU time, and exact API cost.
The fixed score formula is:

`100 * naive_mean_local_cost / (naive_mean_local_cost + policy_mean_local_cost)`

The model artifacts and fixed research results are in `runpod_final/` and
`evaluations/`; the CPU reference rows remain labeled reference-only.

Overall fixed evaluation across 16 seeds:

| Model | Local cost ± stderr | System cost ± stderr | Score |
|---|---:|---:|---:|
| Untrained base | 1538.47 ± 312.27 | 20478.78 ± 5412.61 | 82.89 |
| Final per-turn GRPO | 1120.72 ± 240.84 | 4680.13 ± 559.29 | 86.93 |

Per-bucket local cost ± stderr / exact score:

| Bucket | Base | Final GRPO |
|---|---:|---:|
| In-distribution | 916.88 ± 97.58 / 87.25 | 652.50 ± 26.19 / 90.58 |
| Canonical held-out step | 1146.00 ± 160.74 / 84.30 | 784.25 ± 90.76 / 88.69 |
| Shifted mean / doubled variance | 3196.75 ± 820.94 / 77.37 | 2446.75 ± 598.78 / 81.71 |
| Burst-and-collapse | 894.25 ± 76.33 / 87.84 | 599.38 ± 71.45 / 91.51 |

Both model evaluations had 0% protocol and format failure. Scores use
`100 * naive_mean_local_cost / (naive_mean_local_cost + policy_mean_local_cost)`
with the naive reference mean computed over the same bucket seeds.

Runpod billing was $0.4520515278 total: $0.3905525552 for the A40 run and
$0.0614989726 for the bounded recovery-pod inventory. Both pods were stopped
after transfer; no pods remained running.

The generated CPU reference-only rows (`cpu_reference_results.json`) are:

| Bucket | Naive base-stock local cost ± stderr | Adaptive base-stock local cost ± stderr |
|---|---:|---:|
| In-distribution | 6275.00 ± 73.34 | 228.25 ± 20.09 |
| Canonical held-out step | 6152.00 ± 0.00 | 196.00 ± 0.00 |
| Shifted mean / doubled variance | 10932.25 ± 694.19 | 2442.50 ± 567.01 |
| Burst-and-collapse | 6461.25 ± 139.39 | 267.63 ± 32.54 |

These are not Qwen results and are not used for selection.
