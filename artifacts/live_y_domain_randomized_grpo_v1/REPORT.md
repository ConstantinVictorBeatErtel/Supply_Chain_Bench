# `live-y-domain-randomized-grpo-v1`

## Status

This is a research extension, not the frozen v0.2.0 Tier-5 Y leaderboard.
RL is the intended headline method: the existing critic-free multi-turn
group-relative PPO/GRPO-style trainer is extended with per-turn return-to-go
advantages. SFT is optional, self-generated format-normalization scaffolding
only; it is not policy supervision and is not the headline method.

The CPU implementation and regression tests are complete. The historical
adapter recovery and the new GPU smoke/full run are blocked because Runpod
rejected two bounded starts of stopped pod `s7bbri3e3zilb5` for insufficient
account balance. No GPU pod started and no new Runpod spend accrued. Therefore
this report does not claim a recovered adapter, a trained endpoint, or model
evaluation numbers.

The prescribed smoke was attempted once locally and stopped before model load
with `CUDA is not available`; see `smoke_result.json`. It did not generate a
trajectory or adapter and incurred no spend.

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

The evaluation runner evaluates untrained base, recovered historical RL,
format-scaffold diagnostic, final per-turn GRPO, naive base-stock, and adaptive
base-stock under identical prompts, parsers, decoding, scenarios, and seeds.
It records paired per-seed local/team costs, stderr, the exact score formula,
protocol/format failure rates, bullwhip ratio, normalized order volatility,
weekly costs, return-to-go, advantages, hashes, GPU time, and exact API cost.
The fixed score formula is:

`100 * naive_mean_local_cost / (naive_mean_local_cost + policy_mean_local_cost)`

The current artifact directory contains protocol metadata and is intentionally
not populated with invented model results. CPU reference rows, when generated,
are labeled reference-only and cannot select a checkpoint.

The generated CPU reference-only rows (`cpu_reference_results.json`) are:

| Bucket | Naive base-stock local cost ± stderr | Adaptive base-stock local cost ± stderr |
|---|---:|---:|
| In-distribution | 6275.00 ± 73.34 | 228.25 ± 20.09 |
| Canonical held-out step | 6152.00 ± 0.00 | 196.00 ± 0.00 |
| Shifted mean / doubled variance | 10932.25 ± 694.19 | 2442.50 ± 567.01 |
| Burst-and-collapse | 6461.25 ± 139.39 | 267.63 ± 32.54 |

These are not Qwen results and are not used for selection.
